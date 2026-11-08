"""FLUX가 안 되면 SDXL로 되돌리는 그림 백엔드 (사람이 GPU PC를 만질 수 없을 때를 위한 안전장치).

두 가지를 막는다.
1. 워커 안에서 FLUX가 실패한다(메모리 부족, 패키지 없음, 가중치 없음 등): 그 자리에서 FLUX를 내리고
   SDXL로 같은 컷을 다시 그린다. 그 뒤로는 계속 SDXL을 쓴다.
2. FLUX를 올리다 워커 프로세스가 통째로 죽는다(RAM이 모자라 OS가 끝냄): 올리기 전에 표시 파일을
   쓰고 다 올린 뒤 지운다. 다음에 워커가 시작할 때 표시 파일이 남아 있으면 FLUX를 건너뛰고 SDXL로
   시작한다. 표시 파일을 지우면 다시 FLUX를 시도한다.

작업 입력이 틀린 경우(JobInputError)는 모델 탓이 아니므로 되돌리지 않고 그대로 올린다.
CUDA 메모리 부족은 다른 프로그램이 VRAM을 잡은 때라 SDXL도 같으므로 VramBusyError로 바꿔 작업을
돌려보낸다. 세 번 이어지면 SDXL로 되돌린다.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import JobInputError, Reference, VramBusyError

log = logging.getLogger(__name__)

# VRAM 부족이 이만큼 이어지면 FLUX가 이 GPU에 맞지 않는 것으로 보고 SDXL로 되돌린다
MAX_OOMS = 3


class FallbackImage:
    def __init__(self, primary: Any, make_fallback: Callable[[], Any], marker: Path) -> None:
        self._primary: Any = primary
        self._make_fallback = make_fallback
        self._marker = marker
        self._fallback: Any = None
        self._primary_ready = False
        self._ooms = 0
        if marker.exists():
            log.warning("지난번 FLUX 준비 중 워커가 멈췄다(%s). SDXL로 그린다", marker)
            self._switch(None)

    @property
    def active(self) -> Any:
        return self._fallback if self._fallback is not None else self._primary

    @property
    def fell_back(self) -> bool:
        return self._fallback is not None

    def _switch(self, error: BaseException | None) -> None:
        if error is not None:
            log.error("FLUX가 실패해 SDXL로 되돌린다: %s", error)
        if self._primary is not None:
            try:
                self._primary.unload()
            except Exception:  # 내리기 실패는 되돌리기를 막지 않는다
                log.exception("FLUX 내리기 실패")
        self._primary = None
        self._fallback = self._make_fallback()

    def _prepare_primary(self) -> None:
        """처음 한 번만 표시 파일을 두고 FLUX를 올린다."""
        if self._primary_ready:
            return
        self._marker.parent.mkdir(parents=True, exist_ok=True)
        self._marker.write_text("FLUX 준비 중\n", encoding="utf-8")
        self._primary.load()
        self._marker.unlink(missing_ok=True)
        self._primary_ready = True

    def draw(self, image: dict[str, Any], references: list[Reference] | None = None) -> bytes:
        if self._fallback is None:
            try:
                self._prepare_primary()
                data: bytes = self._primary.draw(image, references)
                self._ooms = 0
                return data
            except JobInputError:
                raise
            except Exception as e:
                self._marker.unlink(missing_ok=True)
                if type(e).__name__ == "OutOfMemoryError" and self._ooms < MAX_OOMS:
                    # 다른 프로그램이 VRAM을 잡은 것이다. SDXL도 같으니 작업을 대기열로 돌려보낸다
                    self._ooms += 1
                    raise VramBusyError(f"FLUX VRAM 부족: {e}") from e
                # 그 밖의 실패, 또는 거듭된 VRAM 부족은 SDXL로 다시 그린다
                self._switch(e)
        data = self._fallback.draw(image, references)
        return data

    def encode_reference(self, image: bytes) -> bytes | None:
        data: bytes | None = self.active.encode_reference(image)
        return data

    def unload(self) -> None:
        self.active.unload()

    def model_info(self) -> dict[str, Any]:
        info: dict[str, Any] = dict(self.active.model_info())
        info["fellBackToSdxl"] = self.fell_back
        return info
