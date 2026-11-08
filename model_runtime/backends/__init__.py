"""모델 실행 백엔드. fake(로컬·CI)와 real(GPU PC)이 같은 인터페이스를 쓴다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class VramBusyError(Exception):
    """남은 VRAM이 모자라 모델을 올릴 수 없다. 작업을 대기열로 돌려보낸다."""


class JobInputError(Exception):
    """작업 입력이 틀렸다. 다시 해도 같으므로 재시도하지 않는다."""


REGIONS = ("full", "left", "right", "center")


@dataclass(frozen=True)
class Reference:
    """IP-Adapter 캐릭터 참조 하나 (캐릭터 DB, conversation-service REQ-018).

    embed는 safetensors 바이트(키 "embeds", [1, 1, 257, 1280] fp16)다. region이 left/right면
    그 절반에만 얼굴을 넣는다(두 명이 나오는 컷). center는 가운데 60%에만 넣는다(캐릭터 한 명과
    참조 없는 다른 사람이 함께 나오는 컷: 전체에 걸면 다른 사람까지 그 얼굴이 됐다).
    """

    character: str
    embed: bytes
    scale: float
    region: str = "full"


def reference_kind(image_backend: object) -> str:
    """그림 백엔드의 참조 방식. reference_kind()가 없는 백엔드(SDXL·FLUX)는 "embed"로 본다."""
    method = getattr(image_backend, "reference_kind", None)
    return str(method()) if callable(method) else "embed"


class Backends(Protocol):
    def transcribe(
        self,
        audio_urls: list[str],
        language: str | None,
        prefix: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """{segments: [{speaker, start, end, text}], language}. prefix는 녹음 중 미리 전사한 앞부분
        ({cut, language, segments: [{start, end, text, words}]}). 있으면 cut 뒤만 전사해 잇는다."""
        ...

    def transcribe_part(
        self, audio_urls: list[str], language: str | None, from_sec: float
    ) -> dict[str, Any]:
        """녹음 중 구간 전사 (FR-015). {segments(단어 포함), cut, language, durationSec, skipped}"""
        ...

    def chat(self, job_input: dict[str, Any]) -> dict[str, Any]:
        """{content: "<JSON 문자열>", evalCount}"""
        ...

    def draw(self, image: dict[str, Any], references: list[Reference] | None = None) -> bytes:
        """PNG 바이트. references가 있으면 IP-Adapter로 그 캐릭터 얼굴을 유지한다."""
        ...

    def encode_reference(self, image: bytes) -> bytes | None:
        """참조 그림 → IP-Adapter embedding(safetensors). IP-Adapter를 끄면 None.

        reference_kind()가 "image"인 모델(FLUX.2 klein)은 그림 바이트를 그대로 돌려준다."""
        ...

    def reference_kind(self) -> str:
        """ "embed": IP-Adapter embedding을 쓴다(저장해 두고 다시 쓴다). "image": 참조 그림을 그대로
        입력으로 받는다(저장한 embedding을 받지도 올리지도 않는다). "none": 참조를 쓰지 않는다."""
        ...

    def prepare(self, model: str) -> None:
        """모델 묶음(WHISPER, LLM, SDXL)을 쓸 수 있게 한다. VRAM이 모자라면 VramBusyError."""
        ...

    def unload(self, model: str) -> None:
        """모델 묶음을 VRAM에서 내린다."""
        ...

    def model_info(self, model: str) -> dict[str, Any]: ...
