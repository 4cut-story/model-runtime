"""모델 실행 백엔드. fake(로컬·CI)와 real(GPU PC)이 같은 인터페이스를 쓴다."""

from __future__ import annotations

from typing import Any, Protocol


class VramBusyError(Exception):
    """남은 VRAM이 모자라 모델을 올릴 수 없다. 작업을 대기열로 돌려보낸다."""


class JobInputError(Exception):
    """작업 입력이 틀렸다. 다시 해도 같으므로 재시도하지 않는다."""


class Backends(Protocol):
    def transcribe(self, audio_urls: list[str], language: str | None) -> dict[str, Any]:
        """{segments: [{speaker, start, end, text}], language}"""
        ...

    def chat(self, job_input: dict[str, Any]) -> dict[str, Any]:
        """{content: "<JSON 문자열>", evalCount}"""
        ...

    def draw(self, image: dict[str, Any]) -> bytes:
        """PNG 바이트"""
        ...

    def prepare(self, model: str) -> None:
        """모델 묶음(WHISPER, LLM, SDXL)을 쓸 수 있게 한다. VRAM이 모자라면 VramBusyError."""
        ...

    def unload(self, model: str) -> None:
        """모델 묶음을 VRAM에서 내린다."""
        ...

    def model_info(self, model: str) -> dict[str, Any]: ...
