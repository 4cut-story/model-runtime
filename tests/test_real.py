from typing import Any

import pytest

from model_runtime.backends import VramBusyError
from model_runtime.backends.real import GIB, RealBackends, free_vram


class Stub:
    def __init__(self) -> None:
        self.unloaded = 0

    def transcribe(self, urls: list[str], language: str | None) -> dict[str, Any]:
        return {"segments": [], "language": language}

    def draw(self, image: dict[str, Any]) -> bytes:
        return b"png"

    def unload(self) -> None:
        self.unloaded += 1

    def model_info(self) -> dict[str, Any]:
        return {"stub": True}

    def chat(self, job_input: dict[str, Any]) -> dict[str, Any]:
        return {"content": "{}"}

    last_model = "m"


def make(free: int | None) -> tuple[RealBackends, Stub, Stub, Stub]:
    whisper, sdxl, ollama = Stub(), Stub(), Stub()
    return RealBackends(whisper, sdxl, ollama, vram=lambda: free), whisper, sdxl, ollama  # type: ignore[arg-type]


def test_VRAM이_모자라면_VRAM_BUSY() -> None:
    backends, *_ = make(5 * GIB)
    with pytest.raises(VramBusyError):
        backends.prepare("SDXL")
    backends.prepare("UNKNOWN")


def test_VRAM을_모르면_그대로_진행한다() -> None:
    backends, whisper, sdxl, ollama = make(None)
    backends.prepare("SDXL")
    assert backends.transcribe([], "ko")["language"] == "ko"
    assert backends.draw({}) == b"png"
    assert backends.chat({}) == {"content": "{}"}
    for model in ("WHISPER", "SDXL", "LLM"):
        backends.unload(model)
        assert backends.model_info(model)
    assert (whisper.unloaded, sdxl.unloaded, ollama.unloaded) == (1, 1, 1)


def test_CUDA가_없으면_남은_VRAM을_모른다() -> None:
    assert free_vram() is None
