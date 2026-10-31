"""GPU PC 백엔드. 한 번에 모델 묶음 하나만 VRAM에 둔다 (Whisper, Ollama LLM, SDXL)."""

from __future__ import annotations

from typing import Any

from . import VramBusyError
from .ollama import OllamaClient

GIB = 1024**3

# 모델 묶음별로 올리기 전에 남아 있어야 하는 VRAM (architecture.md GPU·VRAM 배분 추정치)
REQUIRED_VRAM = {"WHISPER": 7 * GIB, "LLM": 8 * GIB, "SDXL": 10 * GIB}


def free_vram() -> int | None:
    """남은 VRAM(바이트). CUDA가 없으면 None."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return int(free)


class RealBackends:
    def __init__(
        self, whisper: Any, sdxl: Any, ollama: OllamaClient, vram: Any = free_vram
    ) -> None:
        self.whisper = whisper
        self.sdxl = sdxl
        self.ollama = ollama
        self._vram = vram

    def transcribe(self, audio_urls: list[str], language: str | None) -> dict[str, Any]:
        result: dict[str, Any] = self.whisper.transcribe(audio_urls, language)
        return result

    def chat(self, job_input: dict[str, Any]) -> dict[str, Any]:
        return self.ollama.chat(job_input)

    def draw(self, image: dict[str, Any]) -> bytes:
        data: bytes = self.sdxl.draw(image)
        return data

    def prepare(self, model: str) -> None:
        free = self._vram()
        need = REQUIRED_VRAM.get(model, 0)
        if free is not None and free < need:
            raise VramBusyError(
                f"{model}에 필요한 VRAM {need // GIB}GB, 남은 VRAM {free / GIB:.1f}GB"
            )

    def unload(self, model: str) -> None:
        if model == "WHISPER":
            self.whisper.unload()
        elif model == "SDXL":
            self.sdxl.unload()
        elif model == "LLM":
            self.ollama.unload()

    def model_info(self, model: str) -> dict[str, Any]:
        if model == "WHISPER":
            info: dict[str, Any] = self.whisper.model_info()
            return info
        if model == "SDXL":
            info = self.sdxl.model_info()
            return info
        return {"ollama": self.ollama.last_model}
