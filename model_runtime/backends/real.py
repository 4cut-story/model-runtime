"""GPU PC 백엔드. 한 번에 모델 묶음 하나만 VRAM에 둔다 (Whisper, Ollama LLM, SDXL)."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any

from . import Reference, VramBusyError, reference_kind
from .ollama import OllamaClient

GIB = 1024**3

# 모델 묶음별로 올리기 전에 남아 있어야 하는 VRAM (architecture.md GPU·VRAM 배분 추정치)
REQUIRED_VRAM = {"WHISPER": 7 * GIB, "LLM": 8 * GIB, "SDXL": 10 * GIB}


def free_vram() -> int | None:
    """GPU에 실제로 남은 VRAM(바이트). CUDA가 없으면 None.

    Windows(WDDM)에서는 torch.cuda.mem_get_info()가 다른 프로세스가 잡은 VRAM을 빼지 않고
    보고한다(nvidia-smi 2.9GB 남음일 때 13.7GB로 보고). 그래서 nvidia-smi와 같은 값을 주는
    NVML을 먼저 보고, 없으면 nvidia-smi, 그것도 없으면 torch 값을 쓴다.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    index = _gpu_index()
    for probe in (_nvml_free, _nvidia_smi_free):
        free = probe(index)
        if free is not None:
            return free
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def gpu_name() -> str | None:
    """보이는 첫 GPU 이름 (만화 제작 정보). CUDA가 없으면 None."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return str(torch.cuda.get_device_name(0))


def _gpu_index() -> int:
    """NVML은 CUDA_VISIBLE_DEVICES를 모르므로 보이는 첫 GPU의 실제 번호를 쓴다."""
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(first) if first.isdigit() else 0


def _nvml_free(index: int) -> int | None:
    try:
        import pynvml  # nvidia-ml-py
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            return int(pynvml.nvmlDeviceGetMemoryInfo(handle).free)
        finally:
            pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        return None


def _nvidia_smi_free(index: int) -> int | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 - 고정된 인자만 넘긴다
            [exe, f"--id={index}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        return int(out.strip().splitlines()[0]) * 1024**2  # MiB
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class RealBackends:
    def __init__(
        self,
        whisper: Any,
        sdxl: Any,
        ollama: OllamaClient,
        vram: Any = free_vram,
        depth: Any = None,
    ) -> None:
        self.whisper = whisper
        self._depth = depth
        self.sdxl = sdxl
        self.ollama = ollama
        self._vram = vram

    def transcribe(
        self,
        audio_urls: list[str],
        language: str | None,
        prefix: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if prefix is None:
            result: dict[str, Any] = self.whisper.transcribe(audio_urls, language)
        else:
            result = self.whisper.transcribe(audio_urls, language, prefix)
        return result

    def transcribe_part(
        self, audio_urls: list[str], language: str | None, from_sec: float
    ) -> dict[str, Any]:
        result: dict[str, Any] = self.whisper.transcribe_part(audio_urls, language, from_sec)
        return result

    def chat(self, job_input: dict[str, Any]) -> dict[str, Any]:
        return self.ollama.chat(job_input)

    def draw(self, image: dict[str, Any], references: list[Reference] | None = None) -> bytes:
        data: bytes = self.sdxl.draw(image, references)
        return data

    def depth(self, png: bytes) -> bytes:
        """라이브 넓은 그림의 깊이 지도(같은 크기 흑백 PNG, 밝을수록 가깝다)."""
        if self._depth is None:
            from .depth import DepthEstimator

            self._depth = DepthEstimator()
        data: bytes = self._depth.estimate(png)
        return data

    def encode_reference(self, image: bytes) -> bytes | None:
        data: bytes | None = self.sdxl.encode_reference(image)
        return data

    def reference_kind(self) -> str:
        return reference_kind(self.sdxl)

    def prepare(self, model: str) -> None:
        free = self._vram()
        need = REQUIRED_VRAM.get(model, 0)
        if model == "SDXL":
            # 그림 모델이 필요 VRAM을 정해 두었으면 그 값을 쓴다(klein 5GB)
            need = getattr(self.sdxl, "required_vram", None) or need
        if free is not None and free < need:
            raise VramBusyError(
                f"{model}에 필요한 VRAM {need / GIB:.0f}GB, 남은 VRAM {free / GIB:.1f}GB"
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
