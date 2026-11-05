import subprocess
import sys
import types
from typing import Any

import pytest

from model_runtime.backends import VramBusyError, real
from model_runtime.backends.real import GIB, RealBackends, free_vram


class Stub:
    def __init__(self) -> None:
        self.unloaded = 0

    def transcribe(self, urls: list[str], language: str | None) -> dict[str, Any]:
        return {"segments": [], "language": language}

    def draw(self, image: dict[str, Any], references: Any = None) -> bytes:
        self.references = references
        return b"png"

    def encode_reference(self, image: bytes) -> bytes | None:
        return b"embed:" + image

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


def test_그림과_참조_인코딩을_SDXL에_넘긴다() -> None:
    from model_runtime.backends import Reference

    backends, _, sdxl, _ = make(None)
    ref = Reference("c", b"e", 0.6)
    assert backends.draw({"prompt": "p"}, [ref]) == b"png"
    assert sdxl.references == [ref]
    assert backends.encode_reference(b"img") == b"embed:img"


# --- 남은 VRAM 읽기: Windows(WDDM)에서는 torch 값이 다른 프로세스 사용량을 빼지 않는다 ---


def fake_torch(monkeypatch: pytest.MonkeyPatch, free: int) -> None:
    cuda = types.SimpleNamespace(is_available=lambda: True, mem_get_info=lambda: (free, 16 * GIB))
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))


def fake_nvml(monkeypatch: pytest.MonkeyPatch, free: int | None) -> list[str]:
    calls: list[str] = []

    class NVMLError(Exception):
        pass

    def init() -> None:
        calls.append("init")
        if free is None:
            raise NVMLError("no driver")

    module = types.SimpleNamespace(
        NVMLError=NVMLError,
        nvmlInit=init,
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetHandleByIndex=lambda i: calls.append(f"handle{i}") or i,
        nvmlDeviceGetMemoryInfo=lambda h: types.SimpleNamespace(free=free),
    )
    monkeypatch.setitem(sys.modules, "pynvml", module)
    return calls


def test_남은_VRAM은_torch보다_NVML_값을_먼저_쓴다(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch(monkeypatch, 13 * GIB)
    calls = fake_nvml(monkeypatch, 3 * GIB)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert free_vram() == 3 * GIB
    assert calls == ["init", "handle1", "shutdown"]


def test_NVML이_없으면_nvidia_smi_값을_쓴다(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch(monkeypatch, 13 * GIB)
    fake_nvml(monkeypatch, None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(real.shutil, "which", lambda name: "nvidia-smi")
    seen: list[list[str]] = []

    def run(args: list[str], **_: Any) -> Any:
        seen.append(args)
        return types.SimpleNamespace(stdout="2970\n")

    monkeypatch.setattr(real.subprocess, "run", run)
    assert free_vram() == 2970 * 1024**2
    assert seen[0][1] == "--id=0"


def test_NVML도_nvidia_smi도_없으면_torch_값을_쓴다(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch(monkeypatch, 13 * GIB)
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(real.shutil, "which", lambda name: None)
    assert free_vram() == 13 * GIB


def test_nvidia_smi가_실패하면_모른다(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(real.shutil, "which", lambda name: "nvidia-smi")

    def run(args: list[str], **_: Any) -> Any:
        raise subprocess.CalledProcessError(9, args)

    monkeypatch.setattr(real.subprocess, "run", run)
    assert real._nvidia_smi_free(0) is None


def test_CUDA_VISIBLE_DEVICES가_UUID면_0번_GPU를_본다(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")
    assert real._gpu_index() == 0
