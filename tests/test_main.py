from pathlib import Path

import pytest

from model_runtime import __main__ as entry
from model_runtime.backends.fake import FakeBackends
from model_runtime.config import Settings


def settings(backend: str) -> Settings:
    return Settings(api_url="https://x", token="t", worker_id="w", backend=backend)


def test_가짜_백엔드를_고른다() -> None:
    assert isinstance(entry.build_backends(settings("fake")), FakeBackends)
    with pytest.raises(ValueError, match="WORKER_BACKEND"):
        entry.build_backends(settings("cloud"))


def test_main은_설정을_읽고_워커를_돌린다(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "GPU_WORKER_API_URL=https://4cut.example\nGPU_WORKER_TOKEN=t\n", encoding="utf-8"
    )
    monkeypatch.delenv("GPU_WORKER_API_URL", raising=False)
    monkeypatch.delenv("GPU_WORKER_TOKEN", raising=False)
    ran: list[bool] = []
    monkeypatch.setattr(entry.Worker, "run", lambda self, max_polls=None: ran.append(True))
    assert entry.main() == 0
    assert ran == [True]


def test_IMAGE_MODEL이_flux면_FLUX_schnell_백엔드로_그린다() -> None:
    from model_runtime.backends.flux import FluxBackend

    s = Settings.from_env(
        {
            "GPU_WORKER_API_URL": "https://x",
            "GPU_WORKER_TOKEN": "t",
            "IMAGE_MODEL": "FLUX",
            "FLUX_QUANTIZED_DIR": "C:/flux-nf4",
            "FLUX_STEPS": "6",
        }
    )
    image = entry.build_image_backend(s)
    assert isinstance(image, FluxBackend)
    assert image.quantized_dir == "C:/flux-nf4"
    assert image.steps == 6
    assert image.encode_reference(b"png") is None
    assert image.model_info()["flux"] == "black-forest-labs/FLUX.1-schnell"


def test_IMAGE_MODEL이_틀리면_시작하지_않는다() -> None:
    s = Settings(api_url="https://x", token="t", worker_id="w", image_model="dalle")
    with pytest.raises(ValueError, match="IMAGE_MODEL"):
        entry.build_image_backend(s)
