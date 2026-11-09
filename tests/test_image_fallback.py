"""FLUX가 실패하면 SDXL로 되돌리는 안전장치 (사람이 GPU PC를 만질 수 없을 때)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from model_runtime.backends import JobInputError, VramBusyError
from model_runtime.backends.image_fallback import MAX_OOMS, FallbackImage


class OutOfMemoryError(RuntimeError):
    """torch.OutOfMemoryError와 이름이 같은 가짜."""


class Model:
    def __init__(self, name: str, fail: Exception | None = None, load_fail: bool = False) -> None:
        self.name = name
        self.fail = fail
        self.load_fail = load_fail
        self.loads = 0
        self.unloads = 0

    def load(self) -> None:
        self.loads += 1
        if self.load_fail:
            raise ImportError("bitsandbytes 없음")

    def draw(self, image: dict[str, Any], references: Any = None) -> bytes:
        if self.fail is not None:
            raise self.fail
        return self.name.encode()

    def encode_reference(self, image: bytes) -> bytes | None:
        return None if self.name == "flux" else b"embed"

    def unload(self) -> None:
        self.unloads += 1

    def model_info(self) -> dict[str, Any]:
        return {"model": self.name}


def make(tmp_path: Path, flux: Model) -> tuple[FallbackImage, list[Model]]:
    made: list[Model] = []

    def sdxl() -> Model:
        made.append(Model("sdxl"))
        return made[-1]

    return FallbackImage(flux, sdxl, tmp_path / "m" / ".flux-loading"), made


def test_FLUX가_되면_FLUX로_그리고_표시_파일을_지운다(tmp_path: Path) -> None:
    flux = Model("flux")
    image, made = make(tmp_path, flux)
    assert image.draw({"prompt": "p"}) == b"flux"
    assert image.draw({"prompt": "p"}) == b"flux"
    assert flux.loads == 1
    assert made == []
    assert not (tmp_path / "m" / ".flux-loading").exists()
    assert image.encode_reference(b"png") is None
    assert image.model_info() == {"model": "flux", "fellBackToSdxl": False}


def test_FLUX를_올리지_못하면_같은_컷을_SDXL로_그리고_계속_SDXL을_쓴다(tmp_path: Path) -> None:
    flux = Model("flux", load_fail=True)
    image, made = make(tmp_path, flux)
    assert image.draw({"prompt": "p"}) == b"sdxl"
    assert image.draw({"prompt": "p"}) == b"sdxl"
    assert len(made) == 1
    assert flux.unloads == 1
    assert image.encode_reference(b"png") == b"embed"
    assert image.model_info()["fellBackToSdxl"] is True
    image.unload()
    assert made[0].unloads == 1


def test_지난번_FLUX를_올리다_워커가_죽었으면_SDXL로_시작한다(tmp_path: Path) -> None:
    marker = tmp_path / "m" / ".flux-loading"
    marker.parent.mkdir()
    marker.write_text("FLUX 준비 중\n", encoding="utf-8")
    flux = Model("flux")
    image, _made = make(tmp_path, flux)
    assert image.draw({"prompt": "p"}) == b"sdxl"
    assert flux.loads == 0


def test_작업_입력이_틀리면_되돌리지_않고_그대로_올린다(tmp_path: Path) -> None:
    image, made = make(tmp_path, Model("flux", fail=JobInputError("prompt 없음")))
    with pytest.raises(JobInputError):
        image.draw({})
    assert made == []


def test_VRAM_부족은_작업을_돌려보내고_거듭되면_SDXL로_되돌린다(tmp_path: Path) -> None:
    image, made = make(tmp_path, Model("flux", fail=OutOfMemoryError("CUDA")))
    for _ in range(MAX_OOMS):
        with pytest.raises(VramBusyError):
            image.draw({"prompt": "p"})
    assert made == []
    assert image.draw({"prompt": "p"}) == b"sdxl"


def test_필요_VRAM은_지금_그리는_모델을_따르고_SDXL로_되돌리면_SDXL_기준(tmp_path: Path) -> None:
    """FR-008: klein이 정한 필요 VRAM을 넘기고, SDXL로 되돌린 뒤에는 SDXL 기준(None)을 쓴다."""
    flux = Model("flux", load_fail=True)
    flux.required_vram = 5  # type: ignore[attr-defined]
    image, _ = make(tmp_path, flux)
    assert image.required_vram == 5
    image.draw({"prompt": "p"})
    assert image.required_vram is None
