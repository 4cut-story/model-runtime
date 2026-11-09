from pathlib import Path

import pytest

from model_runtime import __main__ as entry
from model_runtime.backends import Reference, reference_kind
from model_runtime.backends.diffusion import DiffusionBackend
from model_runtime.backends.image_fallback import FallbackImage
from model_runtime.config import Settings

BASE = {"GPU_WORKER_API_URL": "https://x", "GPU_WORKER_TOKEN": "t"}


def test_IMAGE_MODEL이_sd35면_SD_3_5_Large를_기본값으로_그린다() -> None:
    """FR-006: SD 3.5 Large는 28 step, CFG 3.5로 negative prompt까지 넘긴다."""
    wrapped = entry.build_image_backend(Settings.from_env({**BASE, "IMAGE_MODEL": "sd35"}))
    assert isinstance(wrapped, FallbackImage)
    image = wrapped.active
    assert isinstance(image, DiffusionBackend)
    assert (image.model_name, image.steps, image.guidance, image.quant) == (
        "stabilityai/stable-diffusion-3.5-large",
        28,
        3.5,
        "nf4",
    )
    args = image.call_args({"prompt": "p", "negativePrompt": "n", "width": 512})
    assert args == {
        "prompt": "p",
        "width": 512,
        "height": 768,
        "num_inference_steps": 28,
        "guidance_scale": 3.5,
        "max_sequence_length": 512,
        "negative_prompt": "n",
    }
    assert image.encode_reference(b"png") is None
    assert image.model_info()["sd35"] == "stabilityai/stable-diffusion-3.5-large"


def test_Qwen_Image_nunchaku_8_step은_negative_prompt를_쓰지_않는다() -> None:
    """FR-006: Lightning 8 step(true_cfg 1)이면 negative prompt 없이 그린다."""
    s = Settings.from_env(
        {
            **BASE,
            "IMAGE_MODEL": "qwen",
            "DIFFUSION_QUANT": "nunchaku",
            "DIFFUSION_FILE": "q.safetensors",
            "DIFFUSION_STEPS": "8",
            "DIFFUSION_GUIDANCE": "1",
        }
    )
    image = entry.build_image_backend(s).active
    assert (image.model_name, image.steps, image.guidance) == ("Qwen/Qwen-Image", 8, 1.0)
    args = image.call_args({"prompt": "p", "negativePrompt": "n"})
    assert args["true_cfg_scale"] == 1.0
    assert "negative_prompt" not in args
    assert DiffusionBackend("qwen").steps == 50


@pytest.mark.parametrize(
    ("kind", "quant", "file", "match"),
    [
        ("dalle", "nf4", "", "IMAGE_MODEL"),
        ("sd35", "int3", "", "DIFFUSION_QUANT"),
        ("sd35", "nunchaku", "x", "nunchaku"),
        ("qwen", "nunchaku", "", "nunchaku"),
    ],
)
def test_틀린_그림_모델_설정이면_시작하지_않는다(
    kind: str, quant: str, file: str, match: str
) -> None:
    """FR-006: 모델 이름·양자화 방식이 틀리면 워커가 시작하지 않는다."""
    with pytest.raises(ValueError, match=match):
        DiffusionBackend(kind, quant=quant, single_file=file)


@pytest.mark.parametrize(
    ("kind", "model", "steps", "guidance"),
    [
        ("klein", "black-forest-labs/FLUX.2-klein-4B", 4, 1.0),
        ("zimage", "Tongyi-MAI/Z-Image-Turbo", 9, 0.0),
        ("chroma", "lodestones/Chroma1-HD", 28, 4.0),
    ],
)
def test_회차30_새_그림_모델은_기본값으로_그린다(
    kind: str, model: str, steps: int, guidance: float
) -> None:
    """FR-006: IMAGE_MODEL=klein·zimage·chroma도 SDXL 안전장치와 함께 쓴다."""
    wrapped = entry.build_image_backend(Settings.from_env({**BASE, "IMAGE_MODEL": kind}))
    assert isinstance(wrapped, FallbackImage)
    image = wrapped.active
    assert (image.model_name, image.steps, image.guidance) == (model, steps, guidance)
    args = image.call_args({"prompt": "p", "negativePrompt": "n"})
    assert args["guidance_scale"] == guidance
    assert "true_cfg_scale" not in args
    # klein은 negative prompt를 받지 않고, zimage는 CFG 0이라 쓰지 않는다
    assert ("negative_prompt" in args) == (kind == "chroma")


def test_REQ018_klein은_참조_그림을_그대로_받고_캐릭터마다_한_장만_넘긴다() -> None:
    """REQ-018: FLUX.2 klein은 IP-Adapter embedding 대신 참조 그림을 입력으로 받는다."""
    klein = DiffusionBackend("klein")
    assert klein.reference_kind() == "image"
    assert klein.encode_reference(b"png-a") == b"png-a"
    refs = [Reference(c, f"png-{c}".encode(), 0.6) for c in "aabcdef"]
    assert klein.reference_images(refs) == [b"png-a", b"png-b", b"png-c", b"png-d"]
    assert klein.model_info()["references"] == "image"
    zimage = DiffusionBackend("zimage")
    assert zimage.reference_kind() == "none"
    assert zimage.encode_reference(b"png") is None
    assert zimage.reference_images(refs) == []


def test_REQ018_참조_방식은_감싼_그림_백엔드를_따른다() -> None:
    """REQ-018: SDXL·FLUX처럼 reference_kind가 없는 백엔드는 embedding 방식으로 본다."""
    klein = FallbackImage(DiffusionBackend("klein"), lambda: object(), Path("/nonexistent"))
    assert klein.reference_kind() == "image"
    assert reference_kind(object()) == "embed"


def test_klein은_필요_VRAM을_5GB로_정하고_다른_모델은_정하지_않는다() -> None:
    """FR-008: klein nf4는 최고 3.68GiB를 써서 SDXL 기준 10GB 대신 5GB만 남아 있으면 그린다."""
    from model_runtime.backends.diffusion import DiffusionBackend

    assert DiffusionBackend("klein").required_vram == 5 * 1024**3
    assert DiffusionBackend("sd35").required_vram is None
