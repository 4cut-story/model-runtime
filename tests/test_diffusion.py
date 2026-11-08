import pytest

from model_runtime import __main__ as entry
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
