from pathlib import Path

import pytest

from model_runtime.config import Settings, load_env_file


def test_필수_환경변수가_없으면_시작하지_않는다() -> None:
    with pytest.raises(ValueError, match="GPU_WORKER_TOKEN"):
        Settings.from_env({"GPU_WORKER_API_URL": "https://4cut.example"})


def test_기본값과_환경변수를_읽는다() -> None:
    s = Settings.from_env(
        {
            "GPU_WORKER_API_URL": "https://4cut.example/",
            "GPU_WORKER_TOKEN": "t",
            "WORKER_BACKEND": "real",
            "SDXL_CPU_OFFLOAD": "false",
            "HF_TOKEN": "",
        }
    )
    assert s.api_url == "https://4cut.example"
    assert s.ollama_url == "http://localhost:11434"
    assert s.worker_id == "gpu-pc-1"
    assert s.backend == "real"
    assert s.sdxl_cpu_offload is False
    assert s.hf_token is None


def test_SDXL은_Lightning_8step_768용_기본값을_쓰고_환경변수로_바꾼다() -> None:
    base = {"GPU_WORKER_API_URL": "https://4cut.example", "GPU_WORKER_TOKEN": "t"}
    s = Settings.from_env(base)
    assert s.sdxl_cpu_offload is False
    assert s.sdxl_lora == "ByteDance/SDXL-Lightning:sdxl_lightning_8step_lora.safetensors"
    assert (s.sdxl_steps, s.sdxl_guidance) == (8, 0.0)
    assert s.sdxl_ip_adapter.endswith("ip-adapter-plus_sdxl_vit-h.safetensors")
    assert (s.sdxl_ip_early_scale, s.sdxl_ip_early_steps, s.sdxl_ip_late_boost) == (0.2, 4, 0.2)
    s = Settings.from_env(
        {
            **base,
            "SDXL_LORA": "",
            "SDXL_STEPS": "30",
            "SDXL_GUIDANCE": "6",
            "SDXL_CPU_OFFLOAD": "true",
        }
    )
    assert (s.sdxl_lora, s.sdxl_steps, s.sdxl_guidance, s.sdxl_cpu_offload) == ("", 30, 6.0, True)


def test_모델을_RAM에_옮겨_두는_설정은_기본으로_켜고_환경변수로_끈다() -> None:
    base = {"GPU_WORKER_API_URL": "https://4cut.example", "GPU_WORKER_TOKEN": "t"}
    assert Settings.from_env(base).keep_in_ram is True
    assert Settings.from_env({**base, "MODEL_KEEP_IN_RAM": "false"}).keep_in_ram is False


def test_env_파일은_없는_값만_채운다(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('# 주석\nA=1\nB="two"\nbad line\nC=3\n', encoding="utf-8")
    env = {"C": "keep"}
    load_env_file(env_file, env)
    assert env == {"A": "1", "B": "two", "C": "keep"}
    load_env_file(tmp_path / "missing", env)


def test_체크포인트_variant는_기본_fp16이고_비우면_variant_없이_읽는다() -> None:
    base = {"GPU_WORKER_API_URL": "https://4cut.example", "GPU_WORKER_TOKEN": "t"}
    assert Settings.from_env(base).sdxl_variant == "fp16"
    assert Settings.from_env({**base, "SDXL_VARIANT": ""}).sdxl_variant == ""


def test_화자_분리_최소_화자_수는_기본_2이고_0으로_끌_수_있다() -> None:
    base = {"GPU_WORKER_API_URL": "https://4cut.example", "GPU_WORKER_TOKEN": "t"}
    assert Settings.from_env(base).diarize_min_speakers == 2
    assert Settings.from_env({**base, "DIARIZE_MIN_SPEAKERS": "0"}).diarize_min_speakers == 0


def test_화자_분리에_최소_화자_수를_넘긴다() -> None:
    from model_runtime.backends.whisper import WhisperBackend

    assert WhisperBackend("m", None, min_speakers=2)._speaker_hint() == {"min_speakers": 2}
    assert WhisperBackend("m", None)._speaker_hint() == {}


def test_FR006_인물_확인은_끄고_뼈대_그림을_기본으로_쓴다() -> None:
    base = {"GPU_WORKER_API_URL": "https://4cut.example", "GPU_WORKER_TOKEN": "t"}
    s = Settings.from_env(base)
    assert s.image_check is False
    assert s.pose_adapter == "TencentARC/t2i-adapter-openpose-sdxl-1.0"
    assert (s.pose_scale, s.pose_factor) == (0.8, 0.5)
    s = Settings.from_env({**base, "POSE_ADAPTER": "", "IMAGE_CHECK": "true", "POSE_SCALE": "1"})
    assert (s.pose_adapter, s.image_check, s.pose_scale) == ("", True, 1.0)
