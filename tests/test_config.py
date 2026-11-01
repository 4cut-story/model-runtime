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


def test_env_파일은_없는_값만_채운다(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('# 주석\nA=1\nB="two"\nbad line\nC=3\n', encoding="utf-8")
    env = {"C": "keep"}
    load_env_file(env_file, env)
    assert env == {"A": "1", "B": "two", "C": "keep"}
    load_env_file(tmp_path / "missing", env)
