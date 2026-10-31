"""환경 변수 설정. 비밀 값(GPU_WORKER_TOKEN, HF_TOKEN)은 GPU PC의 .env에만 둔다."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_env_file(path: Path, environ: dict[str, str] | None = None) -> None:
    """KEY=VALUE 형식의 .env를 읽어 아직 없는 환경 변수만 채운다."""
    target = os.environ if environ is None else environ
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        target.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    api_url: str
    token: str
    worker_id: str
    backend: str = "fake"
    ollama_url: str = "http://localhost:11434"
    whisper_model: str = "large-v3"
    sdxl_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    hf_token: str | None = None
    sdxl_cpu_offload: bool = True
    poll_min: float = 2.0
    poll_max: float = 10.0
    heartbeat_interval: float = 30.0
    request_timeout: float = 30.0

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Settings:
        env = dict(os.environ) if environ is None else environ
        missing = [k for k in ("GPU_WORKER_API_URL", "GPU_WORKER_TOKEN") if not env.get(k)]
        if missing:
            raise ValueError("환경 변수가 없다: " + ", ".join(missing))
        return cls(
            api_url=env["GPU_WORKER_API_URL"].rstrip("/"),
            token=env["GPU_WORKER_TOKEN"],
            worker_id=env.get("WORKER_ID", "gpu-pc-1"),
            backend=env.get("WORKER_BACKEND", "fake"),
            ollama_url=env.get("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            whisper_model=env.get("WHISPER_MODEL", "large-v3"),
            sdxl_model=env.get("SDXL_MODEL", "stabilityai/stable-diffusion-xl-base-1.0"),
            hf_token=env.get("HF_TOKEN") or None,
            sdxl_cpu_offload=env.get("SDXL_CPU_OFFLOAD", "true").lower() != "false",
        )
