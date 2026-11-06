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


DEFAULT_SDXL_IP_ADAPTER = "h94/IP-Adapter:sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"
DEFAULT_SDXL_LORA = "ByteDance/SDXL-Lightning:sdxl_lightning_8step_lora.safetensors"


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
    # 화자 분리 최소 화자 수. 대화 녹음이라 2. 짧은 녹음에서 비슷한 두 목소리를
    # 한 사람으로 합치는 일이 잦았다. 0이면 정하지 않는다
    diarize_min_speakers: int = 2
    # 768px은 16GB에 다 올라간다. offload는 컷마다 GPU↔CPU 이동이 커서 느리다
    sdxl_cpu_offload: bool = False
    # 적은 step용 LoRA "<HF 레포>:<파일>". 비우면 SDXL base만 쓴다 (steps 30, guidance 6 권장)
    sdxl_lora: str = DEFAULT_SDXL_LORA
    # from_pretrained variant. fp16 파일이 없는 체크포인트(Animagine XL 4 등)는 비운다
    sdxl_variant: str = "fp16"
    sdxl_steps: int = 8
    sdxl_guidance: float = 0.0
    # 모델을 내릴 때 지우지 않고 시스템 RAM에 옮겨 둔다 (SDXL, 정렬 모델). RAM 약 7GB를 더 쓴다
    keep_in_ram: bool = True
    # 캐릭터 참조용 IP-Adapter "<HF 레포>:<레포 안 경로>". 비우면 참조를 무시한다
    sdxl_ip_adapter: str = DEFAULT_SDXL_IP_ADAPTER
    # IP-Adapter step별 강도: 앞쪽 early_steps는 early_scale 이하, 뒤쪽은 캐릭터 강도 + late_boost.
    # early_steps=0이면 캐릭터 강도 그대로 (Plus-Face로 되돌릴 때)
    sdxl_ip_early_scale: float = 0.2
    sdxl_ip_early_steps: int = 4
    sdxl_ip_late_boost: float = 0.2
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
            diarize_min_speakers=int(env.get("DIARIZE_MIN_SPEAKERS", "2")),
            sdxl_cpu_offload=env.get("SDXL_CPU_OFFLOAD", "false").lower() == "true",
            sdxl_lora=env.get("SDXL_LORA", DEFAULT_SDXL_LORA).strip(),
            sdxl_variant=env.get("SDXL_VARIANT", "fp16").strip(),
            sdxl_steps=int(env.get("SDXL_STEPS", "8")),
            sdxl_guidance=float(env.get("SDXL_GUIDANCE", "0")),
            keep_in_ram=env.get("MODEL_KEEP_IN_RAM", "true").lower() == "true",
            sdxl_ip_adapter=env.get("SDXL_IP_ADAPTER", DEFAULT_SDXL_IP_ADAPTER).strip(),
            sdxl_ip_early_scale=float(env.get("SDXL_IP_EARLY_SCALE", "0.2")),
            sdxl_ip_early_steps=int(env.get("SDXL_IP_EARLY_STEPS", "4")),
            sdxl_ip_late_boost=float(env.get("SDXL_IP_LATE_BOOST", "0.2")),
        )
