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
DEFAULT_POSE_ADAPTER = "TencentARC/t2i-adapter-openpose-sdxl-1.0"
DEFAULT_FLUX_MODEL = "black-forest-labs/FLUX.1-schnell"


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
    # 그린 뒤 CLIP으로 인물 성별·나이·동물을 확인하고 틀리면 다시 그린다 (VRAM 약 +0.9GB).
    # 회차 16~18에서 점수를 올리지 못해 기본은 끈다
    image_check: bool = False
    # 컷마다 인물 수·자리·키를 뼈대 그림으로 넣는 T2I-Adapter (VRAM 약 +0.2GB). 비우면 끈다
    pose_adapter: str = DEFAULT_POSE_ADAPTER
    pose_scale: float = 0.8
    pose_factor: float = 0.5
    # 그림 모델: "sdxl" 또는 "flux"(FLUX.1-schnell, 회차 28부터 시험).
    # flux면 SDXL_* 설정은 쓰지 않는다
    image_model: str = "sdxl"
    flux_model: str = DEFAULT_FLUX_MODEL
    # NF4로 줄인 transformer·T5를 저장해 둘 폴더. 비우면 시작할 때마다 다시 줄인다
    flux_quantized_dir: str = ""
    flux_steps: int = 4
    flux_guidance: float = 0.0
    # auto(bitsandbytes가 있으면 nf4) | nf4 | int8 | gguf | fp8 | nunchaku. 양자화 방식 비교용
    flux_quant: str = "auto"
    # gguf·fp8·nunchaku용 transformer 한 파일 경로 (예: flux1-schnell-Q8_0.gguf)
    flux_file: str = ""
    # 적은 step으로 그리게 하는 LoRA(예: dev 8 step 가속 LoRA). 폴더·파일 경로 또는 HF 저장소
    flux_lora: str = ""
    # flux_lora가 저장소일 때 그 안의 파일 이름
    flux_lora_weight: str = ""
    flux_lora_scale: float = 1.0
    # IMAGE_MODEL=sd35·qwen용. 비우거나 0·음수면 모델마다 기본값(backends/diffusion.py KINDS)
    diffusion_model: str = ""
    diffusion_steps: int = 0
    diffusion_guidance: float = -1.0
    # nf4 | nunchaku(qwen만, DIFFUSION_FILE 필요) | none
    diffusion_quant: str = "nf4"
    diffusion_file: str = ""
    # FLUX를 올리는 동안 두는 표시 파일. 워커가 시작할 때 남아 있으면 SDXL로 그린다
    flux_marker: str = ".flux-loading"
    # 대사 읽기(SPEECH 작업). 켜면 같은 PC의 GPT-SoVITS api_v2 서버(v4)를 부른다
    speech_enabled: bool = False
    tts_url: str = "http://127.0.0.1:9880"
    # 목소리 참조 음성 폴더: <목소리 ID>/<감정>.wav와 같은 이름의 .txt(녹음의 글)
    voices_dir: str = "voices"
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
            image_check=env.get("IMAGE_CHECK", "false").lower() == "true",
            pose_adapter=env.get("POSE_ADAPTER", DEFAULT_POSE_ADAPTER).strip(),
            pose_scale=float(env.get("POSE_SCALE", "0.8")),
            pose_factor=float(env.get("POSE_FACTOR", "0.5")),
            image_model=env.get("IMAGE_MODEL", "sdxl").strip().lower(),
            flux_model=env.get("FLUX_MODEL", DEFAULT_FLUX_MODEL).strip(),
            flux_quantized_dir=env.get("FLUX_QUANTIZED_DIR", "").strip(),
            flux_steps=int(env.get("FLUX_STEPS", "4")),
            flux_guidance=float(env.get("FLUX_GUIDANCE", "0")),
            flux_quant=env.get("FLUX_QUANT", "auto").strip().lower(),
            flux_file=env.get("FLUX_FILE", "").strip(),
            flux_lora=env.get("FLUX_LORA", "").strip(),
            flux_lora_weight=env.get("FLUX_LORA_WEIGHT", "").strip(),
            flux_lora_scale=float(env.get("FLUX_LORA_SCALE", "1")),
            diffusion_model=env.get("DIFFUSION_MODEL", "").strip(),
            diffusion_steps=int(env.get("DIFFUSION_STEPS", "0")),
            diffusion_guidance=float(env.get("DIFFUSION_GUIDANCE", "-1")),
            diffusion_quant=env.get("DIFFUSION_QUANT", "nf4").strip().lower(),
            diffusion_file=env.get("DIFFUSION_FILE", "").strip(),
            flux_marker=env.get("FLUX_MARKER", ".flux-loading").strip(),
            speech_enabled=env.get("SPEECH_ENABLED", "false").lower() == "true",
            tts_url=env.get("TTS_URL", "http://127.0.0.1:9880").strip().rstrip("/"),
            voices_dir=env.get("VOICES_DIR", "voices").strip(),
        )
