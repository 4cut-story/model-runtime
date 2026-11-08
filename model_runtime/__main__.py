"""실행: python -m model_runtime (GPU PC에서는 WORKER_BACKEND=real)."""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path
from types import FrameType
from typing import Any

import httpx

from .api import WorkerApi
from .backends import Backends
from .backends.diffusion import KINDS
from .backends.fake import FakeBackends
from .config import Settings, load_env_file
from .worker import Worker


def build_backends(settings: Settings) -> Backends:
    if settings.backend == "fake":
        return FakeBackends()
    if settings.backend != "real":
        raise ValueError(f"WORKER_BACKEND는 fake 또는 real이어야 한다: {settings.backend}")
    from .backends.ollama import OllamaClient
    from .backends.real import RealBackends
    from .backends.whisper import WhisperBackend

    ollama = OllamaClient.create(settings.ollama_url)
    # 이전 실행이 남긴 모델을 내리고 시작한다
    ollama.unload()
    return RealBackends(
        WhisperBackend(
            settings.whisper_model,
            settings.hf_token,
            keep_in_ram=settings.keep_in_ram,
            min_speakers=settings.diarize_min_speakers,
        ),
        build_image_backend(settings),
        ollama,
    )


def build_image_backend(settings: Settings) -> Any:
    """IMAGE_MODEL에 맞는 그림 백엔드. 작업의 모델 묶음 이름은 계약대로 SDXL을 그대로 쓴다."""
    if settings.image_model == "flux":
        from .backends.flux import FluxBackend
        from .backends.image_fallback import FallbackImage

        flux = FluxBackend(
            settings.flux_model,
            settings.flux_quantized_dir,
            steps=settings.flux_steps,
            guidance=settings.flux_guidance,
            quant=settings.flux_quant,
            single_file=settings.flux_file,
            lora=settings.flux_lora,
            lora_weight=settings.flux_lora_weight,
            lora_scale=settings.flux_lora_scale,
            keep_in_ram=settings.keep_in_ram,
        )
        # FLUX가 실패하거나 올리다 워커가 죽으면 사람 손 없이 SDXL로 되돌린다
        return FallbackImage(flux, lambda: build_sdxl(settings), Path(settings.flux_marker))
    if settings.image_model in KINDS:
        from .backends.diffusion import DiffusionBackend
        from .backends.image_fallback import FallbackImage

        other = DiffusionBackend(
            settings.image_model,
            settings.diffusion_model,
            steps=settings.diffusion_steps,
            guidance=settings.diffusion_guidance,
            quant=settings.diffusion_quant,
            single_file=settings.diffusion_file,
            keep_in_ram=settings.keep_in_ram,
        )
        return FallbackImage(other, lambda: build_sdxl(settings), Path(settings.flux_marker))
    if settings.image_model != "sdxl":
        raise ValueError(
            f"IMAGE_MODEL은 sdxl, flux, {', '.join(KINDS)} 중 하나여야 한다: {settings.image_model}"
        )
    return build_sdxl(settings)


def build_sdxl(settings: Settings) -> Any:
    from .backends.ip_schedule import IpSchedule
    from .backends.sdxl import SdxlBackend

    return SdxlBackend(
        settings.sdxl_model,
        settings.sdxl_cpu_offload,
        lora=settings.sdxl_lora,
        variant=settings.sdxl_variant,
        steps=settings.sdxl_steps,
        guidance=settings.sdxl_guidance,
        keep_in_ram=settings.keep_in_ram,
        ip_adapter=settings.sdxl_ip_adapter,
        image_check=settings.image_check,
        pose_adapter=settings.pose_adapter,
        pose_scale=settings.pose_scale,
        pose_factor=settings.pose_factor,
        ip_schedule=IpSchedule(
            settings.sdxl_ip_early_scale,
            settings.sdxl_ip_early_steps,
            settings.sdxl_ip_late_boost,
        ),
    )


def _gpu_name() -> str | None:
    from .backends.real import gpu_name

    return gpu_name()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    load_env_file(Path.cwd() / ".env")
    settings = Settings.from_env()
    if settings.backend == "real" and not settings.hf_token:
        logging.getLogger(__name__).warning("HF_TOKEN이 없어 화자 분리 없이 전사한다")
    api = WorkerApi.create(
        settings.api_url, settings.token, settings.worker_id, settings.request_timeout
    )
    worker = Worker(
        api,
        build_backends(settings),
        httpx.Client(timeout=120.0),
        heartbeat_interval=settings.heartbeat_interval,
        poll_min=settings.poll_min,
        poll_max=settings.poll_max,
        device=_gpu_name if settings.backend == "real" else (lambda: None),
    )

    def shutdown(signum: int, frame: FrameType | None) -> None:
        worker.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    worker.run()
    api.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
