"""실행: python -m model_runtime (GPU PC에서는 WORKER_BACKEND=real)."""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path
from types import FrameType

import httpx

from .api import WorkerApi
from .backends import Backends
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
    from .backends.sdxl import SdxlBackend
    from .backends.whisper import WhisperBackend

    ollama = OllamaClient.create(settings.ollama_url)
    # 이전 실행이 남긴 모델을 내리고 시작한다
    ollama.unload()
    return RealBackends(
        WhisperBackend(settings.whisper_model, settings.hf_token),
        SdxlBackend(
            settings.sdxl_model,
            settings.sdxl_cpu_offload,
            lora=settings.sdxl_lora,
            steps=settings.sdxl_steps,
            guidance=settings.sdxl_guidance,
        ),
        ollama,
    )


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
