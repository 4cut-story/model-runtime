"""작업 루프: 가져가기 → 모델 준비 → 실행(heartbeat) → 결과·실패 보고. 한 번에 작업 하나만 한다."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx

from .api import Job, LeaseLostError, WorkerApi
from .backends import REGIONS, Backends, JobInputError, Reference, VramBusyError
from .backends.ollama import OllamaUnavailableError

log = logging.getLogger(__name__)

TYPES = ["TRANSCRIBE", "LLM", "IMAGE"]
MODEL_FOR = {"TRANSCRIBE": "WHISPER", "LLM": "LLM", "IMAGE": "SDXL"}


class Heartbeat(threading.Thread):
    """작업을 하는 동안 주기적으로 임대를 연장한다. 임대를 잃으면 lost가 켜진다."""

    def __init__(self, api: WorkerApi, job: Job, interval: float) -> None:
        super().__init__(daemon=True)
        self.api = api
        self.job = job
        self.interval = interval
        self.lost = threading.Event()
        self._stopped = threading.Event()

    def run(self) -> None:
        while not self._stopped.wait(self.interval):
            try:
                self.api.heartbeat(self.job)
            except LeaseLostError:
                log.warning("임대를 잃었다: job=%s", self.job.job_id)
                self.lost.set()
                return
            except httpx.HTTPError as e:
                log.warning("heartbeat 실패, 다음에 다시: %s", e)

    def stop(self) -> None:
        self._stopped.set()


class Worker:
    def __init__(
        self,
        api: WorkerApi,
        backends: Backends,
        uploader: httpx.Client,
        heartbeat_interval: float = 30.0,
        poll_min: float = 2.0,
        poll_max: float = 10.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api = api
        self.backends = backends
        self.uploader = uploader
        self.heartbeat_interval = heartbeat_interval
        self.poll_min = poll_min
        self.poll_max = poll_max
        self.sleep = sleep
        self.loaded: str | None = None
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self, max_polls: int | None = None) -> None:
        """작업이 없으면 2초부터 10초까지 늘려 가며 묻는다. 작업을 끝내면 바로 다음을 묻는다."""
        delay = self.poll_min
        polls = 0
        while self._running and (max_polls is None or polls < max_polls):
            polls += 1
            try:
                worked = self.run_once()
            except httpx.HTTPError as e:
                log.warning("API 호출 실패: %s", e)
                worked = False
            if worked:
                delay = self.poll_min
                continue
            self.sleep(delay)
            delay = min(delay * 2, self.poll_max)

    def run_once(self) -> bool:
        job = self.api.claim(TYPES, self.loaded)
        if job is None:
            return False
        log.info("작업 시작: job=%s type=%s", job.job_id, job.type)
        heartbeat = Heartbeat(self.api, job, self.heartbeat_interval)
        heartbeat.start()
        started = time.monotonic()
        try:
            result, model = self.execute(job)
        except VramBusyError as e:
            self._report_fail(job, heartbeat, "VRAM_BUSY", str(e), retryable=True)
            return True
        except OllamaUnavailableError as e:
            self._report_fail(job, heartbeat, "OLLAMA_UNAVAILABLE", str(e), retryable=True)
            return True
        except (JobInputError, KeyError, ValueError) as e:
            self._report_fail(job, heartbeat, "BAD_INPUT", str(e), retryable=False)
            return True
        except Exception as e:
            log.exception("작업 실패: job=%s", job.job_id)
            self._report_fail(job, heartbeat, "MODEL_ERROR", f"{type(e).__name__}: {e}", False)
            return True
        if heartbeat.lost.is_set():
            heartbeat.stop()
            return True
        duration = int((time.monotonic() - started) * 1000)
        # 보고를 다시 보내는 동안에도 임대가 끝나지 않게 heartbeat는 보고 뒤에 멈춘다.
        try:
            self.api.result(job, result, self.backends.model_info(model), duration)
            log.info("작업 완료: job=%s %dms", job.job_id, duration)
        except LeaseLostError:
            log.warning("결과를 냈지만 임대가 넘어갔다: job=%s", job.job_id)
        except httpx.HTTPError as e:
            log.error(
                "결과를 보내지 못했다. 임대가 끝나면 다시 배정된다: job=%s (%s)", job.job_id, e
            )
        finally:
            heartbeat.stop()
        return True

    def execute(self, job: Job) -> tuple[dict[str, Any], str]:
        model = MODEL_FOR.get(job.type)
        if model is None:
            raise JobInputError(f"모르는 작업 종류: {job.type}")
        self.switch_to(model)
        data = job.input
        if job.type == "TRANSCRIBE":
            return self.transcribe(data), model
        if job.type == "LLM":
            return self.backends.chat(data), model
        return self.draw_all(data["images"]), model

    def transcribe(self, data: dict[str, Any]) -> dict[str, Any]:
        """mode=part면 녹음 중 구간 전사(FR-015).

        아니면 전체 전사이고, prefixUrl이 있으면 미리 전사한 앞부분을 받아 잇는다.
        """
        urls = data["audioUrls"]
        language = data.get("language")
        if data.get("mode") == "part":
            return self.backends.transcribe_part(urls, language, float(data.get("fromSec") or 0.0))
        prefix = None
        if data.get("prefixUrl"):
            response = self.uploader.get(data["prefixUrl"])
            response.raise_for_status()
            prefix = response.json()
        return self.backends.transcribe(urls, language, prefix)

    def switch_to(self, model: str) -> None:
        """다른 모델 묶음이 올라가 있으면 먼저 내린다 (16GB VRAM에 하나만)."""
        if self.loaded == model:
            return
        if self.loaded is not None:
            self.backends.unload(self.loaded)
            self.loaded = None
        self.backends.prepare(model)
        self.loaded = model

    def draw_all(self, images: list[dict[str, Any]]) -> dict[str, Any]:
        uploaded = []
        embeds: dict[str, bytes | None] = {}
        computed: list[str] = []
        for image in images:
            refs = self.references(image.get("references") or [], embeds, computed)
            png = self.backends.draw(image, refs)
            response = self.uploader.put(
                image["uploadUrl"], content=png, headers={"Content-Type": "image/png"}
            )
            response.raise_for_status()
            uploaded.append(image["name"])
        result: dict[str, Any] = {"uploaded": uploaded}
        if computed:
            result["embeds"] = computed
        return result

    def references(
        self, items: list[dict[str, Any]], embeds: dict[str, bytes | None], computed: list[str]
    ) -> list[Reference]:
        """컷의 캐릭터 참조. 같은 작업에서 한 캐릭터는 한 번만 받거나 계산한다.

        embedUrl이 있으면 저장해 둔 embedding을 받는다. 없으면 참조 그림(imageUrl)으로 계산해
        embedUploadUrl에 올린다. 받거나 계산하지 못한 참조는 빼고 텍스트만으로 그린다.
        """
        refs = []
        for item in items:
            character = str(item["character"])
            if character not in embeds:
                embeds[character] = self._embed(item, computed)
            embed = embeds[character]
            if embed is None:
                continue
            region = str(item.get("region") or "full")
            if region not in REGIONS:
                raise JobInputError(f"참조 위치는 full, left, right, center 중 하나다: {region}")
            scale = min(max(float(item.get("scale", 0.6)), 0.0), 1.0)
            refs.append(Reference(character, embed, scale, region))
        return refs

    def _embed(self, item: dict[str, Any], computed: list[str]) -> bytes | None:
        character = item["character"]
        try:
            if item.get("embedUrl"):
                response = self.uploader.get(item["embedUrl"])
                response.raise_for_status()
                return response.content
            if not item.get("imageUrl"):
                return None
            response = self.uploader.get(item["imageUrl"])
            response.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("캐릭터 참조를 받지 못해 빼고 그린다: %s (%s)", character, e)
            return None
        embed = self.backends.encode_reference(response.content)
        if embed is None:
            return None
        if item.get("embedUploadUrl"):
            try:
                put = self.uploader.put(
                    item["embedUploadUrl"],
                    content=embed,
                    headers={"Content-Type": "application/octet-stream"},
                )
                put.raise_for_status()
                computed.append(character)
            except httpx.HTTPError as e:
                log.warning(
                    "embedding을 올리지 못했다. 다음 그림 때 다시 계산한다: %s (%s)", character, e
                )
        return embed

    def _report_fail(
        self, job: Job, heartbeat: Heartbeat, code: str, message: str, retryable: bool
    ) -> None:
        heartbeat.stop()
        if heartbeat.lost.is_set():
            return
        log.warning("실패 보고: job=%s code=%s retryable=%s", job.job_id, code, retryable)
        try:
            self.api.fail(job, code, message, retryable)
        except LeaseLostError:
            log.warning("실패를 보고했지만 임대가 넘어갔다: job=%s", job.job_id)
