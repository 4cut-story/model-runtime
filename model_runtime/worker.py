"""작업 루프: 가져가기 → 모델 준비 → 실행(heartbeat) → 결과·실패 보고. 한 번에 작업 하나만 한다."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx

from . import __version__
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
        device: Callable[[], str | None] = lambda: None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.api = api
        self.backends = backends
        self.uploader = uploader
        self.heartbeat_interval = heartbeat_interval
        self.poll_min = poll_min
        self.poll_max = poll_max
        self.sleep = sleep
        self.device = device
        self.clock = clock
        self.loaded: str | None = None
        # 작업 하나의 제작 정보(모델 로드 시간, 컷별 시간). 결과 보고의 modelInfo에 붙인다.
        self._stats: dict[str, Any] = {}
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
        started = self.clock()
        self._stats = {}
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
        duration = int((self.clock() - started) * 1000)
        # 보고를 다시 보내는 동안에도 임대가 끝나지 않게 heartbeat는 보고 뒤에 멈춘다.
        try:
            self.api.result(job, result, self.model_info(model), duration)
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

    def model_info(self, model: str) -> dict[str, Any]:
        """백엔드의 모델·조건에 GPU, 모델 로드 시간, 컷별 시간을 더한다 (만화 제작 정보)."""
        info = dict(self.backends.model_info(model))
        info["runtime"] = __version__
        try:
            gpu = self.device()
        except Exception:  # 제작 정보 때문에 결과 보고를 막지 않는다
            gpu = None
        if gpu:
            info["gpu"] = gpu
        info.update(self._stats)
        return info

    def switch_to(self, model: str) -> None:
        """다른 모델 묶음이 올라가 있으면 먼저 내린다 (16GB VRAM에 하나만)."""
        if self.loaded == model:
            return
        started = self.clock()
        if self.loaded is not None:
            self.backends.unload(self.loaded)
            self.loaded = None
        self.backends.prepare(model)
        self.loaded = model
        self._stats["loadMs"] = int((self.clock() - started) * 1000)

    def draw_all(self, images: list[dict[str, Any]]) -> dict[str, Any]:
        uploaded = []
        embeds: dict[str, bytes | None] = {}
        computed: list[str] = []
        panels: list[dict[str, Any]] = []
        self._stats["panels"] = panels
        for image in images:
            refs = self.references(image.get("references") or [], embeds, computed)
            started = self.clock()
            png = self.backends.draw(image, refs)
            stats = _panel_stats(image, refs, int((self.clock() - started) * 1000))
            edit = image.get("edit")
            if edit and self.backends.reference_kind() == "image":
                png = self._edit(image, edit, png, embeds, computed, stats)
            panels.append(stats)
            response = self.uploader.put(
                image["uploadUrl"], content=png, headers={"Content-Type": "image/png"}
            )
            response.raise_for_status()
            uploaded.append(image["name"])
        result: dict[str, Any] = {"uploaded": uploaded}
        if computed:
            result["embeds"] = computed
        return result

    def _edit(
        self,
        image: dict[str, Any],
        edit: dict[str, Any],
        base: bytes,
        embeds: dict[str, bytes | None],
        computed: list[str],
        stats: dict[str, Any],
    ) -> bytes:
        """2단 그리기의 2차: 1차 그림을 첫 입력 그림으로 넣고 edit.prompt로 고쳐 그린다.

        물건을 드는 컷은 1차에 빈손 자세를, 2차에 물건 하나만 더한다. 두 사람 컷은 1차에 앞사람을,
        2차에 뒷사람을 더한다. 한 번에 그리면 물건이나 인물이 둘로 그려졌다(회차 29 콘티 16~25판).
        참조 그림을 그대로 받는 모델(FLUX.2 klein)에서만 쓴다. 아니면 1차 그림을 그대로 쓴다.
        """
        refs = [Reference("__base__", base, 1.0, "full")]
        refs += self.references(edit.get("references") or [], embeds, computed)
        second = {**image, "prompt": str(edit["prompt"])}
        started = self.clock()
        png = self.backends.draw(second, refs)
        stats["editMs"] = int((self.clock() - started) * 1000)
        return png

    def references(
        self, items: list[dict[str, Any]], embeds: dict[str, bytes | None], computed: list[str]
    ) -> list[Reference]:
        """컷의 캐릭터 참조. 같은 작업에서 한 캐릭터는 한 번만 받거나 계산한다.

        embedUrl이 있으면 저장해 둔 embedding을 받는다. 없으면 참조 그림(imageUrl)으로 계산해
        embedUploadUrl에 올린다. 받거나 계산하지 못한 참조는 빼고 텍스트만으로 그린다.
        참조 그림을 그대로 받는 모델(reference_kind "image", FLUX.2 klein)은 저장한 embedding을
        쓰지도 올리지도 않고 늘 imageUrl을 받는다. 저장한 embedding은 SDXL IP-Adapter 것이다.
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
        raw_image = self.backends.reference_kind() == "image"
        try:
            if item.get("embedUrl") and not raw_image:
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
        if item.get("embedUploadUrl") and not raw_image:
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


def _panel_stats(image: dict[str, Any], refs: list[Reference], ms: int) -> dict[str, Any]:
    """컷 하나를 그린 조건과 시간. 참조 계산·업로드 시간은 빼고 그리기만 잰다."""
    stats: dict[str, Any] = {"name": image.get("name"), "ms": ms}
    for key in ("seed", "width", "height"):
        if image.get(key) is not None:
            stats[key] = image[key]
    if refs:
        stats["references"] = len(refs)
    return stats
