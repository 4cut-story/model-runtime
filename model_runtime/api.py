"""conversation-service GPU 작업 API 클라이언트 (게이트웨이 경유, Bearer GPU_WORKER_TOKEN)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from . import __version__

log = logging.getLogger(__name__)

# 결과·실패 보고를 다시 보낼 간격(초). 합쳐 약 1분이면 게이트웨이·서비스 재시작을 넘긴다.
REPORT_RETRY_DELAYS = (2.0, 4.0, 8.0, 16.0, 32.0)


class LeaseLostError(Exception):
    """임대가 넘어갔다(409 LEASE_LOST). 그 작업은 버린다."""


@dataclass(frozen=True)
class Job:
    job_id: str
    type: str
    lease_token: str
    input: dict[str, Any]


class WorkerApi:
    def __init__(
        self,
        client: httpx.Client,
        retry_delays: tuple[float, ...] = REPORT_RETRY_DELAYS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._retry_delays = retry_delays
        self._sleep = sleep

    @classmethod
    def create(cls, api_url: str, token: str, worker_id: str, timeout: float) -> WorkerApi:
        client = httpx.Client(
            base_url=api_url + "/internal/gpu-jobs",
            headers={"Authorization": f"Bearer {token}", "X-Worker-Id": worker_id},
            timeout=timeout,
        )
        return cls(client)

    def claim(self, types: list[str], loaded_model: str | None) -> Job | None:
        body: dict[str, Any] = {"types": types, "version": __version__}
        if loaded_model:
            body["loadedModel"] = loaded_model
        response = self._client.post("/claim", json=body)
        if response.status_code == 204:
            return None
        response.raise_for_status()
        data = response.json()
        return Job(data["jobId"], data["type"], data["leaseToken"], data.get("input") or {})

    def heartbeat(self, job: Job, progress: int | None = None) -> None:
        body: dict[str, Any] = {"leaseToken": job.lease_token}
        if progress is not None:
            body["progress"] = progress
        self._post(f"/{job.job_id}/heartbeat", body)

    def result(
        self, job: Job, result: dict[str, Any], model_info: dict[str, Any], duration_ms: int
    ) -> None:
        self._report(
            f"/{job.job_id}/result",
            {
                "leaseToken": job.lease_token,
                "result": result,
                "modelInfo": model_info,
                "durationMs": duration_ms,
            },
        )

    def fail(self, job: Job, code: str, message: str, retryable: bool) -> None:
        self._report(
            f"/{job.job_id}/fail",
            {
                "leaseToken": job.lease_token,
                "code": code,
                "message": message[:1000],
                "retryable": retryable,
            },
        )

    def _report(self, path: str, body: dict[str, Any]) -> None:
        """결과·실패 보고는 몇 분짜리 작업의 끝이라 잠깐의 통신 오류로 버리지 않는다.

        서버는 같은 임대 토큰으로 다시 낸 결과를 그대로 성공으로 받으므로,
        응답만 잃은 경우에도 다시 보내도 된다.
        409(임대 잃음)와 4xx는 다시 보내도 같으니 바로 올린다.
        """
        for delay in self._retry_delays:
            try:
                self._post(path, body)
                return
            except httpx.HTTPStatusError as e:
                if not _retryable(e.response.status_code):
                    raise
                log.warning(
                    "보고 실패(%d), %.0f초 뒤 다시: %s", e.response.status_code, delay, path
                )
            except httpx.TransportError as e:
                log.warning("보고 통신 오류, %.0f초 뒤 다시: %s (%s)", delay, path, e)
            self._sleep(delay)
        self._post(path, body)

    def _post(self, path: str, body: dict[str, Any]) -> None:
        response = self._client.post(path, json=body)
        if response.status_code == 409:
            raise LeaseLostError(path)
        response.raise_for_status()

    def close(self) -> None:
        self._client.close()


def _retryable(status: int) -> bool:
    return status == 429 or status >= 500
