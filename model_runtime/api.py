"""conversation-service GPU 작업 API 클라이언트 (게이트웨이 경유, Bearer GPU_WORKER_TOKEN)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from . import __version__


class LeaseLostError(Exception):
    """임대가 넘어갔다(409 LEASE_LOST). 그 작업은 버린다."""


@dataclass(frozen=True)
class Job:
    job_id: str
    type: str
    lease_token: str
    input: dict[str, Any]


class WorkerApi:
    def __init__(self, client: httpx.Client) -> None:
        self._client = client

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
        self._post(
            f"/{job.job_id}/result",
            {
                "leaseToken": job.lease_token,
                "result": result,
                "modelInfo": model_info,
                "durationMs": duration_ms,
            },
        )

    def fail(self, job: Job, code: str, message: str, retryable: bool) -> None:
        self._post(
            f"/{job.job_id}/fail",
            {
                "leaseToken": job.lease_token,
                "code": code,
                "message": message[:1000],
                "retryable": retryable,
            },
        )

    def _post(self, path: str, body: dict[str, Any]) -> None:
        response = self._client.post(path, json=body)
        if response.status_code == 409:
            raise LeaseLostError(path)
        response.raise_for_status()

    def close(self) -> None:
        self._client.close()
