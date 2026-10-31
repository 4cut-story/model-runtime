import json
from typing import Any

import httpx
import respx

from model_runtime.api import Job, LeaseLostError
from model_runtime.backends import VramBusyError
from model_runtime.backends.fake import FakeBackends, tiny_png
from model_runtime.backends.ollama import OllamaUnavailableError
from model_runtime.worker import Heartbeat, Worker


class FakeApi:
    def __init__(self, jobs: list[Job | None]) -> None:
        self.jobs = jobs
        self.claims: list[str | None] = []
        self.results: list[tuple[str, dict[str, Any]]] = []
        self.fails: list[tuple[str, str, bool]] = []
        self.heartbeats = 0
        self.lose_lease = False

    def claim(self, types: list[str], loaded: str | None) -> Job | None:
        self.claims.append(loaded)
        if not self.jobs:
            return None
        return self.jobs.pop(0)

    def heartbeat(self, job: Job, progress: int | None = None) -> None:
        self.heartbeats += 1
        if self.lose_lease:
            raise LeaseLostError(job.job_id)

    def result(self, job: Job, result: dict[str, Any], info: dict[str, Any], ms: int) -> None:
        if self.lose_lease:
            raise LeaseLostError(job.job_id)
        self.results.append((job.job_id, result))

    def fail(self, job: Job, code: str, message: str, retryable: bool) -> None:
        if self.lose_lease:
            raise LeaseLostError(job.job_id)
        self.fails.append((job.job_id, code, retryable))


def worker(api: FakeApi, backends: Any = None, **kw: Any) -> Worker:
    return Worker(
        api,  # type: ignore[arg-type]
        backends or FakeBackends(),
        httpx.Client(),
        heartbeat_interval=kw.get("heartbeat", 30.0),
        sleep=kw.get("sleep", lambda s: None),
    )


def test_전사_LLM_그림_작업을_차례로_하고_모델을_바꿔_올린다() -> None:
    images = [
        {"name": f"panel-{i}", "prompt": "p", "uploadUrl": f"http://minio/put/{i}"}
        for i in range(1, 3)
    ]
    api = FakeApi(
        [
            Job("t", "TRANSCRIBE", "lt", {"audioUrls": ["http://a"], "language": "ko"}),
            Job("l", "LLM", "lt", {"jsonSchema": {"required": ["candidates"]}, "prompt": "p"}),
            Job("l2", "LLM", "lt", {"jsonSchema": {"required": ["script", "nameWarnings"]}}),
            Job("l3", "LLM", "lt", {"jsonSchema": {"required": ["title"]}}),
            Job("i", "IMAGE", "lt", {"images": images}),
        ]
    )
    backends = FakeBackends()
    with respx.mock:
        put = respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        w = worker(api, backends)
        w.run(max_polls=6)
    assert [r[0] for r in api.results] == ["t", "l", "l2", "l3", "i"]
    assert api.results[0][1]["language"] == "ko"
    assert json.loads(api.results[1][1]["content"])["candidates"][0]["kind"] == "FUNNY"
    assert "script" in json.loads(api.results[2][1]["content"])
    assert api.results[4][1] == {"uploaded": ["panel-1", "panel-2"]}
    assert put.call_count == 2
    assert put.calls.last.request.headers["Content-Type"] == "image/png"
    assert backends.loaded == ["WHISPER", "LLM", "SDXL"]
    assert backends.unloaded == ["WHISPER", "LLM"]
    # 같은 종류 작업을 먼저 받도록 올려 둔 모델을 알려 준다
    assert api.claims[:3] == [None, "WHISPER", "LLM"]


class Failing(FakeBackends):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def chat(self, job_input: dict[str, Any]) -> dict[str, Any]:
        raise self.error


def test_오류_종류에_따라_재시도_여부를_보고한다() -> None:
    cases = [
        (VramBusyError("부족"), "VRAM_BUSY", True),
        (OllamaUnavailableError("down"), "OLLAMA_UNAVAILABLE", True),
        (KeyError("prompt"), "BAD_INPUT", False),
        (RuntimeError("boom"), "MODEL_ERROR", False),
    ]
    for error, code, retryable in cases:
        api = FakeApi([Job("j", "LLM", "lt", {})])
        assert worker(api, Failing(error)).run_once()
        assert api.fails == [("j", code, retryable)]


def test_모르는_작업_종류는_BAD_INPUT() -> None:
    api = FakeApi([Job("j", "VIDEO", "lt", {})])
    worker(api).run_once()
    assert api.fails == [("j", "BAD_INPUT", False)]


def test_임대를_잃으면_결과를_버린다() -> None:
    api = FakeApi([Job("j", "LLM", "lt", {}), Job("k", "LLM", "lt", {})])
    api.lose_lease = True
    w = worker(api)
    assert w.run_once()
    api2 = FakeApi([Job("k", "LLM", "lt", {})])
    api2.lose_lease = True
    assert worker(api2, Failing(RuntimeError("x"))).run_once()
    assert api.results == [] and api2.fails == []


def test_heartbeat는_임대를_잃으면_멈춘다() -> None:
    api = FakeApi([])
    api.lose_lease = True
    hb = Heartbeat(api, Job("j", "LLM", "lt", {}), 0.01)  # type: ignore[arg-type]
    hb.start()
    hb.join(timeout=2)
    assert hb.lost.is_set()


def test_heartbeat_통신_오류는_다음에_다시_한다() -> None:
    class Flaky(FakeApi):
        def heartbeat(self, job: Job, progress: int | None = None) -> None:
            self.heartbeats += 1
            raise httpx.ConnectError("down")

    api = Flaky([])
    hb = Heartbeat(api, Job("j", "LLM", "lt", {}), 0.01)  # type: ignore[arg-type]
    hb.start()
    while api.heartbeats < 2:
        pass
    hb.stop()
    hb.join(timeout=2)
    assert not hb.lost.is_set()


def test_작업이_없으면_2초부터_10초까지_늘려_기다린다() -> None:
    slept: list[float] = []
    api = FakeApi([])
    w = worker(api, sleep=slept.append)
    w.run(max_polls=5)
    assert slept == [2.0, 4.0, 8.0, 10.0, 10.0]


def test_API_오류가_나도_계속_돈다() -> None:
    class Broken(FakeApi):
        def claim(self, types: list[str], loaded: str | None) -> Job | None:
            raise httpx.ConnectError("down")

    slept: list[float] = []
    w = worker(Broken([]), sleep=slept.append)
    w.run(max_polls=2)
    assert slept == [2.0, 4.0]
    w.stop()
    w.run()


def test_가짜_PNG는_올바른_PNG_머리를_가진다() -> None:
    png = tiny_png(4, 2)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert b"IHDR" in png and png.endswith(b"IEND\xaeB`\x82")
