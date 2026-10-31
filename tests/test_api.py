import httpx
import pytest
import respx

from model_runtime.api import Job, LeaseLostError, WorkerApi

BASE = "https://4cut.example/internal/gpu-jobs"


@pytest.fixture
def api() -> WorkerApi:
    return WorkerApi.create("https://4cut.example", "secret", "gpu-pc-1", 5.0)


@respx.mock
def test_작업을_가져가면_토큰과_워커_이름을_보낸다(api: WorkerApi) -> None:
    route = respx.post(f"{BASE}/claim").mock(
        return_value=httpx.Response(
            200,
            json={"jobId": "j1", "type": "LLM", "leaseToken": "lt", "input": {"prompt": "p"}},
        )
    )
    job = api.claim(["LLM"], "LLM")
    assert job == Job("j1", "LLM", "lt", {"prompt": "p"})
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer secret"
    assert request.headers["X-Worker-Id"] == "gpu-pc-1"
    assert b'"loadedModel":"LLM"' in request.content


@respx.mock
def test_작업이_없으면_None(api: WorkerApi) -> None:
    respx.post(f"{BASE}/claim").mock(return_value=httpx.Response(204))
    assert api.claim(["LLM"], None) is None


@respx.mock
def test_임대를_잃으면_LeaseLostError(api: WorkerApi) -> None:
    job = Job("j1", "LLM", "lt", {})
    respx.post(f"{BASE}/j1/heartbeat").mock(return_value=httpx.Response(409))
    with pytest.raises(LeaseLostError):
        api.heartbeat(job, 10)


@respx.mock
def test_결과와_실패를_보고한다(api: WorkerApi) -> None:
    job = Job("j1", "LLM", "lt", {})
    result = respx.post(f"{BASE}/j1/result").mock(return_value=httpx.Response(200))
    fail = respx.post(f"{BASE}/j1/fail").mock(return_value=httpx.Response(200))
    api.result(job, {"content": "{}"}, {"m": 1}, 10)
    api.fail(job, "MODEL_ERROR", "x" * 2000, False)
    assert b'"durationMs":10' in result.calls.last.request.content
    assert len(fail.calls.last.request.content) < 1200
    respx.post(f"{BASE}/j1/heartbeat").mock(return_value=httpx.Response(500))
    with pytest.raises(httpx.HTTPStatusError):
        api.heartbeat(job)
    api.close()
