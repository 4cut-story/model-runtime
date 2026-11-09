"""라이브 모드 두 번째 그림 (docs specs/003-live-mode FR-014): IMAGE 작업 컷의 motion."""

import httpx
import respx

from model_runtime.backends import Reference
from model_runtime.backends.fake import FakeBackends, tiny_png
from tests.test_references import image, make


def motion_image(i: int) -> dict:
    out = image(i)
    out["motion"] = {
        "type": "twinkle",
        "prompt": "make the stars sparkle",
        "uploadUrl": f"http://minio/put/motion-{i}",
    }
    return out


def test_FR014_motion이_있으면_완성_그림을_바탕으로_두_번째_그림을_올린다() -> None:
    backends = FakeBackends()
    backends.reference_mode = "image"
    w, _ = make(backends)
    with respx.mock:
        put = respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        result = w.draw_all([motion_image(1), image(2)])
    assert result["uploaded"] == ["panel-1", "panel-2"]
    assert result["motions"] == ["panel-1"]
    assert backends.prompts == ["p", "make the stars sparkle", "p"]
    assert backends.drawn[1][1] == [Reference("__base__", tiny_png(), 1.0, "full")]
    assert any(c.request.url.path == "/put/motion-1" for c in put.calls)
    assert "motionMs" in w._stats["panels"][0]


def test_FR014_참조_그림을_받지_않는_모델은_두_번째_그림을_건너뛴다() -> None:
    backends = FakeBackends()
    w, _ = make(backends)
    with respx.mock:
        respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        result = w.draw_all([motion_image(1)])
    assert "motions" not in result
    assert [n for n, _ in backends.drawn] == ["panel-1"]


def test_FR014_두_번째_그림을_올리지_못해도_작업은_성공이다() -> None:
    backends = FakeBackends()
    backends.reference_mode = "image"
    w, _ = make(backends)
    with respx.mock:
        respx.put("http://minio/put/1").mock(return_value=httpx.Response(200))
        respx.put("http://minio/put/motion-1").mock(return_value=httpx.Response(500))
        result = w.draw_all([motion_image(1)])
    assert result["uploaded"] == ["panel-1"]
    assert "motions" not in result
