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


def live_image(i: int, motion: bool = False) -> dict:
    out = motion_image(i) if motion else image(i)
    out["live"] = {
        "prompt": "wide 16:9 shot",
        "width": 1024,
        "height": 576,
        "uploadUrl": f"http://minio/put/wide-{i}",
    }
    return out


def test_FR030_live가_있으면_완성_그림을_바탕으로_넓은_그림을_넓은_크기로_올린다() -> None:
    backends = FakeBackends()
    backends.reference_mode = "image"
    w, _ = make(backends)
    sizes: list[tuple[int, int]] = []
    draw = backends.draw

    def spy(img: dict, refs: list | None = None) -> bytes:
        sizes.append((int(img.get("width", 768)), int(img.get("height", 768))))
        return draw(img, refs)

    backends.draw = spy  # type: ignore[method-assign]
    with respx.mock:
        put = respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        result = w.draw_all([live_image(1, motion=True)])
    assert result["lives"] == ["panel-1"]
    assert result["motions"] == ["panel-1"]
    assert backends.prompts == ["p", "wide 16:9 shot", "make the stars sparkle"]
    assert backends.drawn[1][1] == [Reference("__base__", tiny_png(), 1.0, "full")]
    # 두 번째 그림(FR-034)도 넓은 크기로 넓은 그림을 바탕으로 그린다
    assert sizes[1:] == [(1024, 576), (1024, 576)]
    assert any(c.request.url.path == "/put/wide-1" for c in put.calls)
    assert "liveMs" in w._stats["panels"][0]


def test_FR033_넓은_그림을_올리지_못하면_컷_그림_크기로_움직임을_그린다() -> None:
    backends = FakeBackends()
    backends.reference_mode = "image"
    w, _ = make(backends)
    with respx.mock:
        respx.put("http://minio/put/1").mock(return_value=httpx.Response(200))
        respx.put("http://minio/put/wide-1").mock(return_value=httpx.Response(500))
        respx.put("http://minio/put/motion-1").mock(return_value=httpx.Response(200))
        result = w.draw_all([live_image(1, motion=True)])
    assert result["uploaded"] == ["panel-1"]
    assert "lives" not in result
    assert result["motions"] == ["panel-1"]


def test_FR030_참조_그림을_받지_않는_모델은_넓은_그림을_건너뛴다() -> None:
    backends = FakeBackends()
    w, _ = make(backends)
    with respx.mock:
        respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        result = w.draw_all([live_image(1)])
    assert "lives" not in result
    assert [n for n, _ in backends.drawn] == ["panel-1"]
