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


def test_FR044_넓은_그림_컷은_같은_바탕에서_세_번째_그림도_올린다() -> None:
    backends = FakeBackends()
    backends.reference_mode = "image"
    w, _ = make(backends)
    job = live_image(1, motion=True)
    job["motion"]["next"] = {
        "prompt": "make other stars sparkle",
        "uploadUrl": "http://minio/put/motion-c-1",
    }
    with respx.mock:
        put = respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        w.draw_all([job])
    assert backends.prompts == [
        "p",
        "wide 16:9 shot",
        "make the stars sparkle",
        "make other stars sparkle",
    ]
    # 세 번째 그림도 두 번째 그림이 아니라 넓은 그림을 바탕으로 그린다
    assert backends.drawn[3][1] == backends.drawn[2][1]
    assert any(c.request.url.path == "/put/motion-c-1" for c in put.calls)
    assert "motionNextMs" in w._stats["panels"][0]


def test_FR044_넓은_그림이_없으면_세_번째_그림은_그리지_않는다() -> None:
    backends = FakeBackends()
    backends.reference_mode = "image"
    w, _ = make(backends)
    job = motion_image(1)
    job["motion"]["next"] = {"prompt": "other", "uploadUrl": "http://minio/put/motion-c-1"}
    with respx.mock:
        respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        w.draw_all([job])
    assert backends.prompts == ["p", "make the stars sparkle"]


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


def test_FR048_넓은_그림이_있고_depthUploadUrl이_있으면_깊이_지도를_올린다() -> None:
    backends = FakeBackends()
    backends.reference_mode = "image"
    w, _ = make(backends)
    img = live_image(1)
    img["live"]["depthUploadUrl"] = "http://minio/put/depth-1"
    with respx.mock:
        put = respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        result = w.draw_all([img, live_image(2)])
    assert result["depths"] == ["panel-1"]
    assert backends.depths == 1
    assert any(c.request.url.path == "/put/depth-1" for c in put.calls)
    assert "depthMs" in w._stats["panels"][0]


def test_FR048_깊이_지도를_만들지_못해도_작업은_성공이다() -> None:
    backends = FakeBackends()
    backends.reference_mode = "image"

    def broken(png: bytes) -> bytes:
        raise RuntimeError("no model")

    backends.depth = broken  # type: ignore[method-assign]
    w, _ = make(backends)
    img = live_image(1)
    img["live"]["depthUploadUrl"] = "http://minio/put/depth-1"
    with respx.mock:
        respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        result = w.draw_all([img])
    assert result["lives"] == ["panel-1"]
    assert "depths" not in result


def test_FR049_깊이_지도가_있으면_움직임_그림의_앞쪽_인물은_첫_그림으로_덮는다(monkeypatch) -> None:
    backends = FakeBackends()
    backends.reference_mode = "image"
    w, _ = make(backends)
    calls: list[tuple[bytes, bytes, bytes]] = []

    def fake_keep(base: bytes, frame: bytes, depth: bytes) -> bytes:
        calls.append((base, frame, depth))
        return b"kept"

    monkeypatch.setattr("model_runtime.worker.keep_near", fake_keep)
    img = live_image(1, motion=True)
    img["live"]["depthUploadUrl"] = "http://minio/put/depth-1"
    with respx.mock:
        put = respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        result = w.draw_all([img])
    assert result["motions"] == ["panel-1"]
    assert len(calls) == 1
    motion = next(c for c in put.calls if c.request.url.path == "/put/motion-1")
    assert motion.request.content == b"kept"


def test_FR049_인물을_고정하지_못하면_새_그림을_그대로_올린다(monkeypatch) -> None:
    backends = FakeBackends()
    backends.reference_mode = "image"
    w, _ = make(backends)

    def broken(base: bytes, frame: bytes, depth: bytes) -> bytes:
        raise ValueError("bad png")

    monkeypatch.setattr("model_runtime.worker.keep_near", broken)
    img = live_image(1, motion=True)
    img["live"]["depthUploadUrl"] = "http://minio/put/depth-1"
    with respx.mock:
        put = respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        result = w.draw_all([img])
    assert result["motions"] == ["panel-1"]
    motion = next(c for c in put.calls if c.request.url.path == "/put/motion-1")
    assert motion.request.content == tiny_png()


def test_FR049_넓은_그림_컷은_next를_따라_네_번째_그림까지_올린다() -> None:
    backends = FakeBackends()
    backends.reference_mode = "image"
    w, _ = make(backends)
    job = live_image(1, motion=True)
    job["motion"]["next"] = {
        "prompt": "c",
        "uploadUrl": "http://minio/put/motion-c-1",
        "next": {
            "prompt": "d",
            "uploadUrl": "http://minio/put/motion-d-1",
            "next": {"prompt": "e", "uploadUrl": "http://minio/put/motion-e-1"},
        },
    }
    with respx.mock:
        put = respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        w.draw_all([job])
    # 네 장(첫 그림 + 셋)까지만 그린다
    assert backends.prompts == ["p", "wide 16:9 shot", "make the stars sparkle", "c", "d"]
    assert backends.drawn[4][1] == backends.drawn[2][1]
    paths = [c.request.url.path for c in put.calls]
    assert "/put/motion-d-1" in paths
    assert "/put/motion-e-1" not in paths
    assert "motionNext2Ms" in w._stats["panels"][0]


def test_FR049_세_번째_그림을_못_올리면_네_번째는_그리지_않는다() -> None:
    backends = FakeBackends()
    backends.reference_mode = "image"
    w, _ = make(backends)
    job = live_image(1, motion=True)
    job["motion"]["next"] = {
        "prompt": "c",
        "uploadUrl": "http://minio/put/motion-c-1",
        "next": {"prompt": "d", "uploadUrl": "http://minio/put/motion-d-1"},
    }
    with respx.mock:
        respx.put("http://minio/put/motion-c-1").mock(return_value=httpx.Response(500))
        respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        w.draw_all([job])
    assert "d" not in backends.prompts
