"""캐릭터 참조 (conversation-service 캐릭터 DB, REQ-018): IMAGE 작업의 references를 받아 그린다."""

from typing import Any

import httpx
import respx

from model_runtime.api import Job
from model_runtime.backends import Reference
from model_runtime.backends.fake import FakeBackends
from model_runtime.worker import Worker
from tests.test_worker import FakeApi


def make(backends: FakeBackends) -> tuple[Worker, FakeApi]:
    api = FakeApi([])
    return Worker(api, backends, httpx.Client(), sleep=lambda s: None), api  # type: ignore[arg-type]


def image(i: int, refs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": f"panel-{i}",
        "prompt": "p",
        "uploadUrl": f"http://minio/put/{i}",
    }
    if refs is not None:
        out["references"] = refs
    return out


def fresh(region: str = "full", scale: float = 0.6) -> dict[str, Any]:
    return {
        "character": "char-a",
        "scale": scale,
        "region": region,
        "imageUrl": "http://minio/get/a.png",
        "embedUploadUrl": "http://minio/put/a.safetensors",
    }


def test_REQ018_embedding이_없으면_한_번만_계산해_올리고_모든_컷에_쓴다() -> None:
    backends = FakeBackends()
    w, _ = make(backends)
    with respx.mock:
        respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        get = respx.get("http://minio/get/a.png").mock(
            return_value=httpx.Response(200, content=b"png-a")
        )
        result = w.draw_all([image(1, [fresh()]), image(2, [fresh("left")]), image(3)])
        uploads = [c.request for c in respx.calls if c.request.method == "PUT"]
    assert result == {"uploaded": ["panel-1", "panel-2", "panel-3"], "embeds": ["char-a"]}
    assert get.call_count == 1
    assert backends.encoded == 1
    embed_put = [r for r in uploads if r.url.path.endswith(".safetensors")]
    assert len(embed_put) == 1
    assert embed_put[0].headers["Content-Type"] == "application/octet-stream"
    names = [n for n, _ in backends.drawn]
    assert names == ["panel-1", "panel-2", "panel-3"]
    refs1, refs2, refs3 = (r for _, r in backends.drawn)
    assert refs1 == [Reference("char-a", embed_put[0].content, 0.6, "full")]
    assert refs2[0].region == "left"
    assert refs3 == []


def test_REQ018_저장해_둔_embedding을_받아_쓰고_다시_계산하지_않는다() -> None:
    backends = FakeBackends()
    w, _ = make(backends)
    ref = {"character": "b", "scale": 1.7, "region": "right", "embedUrl": "http://minio/get/b"}
    with respx.mock:
        respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        respx.get("http://minio/get/b").mock(return_value=httpx.Response(200, content=b"emb-b"))
        result = w.draw_all([image(1, [ref])])
    assert result == {"uploaded": ["panel-1"]}
    assert backends.encoded == 0
    # 강도는 0~1로 자른다
    assert backends.drawn[0][1] == [Reference("b", b"emb-b", 1.0, "right")]


def test_REQ018_참조를_받지_못하면_빼고_텍스트만으로_그린다() -> None:
    backends = FakeBackends()
    w, _ = make(backends)
    gone = {"character": "c", "embedUrl": "http://minio/get/c"}
    with respx.mock:
        respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        respx.get("http://minio/get/c").mock(return_value=httpx.Response(404))
        result = w.draw_all([image(1, [gone, {"character": "d"}])])
    assert result == {"uploaded": ["panel-1"]}
    assert backends.drawn[0][1] == []


def test_REQ018_embedding_업로드가_실패해도_그림은_그린다() -> None:
    backends = FakeBackends()
    w, _ = make(backends)
    with respx.mock:
        respx.put("http://minio/put/a.safetensors").mock(return_value=httpx.Response(500))
        respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        respx.get("http://minio/get/a.png").mock(return_value=httpx.Response(200, content=b"x"))
        result = w.draw_all([image(1, [fresh()])])
    assert result == {"uploaded": ["panel-1"]}
    assert len(backends.drawn[0][1]) == 1


class NoAdapter(FakeBackends):
    def encode_reference(self, image: bytes) -> bytes | None:
        return None


def test_REQ018_IP_Adapter를_끄면_참조를_무시한다() -> None:
    backends = NoAdapter()
    w, _ = make(backends)
    with respx.mock:
        respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        respx.get("http://minio/get/a.png").mock(return_value=httpx.Response(200, content=b"x"))
        result = w.draw_all([image(1, [fresh()])])
    assert result == {"uploaded": ["panel-1"]}
    assert backends.drawn[0][1] == []


def test_REQ018_모르는_위치는_BAD_INPUT() -> None:
    backends = FakeBackends()
    w, api = make(backends)
    api.jobs = [Job("i", "IMAGE", "lt", {"images": [image(1, [fresh("top")])]})]
    with respx.mock:
        respx.get("http://minio/get/a.png").mock(return_value=httpx.Response(200, content=b"x"))
        respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        w.run_once()
    assert api.fails == [("i", "BAD_INPUT", False)]


def test_REQ018_가운데_자리_참조를_받는다() -> None:
    backends = FakeBackends()
    w, _ = make(backends)
    ref = {"character": "c", "scale": 0.6, "region": "center", "embedUrl": "http://minio/get/c"}
    with respx.mock:
        respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        respx.get("http://minio/get/c").mock(return_value=httpx.Response(200, content=b"emb-c"))
        w.draw_all([image(1, [ref])])
    assert backends.drawn[0][1] == [Reference("c", b"emb-c", 0.6, "center")]


def test_REQ018_참조_그림을_그대로_받는_모델은_저장한_embedding을_쓰지도_올리지도_않는다() -> None:
    """REQ-018: FLUX.2 klein은 SDXL embedding(embedUrl)을 건너뛰고 늘 참조 그림을 받는다."""
    backends = FakeBackends()
    backends.reference_mode = "image"
    w, _ = make(backends)
    ref = {**fresh(), "embedUrl": "http://minio/get/a.safetensors"}
    with respx.mock:
        respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        stored = respx.get("http://minio/get/a.safetensors").mock(
            return_value=httpx.Response(200, content=b"sdxl-embed")
        )
        respx.get("http://minio/get/a.png").mock(return_value=httpx.Response(200, content=b"png-a"))
        result = w.draw_all([image(1, [ref]), image(2, [ref])])
        puts = [c.request.url.path for c in respx.calls if c.request.method == "PUT"]
    assert result == {"uploaded": ["panel-1", "panel-2"]}
    assert stored.call_count == 0
    assert not any(p.endswith(".safetensors") for p in puts)
    assert backends.encoded == 1


def _png(width: int = 8, height: int = 10) -> bytes:
    import io

    from PIL import Image

    im = Image.new("RGB", (width, height), (240, 240, 240))
    for y in range(height):
        im.putpixel((width // 2, y), (200, 0, 0))  # 위아래로 그은 빨간 줄
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


class RawImage(FakeBackends):
    def encode_reference(self, image: bytes) -> bytes | None:
        self.encoded += 1
        return image


def test_역할_옷을_입힌_인물은_참조_그림의_머리만_넘긴다() -> None:
    """콘티가 역할 옷(군복 등)을 입힌 인물(crop "head")은 참조 그림 아래(옷)를 배경색으로 지운다."""
    import io

    from PIL import Image

    backends = RawImage()
    backends.reference_mode = "image"
    w, _ = make(backends)
    ref = {**fresh(), "crop": "head"}
    with respx.mock:
        respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        respx.get("http://minio/get/a.png").mock(return_value=httpx.Response(200, content=_png()))
        result = w.draw_all([image(1, [ref])])
    assert result == {"uploaded": ["panel-1"]}
    sent = backends.drawn[0][1][0].embed
    with Image.open(io.BytesIO(sent)) as im:
        assert im.size == (8, 10)
        assert im.getpixel((4, 0)) == (200, 0, 0)
        assert im.getpixel((4, 9)) == (240, 240, 240)


def test_머리만_자르기는_읽지_못하는_그림이면_그대로_둔다() -> None:
    from model_runtime.worker import head_only

    assert head_only(b"not-png") == b"not-png"
