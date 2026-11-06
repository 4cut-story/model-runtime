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
