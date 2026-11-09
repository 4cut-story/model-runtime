"""2단 그리기 (conversation-service FR-006, 회차 29 콘티 16~25판): IMAGE 작업 컷의 edit."""

import httpx
import respx

from model_runtime.backends import Reference
from model_runtime.backends.fake import FakeBackends, tiny_png
from tests.test_references import fresh, image, make


def test_FR006_edit이_있으면_1차_그림을_첫_입력으로_2차를_그려_최종만_올린다() -> None:
    backends = FakeBackends()
    backends.reference_mode = "image"
    w, _ = make(backends)
    first = image(1)
    first["edit"] = {"prompt": "add one telescope", "references": [fresh()]}
    with respx.mock:
        put = respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        respx.get("http://minio/get/a.png").mock(return_value=httpx.Response(200, content=b"png-a"))
        result = w.draw_all([first, image(2)])
    assert result["uploaded"] == ["panel-1", "panel-2"]
    assert [n for n, _ in backends.drawn] == ["panel-1", "panel-1", "panel-2"]
    assert backends.prompts == ["p", "add one telescope", "p"]
    base, char = backends.drawn[1][1]
    assert base == Reference("__base__", tiny_png(), 1.0, "full")
    assert char.character == "char-a"
    panel_puts = [
        c
        for c in put.calls
        if c.request.url.path.startswith("/put/") and c.request.url.path[-1].isdigit()
    ]
    assert len(panel_puts) == 2
    assert "editMs" in w._stats["panels"][0]


def test_FR006_참조_그림을_받지_않는_모델은_edit을_건너뛰고_1차_그림을_쓴다() -> None:
    backends = FakeBackends()
    w, _ = make(backends)
    first = image(1)
    first["edit"] = {"prompt": "add one telescope"}
    with respx.mock:
        respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        w.draw_all([first])
    assert [n for n, _ in backends.drawn] == ["panel-1"]
    assert "editMs" not in w._stats["panels"][0]


def test_FR006_컷마다_그림_프롬프트와_참조와_2차_프롬프트를_로그에_남긴다(caplog) -> None:  # type: ignore[no-untyped-def]
    backends = FakeBackends()
    backends.reference_mode = "image"
    w, _ = make(backends)
    first = image(1)
    first["edit"] = {"prompt": "add one telescope", "references": [fresh()]}
    caplog.set_level("INFO", logger="model_runtime.worker")
    with respx.mock:
        respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        respx.get("http://minio/get/a.png").mock(return_value=httpx.Response(200, content=b"png-a"))
        w.draw_all([first])
    line = next(r.getMessage() for r in caplog.records if "프롬프트" in r.getMessage())
    assert "컷 panel-1 프롬프트: p" in line
    assert "2차: add one telescope" in line
