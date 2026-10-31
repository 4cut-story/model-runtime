import json

import httpx
import pytest
import respx

from model_runtime.backends.ollama import OllamaClient, OllamaUnavailableError

URL = "http://localhost:11434"


@respx.mock
def test_스키마를_format으로_넘기고_내용을_돌려준다() -> None:
    route = respx.post(f"{URL}/api/chat").mock(
        return_value=httpx.Response(200, json={"message": {"content": '{"a":1}'}, "eval_count": 7})
    )
    client = OllamaClient.create(URL)
    out = client.chat(
        {
            "model": "qwen3:8b",
            "system": "s",
            "prompt": "p",
            "jsonSchema": {"type": "object"},
            "options": {"num_ctx": 16384},
        }
    )
    assert out == {"content": '{"a":1}', "evalCount": 7}
    body = json.loads(route.calls.last.request.content)
    assert body["format"] == {"type": "object"}
    assert body["options"]["num_ctx"] == 16384
    assert body["keep_alive"] == "5m"
    assert client.last_model == "qwen3:8b"

    unload = respx.post(f"{URL}/api/generate").mock(return_value=httpx.Response(200))
    client.unload()
    assert json.loads(unload.calls.last.request.content)["keep_alive"] == 0
    assert client.last_model is None


@respx.mock
def test_모르면_실행_중인_모델을_모두_내린다() -> None:
    respx.get(f"{URL}/api/ps").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "a"}, {"name": "b"}]})
    )
    unload = respx.post(f"{URL}/api/generate").mock(return_value=httpx.Response(200))
    OllamaClient.create(URL).unload()
    assert unload.call_count == 2


@respx.mock
def test_실행_목록을_못_읽으면_아무것도_안_한다() -> None:
    respx.get(f"{URL}/api/ps").mock(return_value=httpx.Response(500))
    OllamaClient.create(URL).unload()


@respx.mock
def test_연결이_안_되면_재시도_가능한_오류() -> None:
    respx.post(f"{URL}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
    respx.get(f"{URL}/api/ps").mock(side_effect=httpx.ConnectError("refused"))
    client = OllamaClient.create(URL)
    with pytest.raises(OllamaUnavailableError):
        client.chat({"model": "m", "prompt": "p"})
    client.unload()
