"""같은 PC의 Ollama(설치형, localhost:11434)를 부른다.

프롬프트와 스키마는 conversation-service가 만든다.
"""

from __future__ import annotations

from typing import Any

import httpx

# LLM 작업이 이어지는 동안 모델을 올려 둔다. 다른 종류 작업 전에는 keep_alive 0으로 내린다
KEEP_ALIVE = "5m"


class OllamaUnavailableError(Exception):
    """Ollama에 연결할 수 없다. 잠시 뒤 다시 하면 될 수 있다."""


class OllamaClient:
    def __init__(self, client: httpx.Client) -> None:
        self._client = client
        self.last_model: str | None = None

    @classmethod
    def create(cls, base_url: str) -> OllamaClient:
        return cls(httpx.Client(base_url=base_url, timeout=httpx.Timeout(300.0, connect=5.0)))

    def chat(self, job_input: dict[str, Any]) -> dict[str, Any]:
        model = job_input["model"]
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": job_input.get("system", "")},
                {"role": "user", "content": job_input["prompt"]},
            ],
            "format": job_input.get("jsonSchema", "json"),
            "options": job_input.get("options", {}),
            "stream": False,
            "think": False,
            "keep_alive": KEEP_ALIVE,
        }
        try:
            response = self._client.post("/api/chat", json=body)
        except httpx.TransportError as e:
            raise OllamaUnavailableError(str(e)) from e
        response.raise_for_status()
        data = response.json()
        self.last_model = model
        return {
            "content": data.get("message", {}).get("content", ""),
            "evalCount": data.get("eval_count", 0),
        }

    def unload(self) -> None:
        """올려 둔 모델을 내린다 (keep_alive 0). 모르면 실행 중인 모델을 모두 내린다."""
        try:
            models = [self.last_model] if self.last_model else self._running()
            for model in models:
                self._client.post(
                    "/api/generate", json={"model": model, "keep_alive": 0, "prompt": ""}
                )
        except httpx.TransportError:
            return
        self.last_model = None

    def _running(self) -> list[str]:
        response = self._client.get("/api/ps")
        if response.status_code != 200:
            return []
        return [m["name"] for m in response.json().get("models", [])]
