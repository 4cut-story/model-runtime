"""Google Gemini API(generateContent)로 LLM 작업을 돌린다 (실험용, OllamaClient와 같은 모양).

키는 환경 변수 GEMINI_API_KEY에서만 읽는다. 로그·결과 파일에 키를 남기지 않는다.
작업 입력(job)은 OllamaClient.chat과 같다: model, system, prompt, jsonSchema, options.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
# 생각(thinking) 토큰도 출력 한도에 들어가므로 Ollama의 num_predict보다 넉넉히 준다
MAX_OUTPUT_TOKENS = 16384


class GeminiClient:
    def __init__(self, client: httpx.Client, thinking: str | None = None) -> None:
        self._client = client
        self.thinking = thinking
        self.usage: list[dict[str, Any]] = []

    @classmethod
    def create(cls, api_key: str | None = None, thinking: str | None = None) -> GeminiClient:
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY가 없다 (.env 또는 환경 변수)")
        client = httpx.Client(
            base_url=BASE_URL,
            headers={"x-goog-api-key": key},
            timeout=httpx.Timeout(300.0, connect=10.0),
        )
        return cls(client, thinking)

    def chat(self, job_input: dict[str, Any]) -> dict[str, Any]:
        model = job_input["model"]
        options = job_input.get("options", {})
        config: dict[str, Any] = {
            "temperature": options.get("temperature", 0.4),
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json",
        }
        if "seed" in options:
            config["seed"] = options["seed"]
        schema = job_input.get("jsonSchema")
        if isinstance(schema, dict):
            config["responseJsonSchema"] = schema
        if self.thinking:
            config["thinkingConfig"] = {"thinkingLevel": self.thinking}
        body = {
            "systemInstruction": {"parts": [{"text": job_input.get("system", "")}]},
            "contents": [{"role": "user", "parts": [{"text": job_input["prompt"]}]}],
            "generationConfig": config,
        }
        start = time.perf_counter()
        for attempt in range(4):
            response = self._client.post(f"/models/{model}:generateContent", json=body)
            if response.status_code not in (429, 500, 503) or attempt == 3:
                break
            time.sleep(2 ** (attempt + 1))
        if response.status_code >= 400:
            raise RuntimeError(f"Gemini {response.status_code}: {response.text[:300]}")
        data = response.json()
        cand = (data.get("candidates") or [{}])[0]
        parts = cand.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        meta = data.get("usageMetadata", {})
        usage = {
            "model": model,
            "input": meta.get("promptTokenCount", 0),
            "output": meta.get("candidatesTokenCount", 0),
            "thinking": meta.get("thoughtsTokenCount", 0),
            "sec": round(time.perf_counter() - start, 2),
            "finish": cand.get("finishReason"),
        }
        self.usage.append(usage)
        return {"content": text, "evalCount": usage["output"] + usage["thinking"]}

    def unload(self) -> None:
        """원격 API라 내릴 모델이 없다."""
