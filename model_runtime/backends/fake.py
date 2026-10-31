"""가짜 백엔드. GPU 없이 고정 결과를 낸다 (로컬 개발, CI, 연결 확인)."""

from __future__ import annotations

import json
import struct
import zlib
from typing import Any


def tiny_png(width: int = 8, height: int = 8, rgb: tuple[int, int, int] = (240, 220, 180)) -> bytes:
    """단색 PNG를 만든다 (Pillow 없이)."""
    row = b"\x00" + bytes(rgb) * width
    raw = row * height

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class FakeBackends:
    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.unloaded: list[str] = []

    def transcribe(self, audio_urls: list[str], language: str | None) -> dict[str, Any]:
        lang = language or "ko"
        segments = [
            {"speaker": "SPEAKER_00", "start": 1.0, "end": 3.0, "text": "내 접시 누가 다 먹었어?"},
            {"speaker": "SPEAKER_01", "start": 3.5, "end": 5.0, "text": "나 아니야, 진짜로."},
            {"speaker": "SPEAKER_00", "start": 5.5, "end": 8.0, "text": "입가에 소스 묻었는데?"},
        ]
        return {"segments": segments, "language": lang}

    def chat(self, job_input: dict[str, Any]) -> dict[str, Any]:
        schema = job_input.get("jsonSchema") or {}
        required = set(schema.get("required", []))
        content: dict[str, Any]
        if "candidates" in required:
            content = {
                "candidates": [
                    {
                        "kind": "FUNNY",
                        "start": 1.0,
                        "end": 8.0,
                        "summary": "빈 접시 범인 찾기",
                        "reason": "들통나는 반전",
                        "score": 0.9,
                    }
                ]
            }
        elif "script" in required:
            content = {"script": _script(), "nameWarnings": []}
        else:
            content = _script()
        return {"content": json.dumps(content, ensure_ascii=False), "evalCount": 0}

    def draw(self, image: dict[str, Any]) -> bytes:
        return tiny_png()

    def prepare(self, model: str) -> None:
        self.loaded.append(model)

    def unload(self, model: str) -> None:
        self.unloaded.append(model)

    def model_info(self, model: str) -> dict[str, Any]:
        return {"backend": "fake", "model": model}


def _script() -> dict[str, Any]:
    roles = ["기", "승", "전", "결"]
    return {
        "title": "접시 미스터리",
        "characters": [
            {"speaker": "A", "look": "단발머리, 노란 후드티"},
            {"speaker": "B", "look": "안경, 파란 셔츠"},
        ],
        "panels": [
            {
                "role": role,
                "scene": f"식당, 빈 접시 앞의 두 사람 {i + 1}",
                "lines": [{"speaker": "A" if i % 2 == 0 else "B", "text": f"대사 {i + 1}"}],
            }
            for i, role in enumerate(roles)
        ],
    }
