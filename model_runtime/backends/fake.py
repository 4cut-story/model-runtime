"""가짜 백엔드. GPU 없이 고정 결과를 낸다 (로컬 개발, CI, 연결 확인)."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from typing import Any

from . import Reference


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
        self.drawn: list[tuple[str, list[Reference]]] = []
        self.prompts: list[str] = []
        self.encoded = 0
        self.depths = 0
        self.spoken: list[tuple[str, str, str]] = []
        # 테스트에서 "image"로 바꿔 klein처럼 참조 그림을 그대로 받는 경우를 본다
        self.reference_mode = "embed"

    def transcribe_part(
        self, audio_urls: list[str], language: str | None, from_sec: float
    ) -> dict[str, Any]:
        """조각마다 1분이라고 보고, 앞 경계 뒤 단어 하나를 낸다."""
        end = 60.0 * len(audio_urls)
        cut = end - 10.0
        if cut <= from_sec:
            return {
                "segments": [],
                "cut": from_sec,
                "language": language or "ko",
                "durationSec": end,
                "skipped": True,
            }
        word = {"word": f"조각{len(audio_urls)}", "start": from_sec + 1.0, "end": from_sec + 2.0}
        segment = {
            "start": word["start"],
            "end": word["end"],
            "text": word["word"],
            "words": [word],
        }
        return {
            "segments": [segment],
            "cut": cut,
            "language": language or "ko",
            "durationSec": end,
            "skipped": False,
        }

    def transcribe(
        self,
        audio_urls: list[str],
        language: str | None,
        prefix: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        lang = language or "ko"
        if prefix and prefix.get("segments"):
            joined = [
                {"speaker": "SPEAKER_00", "start": s["start"], "end": s["end"], "text": s["text"]}
                for s in prefix["segments"]
            ]
            return {"segments": joined, "language": prefix.get("language") or lang}
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

    def draw(self, image: dict[str, Any], references: list[Reference] | None = None) -> bytes:
        self.drawn.append((str(image.get("name")), list(references or [])))
        self.prompts.append(str(image.get("prompt")))
        return tiny_png()

    def speak(
        self, voice_id: str, emotion: str, text: str, speed: float = 1.0
    ) -> tuple[bytes, int]:
        """가짜 MP3: 글자마다 100ms."""
        self.spoken.append((voice_id, emotion, text))
        return b"ID3fake-mp3", int(len(text) * 100 / speed)

    def depth(self, png: bytes) -> bytes:
        """가짜 깊이 지도: 아주 작은 PNG."""
        self.depths += 1
        return tiny_png()

    def encode_reference(self, image: bytes) -> bytes | None:
        """참조 그림 내용에 따라 정해지는 가짜 embedding."""
        self.encoded += 1
        return b"fake-embed:" + hashlib.sha256(image).hexdigest().encode()

    def reference_kind(self) -> str:
        return self.reference_mode

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
