"""대사 읽기 (GPT-SoVITS v4). 가상 인물 목소리의 감정별 참조 음성을 흉내 내 대사를 MP3로 만든다.

GPT-SoVITS는 저장소와 Python 환경이 따로라 같은 GPU PC에서 도는 api_v2 서버(기본
http://127.0.0.1:9880, v4 설정, CUDAGraph 끔)를 부른다. 참조 음성은 VOICES_DIR 아래
``<목소리 ID>/<감정>.wav``와 그 녹음의 글 ``<목소리 ID>/<감정>.txt``다.
감정 참조가 없으면 기본(neutral)을 쓴다.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from . import JobInputError
from .ko_numbers import normalize

EMOTIONS = ("neutral", "happy", "angry", "sad", "surprised")
VOICE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")
MODEL = "GPT-SoVITS v4"


class SpeechUnavailableError(Exception):
    """대사 읽기 서버에 닿지 않는다. 작업을 대기열로 돌려보낸다."""


def wav_ms(data: bytes) -> int:
    """WAV 길이(ms)."""
    with wave.open(io.BytesIO(data)) as w:
        return int(w.getnframes() * 1000 / w.getframerate())


Runner = Callable[[list[str], bytes], bytes]

# 대사 음성 크기를 -16 LUFS(최대 -1.5 dBTP)로 맞춘다.
# 목소리마다 -20~-25 LUFS로 달라 배경음에 묻혔다.
# loudnorm은 출력을 192kHz로 올리므로 44.1kHz로 되돌린다.
LOUDNORM = "loudnorm=I=-16:TP=-1.5:LRA=11"
SAMPLE_RATE = "44100"


def _ffmpeg(args: list[str], data: bytes) -> bytes:
    out = subprocess.run(  # noqa: S603 - 고정된 인자만 넘긴다
        args, input=data, capture_output=True, timeout=60, check=True
    )
    return out.stdout


class SovitsSpeaker:
    def __init__(
        self,
        url: str,
        voices_dir: str,
        client: httpx.Client | None = None,
        ffmpeg: str | None = None,
        run: Runner = _ffmpeg,
        bitrate: str = "64k",
        wait_s: float = 90.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url = url.rstrip("/")
        self.voices = Path(voices_dir)
        self.client = client or httpx.Client(timeout=120.0)
        self.ffmpeg = ffmpeg or shutil.which("ffmpeg") or "ffmpeg"
        self.run = run
        self.bitrate = bitrate
        self.wait_s = wait_s
        self.sleep = sleep
        self.clock = clock

    def reference(self, voice_id: str, emotion: str) -> tuple[Path, str]:
        """감정 참조 음성과 그 글. 감정 참조가 없으면 기본 목소리."""
        if not VOICE_ID.match(voice_id):
            raise JobInputError(f"목소리 ID 형식이 아니다: {voice_id}")
        if emotion not in EMOTIONS:
            raise JobInputError(f"감정은 {', '.join(EMOTIONS)} 중 하나다: {emotion}")
        for e in (emotion, "neutral"):
            wav = self.voices / voice_id / f"{e}.wav"
            text = wav.with_suffix(".txt")
            if wav.is_file() and text.is_file():
                return wav, text.read_text(encoding="utf-8").strip()
        raise JobInputError(f"참조 음성이 없는 목소리: {voice_id}")

    def speak(
        self, voice_id: str, emotion: str, text: str, speed: float = 1.0
    ) -> tuple[bytes, int]:
        """MP3 바이트와 길이(ms)."""
        ref, prompt = self.reference(voice_id, emotion)
        body: dict[str, Any] = {
            # 읽기 모델은 아라비아 숫자를 제대로 읽지 못해 한글 읽기로 바꿔 넘긴다
            "text": normalize(text),
            "text_lang": "ko",
            "ref_audio_path": str(ref.resolve()),
            "prompt_text": prompt,
            "prompt_lang": "ko",
            "text_split_method": "cut5",
            "batch_size": 1,
            "speed_factor": speed,
            "media_type": "wav",
            "streaming_mode": False,
        }
        response = self._post(body)
        response.raise_for_status()
        wav = response.content
        mp3 = self.run(
            [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "wav",
                "-i",
                "pipe:0",
                "-af",
                LOUDNORM,
                "-ar",
                SAMPLE_RATE,
                "-codec:a",
                "libmp3lame",
                "-b:a",
                self.bitrate,
                "-f",
                "mp3",
                "pipe:1",
            ],
            wav,
        )
        return mp3, wav_ms(wav)

    def _post(self, body: dict[str, Any]) -> httpx.Response:
        """서버가 잠깐 내려가 있으면(재시작 중) wait_s까지 기다리며 다시 부른다.

        서버는 실패한 작업을 곧바로 다시 주므로, 몇 초짜리 재시작에도
        작업이 실패로 끝나지 않게 여기서 기다린다.
        """
        until = self.clock() + self.wait_s
        while True:
            try:
                return self.client.post(self.url + "/tts", json=body)
            except httpx.TransportError as e:
                if self.clock() >= until:
                    raise SpeechUnavailableError(f"대사 읽기 서버에 닿지 않는다: {e}") from e
                self.sleep(3.0)

    def model_info(self) -> dict[str, Any]:
        return {"tts": MODEL}
