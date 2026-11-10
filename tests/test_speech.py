"""대사 읽기 (라이브 모드 대사 음성): SPEECH 작업, GPT-SoVITS 호출, 참조 음성 고르기."""

import io
import json
import wave
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from model_runtime.__main__ import build_speaker, job_types
from model_runtime.api import Job
from model_runtime.backends import JobInputError
from model_runtime.backends.fake import FakeBackends
from model_runtime.backends.real import RealBackends
from model_runtime.backends.speech import SovitsSpeaker, SpeechUnavailableError, wav_ms
from model_runtime.config import Settings
from tests.test_worker import FakeApi, worker


def wav_bytes(ms: int, rate: int = 32000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * (rate * ms // 1000))
    return buf.getvalue()


def voices(tmp: Path) -> Path:
    for emotion in ("neutral", "happy"):
        d = tmp / "f-child-bright"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{emotion}.wav").write_bytes(wav_bytes(100))
        (d / f"{emotion}.txt").write_text(f"{emotion} 참조 글\n", encoding="utf-8")
    return tmp


def speech_job(lines: list[dict[str, Any]]) -> Job:
    return Job("s", "SPEECH", "lt", {"conversationId": "c", "speed": 1.0, "lines": lines})


def line(key: str, text: str = "안녕", voice: str = "f-child-bright") -> dict[str, Any]:
    return {
        "key": key,
        "voiceId": voice,
        "emotion": "happy",
        "text": text,
        "uploadUrl": f"http://minio/put/{key}",
    }


def test_대사_작업은_줄마다_MP3를_올리고_길이를_보고한다() -> None:
    api = FakeApi([speech_job([line("p1-1", "안녕하세요"), line("p2-1", "응")])])
    backends = FakeBackends()
    with respx.mock:
        put = respx.put(url__startswith="http://minio/put/").mock(return_value=httpx.Response(200))
        worker(api, backends).run(max_polls=1)
    assert api.results == [
        ("s", {"lines": [{"key": "p1-1", "durationMs": 500}, {"key": "p2-1", "durationMs": 100}]})
    ]
    assert put.call_count == 2
    assert put.calls.last.request.headers["Content-Type"] == "audio/mpeg"
    assert backends.spoken[0] == ("f-child-bright", "happy", "안녕하세요")
    # 그림 모델 묶음을 내리지 않는다
    assert backends.loaded == ["SDXL"]


def test_못_읽은_줄은_빼고_나머지를_보고한다() -> None:
    api = FakeApi([speech_job([line("p1-1"), line("p1-2")])])
    with respx.mock:
        respx.put("http://minio/put/p1-1").mock(return_value=httpx.Response(500))
        respx.put("http://minio/put/p1-2").mock(return_value=httpx.Response(200))
        worker(api).run(max_polls=1)
    assert api.results == [("s", {"lines": [{"key": "p1-2", "durationMs": 200}]})]


def test_한_줄도_못_읽으면_작업_실패다() -> None:
    api = FakeApi([speech_job([line("p1-1")])])
    with respx.mock:
        respx.put("http://minio/put/p1-1").mock(return_value=httpx.Response(500))
        worker(api).run(max_polls=1)
    assert api.results == []
    assert api.fails == [("s", "MODEL_ERROR", False)]


def test_읽기_서버가_없으면_다시_하도록_돌려보낸다() -> None:
    api = FakeApi([speech_job([line("p1-1")])])
    real = RealBackends(None, None, None, vram=lambda: None)  # type: ignore[arg-type]
    worker(api, real).run(max_polls=1)
    assert api.fails == [("s", "TTS_UNAVAILABLE", True)]


def test_대사_읽기는_켤_때만_가져간다() -> None:
    env = {"GPU_WORKER_API_URL": "http://api", "GPU_WORKER_TOKEN": "t"}
    off = Settings.from_env(env)
    on = Settings.from_env({**env, "SPEECH_ENABLED": "true", "VOICES_DIR": "D:/voices"})
    assert "SPEECH" not in job_types(off)
    assert job_types(on)[-1] == "SPEECH"
    assert build_speaker(off) is None
    speaker = build_speaker(on)
    assert isinstance(speaker, SovitsSpeaker)
    assert speaker.url == "http://127.0.0.1:9880"
    api = FakeApi([])
    w = worker(api)
    w.types = job_types(on)
    w.run_once()
    assert api.claims == [None]


def test_GPT_SoVITS에_감정_참조_음성과_그_글을_넘기고_MP3로_바꾼다(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def run(args: list[str], data: bytes) -> bytes:
        calls.append(args)
        return b"ID3" + data[:4]

    speaker = SovitsSpeaker("http://tts:9880/", str(voices(tmp_path)), ffmpeg="ffmpeg", run=run)
    with respx.mock:
        tts = respx.post("http://tts:9880/tts").mock(
            return_value=httpx.Response(200, content=wav_bytes(1500))
        )
        mp3, ms = speaker.speak("f-child-bright", "happy", "반가워", 1.1)
    body = json.loads(tts.calls.last.request.content)
    assert body["text"] == "반가워"
    assert body["prompt_text"] == "happy 참조 글"
    assert body["ref_audio_path"].endswith("happy.wav")
    assert body["speed_factor"] == 1.1
    assert body["text_lang"] == "ko"
    assert mp3.startswith(b"ID3")
    assert ms == 1500
    assert "libmp3lame" in calls[0]
    assert speaker.model_info() == {"tts": "GPT-SoVITS v4"}


def test_감정_참조가_없으면_기본_목소리를_쓰고_틀린_ID는_거절한다(tmp_path: Path) -> None:
    speaker = SovitsSpeaker("http://tts", str(voices(tmp_path)))
    ref, text = speaker.reference("f-child-bright", "sad")
    assert ref.name == "neutral.wav"
    assert text == "neutral 참조 글"
    with pytest.raises(JobInputError):
        speaker.reference("../etc", "neutral")
    with pytest.raises(JobInputError):
        speaker.reference("f-child-bright", "bored")
    with pytest.raises(JobInputError):
        speaker.reference("m-none", "neutral")


def test_읽기_서버에_기다려도_닿지_않으면_SpeechUnavailable(tmp_path: Path) -> None:
    now = [0.0]
    naps: list[float] = []

    def sleep(s: float) -> None:
        naps.append(s)
        now[0] += s

    speaker = SovitsSpeaker(
        "http://tts", str(voices(tmp_path)), wait_s=10, sleep=sleep, clock=lambda: now[0]
    )
    with respx.mock:
        respx.post("http://tts/tts").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(SpeechUnavailableError):
            speaker.speak("f-child-bright", "neutral", "안녕")
    assert sum(naps) >= 10


def test_읽기_서버가_재시작_중이면_다시_뜰_때까지_기다렸다_읽는다(tmp_path: Path) -> None:
    now = [0.0]

    def sleep(s: float) -> None:
        now[0] += s

    speaker = SovitsSpeaker(
        "http://tts",
        str(voices(tmp_path)),
        run=lambda args, data: b"ID3",
        sleep=sleep,
        clock=lambda: now[0],
    )
    with respx.mock:
        respx.post("http://tts/tts").mock(
            side_effect=[
                httpx.ConnectError("refused"),
                httpx.ConnectError("refused"),
                httpx.Response(200, content=wav_bytes(800)),
            ]
        )
        mp3, ms = speaker.speak("f-child-bright", "neutral", "안녕")
    assert mp3 == b"ID3"
    assert ms == 800
    assert now[0] == 6.0


def test_실제_백엔드는_읽기와_모델_정보를_읽기_백엔드에_넘긴다() -> None:
    class Speaker:
        def speak(self, v: str, e: str, t: str, s: float) -> tuple[bytes, int]:
            return b"mp3", 1234

        def model_info(self) -> dict[str, Any]:
            return {"tts": "x"}

    real = RealBackends(None, None, None, vram=lambda: None, speaker=Speaker())  # type: ignore[arg-type]
    assert real.speak("v", "neutral", "t") == (b"mp3", 1234)
    assert real.model_info("SPEECH") == {"tts": "x"}


def test_WAV_길이() -> None:
    assert wav_ms(wav_bytes(2310, 16000)) == 2310
