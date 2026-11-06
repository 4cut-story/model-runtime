"""WhisperX large-v3 전사 + pyannote 화자 분리 (GPU PC 전용).

HF_TOKEN이 없으면 화자 분리를 건너뛰고 모든 발화를 한 화자(SPEAKER_00)로 낸다.

정렬 모델(wav2vec2)은 언어별로 한 번만 불러 두고 다시 쓴다 (GPU PC 측정: 매번 새로 올리면 3~33초).
keep_in_ram이면 내릴 때 정렬 모델을 시스템 RAM으로 옮겨 둔다. Whisper 본체는 RAM 32GB에서
Ollama 모델 파일 캐시가 밀려나지 않도록 지금처럼 내린다.
"""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path
from typing import Any

import httpx

from .. import segments

SAMPLE_RATE = 16000

# 구간 전사는 새로 쌓인 음성이 이만큼은 돼야 한다. 모자라면 다음 조각 때 한다.
MIN_NEW_SEC = 45.0


class WhisperBackend:
    def __init__(
        self,
        model_name: str,
        hf_token: str | None,
        device: str = "cuda",
        keep_in_ram: bool = False,
        min_speakers: int = 0,
    ) -> None:
        self.model_name = model_name
        self.min_speakers = min_speakers
        self.hf_token = hf_token
        self.device = device
        self.keep_in_ram = keep_in_ram
        self._model: Any = None
        self._align: dict[str, tuple[Any, Any]] = {}
        self._diarize: Any = None
        self._tf32: tuple[bool, bool] | None = None

    def _restore_tf32(self) -> None:
        """pyannote는 재현성을 위해 TF32를 끄고 되돌리지 않는다. 같은 프로세스의 SDXL이 장당 약 13%
        느려지므로(1.33초 → 1.5초, GPU PC 측정) 불러오기 전 값으로 되돌린다."""
        import torch

        if self._tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = self._tf32

    def _speaker_hint(self) -> dict[str, int]:
        """pyannote에 줄 최소 화자 수. 0 이하면 정하지 않는다."""
        return {"min_speakers": self.min_speakers} if self.min_speakers > 0 else {}

    def load(self) -> None:
        import torch

        if self._tf32 is None:
            self._tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        import whisperx

        if self._model is None:
            self._model = whisperx.load_model(self.model_name, self.device, compute_type="float16")
        if self._diarize is None and self.hf_token:
            from whisperx.diarize import DiarizationPipeline

            self._diarize = DiarizationPipeline(token=self.hf_token, device=self.device)
        self._restore_tf32()

    def unload(self) -> None:
        import torch

        self._model = None
        self._diarize = None
        if self.keep_in_ram:
            for align_model, _ in self._align.values():
                align_model.to("cpu")
        else:
            self._align.clear()
        self._restore_tf32()
        gc.collect()
        torch.cuda.empty_cache()

    def transcribe(
        self,
        audio_urls: list[str],
        language: str | None,
        prefix: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """전체 전사 + 화자 분리.

        prefix(녹음 중 미리 전사한 앞부분)가 있으면 그 경계 뒤만 전사해 잇는다 (FR-015).
        """
        import whisperx

        self.load()
        audio = self._load_audio(audio_urls)
        if prefix and prefix.get("segments") is not None:
            cut = float(prefix.get("cut") or 0.0)
            detected = prefix.get("language") or language
            lo = max(0.0, cut - segments.OVERLAP_SEC)
            tail, detected = self._asr(audio[int(lo * SAMPLE_RATE) :], detected)
            tail = segments.keep_between(segments.shift(tail, lo), cut, float("inf"))
            prior = segments.shift(prefix["segments"], 0.0)
            merged = prior + tail
        else:
            merged, detected = self._asr(audio, language)
            merged = segments.shift(merged, 0.0)
        result: dict[str, Any] = {"segments": merged}
        if self._diarize is not None:
            try:
                speakers = self._diarize(audio, **self._speaker_hint())
            finally:
                self._restore_tf32()
            result = whisperx.assign_word_speakers(speakers, result)
        return {"segments": segments.to_output(result["segments"]), "language": detected}

    def transcribe_part(
        self, audio_urls: list[str], language: str | None, from_sec: float
    ) -> dict[str, Any]:
        """녹음 중 구간 전사 (화자 분리 없음).

        지금까지 녹음된 음성에서 from_sec 뒤를 전사하고 다음 경계(cut)를 정한다.
        """
        self.load()
        audio = self._load_audio(audio_urls)
        duration = len(audio) / SAMPLE_RATE
        if duration - from_sec < MIN_NEW_SEC:
            return {
                "segments": [],
                "cut": from_sec,
                "language": language,
                "durationSec": duration,
                "skipped": True,
            }
        lo = max(0.0, from_sec - segments.OVERLAP_SEC)
        part, detected = self._asr(audio[int(lo * SAMPLE_RATE) :], language)
        part = segments.shift(part, lo)
        cut = segments.choose_cut(part, duration)
        return {
            "segments": segments.keep_between(part, from_sec, cut),
            "cut": cut,
            "language": detected,
            "durationSec": duration,
            "skipped": False,
        }

    def _load_audio(self, audio_urls: list[str]) -> Any:
        import whisperx

        # 조각은 한 녹음(또는 한 파일)을 순서대로 자른 바이트다. 브라우저 MediaRecorder 조각은
        # 첫 조각에만 컨테이너 헤더가 있으므로 하나로 이어 붙인 뒤 한 번에 디코딩한다.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audio"
            with httpx.Client(timeout=120.0) as client, path.open("wb") as out:
                for url in audio_urls:
                    response = client.get(url)
                    response.raise_for_status()
                    out.write(response.content)
            return whisperx.load_audio(str(path))

    def _asr(self, audio: Any, language: str | None) -> tuple[list[dict[str, Any]], str]:
        """전사 + 단어 정렬. 시각은 audio 시작 기준."""
        import whisperx

        result = self._model.transcribe(audio, batch_size=8, language=language)
        detected = result.get("language") or language or "ko"
        align_model, metadata = self._align_model(detected)
        aligned = whisperx.align(result["segments"], align_model, metadata, audio, self.device)
        return list(aligned["segments"]), detected

    def _align_model(self, language: str) -> tuple[Any, Any]:
        """언어별 정렬 모델. 처음 한 번만 불러 오고, RAM에 옮겨 둔 것은 GPU로 다시 옮긴다."""
        import whisperx

        if language not in self._align:
            self._align[language] = whisperx.load_align_model(
                language_code=language, device=self.device
            )
        align_model, metadata = self._align[language]
        align_model.to(self.device)
        return align_model, metadata

    def model_info(self) -> dict[str, Any]:
        return {"whisper": self.model_name, "diarization": self._diarize is not None}
