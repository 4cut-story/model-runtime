"""WhisperX large-v3 전사 + pyannote 화자 분리 (GPU PC 전용).

HF_TOKEN이 없으면 화자 분리를 건너뛰고 모든 발화를 한 화자(SPEAKER_00)로 낸다.
"""

from __future__ import annotations

import gc
import tempfile
from pathlib import Path
from typing import Any

import httpx

SAMPLE_RATE = 16000


class WhisperBackend:
    def __init__(self, model_name: str, hf_token: str | None, device: str = "cuda") -> None:
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = device
        self._model: Any = None
        self._diarize: Any = None

    def load(self) -> None:
        import whisperx

        if self._model is None:
            self._model = whisperx.load_model(self.model_name, self.device, compute_type="float16")
        if self._diarize is None and self.hf_token:
            from whisperx.diarize import DiarizationPipeline

            self._diarize = DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)

    def unload(self) -> None:
        import torch

        self._model = None
        self._diarize = None
        gc.collect()
        torch.cuda.empty_cache()

    def transcribe(self, audio_urls: list[str], language: str | None) -> dict[str, Any]:
        import numpy as np
        import whisperx

        self.load()
        with tempfile.TemporaryDirectory() as tmp:
            parts = []
            with httpx.Client(timeout=120.0) as client:
                for i, url in enumerate(audio_urls):
                    path = Path(tmp) / f"chunk-{i:02d}"
                    response = client.get(url)
                    response.raise_for_status()
                    path.write_bytes(response.content)
                    parts.append(whisperx.load_audio(str(path)))
            audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        result = self._model.transcribe(audio, batch_size=8, language=language)
        detected = result.get("language") or language or "ko"
        align_model, metadata = whisperx.load_align_model(
            language_code=detected, device=self.device
        )
        result = whisperx.align(result["segments"], align_model, metadata, audio, self.device)
        del align_model
        if self._diarize is not None:
            speakers = self._diarize(audio)
            result = whisperx.assign_word_speakers(speakers, result)
        segments = [
            {
                "speaker": s.get("speaker", "SPEAKER_00"),
                "start": float(s["start"]),
                "end": float(s["end"]),
                "text": s.get("text", "").strip(),
            }
            for s in result["segments"]
        ]
        return {"segments": segments, "language": detected}

    def model_info(self) -> dict[str, Any]:
        return {"whisper": self.model_name, "diarization": self._diarize is not None}
