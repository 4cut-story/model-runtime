"""단계별로 "모델 올리기"와 "실제 처리" 시간을 나눠 잰다 (GPU PC 전용).

워커는 16GB VRAM에 모델 묶음 하나만 두므로 대화마다 전사 → LLM → 그림 순서로 모델을 갈아 끼운다.
이 스크립트는 그 비용과 튜닝 후보의 효과를 잰다.

    python eval/stage_timing.py --audio D:/4cut/samples/01.m4a --out D:/4cut/eval-out/timing.json

잰다:
1. 전사 (처음): Whisper·정렬·화자 분리 모델 올리기, 받아쓰기, 정렬, 화자 분리
2. 전사 (모델을 올려 둔 채 다시): batch 8, batch 16
3. (--turbo) large-v3-turbo로 받아쓰기 시간과 large-v3 대비 글자 차이
4. LLM: Ollama 모델 내린 뒤 처음 부르기(load_duration)와 이어 부르기
5. 그림: SDXL 올리기, 4장 그리기
6. Whisper와 SDXL을 함께 올렸을 때 VRAM (둘 다 올려 둘 수 있는지)
7. 정렬·화자 분리를 CPU에서 돌렸을 때 시간 (작은 모델만 CPU로 옮기는 안)

전사 글은 --out 옆 *.text.txt에만 남는다. 녹음·전사는 커밋하지 않는다.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

PROMPT = (
    "simple cartoon, thick outlines, flat colors, plain background, medium shot, two people, "
    "(young woman, long black hair, yellow hoodie) talks to (young man, short brown hair, "
    "green jacket), cafe"
)


def vram() -> dict[str, Any]:
    import torch

    exe = shutil.which("nvidia-smi") or "nvidia-smi"
    try:
        used = subprocess.run(  # noqa: S603 - 고정 인자로 nvidia-smi만 실행
            [exe, "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        used = f"nvidia-smi 실패: {e}"
    return {"torchReservedMB": round(torch.cuda.memory_reserved() / 2**20), "nvidiaSmi": used}


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    import torch

    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, round(time.perf_counter() - start, 2)


def text_of(segments: list[dict[str, Any]]) -> str:
    return " ".join(s.get("text", "").strip() for s in segments)


def transcribe_parts(whisper: Any, audio: Any, batch: int, report: dict[str, Any]) -> list[Any]:
    """WhisperBackend.transcribe_file과 같은 순서로 하되 단계마다 시간을 잰다."""
    import whisperx

    result, report["transcribeSec"] = timed(
        lambda: whisper._model.transcribe(audio, batch_size=batch, language="ko")
    )
    (align, meta), report["alignLoadSec"] = timed(
        lambda: whisperx.load_align_model(language_code="ko", device="cuda")
    )
    aligned, report["alignSec"] = timed(
        lambda: whisperx.align(result["segments"], align, meta, audio, "cuda")
    )
    if whisper._diarize is not None:
        speakers, report["diarizeSec"] = timed(lambda: whisper._diarize(audio))
        aligned = whisperx.assign_word_speakers(speakers, aligned)
    segments: list[Any] = aligned["segments"]
    return segments


def measure_stt(s: Any, audio_path: Path, turbo: bool, out: Path) -> dict[str, Any]:
    import whisperx

    from model_runtime.backends.whisper import WhisperBackend

    report: dict[str, Any] = {}
    audio, report["audioDecodeSec"] = timed(lambda: whisperx.load_audio(str(audio_path)))
    report["audioSec"] = round(len(audio) / 16000, 1)

    whisper = WhisperBackend(s.whisper_model, s.hf_token)
    _, report["whisperLoadSec"] = timed(
        lambda: setattr(
            whisper,
            "_model",
            whisperx.load_model(whisper.model_name, "cuda", compute_type="float16"),
        )
    )
    if s.hf_token:
        from whisperx.diarize import DiarizationPipeline

        _, report["diarizeLoadSec"] = timed(
            lambda: setattr(
                whisper, "_diarize", DiarizationPipeline(token=s.hf_token, device="cuda")
            )
        )
    cold: dict[str, Any] = {}
    base = transcribe_parts(whisper, audio, 8, cold)
    base_result = whisper._model.transcribe(audio, batch_size=8, language="ko")["segments"]
    report["cold"] = cold
    report["vramWhisper"] = vram()
    for batch in (8, 16):
        warm: dict[str, Any] = {}
        transcribe_parts(whisper, audio, batch, warm)
        report[f"warmBatch{batch}"] = warm
    base_text = text_of(base)
    Path(f"{out}.large-v3.text.txt").write_text(base_text, "utf-8")

    if turbo:
        whisper._model = None
        model, report["turboLoadSec"] = timed(
            lambda: whisperx.load_model("large-v3-turbo", "cuda", compute_type="float16")
        )
        result, report["turboTranscribeSec"] = timed(
            lambda: model.transcribe(audio, batch_size=16, language="ko")
        )
        turbo_text = text_of(result["segments"])
        Path(f"{out}.turbo.text.txt").write_text(turbo_text, "utf-8")
        ratio = difflib.SequenceMatcher(None, base_text, turbo_text, autojunk=False).ratio()
        report["turboSameAsLargeV3"] = round(ratio, 3)
        del model
    report["cpu"] = measure_cpu_small(s, audio, base_result)
    whisper.unload()
    report["vramAfterUnload"] = vram()
    return report


def measure_cpu_small(s: Any, audio: Any, segments: list[Any]) -> dict[str, Any]:
    """정렬(wav2vec2)과 화자 분리(pyannote)를 CPU에서 돌린다. 받아쓰기는 GPU 결과를 쓴다."""
    import torch
    import whisperx

    report: dict[str, Any] = {"threads": torch.get_num_threads()}
    start = time.perf_counter()
    align, meta = whisperx.load_align_model(language_code="ko", device="cpu")
    report["alignLoadSec"] = round(time.perf_counter() - start, 2)
    start = time.perf_counter()
    whisperx.align(segments, align, meta, audio, "cpu")
    report["alignSec"] = round(time.perf_counter() - start, 2)
    del align
    if s.hf_token:
        from whisperx.diarize import DiarizationPipeline

        start = time.perf_counter()
        diarize = DiarizationPipeline(token=s.hf_token, device="cpu")
        report["diarizeLoadSec"] = round(time.perf_counter() - start, 2)
        start = time.perf_counter()
        diarize(audio)
        report["diarizeSec"] = round(time.perf_counter() - start, 2)
    return report


def measure_llm(ollama_url: str, model: str) -> dict[str, Any]:
    client = httpx.Client(base_url=ollama_url, timeout=300.0)
    client.post("/api/generate", json={"model": model, "keep_alive": 0, "prompt": ""})
    report: dict[str, Any] = {"model": model}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "한 문장으로 오늘 날씨 이야기를 지어 줘."}],
        "stream": False,
        "think": False,
        "keep_alive": "5m",
    }
    for label in ("cold", "warm"):
        start = time.perf_counter()
        data = client.post("/api/chat", json=body).json()
        report[label] = {
            "totalSec": round(time.perf_counter() - start, 2),
            "loadSec": round(data.get("load_duration", 0) / 1e9, 2),
            "tokensPerSec": round(
                data.get("eval_count", 0) / max(data.get("eval_duration", 1) / 1e9, 1e-9), 1
            ),
        }
    ps = client.get("/api/ps").json().get("models", [])
    report["vramMB"] = [round(m.get("size_vram", 0) / 2**20) for m in ps]
    client.post("/api/generate", json={"model": model, "keep_alive": 0, "prompt": ""})
    return report


def measure_sdxl(s: Any, with_whisper: bool) -> dict[str, Any]:
    from model_runtime.backends.sdxl import SdxlBackend

    sdxl = SdxlBackend(s.sdxl_model, s.sdxl_cpu_offload, s.sdxl_lora, s.sdxl_steps, s.sdxl_guidance)
    report: dict[str, Any] = {}
    _, report["loadSec"] = timed(sdxl.load)
    report["imageSec"] = [
        timed(lambda i=i: sdxl.draw({"prompt": PROMPT, "seed": i}))[1] for i in range(4)
    ]
    report["vramSdxl"] = vram()
    if with_whisper:
        import whisperx

        model = whisperx.load_model(s.whisper_model, "cuda", compute_type="float16")
        report["vramSdxlPlusWhisper"] = vram()
        report["imageSecWithWhisperLoaded"] = [
            timed(lambda i=i: sdxl.draw({"prompt": PROMPT, "seed": i}))[1] for i in range(2)
        ]
        del model
    sdxl.unload()
    return report


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("stage_timing.json"))
    p.add_argument("--model", default="qwen3:8b")
    p.add_argument("--turbo", action="store_true", help="large-v3-turbo도 재 본다")
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model_runtime.config import Settings, load_env_file

    load_env_file(Path(".env"))
    env = {k: v for k, v in os.environ.items() if v}
    s = Settings.from_env({"GPU_WORKER_API_URL": "x", "GPU_WORKER_TOKEN": "x", **env})

    report: dict[str, Any] = {"vramStart": vram()}
    report["stt"] = measure_stt(s, args.audio, args.turbo, args.out)
    report["llm"] = measure_llm(s.ollama_url, args.model)
    report["sdxl"] = measure_sdxl(s, with_whisper=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
