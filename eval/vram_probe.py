"""전사 뒤 그림이 느려지는 원인 찾기 (GPU PC 전용).

같은 프로세스에서 WhisperX + pyannote 전사를 한 뒤 SDXL 그림이 4배쯤 느려진다(comic_eval 1·4회차).
이 스크립트는 단계마다 GPU 메모리와 TF32 설정을 찍고 그림 시간을 잰다.

    python eval/vram_probe.py --audio <wav> --out vram_probe.json

1. 전사 전에 그림 2장 (기준)
2. 전사 → unload 뒤 메모리·TF32 설정, 그림 2장
3. TF32를 다시 켠 뒤 그림 2장
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROMPT = (
    "simple cartoon, thick outlines, flat colors, plain background, medium shot, two people, "
    "(young woman, long black hair, yellow hoodie) talks to (young man, short brown hair, "
    "green jacket), cafe"
)


def snapshot(label: str) -> dict[str, Any]:
    import torch

    smi = ""
    exe = shutil.which("nvidia-smi") or "nvidia-smi"
    try:
        smi = subprocess.run(  # noqa: S603 - 고정 인자로 nvidia-smi만 실행
            [exe, "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        smi = f"nvidia-smi 실패: {e}"
    return {
        "label": label,
        "allocatedMB": round(torch.cuda.memory_allocated() / 2**20),
        "reservedMB": round(torch.cuda.memory_reserved() / 2**20),
        "nvidiaSmi": smi,
        "matmulTf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnnTf32": torch.backends.cudnn.allow_tf32,
    }


def draw_times(sdxl: Any, n: int) -> list[float]:
    out = []
    for i in range(n):
        start = time.perf_counter()
        sdxl.draw({"prompt": PROMPT, "seed": i, "width": 768, "height": 768})
        out.append(round(time.perf_counter() - start, 2))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("vram_probe.json"))
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model_runtime.backends.sdxl import SdxlBackend
    from model_runtime.backends.whisper import WhisperBackend
    from model_runtime.config import Settings, load_env_file

    load_env_file(Path(".env"))
    env = {k: v for k, v in os.environ.items() if v}
    s = Settings.from_env({"GPU_WORKER_API_URL": "x", "GPU_WORKER_TOKEN": "x", **env})
    report: dict[str, Any] = {"steps": []}

    sdxl = SdxlBackend(s.sdxl_model, s.sdxl_cpu_offload, s.sdxl_lora, s.sdxl_steps, s.sdxl_guidance)
    report["steps"].append(snapshot("시작"))
    report["beforeTranscribe"] = draw_times(sdxl, 3)
    report["steps"].append(snapshot("그림 뒤(전사 전)"))
    sdxl.unload()
    report["steps"].append(snapshot("SDXL unload"))

    whisper = WhisperBackend(s.whisper_model, s.hf_token)
    start = time.perf_counter()
    whisper.transcribe_file(str(args.audio), "ko")
    report["transcribeSec"] = round(time.perf_counter() - start, 2)
    report["steps"].append(snapshot("전사 직후"))
    whisper.unload()
    del whisper
    report["steps"].append(snapshot("Whisper unload"))

    report["afterTranscribe"] = draw_times(sdxl, 3)
    report["steps"].append(snapshot("그림 뒤(전사 후)"))

    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    report["afterTf32Restore"] = draw_times(sdxl, 3)
    report["steps"].append(snapshot("TF32 다시 켠 뒤"))

    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
