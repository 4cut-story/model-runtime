"""워커와 같은 순서로 대화 몇 건을 돌려 모델 갈아 끼우기 시간을 잰다 (GPU PC 전용).

워커처럼 받아쓰기 → LLM 2번 → 그림 4장 순서로, 다른 모델 묶음으로 넘어갈 때마다 앞 묶음을 내린다.
MODEL_KEEP_IN_RAM=true(기본)와 false를 같은 녹음으로 돌려 비교한다.

    python eval/swap_check.py --audio D:/4cut/samples/01.wav --rounds 3 --out swap_check.json
    set MODEL_KEEP_IN_RAM=false && python eval/swap_check.py --audio ... --out swap_check_off.json

녹음은 이 PC 안의 임시 HTTP 서버로만 넘긴다. 전사 글은 저장하지 않는다.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

PROMPT = (
    "simple cartoon, thick outlines, flat colors, plain background, medium shot, two people, "
    "(young woman, long black hair, yellow hoodie) talks to (young man, short brown hair, "
    "green jacket), cafe"
)


def ram() -> dict[str, Any]:
    try:
        import psutil
    except ImportError:
        return {"error": "psutil 없음"}
    vm = psutil.virtual_memory()
    return {
        "availableMB": vm.available // 2**20,
        "rssMB": psutil.Process().memory_info().rss // 2**20,
    }


def serve(path: Path) -> tuple[http.server.ThreadingHTTPServer, str]:
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(path.resolve().parent)
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/{path.name}"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--model", default="qwen3:8b")
    p.add_argument("--out", type=Path, default=Path("swap_check.json"))
    args = p.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model_runtime.__main__ import build_backends
    from model_runtime.config import Settings, load_env_file

    load_env_file(Path(".env"))
    env = {k: v for k, v in os.environ.items() if v}
    settings = Settings.from_env(
        {"GPU_WORKER_API_URL": "x", "GPU_WORKER_TOKEN": "x", **env, "WORKER_BACKEND": "real"}
    )
    backends: Any = build_backends(settings)
    server, url = serve(args.audio)
    chat = {
        "model": args.model,
        "system": "",
        "prompt": "한 문장으로 오늘 날씨 이야기를 지어 줘.",
        "options": {},
    }
    report: dict[str, Any] = {"keepInRam": settings.keep_in_ram, "ramStart": ram(), "rounds": []}
    loaded: str | None = None

    def switch(model: str) -> float:
        nonlocal loaded
        start = time.perf_counter()
        if loaded not in (None, model):
            backends.unload(loaded)
        backends.prepare(model)
        loaded = model
        return round(time.perf_counter() - start, 2)

    def timed(fn: Any) -> float:
        start = time.perf_counter()
        fn()
        return round(time.perf_counter() - start, 2)

    try:
        for n in range(args.rounds):
            r: dict[str, Any] = {"round": n + 1}
            r["unloadToWhisperSec"] = switch("WHISPER")
            r["transcribeSec"] = timed(lambda: backends.transcribe([url], "ko"))
            r["unloadToLlmSec"] = switch("LLM")
            r["llmSec"] = [timed(lambda: backends.chat(chat)) for _ in range(2)]
            r["unloadToSdxlSec"] = switch("SDXL")
            r["imageSec"] = [
                timed(lambda i=i: backends.draw({"prompt": PROMPT, "seed": i})) for i in range(4)
            ]
            r["totalSec"] = round(
                r["unloadToWhisperSec"]
                + r["transcribeSec"]
                + r["unloadToLlmSec"]
                + sum(r["llmSec"])
                + r["unloadToSdxlSec"]
                + sum(r["imageSec"]),
                1,
            )
            r["ram"] = ram()
            report["rounds"].append(r)
            args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
            print(json.dumps(r, ensure_ascii=False))
    finally:
        if loaded:
            backends.unload(loaded)
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
