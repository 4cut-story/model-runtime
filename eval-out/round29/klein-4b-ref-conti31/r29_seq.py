"""round29 나눠 그리기(A 먼저, 그 위에 B). r29_ref.py와 같고, stage 2 컷은 baseFrom 그림을 첫 입력 그림으로 넣는다.

round29 참조 그림 렌더. render_env.py와 같지만 컷마다 캐릭터 참조 그림을 받아 draw(image, refs)로 넘긴다.

    python r29_ref.py panels.json refs.json OUT_DIR KEY=VAL ...   (REFS=0 이면 참조 없이)
참조: [{character, imageUrl: https://4cut.java21.net/api/images/looks/{lookId}, region, scale 0.6}]
두 명이면 left/right(inScene 순서), 한 명이면 center. lookId 없는 캐릭터는 뺀다.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\4cut\model-runtime")

import httpx  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

from model_runtime.__main__ import build_image_backend  # noqa: E402
from model_runtime.backends import Reference  # noqa: E402
from model_runtime.config import Settings  # noqa: E402

panels_path, refs_path, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
overrides = dict(kv.split("=", 1) for kv in sys.argv[4:])
use_refs = overrides.pop("REFS", "1") != "0"
env = dict(os.environ)
env.update({"GPU_WORKER_API_URL": "http://offline.invalid", "GPU_WORKER_TOKEN": "offline", "WORKER_BACKEND": "real"})
env.update(overrides)
panels = json.loads(Path(panels_path).read_text(encoding="utf-8"))
refmap = json.loads(Path(refs_path).read_text(encoding="utf-8"))["panels"]
out.mkdir(parents=True, exist_ok=True)
info = {"settings": overrides, "reference": use_refs}
if (out / "timing.json").exists():
    old = json.loads((out / "timing.json").read_text(encoding="utf-8"))
    info["panelSeconds"] = old.get("panelSeconds", {})
    info["referencesUsed"] = old.get("referencesUsed", {})


def save(extra: dict) -> None:
    info.update(extra)
    info["peakRamGiB"] = round(psutil.Process().memory_info().peak_wset / 2**30, 2)
    (out / "timing.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")


def ref_items(name: str) -> list[dict]:
    chars = [c for c in (refmap.get(name) or {}).get("inScene", []) if c.get("lookId")]
    total = len((refmap.get(name) or {}).get("inScene", []))
    items = []
    for i, c in enumerate(chars):
        region = "center" if total == 1 else ("left" if i == 0 else "right")
        items.append({"character": c.get("storyName") or c["speaker"],
                      "imageUrl": f"https://4cut.java21.net/api/images/looks/{c['lookId']}",
                      "region": region, "scale": 0.6})
    return items


t = time.perf_counter()
try:
    backend = build_image_backend(Settings.from_env(env))
    be = getattr(backend, "active", backend)
    if hasattr(be, "load"):
        be.load()
except Exception as e:  # 로드 실패도 남긴다
    save({"loadError": f"{type(e).__name__}: {e}"[:1500]})
    raise
info["loadSeconds"] = round(time.perf_counter() - t, 1)
info["backend"] = type(be).__name__
torch.cuda.reset_peak_memory_stats()
cache: dict[str, bytes] = {}
http = httpx.Client(timeout=60, follow_redirects=True)
times = info.setdefault("panelSeconds", {})
used = info.setdefault("referencesUsed", {})
failed = {}
for p in panels:
    name = p["name"]
    if (out / f"{name}.png").exists():
        continue
    refs = []
    if use_refs:
        for it in ref_items(name):
            try:
                if it["imageUrl"] not in cache:
                    r = http.get(it["imageUrl"])
                    r.raise_for_status()
                    cache[it["imageUrl"]] = be.encode_reference(r.content)
                emb = cache[it["imageUrl"]]
                if emb is not None:
                    refs.append(Reference(it["character"], emb, it["scale"], it["region"]))
            except Exception as e:  # noqa: BLE001
                print(name, "ref fail", it["character"], e, flush=True)
        used[name] = [{"character": r.character, "region": r.region} for r in refs]
    if p.get("stage") == 2:
        base = (out / f"{p['baseFrom']}.png").read_bytes()
        refs.insert(0, Reference("__base__", base, 1.0, "full"))
        used[name] = [{"character": "__base__:" + p["baseFrom"], "region": "full"}] + used.get(name, [])
    t = time.perf_counter()
    try:
        (out / f"{name}.png").write_bytes(be.draw(p, refs) if use_refs else be.draw(p))
        times[name] = round(time.perf_counter() - t, 2)
        info.setdefault("stageSeconds", {}).setdefault(str(p.get("stage", 1)), {})[name] = times[name]
        print(name, times[name], "s refs", len(refs), flush=True)
    except Exception as e:  # 한 컷이 실패해도 나머지는 그린다
        failed[name] = f"{type(e).__name__}: {e}"[:300]
        print(name, "FAILED", failed[name], flush=True)
        torch.cuda.empty_cache()
    save({"failed": failed})
if hasattr(be, "model_info"):
    try:
        info["modelInfo"] = be.model_info()
    except Exception:  # noqa: BLE001
        pass
secs = list(times.values())
save({"failed": failed, "peakVramGiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
      "secondsPerPanel": round(sum(secs) / len(secs), 2) if secs else None})
print("RENDERDONE", json.dumps({k: info[k] for k in ("loadSeconds", "peakVramGiB", "secondsPerPanel")}), flush=True)
