"""테마별 캐릭터 참조 그림 후보를 그린다 (GPU PC).

운영 워커와 같은 SdxlBackend(SDXL base + Lightning 8 step, 768px)를 그대로 쓴다. 캐릭터마다 시드 N개로 후보를
그리고, 후보마다 IP-Adapter Plus-Face embedding도 미리 계산해 둔다 (register.py가 함께 올리면 워커가 첫 그림에서
다시 계산하지 않는다). 테마마다 비교 시트(sheet_<THEME>.png)를 만든다.

  python draw.py --catalog catalog.yaml --out out [--candidates 4] [--theme JOSEON] [--only 김선비]

PYTHONPATH에 model-runtime 체크아웃이 있어야 한다 (import model_runtime).
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import yaml

FRAME = "character reference portrait, single {look}, upper body, facing the viewer, centered, plain light background"
BASE_SEED = 1000


def slug(theme: str, index: int) -> str:
    return f"{theme.lower()}_{index:02d}"


def prompt(style: str, look: str) -> str:
    return f"{style}, " + FRAME.format(look=look)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="catalog.yaml")
    ap.add_argument("--out", default="out")
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--theme", action="append", help="이 테마만 (여러 번 가능)")
    ap.add_argument("--only", action="append", help="이 캐릭터 이름만 (여러 번 가능)")
    ap.add_argument("--no-embed", action="store_true", help="IP-Adapter embedding을 계산하지 않는다")
    args = ap.parse_args()

    from model_runtime.backends.sdxl import SdxlBackend
    from model_runtime.config import DEFAULT_SDXL_IP_ADAPTER, DEFAULT_SDXL_LORA

    catalog = yaml.safe_load(Path(args.catalog).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend = SdxlBackend(
        model_name="stabilityai/stable-diffusion-xl-base-1.0",
        lora=DEFAULT_SDXL_LORA,
        steps=8,
        guidance=0.0,
        ip_adapter="" if args.no_embed else DEFAULT_SDXL_IP_ADAPTER,
    )
    t0 = time.time()
    backend.load()
    print(f"SDXL 올림 {time.time() - t0:.1f}s", flush=True)

    manifest: dict[str, dict] = {}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for theme, spec in catalog["themes"].items():
        if args.theme and theme not in args.theme:
            continue
        rows = []
        for i, ch in enumerate(spec["characters"]):
            if args.only and ch["name"] not in args.only:
                continue
            d = out / theme / slug(theme, i)
            d.mkdir(parents=True, exist_ok=True)
            p = prompt(spec["style"], ch["look"])
            cands = []
            for k in range(args.candidates):
                seed = BASE_SEED + k
                t = time.time()
                png = backend.draw(
                    {"prompt": p, "negativePrompt": spec.get("negative") or None,
                     "width": 768, "height": 768, "seed": seed}
                )
                (d / f"seed{seed}.png").write_bytes(png)
                if not args.no_embed:
                    emb = backend.encode_reference(png)
                    if emb:
                        (d / f"seed{seed}.ipadapter.safetensors").write_bytes(emb)
                cands.append(seed)
                print(f"{theme} {ch['name']} seed{seed} {time.time() - t:.2f}s", flush=True)
            manifest[ch["name"]] = {"theme": theme, "dir": str(d.relative_to(out)), "prompt": p, "seeds": cands}
            rows.append((ch["name"], d, cands))
        if rows:
            sheet(out / f"sheet_{theme}.png", rows)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"끝 {time.time() - t0:.1f}s", flush=True)


def sheet(path: Path, rows: list[tuple[str, Path, list[int]]]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    cell, label = 256, 28
    cols = max(len(c) for _, _, c in rows)
    img = Image.new("RGB", (cols * cell, len(rows) * (cell + label)), "white")
    draw = ImageDraw.Draw(img)
    font = None
    for f in ("C:/Windows/Fonts/malgun.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc"):
        try:
            font = ImageFont.truetype(f, 18)
            break
        except OSError:
            continue
    for r, (name, d, seeds) in enumerate(rows):
        y = r * (cell + label)
        for c, seed in enumerate(seeds):
            im = Image.open(io.BytesIO((d / f"seed{seed}.png").read_bytes())).resize((cell, cell))
            img.paste(im, (c * cell, y + label))
            draw.text((c * cell + 4, y + 4), f"{name} · seed{seed}", fill="black", font=font)
    img.save(path)


if __name__ == "__main__":
    main()
