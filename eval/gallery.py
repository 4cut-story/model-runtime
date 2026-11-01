"""comic_eval.py 결과 폴더들을 한 장의 HTML 갤러리로 묶는다 (이미지는 JPEG로 줄여 HTML 안에 넣는다).

예:
    python eval/gallery.py D:/4cut/eval-out/20261101-140000 D:/4cut/eval-out/20261101-150000 \
        --out gallery.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path
from typing import Any

STEPS = ["extract", "script", "draw", "compose"]


def jpeg(path: Path, width: int) -> str:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def run_card(run_dir: Path, report: dict[str, Any]) -> str:
    t = report.get("timings", {})
    cells = "".join(f"<td>{t.get(s, '-')}</td>" for s in STEPS)
    script_path = run_dir / "script.json"
    cast = ""
    if script_path.exists():
        script = json.loads(script_path.read_text(encoding="utf-8"))
        cast = ", ".join(
            f"{html.escape(c.get('name') or c['speaker'])}({html.escape(c.get('persona') or '')})"
            for c in script.get("characters", [])
        )
    comic = run_dir / "comic.png"
    img = f'<img src="{jpeg(comic, 520)}" alt="comic">' if comic.exists() else ""
    fails = "; ".join(report.get("scriptFailures") or [])
    sc = report.get("scene") or {}
    scene = (
        f'<p class="meta">장면 {html.escape(str(sc.get("kind")))} '
        f"{sc.get('start')}~{sc.get('end')}초"
        f" · 점수 {sc.get('score')} · 추천 컨셉 {html.escape(str(sc.get('concept') or '-'))}"
        f" · {html.escape(str(sc.get('summary')))}</p>"
        if sc
        else ""
    )
    heading = html.escape(report.get("title") or report.get("result", ""))
    if report.get("conceptLabel"):
        heading = f"[{html.escape(report['conceptLabel'])}] {heading}"
    return f"""
<section class="run">
  <h3>{html.escape(report.get("dir") or run_dir.name)} · {heading}</h3>
  <table><tr><th>추출</th><th>콘티</th><th>그림</th><th>합성</th><th>합계</th></tr>
  <tr>{cells}<td>{report.get("totalSec", "-")}</td></tr></table>
  <p class="meta">결과 {html.escape(report.get("result", ""))} · 콘티 시도
  {report.get("scriptAttempts", "-")} · 인물 {cast}</p>
  {scene}
  {f'<p class="fail">콘티 재시도 사유: {html.escape(fails)}</p>' if fails else ""}
  {img}
</section>"""


def batch(root: Path) -> str:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    note = (root / "NOTE.md").read_text(encoding="utf-8") if (root / "NOTE.md").exists() else ""
    runs = "".join(
        run_card(root / r["dir"] if r.get("dir") else root / f"run-{i + 1:02d}", r)
        for i, r in enumerate(summary.get("runs", []))
    )
    sdxl = html.escape(json.dumps(summary.get("sdxl", {}), ensure_ascii=False))
    return f"""
<article class="batch">
  <h2>{html.escape(root.name)} · 프롬프트 {html.escape(summary.get("promptVersion", ""))}</h2>
  <p class="meta">녹음 {summary.get("audioSec", "-")}초 · 전사 {summary.get("transcribeSec", "-")}초
  · {sdxl}</p>
  {f"<p>{html.escape(note)}</p>" if note else ""}
  <div class="runs">{runs}</div>
</article>"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("gallery.html"))
    args = p.parse_args()
    body = "".join(batch(r) for r in args.roots)
    args.out.write_text(
        f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>4컷 만화 반복 결과</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#111}}
.runs{{display:flex;flex-wrap:wrap;gap:24px}}.run{{width:540px}}
img{{width:520px;border:1px solid #ddd}}table{{border-collapse:collapse;margin:6px 0}}
td,th{{border:1px solid #ccc;padding:2px 8px;font-size:13px}}.meta{{color:#555;font-size:13px}}
.fail{{color:#b00;font-size:13px}}</style></head><body><h1>4컷 만화 반복 결과</h1>{body}
</body></html>""",
        encoding="utf-8",
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
