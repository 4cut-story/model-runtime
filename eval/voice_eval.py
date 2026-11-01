"""말투 바꾸기만 여러 방식으로 다시 돌려 비교한다 (그림 없이 LLM만, 몇 분이면 끝난다).

comic_eval.py가 남긴 컨셉별 script.json을 읽어 원래 대사로 되돌린 뒤, 변형(모델, 생각 모드)마다
ReviewHandler와 같은 말투 바꾸기를 부르고, 대사별 결과와 버린 까닭을 HTML 표로 남긴다.

예:
    python eval/voice_eval.py C:/claude-work/4cut/eval-out/20261101-231145 \
        --variants qwen3:8b,qwen3:8b+think,qwen3:14b --out voice-compare.html
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from comic_eval import (
    chat,
    clean_voice,
    content_json,
    keeps_facts,
    line_texts,
    llm_job,
    voice_prompt,
    voice_rejections,
    voice_schema,
)


def original(script: dict[str, Any]) -> dict[str, Any]:
    """comic_eval이 말투로 바꾼 콘티를 원래 대사로 되돌린다."""
    s = json.loads(json.dumps(script))
    for p in s.get("panels") or []:
        for line in p.get("lines") or []:
            if line.get("said"):
                line["text"] = line.pop("said")
    if s.get("originalTitle"):
        s["title"] = s.pop("originalTitle")
    return s


def scripts(roots: list[Path], concepts: dict[str, dict[str, Any]]) -> list[tuple[str, str, dict]]:
    out = []
    for root in roots:
        for path in sorted(root.glob("run-*/*/script.json")):
            cid = path.parent.name
            if concepts.get(cid, {}).get("voiceSystem"):
                name = f"{root.name}/{path.parent.parent.name}/{cid}"
                out.append((name, cid, original(json.loads(path.read_text(encoding="utf-8")))))
    return out


def variant(spec: str) -> tuple[str, bool]:
    model, _, flag = spec.partition("+")
    return model, flag == "think"


def run_one(
    llm: Any, bundle: dict[str, Any], concept: dict[str, Any], script: dict[str, Any], spec: str
) -> dict[str, Any]:
    model, think = variant(spec)
    n = len(line_texts(script))
    job = llm_job(
        bundle,
        "voice",
        voice_prompt(script),
        1,
        model,
        concept["voiceSystem"],
        voice_schema(bundle, n),
    )
    t0 = time.perf_counter()
    res = chat(llm, job, think)
    sec = round(time.perf_counter() - t0, 2)
    r: dict[str, Any] = {"variant": spec, "sec": sec, "raw": res["content"]}
    try:
        out = content_json(res["content"])
    except ValueError as e:
        return {**r, "error": f"파싱 실패: {e}", "lines": [], "used": [], "title": ""}
    lines = [clean_voice(x) for x in out.get("lines") or []]
    origin = line_texts(script)
    used = (
        [keeps_facts(o, v) for o, v in zip(origin, lines, strict=True)]
        if len(lines) == n
        else [False] * n
    )
    return {
        **r,
        "title": out.get("title", ""),
        "lines": lines,
        "used": used,
        "rejections": voice_rejections(script, lines),
        "unchanged": sum(1 for o, v in zip(origin, lines, strict=False) if o.strip() == v.strip()),
    }


def table(name: str, label: str, script: dict[str, Any], results: list[dict[str, Any]]) -> str:
    origin = line_texts(script)
    head = "".join(
        f"<th>{html.escape(r['variant'])}<br><small>{r['sec']}초</small></th>" for r in results
    )
    title_row = "".join(f"<td>{html.escape(r.get('title') or '')}</td>" for r in results)
    rows = []
    for i, o in enumerate(origin):
        cells = []
        for r in results:
            v = r["lines"][i] if i < len(r["lines"]) else ""
            cls = "" if i < len(r["used"]) and r["used"][i] else ' class="bad"'
            cells.append(f"<td{cls}>{html.escape(v)}</td>")
        rows.append(f"<tr><td>{html.escape(o)}</td>{''.join(cells)}</tr>")
    why = ["; ".join(r.get("rejections") or []) or r.get("error", "") for r in results]
    notes = "".join(
        f"<td class='meta'>{html.escape(w)}"
        f"{' · 그대로 ' + str(r['unchanged']) if r.get('unchanged') else ''}</td>"
        for r, w in zip(results, why, strict=True)
    )
    return f"""<section><h3>[{html.escape(label)}] {html.escape(name)}</h3>
<table><tr><th>원래 대사</th>{head}</tr>
<tr><td><b>{html.escape(script.get("title") or "")}</b></td>{title_row}</tr>
{"".join(rows)}<tr><td class="meta">버린 대사</td>{notes}</tr></table></section>"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--bundle", type=Path, default=Path(__file__).with_name("prompt_bundle.json"))
    p.add_argument("--variants", default="qwen3:8b,qwen3:8b+think")
    p.add_argument("--ollama", default="http://localhost:11434")
    p.add_argument("--out", type=Path, default=Path("voice-compare.html"))
    args = p.parse_args(argv)

    from model_runtime.backends.ollama import OllamaClient

    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    concepts = {c["id"]: c for c in bundle["concepts"]}
    llm = OllamaClient.create(args.ollama)
    specs = [v.strip() for v in args.variants.split(",") if v.strip()]
    items = scripts(args.roots, concepts)
    # 모델을 오가며 올리고 내리지 않도록 변형마다 모아서 돈다
    results: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in items}
    for spec in specs:
        for name, cid, script in items:
            results[name].append(run_one(llm, bundle, concepts[cid], script, spec))
            print(spec, name, results[name][-1]["sec"], flush=True)
    summary = {
        spec: {
            "lines": sum(len(line_texts(s)) for _, _, s in items),
            "used": sum(
                sum(r["used"]) for n, _, _ in items for r in results[n] if r["variant"] == spec
            ),
            "unchanged": sum(
                r.get("unchanged", 0)
                for n, _, _ in items
                for r in results[n]
                if r["variant"] == spec
            ),
            "sec": round(
                sum(r["sec"] for n, _, _ in items for r in results[n] if r["variant"] == spec), 1
            ),
        }
        for spec in specs
    }
    body = "".join(
        table(name, concepts[cid]["label"], script, results[name]) for name, cid, script in items
    )
    head = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v['used']}/{v['lines']}</td><td>{v['unchanged']}</td>"
        f"<td>{v['sec']}</td></tr>"
        for k, v in summary.items()
    )
    args.out.write_text(
        f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>말투 바꾸기 비교</title><style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#111}}
table{{border-collapse:collapse;margin:6px 0 24px}}
td,th{{border:1px solid #ccc;padding:3px 8px;font-size:14px;vertical-align:top}}
.bad{{background:#fde8e8;color:#900}}.meta{{color:#555;font-size:12px}}</style></head><body>
<h1>말투 바꾸기 비교 · 프롬프트 {html.escape(bundle["promptVersion"])}</h1>
<p class="meta">빨간 칸은 쓰지 못해 원래 대사로 두는 문장(개수 불일치·숫자 다름·너무 김)</p>
<table><tr><th>변형</th><th>쓴 대사</th><th>그대로 둔 대사</th><th>시간(초)</th></tr>{head}</table>
{body}</body></html>""",
        encoding="utf-8",
    )
    (args.out.with_suffix(".json")).write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(args.out, json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
