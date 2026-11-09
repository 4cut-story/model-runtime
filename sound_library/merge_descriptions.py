"""손으로 쓴 설명(한 줄에 하나)을 register_sounds.py가 읽는 descriptions.jsonl로 바꾼다.

  python merge_descriptions.py --raw desc.raw [--base gemini.jsonl] --out descriptions.jsonl

줄 형식: key|제목|설명|분위기,…|장소,…|x 또는 카테고리|카테고리
- 여섯째 칸이 x면 등록하지 않는다(중복, 말소리가 중심인 클립, 카테고리와 다른 소리 등).
- 여섯째나 일곱째 칸의 카테고리 코드는 클립을 그 카테고리로 옮긴다.
- 같은 key가 여러 번 나오면 처음 것을 쓴다. --base에 있는 key는 raw가 덮어쓴다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from categories import ALL, MOODS

CODES = {c.code for c in ALL}


def parse(row: str) -> dict[str, object]:
    p = [x.strip() for x in row.split("|")]
    f5 = p[5] if len(p) > 5 else ""
    cat = p[6] if len(p) > 6 and p[6] else (f5 if f5 and f5 != "x" else None)
    if cat is not None and cat not in CODES:
        raise SystemExit(f"모르는 카테고리 {cat}: {p[0]}")
    moods = [m for m in p[3].split(",") if m]
    bad = [m for m in moods if m not in MOODS]
    if bad:
        raise SystemExit(f"모르는 분위기 {bad}: {p[0]}")
    return {
        "key": p[0],
        "model": "claude",
        "title": p[1],
        "description": p[2],
        "moods": moods,
        "places": [x for x in p[4].split(",") if x],
        "exclude": f5 == "x",
        "category": cat,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--base", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out: dict[str, dict[str, object]] = {}
    if a.base:
        for r in a.base.read_text(encoding="utf-8").splitlines():
            d = json.loads(r)
            out[d["key"]] = d
    seen: set[str] = set()
    for r in a.raw.read_text(encoding="utf-8").splitlines():
        if not r.strip():
            continue
        d = parse(r)
        k = str(d["key"])
        if k in seen:
            continue
        seen.add(k)
        out[k] = d
    with a.out.open("w", encoding="utf-8") as f:
        for d in out.values():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{len(out)}개, 뺄 것 {sum(1 for d in out.values() if d.get('exclude'))}개")


if __name__ == "__main__":
    main()
