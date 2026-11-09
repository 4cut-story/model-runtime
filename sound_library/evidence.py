"""설명을 쓸 근거를 클립마다 한 줄로 보여 준다. 아직 설명이 없는 클립만 나온다.

  python evidence.py --candidates candidates.jsonl --clap clap.jsonl --done descriptions.jsonl \
      --start 0 --n 50

설명은 이 줄(원본 제목·태그·FSD50K 정답 라벨, CLAP이 들은 소리·분위기·악기·템포)을 보고 사람이
(또는 이 대화의 Claude가) 직접 쓴다. 유료 API는 쓰지 않는다.
쓴 설명은 key|제목|설명|분위기,…|장소,…|x(뺄 때)|카테고리(옮길 때) 한 줄씩 적고
merge_descriptions.py로 descriptions.jsonl을 만든다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from categories import ALL


def key(c: dict[str, object]) -> str:
    return f"{str(c['source']).lower()}-{c['source_id']}"


def line(c: dict[str, object], x: dict[str, object] | None, label: str) -> str:
    if x is None:  # CLAP 전: 원본 정보만
        tags = ",".join(c["tags"][:8])  # type: ignore[index]
        labels = ",".join(c["labels"][:4])  # type: ignore[index]
        note = str(c.get("note") or "").replace("\n", " ")[:90]
        return (
            f"{key(c)} | {label} | {str(c['title_original'])[:50]} | {tags} | {labels} | {note}"
            f" | {c['duration']}s"
        )
    moods = x["moods"]
    assert isinstance(moods, dict)
    top = sorted(moods, key=lambda q: -moods[q])[:3]
    heard = ",".join(x["heard"][:3])  # type: ignore[index]
    attrs = x["attributes"]
    assert isinstance(attrs, dict)
    extra = ""
    if c["kind"] == "MUSIC":
        extra = (
            f" | 작가:{c['author']} | 악기:{','.join(x['instruments'])}"  # type: ignore[arg-type]
            f" | {attrs['tempo']}, {attrs['loud']}, bpm {attrs.get('bpm')}, {attrs.get('dynamics')}"
        )
    tags = ",".join(c["tags"][:8])  # type: ignore[index]
    labels = ",".join(c["labels"][:4])  # type: ignore[index]
    note = str(c.get("note") or "").replace("\n", " ")[:90]
    return (
        f"{key(c)} | {label} | {str(c['title_original'])[:50]} | {tags} | {labels} | {note}"
        f" | 들림:{heard}{extra} | CLAP분위기:{','.join(top)} | {c['duration']}s"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--clap", type=Path, required=True)
    ap.add_argument("--done", type=Path, nargs="*", default=[])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--kind", help="AMBIENCE·EFFECT·MUSIC 중 하나만")
    ap.add_argument("--no-clap", action="store_true", help="CLAP 결과 없이 원본 정보만 본다")
    a = ap.parse_args()
    labels = {c.code: c.label for c in ALL}
    clap = {}
    for row in a.clap.read_text(encoding="utf-8").splitlines():
        x = json.loads(row)
        clap[x["key"]] = x
    done: set[str] = set()
    for f in a.done:
        if f.exists():
            for r in f.read_text(encoding="utf-8").splitlines():
                if r.strip():
                    done.add(r.split("|")[0].strip() if "|" in r else json.loads(r)["key"])
    cands = [json.loads(r) for r in a.candidates.read_text(encoding="utf-8").splitlines()]
    todo = [
        c
        for c in cands
        if (a.no_clap or key(c) in clap)
        and key(c) not in done
        and (not a.kind or c["kind"] == a.kind)
    ]
    for c in todo[a.start : a.start + a.n]:
        print(line(c, clap.get(key(c)), labels[c["category"]]))
    print("남은 수", len(todo))


if __name__ == "__main__":
    main()
