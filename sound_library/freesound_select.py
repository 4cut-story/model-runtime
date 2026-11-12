"""Freesound에서 CC0 클립만 카테고리별로 골라 후보를 만든다 (4차부터).

  FREESOUND_API_KEY=... python freesound_select.py --total 13000 --out data/candidates-4.jsonl \
      --exclude data/candidates-1.jsonl data/candidates-2.jsonl data/candidates-3.jsonl

- 라이선스는 CC0 1.0(저작자 표시 없이 상업적으로도 자유롭게 쓸 수 있음)만 받는다.
- 카테고리(categories.py)의 영어 문장과 키워드로 찾고, 내려받기 많은 순으로 작가가 겹치지 않게 고른다.
- FSD50K 클립 번호는 Freesound 번호와 같아서, 앞 차수 후보와 겹치는 소리는 뺀다.
- 말소리가 중심인 클립, 너무 짧은 배경음은 뺀다. 미리 듣기 MP3(HQ) 주소를 같이 적어 fetch_clips.py가 받는다.
- 검색은 카테고리마다 몇 번만 부른다(하루 2,000번 제한 안). 키는 출력하지 않는다.
- 카테고리마다 바로 --out에 덧붙인다. 중간에 끊기면 다시 돌려 이미 고른 카테고리는 건너뛴다.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from categories import ALL, Category
from select_clips import jsonline

API = "https://freesound.org/apiv2/search/text/"
FIELDS = "id,name,tags,description,username,license,duration,previews,url,num_downloads"
CC0 = "http://creativecommons.org/publicdomain/zero/1.0/"
SPEECH_TAGS = {"speech", "voice", "talking", "narration", "spoken", "vocal", "singing", "words"}
PAGE = 150


def license_code(url: str) -> str | None:
    return "CC0-1.0" if "publicdomain/zero" in url else None


def search(token: str, query: str, filt: str, page: int) -> dict:
    params = {
        "query": query,
        "filter": filt,
        "fields": FIELDS,
        "page_size": str(PAGE),
        "page": str(page),
        "sort": "downloads_desc",
    }
    req = urllib.request.Request(
        API + "?" + urllib.parse.urlencode(params),
        headers={"Authorization": f"Token {token}", "User-Agent": "4cut-sound-library"},
    )
    for wait in (0, 5, 20, 60):
        time.sleep(wait)
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 404:  # 페이지 끝
                return {"results": []}
            if e.code not in (429, 500, 502, 503):
                raise
    return {"results": []}


def candidate(x: dict, cat: Category) -> dict | None:
    code = license_code(str(x.get("license", "")))
    if code is None:
        return None
    tags = [str(t).lower() for t in x.get("tags", [])]
    if SPEECH_TAGS & set(tags):
        return None
    duration = float(x.get("duration") or 0)
    if duration < max(cat.min_seconds, 1.0) or (cat.max_seconds and duration > cat.max_seconds):
        return None
    preview = (x.get("previews") or {}).get("preview-hq-mp3")
    if not preview:
        return None
    return {
        "source": "Freesound",
        "source_id": str(x["id"]),
        "duration": round(duration, 2),
        "title_original": x.get("name", ""),
        "author": x.get("username", ""),
        "license": code,
        "license_url": CC0,
        "source_url": x.get("url") or f"https://freesound.org/s/{x['id']}/",
        "labels": [],
        "tags": tags[:20],
        "note": str(x.get("description") or "")[:600],
        "preview_url": preview,
        "kind": cat.kind,
        "category": cat.code,
    }


def queries(cat: Category) -> list[str]:
    out = [cat.prompt] if cat.prompt else []
    out += [k for k in cat.keywords[:3] if k not in out]
    return out or [cat.label]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--total", type=int, default=13000, help="뽑을 후보 수 (설명 단계에서 빠질 몫까지)"
    )
    ap.add_argument("--kinds", default="AMBIENCE,EFFECT")
    ap.add_argument("--exclude", type=Path, nargs="*", default=[])
    ap.add_argument("--per-author", type=int, default=3)
    ap.add_argument("--max-pages", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    token = os.environ.get("FREESOUND_API_KEY", "")
    if not token:
        raise SystemExit("FREESOUND_API_KEY가 없다")

    seen: set[str] = set()
    for path in a.exclude:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(str(json.loads(line)["source_id"]))
    kinds = set(a.kinds.split(","))
    cats = [c for c in ALL if c.kind in kinds]
    codes: dict[str, Category] = {}
    for c in cats:
        codes.setdefault(c.code, c)
    per = max(1, a.total // len(codes))
    filt = 'license:"Creative Commons 0" duration:[1 TO 300]'

    chosen: list[dict] = []
    done: set[str] = set()
    if a.out.exists():
        for line in a.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                x = json.loads(line)
                chosen.append(x)
                seen.add(str(x["source_id"]))
                done.add(x["category"])
        print(f"이어서 고른다: 이미 {len(chosen)}개, 카테고리 {len(done)}개", flush=True)
    for cat in codes.values():
        if cat.code in done:
            continue
        got: list[dict] = []
        authors: dict[str, int] = {}
        for q in queries(cat):
            for page in range(1, a.max_pages + 1):
                res = search(token, q, filt, page).get("results", [])
                for x in res:
                    c = candidate(x, cat)
                    if c is None or c["source_id"] in seen:
                        continue
                    if authors.get(c["author"], 0) >= a.per_author:
                        continue
                    seen.add(c["source_id"])
                    authors[c["author"]] = authors.get(c["author"], 0) + 1
                    got.append(c)
                if len(got) >= per or len(res) < PAGE:
                    break
            if len(got) >= per:
                break
        chosen += got[:per]
        with open(a.out, "a", encoding="utf-8") as f:
            for c in got[:per]:
                f.write(jsonline(c) + "\n")
        print(f"{cat.code}: {len(got[:per])}개 (누적 {len(chosen)})", flush=True)
    print(f"{len(chosen)}개, 카테고리 {len({c['category'] for c in chosen})}개를 골랐다 → {a.out}")


if __name__ == "__main__":
    main()
