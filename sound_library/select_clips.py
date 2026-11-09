"""FSD50K(환경음·효과음)와 FMA(배경음악)에서 상업적으로 쓸 수 있는(CC0, CC BY) 클립만 골라 후보 목록을 만든다.

  python select_clips.py --fsd FSD50K_메타데이터_폴더 --fma fma_metadata/raw_tracks.csv --total 1000 --out candidates.jsonl

- NC(비상업), ND(변경 금지), Sampling+, SA(동일 조건 변경 허락)는 받지 않는다.
- 카테고리마다 같은 수를 목표로 고르고, 모자라면 다른 카테고리로 넘긴다.
- --exclude로 이미 등록한 클립을 빼면 2차(5,000개)도 같은 규칙으로 이어서 고른다.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
from pathlib import Path

from categories import AMBIENCE, EFFECT, FMA_GENRES, Category
from remote_zip import RemoteZip

ZENODO = "https://zenodo.org/records/4060432/files/"
FSD_PARTS = {
    "eval": [
        f"{ZENODO}FSD50K.eval_audio.z01?download=1",
        f"{ZENODO}FSD50K.eval_audio.zip?download=1",
    ],
    "dev": [f"{ZENODO}FSD50K.dev_audio.z0{i}?download=1" for i in range(1, 6)]
    + [f"{ZENODO}FSD50K.dev_audio.zip?download=1"],
}
FMA_ZIP = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"

LICENSES = {
    "http://creativecommons.org/publicdomain/zero/1.0/": "CC0-1.0",
    "http://creativecommons.org/licenses/by/3.0/": "CC-BY-3.0",
}
AMBIENT_TAGS = {
    "ambience",
    "ambient",
    "ambiance",
    "atmosphere",
    "atmo",
    "field-recording",
    "soundscape",
    "fieldrecording",
    "environment",
    "background",
    "room-tone",
    "nature",
    "loop",
}


def fma_license(url: str) -> str | None:
    m = re.match(r"https?://creativecommons\.org/(licenses/by|publicdomain/zero)/(\d\.\d)", url)
    if not m:
        return None
    return ("CC-BY-" if m.group(1) == "licenses/by" else "CC0-") + m.group(2)


def fsd_pool(meta: Path) -> list[dict]:
    labels: dict[str, list[str]] = {}
    for split in ("eval", "dev"):
        with open(meta / "FSD50K.ground_truth" / f"{split}.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels[row["fname"]] = row["labels"].split(",")
    out = []
    for split in ("eval", "dev"):
        info = json.loads(
            (meta / "FSD50K.metadata" / f"{split}_clips_info_FSD50K.json").read_text()
        )
        zipped = RemoteZip(FSD_PARTS[split])
        for fid, v in info.items():
            lic = LICENSES.get(v["license"])
            name = f"FSD50K.{split}_audio/{fid}.wav"
            if lic is None or name not in zipped.entries:
                continue
            size = zipped.entries[name].size
            out.append(
                {
                    "source": "FSD50K",
                    "split": split,
                    "source_id": fid,
                    "zip_name": name,
                    "duration": round((size - 44) / (44100 * 2), 2),
                    "title_original": v["title"],
                    "author": v["uploader"],
                    "license": lic,
                    "license_url": v["license"],
                    "source_url": f"https://freesound.org/s/{fid}/",
                    "labels": labels.get(fid, []),
                    "tags": v.get("tags", []),
                    "note": re.sub(r"<[^>]+>", "", v.get("description", ""))[:600],
                }
            )
    return out


def fma_pool(raw_tracks: Path) -> list[dict]:
    csv.field_size_limit(10**9)
    zipped = RemoteZip([FMA_ZIP])
    out = []
    with open(raw_tracks, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lic = fma_license(row["license_url"])
            tid = int(row["track_id"])
            name = f"fma_medium/{tid // 1000:03d}/{tid:06d}.mp3"
            if lic is None or name not in zipped.entries:
                continue
            genres = [g["genre_title"] for g in ast.literal_eval(row["track_genres"] or "[]")]
            out.append(
                {
                    "source": "FMA",
                    "source_id": str(tid),
                    "zip_name": name,
                    "duration": 30.0,
                    "title_original": row["track_title"],
                    "author": row["artist_name"],
                    "license": lic,
                    "license_url": row["license_url"],
                    "source_url": row["track_url"],
                    "labels": genres,
                    "tags": ast.literal_eval(row["tags"] or "[]"),
                    "instrumental": row["track_instrumental"] == "1",
                    "note": re.sub(r"<[^>]+>", "", row["track_information"] or "")[:600],
                }
            )
    return out


def score(c: dict, cat: Category) -> float:
    tags = {t.lower() for t in c["tags"]}
    s = min(c["duration"], 30.0) / 30.0
    if cat.kind == "AMBIENCE":
        s += 0.5 * bool(tags & AMBIENT_TAGS)
    s += 0.3 * bool(tags & set(cat.keywords))
    if cat.kind == "MUSIC":
        s += 0.7 * c.get("instrumental", False)
    return s + random.random() * 0.4


def classify(c: dict, cats: list[Category]) -> Category | None:
    labels = set(c["labels"])
    tags = {t.lower() for t in c["tags"]}
    for cat in cats:
        if cat.min_seconds and c["duration"] < cat.min_seconds:
            continue
        if cat.max_seconds and c["duration"] > cat.max_seconds:
            continue
        if cat.exclude & labels:
            continue
        if cat.all_labels and not cat.all_labels <= labels:
            continue
        if (cat.labels & labels) or (
            cat.keywords and set(cat.keywords) & tags and cat.keyword_only
        ):
            return cat
    return None


def pick(pool: list[dict], cats: list[Category], total: int, seen: set[str]) -> list[dict]:
    buckets: dict[str, list[dict]] = {c.code: [] for c in cats}
    for c in pool:
        if f"{c['source']}:{c['source_id']}" in seen:
            continue
        cat = classify(c, cats)
        if cat:
            c["kind"], c["category"] = cat.kind, cat.code
            buckets[cat.code].append(c)
    by_code = {c.code: c for c in cats}
    for code, items in buckets.items():
        items.sort(key=lambda x: score(x, by_code[code]), reverse=True)
    chosen: list[dict] = []
    left = total
    open_codes = [c.code for c in cats if buckets[c.code]]
    while left > 0 and open_codes:
        share = max(1, left // len(open_codes))
        nxt = []
        for code in open_codes:
            take = buckets[code][:share]
            buckets[code] = buckets[code][share:]
            chosen += take
            left -= len(take)
            if buckets[code]:
                nxt.append(code)
            if left <= 0:
                break
        open_codes = nxt
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fsd", type=Path, required=True)
    ap.add_argument("--fma", type=Path, required=True)
    ap.add_argument("--ambience", type=int, default=550)
    ap.add_argument("--effect", type=int, default=150)
    ap.add_argument("--music", type=int, default=300)
    ap.add_argument("--exclude", type=Path, help="이미 고른 candidates.jsonl (2차에서 뺀다)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    random.seed(a.seed)
    seen: set[str] = set()
    if a.exclude and a.exclude.exists():
        for line in a.exclude.read_text(encoding="utf-8").splitlines():
            c = json.loads(line)
            seen.add(f"{c['source']}:{c['source_id']}")
    fsd = fsd_pool(a.fsd)
    chosen = pick(fsd, AMBIENCE, a.ambience, seen)
    seen |= {f"{c['source']}:{c['source_id']}" for c in chosen}
    chosen += pick(fsd, EFFECT, a.effect, seen)
    chosen += pick(fma_pool(a.fma), FMA_GENRES, a.music, seen)
    with open(a.out, "w", encoding="utf-8") as f:
        for c in chosen:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(chosen)}개를 골랐다 → {a.out}")


if __name__ == "__main__":
    main()
