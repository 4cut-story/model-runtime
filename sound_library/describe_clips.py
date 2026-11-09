"""Gemini가 클립을 직접 듣고 한국어 제목·설명·분위기·장소 태그를 쓴다.

  GEMINI_API_KEY=... python describe_clips.py --candidates candidates.jsonl --audio audio/ \
      --clap clap.jsonl --out descriptions.jsonl

- 원본 제목·태그·FSD50K 정답 라벨과 CLAP 분위기 점수를 참고로 주되, 들리는 소리를 기준으로 쓰게 한다.
- 카테고리와 다른 소리(matches=false)나 노래가 들어간 음악(has_vocals=true)은 등록 단계에서 뺀다.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from categories import ALL, MOODS
from fetch_clips import key

MODELS = ["gemini-2.5-flash", "gemini-3-flash-preview"]
URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
CATS = {c.code: c for c in ALL}
KIND = {"AMBIENCE": "장소·환경음", "EFFECT": "효과음", "MUSIC": "배경음악"}

PROMPT = """너는 만화 서비스의 배경음 사서다. 첨부한 소리를 끝까지 듣고, 들리는 그대로 한국어로 정리해라.
참고 정보는 원본 올린 사람이 쓴 것이라 틀릴 수 있다. 실제로 들리는 소리를 기준으로 쓴다.
사람 말소리가 들려도 말한 내용을 옮겨 적지 말고 "아이들이 떠드는 소리"처럼 소리의 종류만 쓴다. 사람 이름, 상표, 노래 제목은 쓰지 않는다.

[분류] {kind} / {label}
[원본 제목] {title}
[원본 태그] {tags}
[원본 라벨] {labels}
[원본 설명] {note}
[CLAP이 고른 분위기 후보] {clap}
[길이] {duration:.1f}초

다음 JSON 하나만 답한다.
{{
 "matches": 이 소리가 분류 "{label}"에 맞으면 true,
 "title": "짧은 한국어 제목 (20자 이내, 예: 밤바다에 밀려오는 잔잔한 파도)",
 "description": "3~4문장. 1) 무엇이 들리는지 구체적으로(소리 나는 것, 크기, 리듬, 악기·템포) 2) 어떤 장소·시간·날씨가 떠오르는지 3) 어떤 분위기인지 4) 만화의 어떤 장면에 어울리는지",
 "moods": [아래 분위기 코드 중 맞는 것 1~3개],
 "places": ["어울리는 장소 한국어 낱말 0~3개 (예: 바닷가, 교실, 카페)"],
 "has_vocals": 노래 가사나 말소리가 또렷하게 들리면 true,
 "quality": 녹음 상태와 배경음으로 쓰기 좋은 정도 1~5
}}
분위기 코드: {moods}
"""


def ask(model: str, api_key: str, audio: bytes, text: str) -> dict:
    body = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "audio/mp3",
                            "data": base64.b64encode(audio).decode(),
                        }
                    },
                    {"text": text},
                ]
            }
        ],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
    }
    req = urllib.request.Request(
        URL.format(model),
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        res = json.loads(r.read())
    out = res["candidates"][0]["content"]["parts"]
    raw = "".join(p.get("text", "") for p in out if not p.get("thought"))
    data = json.loads(raw)
    if isinstance(data, list):
        data = data[0]
    return dict(data)


def clean(d: dict) -> dict:
    moods = [m for m in d.get("moods", []) if m in MOODS][:3]
    places = [str(p)[:20] for p in d.get("places", []) if str(p).strip()][:3]
    return {
        "matches": bool(d.get("matches", True)),
        "title": str(d["title"]).strip()[:40],
        "description": str(d["description"]).strip()[:1000],
        "moods": moods,
        "places": places,
        "has_vocals": bool(d.get("has_vocals", False)),
        "quality": max(1, min(5, int(d.get("quality", 3)))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--clap", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--interval", type=float, default=6.5, help="같은 모델을 부르는 최소 간격(초)")
    a = ap.parse_args()
    api_key = os.environ["GEMINI_API_KEY"]
    clap = {}
    for line in a.clap.read_text(encoding="utf-8").splitlines():
        x = json.loads(line)
        clap[x["key"]] = x
    done = set()
    if a.out.exists():
        done = {json.loads(x)["key"] for x in a.out.read_text(encoding="utf-8").splitlines()}
    cands = [json.loads(x) for x in a.candidates.read_text(encoding="utf-8").splitlines()]
    todo = [c for c in cands if key(c) not in done and key(c) in clap]
    lock = Lock()
    exhausted: set[str] = set()
    last: dict[str, float] = {}
    mood_list = ", ".join(f"{k}({v[0]})" for k, v in MOODS.items())

    def one(c: dict) -> None:
        k = key(c)
        cat = CATS[c["category"]]
        moods = clap[k]["moods"]
        top = sorted(moods, key=lambda m: moods[m], reverse=True)[:4]
        text = PROMPT.format(
            kind=KIND[cat.kind],
            label=cat.label,
            title=c["title_original"],
            tags=", ".join(c["tags"][:20]),
            labels=", ".join(c["labels"]),
            note=c["note"][:400],
            clap=", ".join(f"{m}({MOODS[m][0]})" for m in top),
            duration=c["duration"],
            moods=mood_list,
        )
        audio = (a.audio / f"{k}.mp3").read_bytes()
        for attempt in range(6):
            live = [m for m in MODELS if m not in exhausted]
            if not live:
                return
            model = live[0]
            with lock:  # 무료 등급 분당 한도 안에서 부른다
                wait = last.get(model, 0) + a.interval - time.time()
                last[model] = time.time() + max(0, wait)
            time.sleep(max(0, wait))
            try:
                d = clean(ask(model, api_key, audio, text))
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                if e.code == 429 and "PerDay" in body:
                    exhausted.add(model)  # 하루 무료 한도를 다 썼다
                time.sleep(min(60, 5 * 2**attempt))
            except (OSError, ValueError, KeyError, TypeError):
                time.sleep(2)
        else:
            print(f"fail {k}")
            return
        with lock, open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": k, "model": model, **d}, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one, todo))


if __name__ == "__main__":
    main()
