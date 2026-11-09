"""CLAP(laion/larger_clap_general, Apache-2.0)으로 클립마다 소리 임베딩(512차원)과 분위기·카테고리 점수를 만든다.

  python embed_clips.py --candidates candidates.jsonl --audio audio/ --out clap.jsonl

- 소리는 48kHz 모노로 읽어 10초 구간마다 임베딩한 뒤 평균 내고 길이 1로 맞춘다.
- 같은 공간의 글 임베딩과 견줘 분위기(MOODS)와 카테고리 확인 점수를 남긴다. 설명을 쓸 때 근거로 쓴다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch
from categories import ALL, MOODS
from fetch_clips import key
from transformers import ClapModel, ClapProcessor

# CLAP이 소리에서 직접 고르는 낱말. 설명을 쓸 때 "실제로 들린 것"의 근거가 된다
EVENTS = [
    "ocean waves",
    "rain",
    "thunder",
    "wind",
    "flowing water",
    "birds chirping",
    "crickets",
    "frogs",
    "insects buzzing",
    "seagulls",
    "dog barking",
    "cat meowing",
    "rooster crowing",
    "cows mooing",
    "children shouting",
    "children laughing",
    "crowd chatter",
    "people talking",
    "applause",
    "cheering",
    "laughter",
    "a baby crying",
    "a person crying",
    "screaming",
    "footsteps",
    "car traffic",
    "car horn",
    "a motorcycle",
    "a train",
    "a subway announcement",
    "an airplane",
    "a boat engine",
    "a siren",
    "church bells",
    "a school bell",
    "wind chimes",
    "a clock ticking",
    "a door closing",
    "knocking",
    "a doorbell",
    "a phone ringing",
    "typing on a keyboard",
    "dishes clinking",
    "cooking and sizzling",
    "boiling water",
    "a crackling fire",
    "fireworks",
    "an explosion",
    "glass breaking",
    "a whoosh",
    "a camera shutter",
    "silence and room tone",
    "electronic beeps",
    "music playing",
]
INSTRUMENTS = [
    "piano",
    "acoustic guitar",
    "electric guitar",
    "bass guitar",
    "drums",
    "synthesizer",
    "violin and strings",
    "cello",
    "flute",
    "saxophone",
    "trumpet and brass",
    "organ",
    "harp",
    "bells and glockenspiel",
    "choir",
    "orchestra",
    "ukulele",
    "accordion",
    "8-bit chiptune",
]
ATTRIBUTES = {
    "vocals": ("a person singing with lyrics", "instrumental music without vocals"),
    "tempo": ("fast tempo music", "slow tempo music"),
    "place": ("recorded indoors", "recorded outdoors"),
    "time": ("at night", "during the day"),
    "loud": ("loud and intense", "quiet and soft"),
}

MODEL = "laion/larger_clap_general"
SR = 48_000
WINDOW = 10 * SR


def _tensor(out: object) -> torch.Tensor:
    """transformers 5는 출력 객체를 돌려준다 (pooler_output이 투영된 임베딩)."""
    if isinstance(out, torch.Tensor):
        return out
    return out.pooler_output  # type: ignore[attr-defined, no-any-return]


def windows(y: np.ndarray) -> list[np.ndarray]:
    if len(y) <= WINDOW:
        return [y]
    starts = range(0, len(y) - WINDOW // 2, WINDOW)
    return [y[s : s + WINDOW] for s in starts if len(y[s : s + WINDOW]) >= SR * 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = ClapModel.from_pretrained(MODEL).eval()
    proc = ClapProcessor.from_pretrained(MODEL)
    mood_codes = list(MOODS)
    texts = [f"{MOODS[m][1]} sound" for m in mood_codes] + [c.prompt for c in ALL]
    texts += [f"the sound of {x}" for x in EVENTS] + [f"music played on {x}" for x in INSTRUMENTS]
    for pos, neg in ATTRIBUTES.values():
        texts += [pos, neg]
    with torch.no_grad():
        t = _tensor(model.get_text_features(**proc(text=texts, return_tensors="pt", padding=True)))
        t = torch.nn.functional.normalize(t, dim=-1)
    done = set()
    if a.out.exists():
        done = {json.loads(x)["key"] for x in a.out.read_text(encoding="utf-8").splitlines()}
    cands = [json.loads(x) for x in a.candidates.read_text(encoding="utf-8").splitlines()]
    with open(a.out, "a", encoding="utf-8") as f:
        for c in cands:
            k = key(c)
            path = a.audio / f"{k}.mp3"
            if k in done or not path.exists():
                continue
            y, _ = librosa.load(path, sr=SR, mono=True)
            with torch.no_grad():
                inp = proc(audio=windows(y), sampling_rate=SR, return_tensors="pt")
                e = _tensor(model.get_audio_features(**inp))
                e = torch.nn.functional.normalize(e, dim=-1).mean(0)
                e = torch.nn.functional.normalize(e, dim=-1)
            sims = (t @ e).tolist()
            mood = dict(zip(mood_codes, sims[: len(mood_codes)], strict=True))
            n_m, n_c, n_e = len(mood_codes), len(ALL), len(EVENTS)
            cat = dict(zip([x.code for x in ALL], sims[n_m : n_m + n_c], strict=True))
            ev = sims[n_m + n_c : n_m + n_c + n_e]
            ins = sims[n_m + n_c + n_e : n_m + n_c + n_e + len(INSTRUMENTS)]
            attr = sims[n_m + n_c + n_e + len(INSTRUMENTS) :]
            heard = [EVENTS[i] for i in np.argsort(ev)[::-1][:5]]
            instruments = [INSTRUMENTS[i] for i in np.argsort(ins)[::-1][:3]]
            attributes = {
                name: ATTRIBUTES[name][0] if attr[2 * i] > attr[2 * i + 1] else ATTRIBUTES[name][1]
                for i, name in enumerate(ATTRIBUTES)
            }
            if c["kind"] == "MUSIC":
                bpm = float(np.atleast_1d(librosa.beat.beat_track(y=y[: SR * 30], sr=SR)[0])[0])
                attributes["bpm"] = str(round(bpm))
            rms = librosa.feature.rms(y=y)[0]
            attributes["dynamics"] = "steady" if rms.std() < 0.5 * rms.mean() else "changing"
            top = sorted(cat, key=lambda x: cat[x], reverse=True)[:3]
            rank = sorted(cat, key=lambda x: cat[x], reverse=True).index(c["category"])
            f.write(
                json.dumps(
                    {
                        "key": k,
                        "embedding": [round(v, 6) for v in e.tolist()],
                        "moods": {m: round(v, 4) for m, v in mood.items()},
                        "heard": heard,
                        "instruments": instruments,
                        "attributes": attributes,
                        "top_categories": top,
                        "category_rank": rank,
                        "category_score": round(cat[c["category"]], 4),
                    }
                )
                + "\n"
            )
            f.flush()


if __name__ == "__main__":
    main()
