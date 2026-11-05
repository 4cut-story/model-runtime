"""녹음 중 선행 전사(FR-015)의 구간 나누기와 이어 붙이기. GPU 없이 테스트한다.

구간 전사는 [앞 구간 경계 - 겹침, 지금까지 녹음된 끝]을 전사하고,
끝 직전(끝에서 5~15초 전)의 단어 사이 빈틈이 가장 큰 곳을 다음 경계로 정한다.
이번 구간은 [앞 경계, 새 경계) 안에서 시작한 단어만 쓴다.
구간 끝(소리가 잘린 곳)과 다음 구간의 앞 가장자리 모두에서 5초 이상 떨어진 곳,
말이 쉬는 틈에서 자르므로 경계에서 단어가 빠지지 않는다
(GPU PC 60분 샘플: 경계에서만 자르면 -2.2%, 이 방식은 -0.13%).
"""

from __future__ import annotations

import itertools
from typing import Any

OVERLAP_SEC = 20.0
CUT_WINDOW = (5.0, 15.0)


def shift(segments: list[dict[str, Any]], offset: float) -> list[dict[str, Any]]:
    """구간 안 시각에 구간 시작 시각을 더한다.

    정렬에 실패한 단어는 앞 단어(없으면 세그먼트) 시각을 쓴다.
    """
    out = []
    for seg in segments:
        start = float(seg["start"]) + offset
        end = float(seg["end"]) + offset
        words = []
        last = start
        for w in seg.get("words") or []:
            ws = float(w["start"]) + offset if w.get("start") is not None else last
            we = float(w["end"]) + offset if w.get("end") is not None else ws
            last = ws
            words.append({"word": w.get("word", ""), "start": ws, "end": we})
        out.append({"start": start, "end": end, "text": seg.get("text", ""), "words": words})
    return out


def choose_cut(
    segments: list[dict[str, Any]], end: float, window: tuple[float, float] = CUT_WINDOW
) -> float:
    """[end - window[1], end - window[0]]에서 시작한 단어 사이 빈틈이 가장 큰 곳의 가운데.

    단어가 모자라면 그 범위 가운데.
    """
    lo, hi = end - window[1], end - window[0]
    words = sorted(
        (w for s in segments for w in s["words"] if lo <= w["start"] <= hi),
        key=lambda w: w["start"],
    )
    best_gap, cut = -1.0, end - (window[0] + window[1]) / 2
    for a, b in itertools.pairwise(words):
        gap = b["start"] - a["end"]
        if gap > best_gap:
            best_gap, cut = gap, (a["end"] + b["start"]) / 2
    return cut


def keep_between(segments: list[dict[str, Any]], frm: float, to: float) -> list[dict[str, Any]]:
    """[frm, to) 안에서 시작한 단어만 남긴다.

    단어가 다 남은 세그먼트는 원래 글을, 잘린 세그먼트는 남은 단어로 글을 다시 만든다.
    """
    out = []
    for seg in segments:
        words = seg["words"]
        if not words:
            if frm <= seg["start"] < to and seg.get("text", "").strip():
                out.append(seg)
            continue
        kept = [w for w in words if frm <= w["start"] < to]
        if not kept:
            continue
        if len(kept) == len(words):
            out.append(seg)
            continue
        out.append(
            {
                "start": kept[0]["start"],
                "end": kept[-1]["end"],
                "text": " ".join(w["word"].strip() for w in kept if w["word"].strip()),
                "words": kept,
            }
        )
    return out


def to_output(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """워커가 내는 전사 형식 ({speaker, start, end, text})."""
    return [
        {
            "speaker": s.get("speaker", "SPEAKER_00"),
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": s.get("text", "").strip(),
        }
        for s in segments
    ]
