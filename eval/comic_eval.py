"""4컷 만화 생성 반복 실행 (GPU PC).

서버와 같은 프롬프트로 전사 → 장면 추출 → 콘티 → 그림 → 합성을 돌리고
단계별 시간과 결과물을 남긴다.

프롬프트·스키마·그림 설정은 conversation-service가 내보낸 ``eval/prompt_bundle.json``을 쓴다
(conversation-service ``PromptBundleExport``). 프롬프트 조립·검증·그림 프롬프트 규칙은 서버 코드
(ExtractHandler, ScriptHandler, ComicScript, DrawHandler)를 그대로 옮겼다.

예:
    python eval/comic_eval.py --audio D:/4cut/samples/01.m4a --runs 3 --out D:/4cut/eval-out
    python eval/comic_eval.py --transcript out/run-01/transcript.json --runs 5 --skip-transcribe

결과: <out>/<타임스탬프>/run-NN/{candidates,script,draw,timings}.json, panel-N.png, comic.png
와 summary.json. 녹음·전사는 커밋하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

ROLES = ["기", "승", "전", "결"]
SHOTS = {"wide", "medium", "close-up"}
STOPWORDS = {"a", "an", "the", "and", "at", "in", "on", "of", "with", "to", "is", "are", "nearby"}
MAX_SCENE_OVERLAP = 0.8
UNKNOWN = "친구"
FILLER = re.compile(r"(?<!\S)(?:어+|음+|으+음+|에+|uh+|um+|hmm+)[.,…!?~]*(?=\s|$)", re.IGNORECASE)


# ---------- 전사 ----------


def letters(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """워커 화자 이름(SPEAKER_00…)을 처음 나온 순서대로 A, B, C…로 바꾼다 (TranscribeHandler)."""
    order: dict[str, str] = {}
    out = []
    for s in raw:
        name = s.get("speaker", "SPEAKER_00")
        if name not in order:
            i = len(order)
            order[name] = chr(ord("A") + i) if i < 26 else f"S{i + 1}"
        out.append({**s, "speaker": order[name]})
    return out


def load_transcript(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "utterances" in data:
        return dict(data)
    utterances = letters(data["segments"])
    speakers = list(dict.fromkeys(u["speaker"] for u in utterances))
    return {"language": data.get("language", "ko"), "speakers": speakers, "utterances": utterances}


def between(t: dict[str, Any], start: float, end: float) -> list[dict[str, Any]]:
    return [u for u in t["utterances"] if u["end"] >= start and u["start"] <= end]


def lines(utterances: list[dict[str, Any]]) -> str:
    out = []
    for u in utterances:
        s = int(u["start"])
        out.append(f"[{s // 60:02d}:{s % 60:02d}] {u['speaker']}: {u['text']}\n")
    return "".join(out)


def clean(text: str | None) -> str:
    return re.sub(r"\s{2,}", " ", FILLER.sub("", text or "")).strip()


def compact(utterances: list[dict[str, Any]]) -> str:
    """같은 화자가 2초 안에 이어 말하면 합치고 군말을 지운다 (TranscriptText)."""
    return lines(merge(utterances))


def merge(utterances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for u in utterances:
        text = clean(u["text"])
        if not text:
            continue
        if merged and merged[-1]["speaker"] == u["speaker"] and u["start"] - merged[-1]["end"] <= 2:
            merged[-1] = {**merged[-1], "end": u["end"], "text": merged[-1]["text"] + " " + text}
            continue
        merged.append({**u, "text": text})
    return merged


def tag_seconds(tag: Any) -> int:
    """ "04:11" → 251, 형식이 다르면 -1 (SceneTimes.seconds)."""
    m = re.fullmatch(r"(\d+):(\d+)", str(tag or "").strip())
    if not m or int(m.group(2)) >= 60:
        return -1
    return int(m.group(1)) * 60 + int(m.group(2))


def resolve_times(c: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """from/to 줄 표시를 실제 발화 시각으로 바꾼다. 30초~3분으로 맞춘다 (SceneTimes)."""
    if "from" not in c:
        return c

    def index(sec: int) -> int:
        if sec < 0:
            return -1
        for i, u in enumerate(rows):
            if int(u["start"]) == sec:
                return i
        return next((i for i, u in enumerate(rows) if u["start"] <= sec <= u["end"]), -1)

    out = {k: v for k, v in c.items() if k not in ("from", "to")}
    first, last = index(tag_seconds(c.get("from"))), index(tag_seconds(c.get("to")))
    if first < 0 or last < first:
        return {**out, "start": 0, "end": 0}
    start, end = rows[first]["start"], rows[last]["end"]
    while end - start < 30 and last + 1 < len(rows):
        last += 1
        end = rows[last]["end"]
    return {**out, "start": start, "end": min(end, start + 180), "tags": [c["from"], c["to"]]}


def spoken(t: dict[str, Any]) -> str:
    return "".join(u["text"] + "\n" for u in t["utterances"])


# ---------- LLM ----------


def content_json(content: str) -> dict[str, Any]:
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM 결과가 JSON이 아니다")
    result: dict[str, Any] = json.loads(content[start : end + 1])
    return result


def llm_job(
    bundle: dict[str, Any],
    step: str,
    prompt: str,
    seed: int,
    model: str,
    system: str | None = None,
) -> dict[str, Any]:
    spec = bundle[step]
    return {
        "model": model,
        "system": system or spec["system"],
        "prompt": prompt,
        "jsonSchema": spec["schema"],
        "options": {
            "num_ctx": bundle["numCtx"],
            "temperature": bundle["temperature"],
            "seed": seed,
            "num_predict": spec["maxTokens"],
        },
    }


def extract_prompt(t: dict[str, Any], window: int, size: int, part: list[dict[str, Any]]) -> str:
    return (
        f"대화 언어: {t['language']}\n화자: {', '.join(t['speakers'])}\n\n"
        f"전사 ({window * size // 60}분부터):\n{compact(part)}"
    )


def cast_names(t: dict[str, Any], pool: list[str], seed: int) -> dict[str, str]:
    """화자별 가명을 코드로 정한다 (CastNames). 대화에 나온 이름은 건너뛴다."""
    text = spoken(t)
    free = [n for n in pool if n not in text]
    if not free:
        return {}
    start = seed % len(free)
    return {sp: free[(start + i) % len(free)] for i, sp in enumerate(t["speakers"][: len(free)])}


TITLES = set(
    [
        "팀장",
        "부장",
        "과장",
        "차장",
        "대리",
        "사원",
        "사장",
        "실장",
        "선배",
        "후배",
        "막내",
        "총무",
        "언니",
        "오빠",
        "누나",
        "형님",
        "엄마",
        "아빠",
        "친구",
        "선생",
        "교수",
        "이모",
        "삼촌",
        "고모",
        "할머니",
        "할아버지",
    ]
)


def plausible(word: str, text: str, pool: list[str]) -> bool:
    """실명으로 볼 만한 단어 (RealNames.plausible)."""
    return (
        re.fullmatch(r"[가-힣]{2,4}", word or "") is not None
        and word in text
        and not word.endswith("님")
        and word not in TITLES
        and word not in pool
    )


def merge_names(known: list[str], found: list[Any], text: str, pool: list[str]) -> list[str]:
    """구간마다 찾은 실명을 합친다 (ExtractHandler.mergeNames)."""
    names = list(known)
    for n in found or []:
        word = (n if isinstance(n, str) else "").strip()
        if plausible(word, text, pool) and word not in names:
            names.append(word)
    return names


def whose(names: list[str], t: dict[str, Any]) -> dict[str, str]:
    """실명 → 그 이름으로 불리는 화자 (RealNames.whose)."""
    out = {}
    speakers = t["speakers"]
    for name in names:
        who = ""
        if len(speakers) == 2:
            voc = re.compile(re.escape(name) + r"(?:야|아)(?![가-힣])")
            callers = {u["speaker"] for u in t["utterances"] if voc.search(u["text"])}
            if len(callers) == 1:
                caller = callers.pop()
                who = speakers[1] if speakers[0] == caller else speakers[0]
        out[name] = who
    return out


def has_final(word: str) -> bool:
    if not word:
        return False
    c = ord(word[-1])
    return 0xAC00 <= c <= 0xD7A3 and (c - 0xAC00) % 28 != 0


def real_name_replacements(found: dict[str, str], cast: dict[str, str]) -> dict[str, str]:
    """실명 → 화자 가명(모르면 친구), 받침에 맞춘 호격·조사 (RealNames.replacements)."""
    out: dict[str, str] = {}
    for name, speaker in found.items():
        n = (name or "").strip()
        if len(n) < 2 or n in cast or n in cast.values():
            continue
        alias = cast.get((speaker or "").strip(), UNKNOWN)
        alias_final = has_final(alias)
        vocative = alias + ("아" if alias_final else "야")
        out[n + "야"] = vocative
        out[n + "아"] = vocative
        if has_final(n):
            out[n + "이"] = alias + "이" if alias_final else alias
            out[n] = alias
        else:
            out[n] = alias + "이" if alias_final and alias != UNKNOWN else alias
    return out


def section(t: dict[str, Any], keep: Any, rename: dict[str, str]) -> str:
    part = [u for u in t["utterances"] if keep(u)]
    return replace_all(compact(part), rename) or "" if part else "(없음)"


def script_prompt(
    t: dict[str, Any],
    scene: dict[str, Any],
    context: float,
    cast: dict[str, str],
    found: dict[str, str],
) -> str:
    rename = real_name_replacements(found, cast)
    a, b = scene["start"], scene["end"]
    before = section(t, lambda u: u["start"] < a and u["end"] >= a - context, rename)
    during = section(t, lambda u: a <= u["start"] <= b, rename)
    after = section(t, lambda u: b < u["start"] <= b + context, rename)
    names = ", ".join(f"{k}={v}" for k, v in cast.items())
    return (
        f"대화 언어: {t['language']}\n화자: {', '.join(t['speakers'])}\n인물 가명: {names}\n"
        f"장면 종류: {scene['kind']}\n장면 요약: {replace_all(scene['summary'], rename)}\n"
        f"고른 이유: {replace_all(scene['reason'], rename)}\n\n"
        f"[장면 앞 대화 (참고만)]\n{before}\n\n"
        f"[장면 대화 (대사는 여기서 고른다)]\n{during}\n\n"
        f"[장면 뒤 대화 (참고만)]\n{after}"
    )


def scored(c: dict[str, Any]) -> dict[str, Any]:
    """항목 점수로 score를 계산한다 (SceneCandidate.scored)."""
    if any(not isinstance(c.get(k), int) for k in ("story", "visual", "relatable")):
        return c

    def clamp(v: int) -> int:
        return max(1, min(5, v))

    s = (0.4 * clamp(c["story"]) + 0.3 * clamp(c["visual"]) + 0.3 * clamp(c["relatable"])) / 5
    if c.get("intro") is True:
        s -= 0.3
    return {**c, "score": round(max(0.0, s), 2)}


def valid_candidate(c: dict[str, Any]) -> bool:
    return (
        c.get("kind") in ("CORE", "FUNNY", "IMPORTANT")
        and c.get("end", 0) > c.get("start", 0)
        and bool((c.get("summary") or "").strip())
        and 0 <= c.get("score", -1) <= 1
    )


# ---------- 콘티 검증 (ComicScript) ----------


def display_name(c: dict[str, Any]) -> str:
    name = (c.get("name") or "").strip()
    return name or c["speaker"]


def normalized(script: dict[str, Any]) -> dict[str, Any]:
    by_name = {
        (c.get("name") or "").strip(): c["speaker"]
        for c in script.get("characters") or []
        if (c.get("name") or "").strip()
    }
    for p in script.get("panels") or []:
        for line in p.get("lines") or []:
            sp = (line.get("speaker") or "").strip()
            line["speaker"] = by_name.get(sp, sp)
        speakers_in_scene(p)
    return script


def speakers_in_scene(p: dict[str, Any]) -> None:
    """대사를 하는 인물이 장면 설명에 없으면 붙인다 (ComicScript.withSpeakersInScene)."""
    if p.get("scene") is None or p.get("lines") is None:
        return
    scene = p["scene"].strip()
    for sp in dict.fromkeys(line.get("speaker") for line in p["lines"]):
        if sp and "{" + sp + "}" not in scene:
            scene += ", {" + sp + "} nearby"
    p["scene"] = scene


def names_found_in(script: dict[str, Any], text: str) -> list[str]:
    return [n for n in (display_name(c) for c in script["characters"]) if n and n in text]


def with_names(script: dict[str, Any], assigned: dict[str, str]) -> dict[str, Any]:
    rename = {}
    for c in script.get("characters") or []:
        name = assigned.get(c.get("speaker"))
        if name is None:
            continue
        old = (c.get("name") or "").strip()
        if old and old != name:
            rename[old] = name
        c["name"] = name
    for p in script.get("panels") or []:
        for line in p.get("lines") or []:
            for a, b in rename.items():
                line["text"] = (line.get("text") or "").replace(a, b)
    for a, b in rename.items():
        script["title"] = (script.get("title") or "").replace(a, b)
    return script


def with_fictional_names(script: dict[str, Any], text: str, spare: list[str]) -> dict[str, Any]:
    clashes = names_found_in(script, text)
    if not clashes:
        return script
    used = {display_name(c) for c in script["characters"]}
    pool = iter(n for n in spare if n not in text and n not in used)
    rename = {}
    for old in clashes:
        new = next(pool, None)
        if new:
            rename[old] = new

    def sub(s: str | None) -> str | None:
        if s is None:
            return None
        for a, b in rename.items():
            s = s.replace(a, b)
        return s

    for c in script["characters"]:
        if display_name(c) in rename:
            c["name"] = rename[display_name(c)]
    for p in script["panels"]:
        for line in p.get("lines") or []:
            line["text"] = sub(line.get("text"))
    script["title"] = sub(script.get("title"))
    return script


def replace_all(text: str | None, rename: dict[str, str]) -> str | None:
    if text is None:
        return None
    for k in sorted(rename, key=len, reverse=True):
        text = text.replace(k, rename[k])
    return text


def replaced(script: dict[str, Any], rename: dict[str, str]) -> dict[str, Any]:
    for p in script.get("panels") or []:
        for line in p.get("lines") or []:
            line["text"] = replace_all(line.get("text"), rename)
    script["title"] = replace_all(script.get("title"), rename)
    return script


def called(word: str, text: str) -> bool:
    """대화에서 "민수야"처럼 불린 적이 있는 말인지 (RealNames.called)."""
    return len(word) >= 2 and re.search(re.escape(word) + r"(?:야|아)(?![가-힣])", text) is not None


def scrubbed(script: dict[str, Any], text: str) -> dict[str, Any]:
    """nameWarnings 중 대화에서 불린 이름만 대사·제목에서 "친구"로 바꾸고 JSON 조각을 지운다."""
    rename: dict[str, str] = {}
    for w in script.get("nameWarnings") or []:
        word = (w or "").strip()
        stem = word[:-1] if len(word) > 2 and word.endswith("이") else word
        if len(stem) < 2 or not (called(word, text) or called(stem, text)):
            continue
        rename[word] = UNKNOWN
        rename[stem] = UNKNOWN
    for p in script.get("panels") or []:
        for line in p.get("lines") or []:
            text = replace_all(line.get("text"), rename) or ""
            line["text"] = re.sub(r",\s*$", "", re.sub(r'[{}"]', "", text)).strip()
    script["title"] = replace_all(script.get("title"), rename)
    return script


def words(s: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9]+", s.lower()) if len(w) > 1 and w not in STOPWORDS}


def overlap(a: str, b: str) -> float:
    x, y = words(a), words(b)
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def violation(s: dict[str, Any]) -> str | None:
    title = s.get("title") or ""
    if not title.strip() or len(title) > 100:
        return "제목이 없거나 100자를 넘는다"
    cast, panels = s.get("characters") or [], s.get("panels") or []
    if not cast or len(panels) != 4:
        return "등장인물이 없거나 컷이 4개가 아니다"
    names = set()
    for c in cast:
        name = (c.get("name") or "").strip()
        if not name or len(name) > 10 or name == c.get("speaker"):
            return f"{c.get('speaker')}의 가명이 없거나 틀렸다"
        if name in names:
            return f"가명 {name}이(가) 겹친다"
        names.add(name)
        if not (c.get("look") or "").strip():
            return f"{c.get('speaker')}의 외모 설명이 없다"
    speakers = {c.get("speaker") for c in cast}
    for i, p in enumerate(panels):
        if p.get("role") != ROLES[i] or not (p.get("scene") or "").strip():
            return f"{i + 1}번 컷의 역할이나 장면 설명이 틀렸다"
        if p.get("shot") not in SHOTS:
            return f"{i + 1}번 컷의 구도(shot)가 틀렸다"
        ls = p.get("lines") or []
        if len(ls) > 2:
            return f"{i + 1}번 컷 대사가 2개를 넘는다"
        for line in ls:
            if not (line.get("text") or "").strip() or line.get("speaker") not in speakers:
                return f"{i + 1}번 컷 대사의 화자나 내용이 틀렸다"
    if len({p.get("shot") for p in panels}) == 1:
        return "네 컷의 구도가 모두 같다"
    for i in range(4):
        for j in range(i + 1, 4):
            if overlap(panels[i]["scene"], panels[j]["scene"]) >= MAX_SCENE_OVERLAP:
                return f"{i + 1}번과 {j + 1}번 컷 장면이 거의 같다"
    return None


# ---------- 그림 (DrawHandler) ----------


MAX_LINE_CHARS = 40
MIN_SIMILARITY = 0.75


def sentences(text: str | None) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.?!])\s+", (text or "").strip()) if x.strip()]


def distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(b)]


def similarity(a: str, b: str) -> float:
    m = max(len(a), len(b))
    return 1.0 if m == 0 else 1 - distance(a, b) / m


def snap(line: str | None, source: list[str]) -> str | None:
    """대사를 원문 문장으로 맞추고 말풍선 길이를 넘는 뒷문장을 버린다 (SpokenLines.snap)."""
    if line is None:
        return None
    out = ""
    for s in sentences(line):
        best, score = s, MIN_SIMILARITY
        for c in source:
            sim = similarity(s, c)
            if sim >= score:
                best, score = c, sim
        if out and len(out) + 1 + len(best) > MAX_LINE_CHARS:
            break
        out = f"{out} {best}" if out else best
    return out or line.strip()


COUNTS = ["one", "two", "three", "four"]


def draw_prompt(
    style: str,
    script: dict[str, Any],
    panel: dict[str, Any],
    concept: dict[str, Any] | None = None,
) -> str:
    """DrawHandler.prompt: 그림체 → 구도 → 인원 → 장면."""
    scene = panel["scene"].strip()
    legacy = "{" not in scene
    cast = ""
    people = 0
    for c in script["characters"]:
        mark = "{" + c["speaker"] + "}"
        if legacy:
            cast += "; " + c["look"].strip()
        elif mark in scene:
            people += 1
            scene = scene.replace(mark, "(" + c["look"].strip() + ")")
    crowd = (concept or {}).get("crowd", "people")
    scene = re.sub(r"\{[A-Z]\}", (concept or {}).get("friend", "a friend"), scene)
    shot = panel["shot"].replace("-", " ") + " shot, " if panel.get("shot") else ""
    count = f"{COUNTS[min(people, len(COUNTS)) - 1]} {crowd}, " if people >= 2 else ""
    return f"{style}, {shot}{count}{scene}{cast}"


def draw_images(
    bundle: dict[str, Any],
    script: dict[str, Any],
    seed: int,
    same_seed: bool = False,
    concept: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    d = bundle["draw"]
    style = concept["style"] if concept else d["style"]
    negative = d["negativePrompt"]
    if concept and concept.get("negative"):
        negative += ", " + concept["negative"]
    return [
        {
            "name": f"panel-{i + 1}",
            "prompt": draw_prompt(style, script, p, concept),
            "negativePrompt": negative,
            "seed": seed if same_seed else seed + i,  # 서버는 같은 시드 (DrawHandler)
            "width": d["imageSize"],
            "height": d["imageSize"],
        }
        for i, p in enumerate(script["panels"])
    ]


# ---------- 합성 (ComicComposer와 같은 배치) ----------

WIDTH, MARGIN, GAP, TITLE_H = 800, 24, 20, 90
PANEL = WIDTH - 2 * MARGIN
FONTS = [
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]


def font(size: int) -> Any:
    from PIL import ImageFont

    for f in FONTS:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def bubble_text(name: str, speaker: str, text: str) -> str:
    text = text.strip()
    changed = True
    while changed:
        changed = False
        for who in (speaker, name):
            for sep in (":", "："):
                if who and text.startswith(who + sep):
                    text = text[len(who) + len(sep) :].strip()
                    changed = True
    return f"{name}: {text}"


def wrap(draw: Any, text: str, fnt: Any, width: int) -> list[str]:
    rows, row = [], ""
    for ch in text:
        if row and draw.textlength(row + ch, font=fnt) > width:
            rows.append(row.strip())
            row = ""
        row += ch
    if row.strip():
        rows.append(row.strip())
    return rows


def compose(script: dict[str, Any], panels: list[Any], out: Path) -> None:
    from PIL import Image, ImageDraw

    height = MARGIN + TITLE_H + 4 * (PANEL + GAP) - GAP + MARGIN
    img = Image.new("RGB", (WIDTH, height), "white")
    g = ImageDraw.Draw(img)
    title = script["title"]
    for size in range(40, 22, -2):
        tf = font(size)
        if g.textlength(title, font=tf) <= WIDTH - 2 * MARGIN:
            break
    g.text((WIDTH / 2, MARGIN + TITLE_H / 2), title, fill="black", font=tf, anchor="mm")
    names = {c["speaker"]: display_name(c) for c in script["characters"]}
    bf = font(28)
    y = MARGIN + TITLE_H
    for i, panel in enumerate(panels):
        img.paste(panel.resize((PANEL, PANEL)), (MARGIN, y))
        g.rectangle([MARGIN, y, MARGIN + PANEL, y + PANEL], outline="black", width=4)
        for k, line in enumerate(script["panels"][i].get("lines") or []):
            text = bubble_text(
                names.get(line["speaker"], line["speaker"]), line["speaker"], line["text"]
            )
            rows = wrap(g, text, bf, int(PANEL * 0.72) - 28)
            lh = 36
            w = max(int(g.textlength(r, font=bf)) for r in rows) + 28
            h = len(rows) * lh + 28
            x = MARGIN + 16 if k % 2 == 0 else MARGIN + PANEL - 16 - w
            by = y + 16 if k % 2 == 0 else y + PANEL - 16 - h
            g.rounded_rectangle(
                [x, by, x + w, by + h], radius=14, fill="white", outline="black", width=3
            )
            for r_i, r in enumerate(rows):
                g.text((x + 14, by + 14 + r_i * lh), r, fill="black", font=bf)
        y += PANEL + GAP
    img.save(out)


# ---------- 실행 ----------


class Timer:
    def __init__(self) -> None:
        self.times: dict[str, float] = {}

    def run(self, key: str, fn: Callable[[], Any]) -> Any:
        start = time.perf_counter()
        try:
            return fn()
        finally:
            self.times[key] = round(self.times.get(key, 0) + time.perf_counter() - start, 2)


def run_once(
    args: argparse.Namespace,
    bundle: dict[str, Any],
    t: dict[str, Any],
    llm: Any,
    sdxl: Any,
    out: Path,
    seed: int,
    concepts: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """한 번 실행: 장면 추출·선정은 한 번, 콘티·그림은 컨셉마다 (컨셉이 없으면 기본 하나)."""
    out.mkdir(parents=True, exist_ok=True)
    timer = Timer()
    report: dict[str, Any] = {"seed": seed, "promptVersion": bundle["promptVersion"]}
    size = bundle["windowSec"]
    last = int(max(u["end"] for u in t["utterances"])) // size
    candidates: list[dict[str, Any]] = []
    found_names: list[str] = []
    for w in range(last + 1):
        part = between(t, w * size, (w + 1) * size)
        if not part:
            continue
        job = llm_job(bundle, "extract", extract_prompt(t, w, size, part), seed + w, args.model)
        res = timer.run("extract", lambda job=job: llm.chat(job))
        try:
            body = content_json(res["content"])
        except ValueError:
            body = {}
        found = body.get("candidates", [])
        found_names = merge_names(
            found_names, body.get("names", []), spoken(t), bundle["castNames"]
        )
        rows = merge(part)
        resolved = [scored(resolve_times(c, rows)) for c in found]
        candidates += [c for c in resolved if valid_candidate(c)]
    (out / "candidates.json").write_text(
        json.dumps({"candidates": candidates, "names": found_names}, ensure_ascii=False, indent=1),
        "utf-8",
    )
    if not candidates:
        report["result"] = "NO_SCENE"
        report["timings"] = timer.times
        report["dir"] = str(out.relative_to(out.parent))
        return [report]
    scene = max(candidates, key=lambda c: c["score"])
    report["scene"] = {
        k: scene.get(k) for k in ("kind", "start", "end", "summary", "score", "concept")
    }
    cast = cast_names(t, bundle["castNames"], seed)
    reports = []
    for concept in with_auto(concepts, bundle, scene, args.auto_concept):
        cdir = out / concept["id"] if concept else out
        cdir.mkdir(parents=True, exist_ok=True)
        r = {**report, "timings": dict(timer.times), "dir": cdir}
        if concept:
            r["concept"] = concept["id"]
            r["conceptLabel"] = concept["label"]
        stimer = Timer()
        script = make_script(
            args, bundle, t, llm, scene, cast, found_names, seed, concept, stimer, r
        )
        r["timings"]["script"] = stimer.times.get("script", 0)
        r["script"] = script
        reports.append(r)
    for r in reports:
        script = r.pop("script")
        if script is None:
            continue
        cdir = r["dir"]
        concept = next((c for c in concepts or [] if c["id"] == r.get("concept")), None)
        (cdir / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=1), "utf-8")
        images = draw_images(bundle, script, seed * 10, not args.per_panel_seed, concept)
        (cdir / "draw.json").write_text(json.dumps(images, ensure_ascii=False, indent=1), "utf-8")
        r["title"] = script["title"]
        if sdxl is not None:
            from PIL import Image

            llm.unload()
            dtimer = Timer()
            pngs = []
            for img in images:
                data = dtimer.run("draw", lambda img=img: sdxl.draw(img))
                path = cdir / f"{img['name']}.png"
                path.write_bytes(data)
                pngs.append(Image.open(path).convert("RGB"))
            dtimer.run(
                "compose",
                lambda s=script, ps=pngs, d=cdir: compose(s, ps, d / "comic.png"),
            )
            r["timings"].update(dtimer.times)
        r["result"] = "DONE"
    for r in reports:
        r["dir"] = str(Path(r["dir"]).relative_to(out.parent))
    return reports


def with_auto(
    concepts: list[dict[str, Any]] | None,
    bundle: dict[str, Any],
    scene: dict[str, Any],
    auto: bool,
) -> list[dict[str, Any] | None]:
    """--auto-concept면 장면 추출이 고른 컨셉을 더한다 (서버 Concept.chosen처럼 모르면 DAILY)."""
    out: list[dict[str, Any] | None] = list(concepts or [])
    if auto:
        known = {c["id"]: c for c in bundle.get("concepts", [])}
        suggested = known.get(str(scene.get("concept") or "").upper(), known.get("DAILY"))
        if suggested is not None and all(c is None or c["id"] != suggested["id"] for c in out):
            out.append(suggested)
    return out or [None]


def make_script(
    args: argparse.Namespace,
    bundle: dict[str, Any],
    t: dict[str, Any],
    llm: Any,
    scene: dict[str, Any],
    cast: dict[str, str],
    found_names: list[str],
    seed: int,
    concept: dict[str, Any] | None,
    timer: Any,
    report: dict[str, Any],
) -> dict[str, Any] | None:
    script = None
    failures = []
    for attempt in range(3):
        job = llm_job(
            bundle,
            "script",
            script_prompt(t, scene, bundle["contextSec"], cast, whose(found_names, t)),
            seed + attempt,
            args.model,
            concept["scriptSystem"] if concept else None,
        )
        res = timer.run("script", lambda job=job: llm.chat(job))
        try:
            s = normalized(content_json(res["content"]))
            s = with_fictional_names(with_names(s, cast), spoken(t), bundle["castNames"])
            rename = real_name_replacements(whose(found_names, t), cast)
            s = scrubbed(replaced(s, rename), spoken(t))
            source = sentences(replace_all(spoken(t), rename))
            for p in s.get("panels") or []:
                for line in p.get("lines") or []:
                    line["text"] = snap(line.get("text"), source)
            v = violation(s)
            if v is None:
                script = s
                break
            failures.append(v)
        except (ValueError, KeyError, TypeError) as e:
            failures.append(f"파싱 실패: {e}")
    report["scriptAttempts"] = len(failures) + (1 if script else 0)
    report["scriptFailures"] = failures
    if script is None:
        report["result"] = "SCRIPT_FAILED"
    return script


def env_set() -> dict[str, str]:
    """값이 빈 환경 변수(.env의 "GPU_WORKER_TOKEN=")는 없는 것으로 본다."""
    import os

    return {k: v for k, v in os.environ.items() if v}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--audio", type=Path, help="녹음 파일 (한 번 전사해 transcript.json으로 남긴다)"
    )
    src.add_argument(
        "--transcript", type=Path, help="전사 JSON (서버 형식 또는 워커 segments 형식)"
    )
    p.add_argument("--bundle", type=Path, default=Path(__file__).with_name("prompt_bundle.json"))
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("eval-out"))
    p.add_argument("--model", default="qwen3:8b")
    p.add_argument("--ollama", default="http://localhost:11434")
    p.add_argument("--language", default="ko")
    p.add_argument("--skip-images", action="store_true", help="LLM 단계만 돌린다")
    p.add_argument(
        "--concepts",
        default="",
        help="쉼표로 나눈 컨셉 id(DAILY,INSECTS,…) 또는 all. 장면은 한 번, 콘티·그림은 컨셉마다",
    )
    p.add_argument(
        "--auto-concept",
        action="store_true",
        help="장면 추출이 고른 컨셉도 그린다(사용자가 컨셉을 고르지 않았을 때와 같다)",
    )
    p.add_argument(
        "--per-panel-seed", action="store_true", help="컷마다 시드를 달리한다(예전 방식 비교용)"
    )
    args = p.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model_runtime.backends.ollama import OllamaClient
    from model_runtime.config import Settings, load_env_file

    load_env_file(Path(".env"))
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    all_concepts = bundle.get("concepts", [])
    wanted = [c.strip() for c in args.concepts.split(",") if c.strip()]
    concepts = [c for c in all_concepts if args.concepts == "all" or c["id"] in wanted]
    root = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"promptVersion": bundle["promptVersion"], "runs": []}

    if args.audio:
        from model_runtime.backends.whisper import WhisperBackend

        env = Settings.from_env({"GPU_WORKER_API_URL": "x", "GPU_WORKER_TOKEN": "x", **env_set()})
        whisper = WhisperBackend(env.whisper_model, env.hf_token)
        start = time.perf_counter()
        raw = whisper.transcribe_file(str(args.audio), args.language)
        summary["transcribeSec"] = round(time.perf_counter() - start, 2)
        whisper.unload()
        (root / "transcript.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=1), "utf-8"
        )
        t = load_transcript(root / "transcript.json")
    else:
        t = load_transcript(args.transcript)
    summary["audioSec"] = round(max(u["end"] for u in t["utterances"]), 1)

    llm = OllamaClient.create(args.ollama)
    sdxl = None
    if not args.skip_images:
        from model_runtime.backends.sdxl import SdxlBackend

        s = Settings.from_env({"GPU_WORKER_API_URL": "x", "GPU_WORKER_TOKEN": "x", **env_set()})
        sdxl = SdxlBackend(
            s.sdxl_model, s.sdxl_cpu_offload, s.sdxl_lora, s.sdxl_steps, s.sdxl_guidance
        )
        summary["sdxl"] = sdxl.model_info()

    for i in range(args.runs):
        seed = args.seed + i * 100
        run_dir = root / f"run-{i + 1:02d}"
        start = time.perf_counter()
        reports = run_once(args, bundle, t, llm, sdxl, run_dir, seed, concepts)
        for report in reports:
            report["totalSec"] = round(sum(report.get("timings", {}).values()), 2)
            (root / report["dir"] / "timings.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=1), "utf-8"
            )
            summary["runs"].append(report)
            print(json.dumps(report, ensure_ascii=False), flush=True)
        summary.setdefault("runSec", []).append(round(time.perf_counter() - start, 2))
        if sdxl is not None:
            sdxl.unload()
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    print(f"결과: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
