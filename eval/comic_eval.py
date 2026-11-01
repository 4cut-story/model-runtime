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
STOPWORDS = {"a", "an", "the", "and", "at", "in", "on", "of", "with", "to", "is", "are"}
MAX_SCENE_OVERLAP = 0.8
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
    merged: list[dict[str, Any]] = []
    for u in utterances:
        text = clean(u["text"])
        if not text:
            continue
        if merged and merged[-1]["speaker"] == u["speaker"] and u["start"] - merged[-1]["end"] <= 2:
            merged[-1] = {**merged[-1], "end": u["end"], "text": merged[-1]["text"] + " " + text}
            continue
        merged.append({**u, "text": text})
    return lines(merged)


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
    bundle: dict[str, Any], step: str, prompt: str, seed: int, model: str
) -> dict[str, Any]:
    spec = bundle[step]
    return {
        "model": model,
        "system": spec["system"],
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


def script_prompt(
    t: dict[str, Any], scene: dict[str, Any], context: float, cast: dict[str, str]
) -> str:
    part = between(t, scene["start"] - context, scene["end"] + context)
    names = ", ".join(f"{k}={v}" for k, v in cast.items())
    return (
        f"대화 언어: {t['language']}\n화자: {', '.join(t['speakers'])}\n인물 가명: {names}\n"
        f"장면 종류: {scene['kind']}\n장면 요약: {scene['summary']}\n"
        f"고른 이유: {scene['reason']}\n\n"
        f"장면 앞뒤 대화:\n{compact(part)}"
    )


def valid_candidate(c: dict[str, Any]) -> bool:
    return (
        c.get("kind") in ("FUNNY", "IMPORTANT")
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
    return script


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


def scrubbed(script: dict[str, Any]) -> dict[str, Any]:
    """nameWarnings 실명을 대사·제목에서 "친구"로 바꾸고 JSON 조각을 지운다 (ComicScript)."""
    rename: dict[str, str] = {}
    for w in script.get("nameWarnings") or []:
        word = (w or "").strip()
        if len(word) < 2:
            continue
        rename[word] = "친구"
        if len(word) > 2 and word.endswith("이"):
            rename[word[:-1]] = "친구"
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


def draw_prompt(style: str, script: dict[str, Any], panel: dict[str, Any]) -> str:
    scene = panel["scene"].strip()
    legacy = "{" not in scene
    cast = ""
    for c in script["characters"]:
        if legacy:
            cast += "; " + c["look"].strip()
        else:
            scene = scene.replace("{" + c["speaker"] + "}", "(" + c["look"].strip() + ")")
    shot = panel["shot"].replace("-", " ") + " shot, " if panel.get("shot") else ""
    return f"{style}, {shot}{scene}{cast}"


def draw_images(bundle: dict[str, Any], script: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    d = bundle["draw"]
    return [
        {
            "name": f"panel-{i + 1}",
            "prompt": draw_prompt(d["style"], script, p),
            "negativePrompt": d["negativePrompt"],
            "seed": seed + i,
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
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    timer = Timer()
    report: dict[str, Any] = {"seed": seed, "promptVersion": bundle["promptVersion"]}
    size = bundle["windowSec"]
    last = int(max(u["end"] for u in t["utterances"])) // size
    candidates: list[dict[str, Any]] = []
    for w in range(last + 1):
        part = between(t, w * size, (w + 1) * size)
        if not part:
            continue
        job = llm_job(bundle, "extract", extract_prompt(t, w, size, part), seed + w, args.model)
        res = timer.run("extract", lambda job=job: llm.chat(job))
        try:
            found = content_json(res["content"]).get("candidates", [])
        except ValueError:
            found = []
        candidates += [c for c in found if valid_candidate(c)]
    (out / "candidates.json").write_text(
        json.dumps(candidates, ensure_ascii=False, indent=1), "utf-8"
    )
    if not candidates:
        report["result"] = "NO_SCENE"
        report["timings"] = timer.times
        return report
    scene = max(candidates, key=lambda c: c["score"])
    cast = cast_names(t, bundle["castNames"], seed)
    script = None
    failures = []
    for attempt in range(3):
        job = llm_job(
            bundle,
            "script",
            script_prompt(t, scene, bundle["contextSec"], cast),
            seed + attempt,
            args.model,
        )
        res = timer.run("script", lambda job=job: llm.chat(job))
        try:
            s = normalized(content_json(res["content"]))
            s = with_fictional_names(with_names(s, cast), spoken(t), bundle["castNames"])
            s = scrubbed(s)
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
        report["timings"] = timer.times
        return report
    (out / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=1), "utf-8")
    images = draw_images(bundle, script, seed * 10)
    (out / "draw.json").write_text(json.dumps(images, ensure_ascii=False, indent=1), "utf-8")
    report["title"] = script["title"]
    if sdxl is not None:
        from PIL import Image

        llm.unload()
        pngs = []
        for img in images:
            data = timer.run("draw", lambda img=img: sdxl.draw(img))
            path = out / f"{img['name']}.png"
            path.write_bytes(data)
            pngs.append(Image.open(path).convert("RGB"))
        timer.run("compose", lambda: compose(script, pngs, out / "comic.png"))
    report["result"] = "DONE"
    report["timings"] = timer.times
    return report


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
    args = p.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model_runtime.backends.ollama import OllamaClient
    from model_runtime.config import Settings, load_env_file

    load_env_file(Path(".env"))
    bundle = json.loads(args.bundle.read_text(encoding="utf-8"))
    root = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"promptVersion": bundle["promptVersion"], "runs": []}

    if args.audio:
        from model_runtime.backends.whisper import WhisperBackend

        env = Settings.from_env(
            {"GPU_WORKER_API_URL": "x", "GPU_WORKER_TOKEN": "x", **dict(__import__("os").environ)}
        )
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

        s = Settings.from_env(
            {"GPU_WORKER_API_URL": "x", "GPU_WORKER_TOKEN": "x", **dict(__import__("os").environ)}
        )
        sdxl = SdxlBackend(
            s.sdxl_model, s.sdxl_cpu_offload, s.sdxl_lora, s.sdxl_steps, s.sdxl_guidance
        )
        summary["sdxl"] = sdxl.model_info()

    for i in range(args.runs):
        seed = args.seed + i * 100
        run_dir = root / f"run-{i + 1:02d}"
        start = time.perf_counter()
        report = run_once(args, bundle, t, llm, sdxl, run_dir, seed)
        report["totalSec"] = round(time.perf_counter() - start, 2)
        (run_dir / "timings.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=1), "utf-8"
        )
        summary["runs"].append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if sdxl is not None:
            sdxl.unload()
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), "utf-8")
    print(f"결과: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
