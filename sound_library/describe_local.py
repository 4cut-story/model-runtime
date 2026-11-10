"""GPU PC의 로컬 LLM(Ollama)이 근거를 보고 소리마다 항목별 한국어 설명을 쓴다 (4차부터, 기존 소리 다시 쓰기에도).

  OLLAMA_URL=http://localhost:11434 python describe_local.py --candidates data/candidates-4.jsonl \
      --clap data/clap-4.jsonl.gz --out data/descriptions-4.jsonl [--model qwen3:8b]

- 근거: 원본 제목·태그·올린 사람의 설명, CLAP이 들은 소리 5개와 실내/바깥·밤/낮 점수.
- 벡터로 만드는 설명은 "들리는 소리" 기준 항목으로만 쓴다(소리 나는 것, 소리 결, 이어짐, 장소, 시간대, 날씨,
  분위기, 함께 들리는 소리). "어떤 장면에 어울린다"는 쓰지 않는다. 장면 추천 한 줄이 실제 소리와 다른 장면으로
  검색을 끌고 가기 때문이다(마트 장면에 잡힌 와인병 부딪히는 소리).
- 말소리가 중심이거나, 분류와 다르거나, 녹음 품질이 낮으면 exclude=true로 남기고 등록하지 않는다.
- 이미 쓴 key는 건너뛰므로 끊겨도 다시 돌리면 이어서 쓴다.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from categories import ALL, MOODS
from fetch_clips import key
from gpu_gate import wait_until_free
from select_clips import jsonline

CATS = {c.code: c for c in ALL}
KIND = {
    "AMBIENCE": "장소·환경음(계속 깔리는 소리)",
    "EFFECT": "효과음(한 번 나는 소리)",
    "MUSIC": "배경음악(곡). 소리를 내는 것에는 악기를 쓴다",
}
# 원본 태그·FSD50K 라벨에 이게 있으면 사람 목소리가 섞인 소리로 본다
VOICE_TAGS = {
    "speech",
    "voice",
    "voices",
    "talking",
    "talk",
    "conversation",
    "chatter",
    "chat",
    "crowd",
    "people",
    "walla",
    "babble",
    "murmur",
    "laughter",
    "laughing",
    "laugh",
    "shout",
    "shouting",
    "yelling",
    "scream",
    "screaming",
    "singing",
    "vocal",
    "vocals",
    "narration",
    "spoken",
    "words",
    "child",
    "children",
    "kids",
    "baby",
    "male_speech_and_man_speaking",
    "female_speech_and_woman_speaking",
    "child_speech_and_kid_speaking",
    "male_singing",
    "female_singing",
}
# 사람 소리가 바로 그 소리인 카테고리(환호성, 웃음, 학교 아이들, 붐비는 가게 등)는 목소리로 빼지 않는다
VOICE_OK = {
    "SCHOOL_KIDS",
    "CHEERING",
    "PARTY_CROWD",
    "LAUGHTER",
    "SCREAM",
    "CRYING",
    "GASP_SIGH",
    "COUGH",
    "SNEEZE",
    "WHISPER",
    "BABY",
    "SNORE",
    "YAWN",
    "BREATH",
    "BURP",
    "VOICE",
    "MAN_TALKING",
    "WOMAN_TALKING",
    "ROBOT_VOICE",
    "FIGHT",
    "BATTLE",
    "STADIUM",
    "SPORTS",
    "CONCERT",
    "PARADE",
    "PLAYGROUND",
    "SWIMMING_POOL",
    "CAFE",
    "RESTAURANT",
    "FASTFOOD",
    "BAR",
    "MARKET",
    "SHOPPING",
}
UNKNOWN = {"", "모름", "알 수 없음", "없음", "unknown", "none"}

PROMPT = """너는 만화 서비스의 배경음 사서다. 아래 근거로 이 소리가 실제로 어떻게 들리는지 한국어로 자세히 정리해라.
근거는 올린 사람이 쓴 것과 자동 분석이라 틀릴 수 있다. 서로 맞는 내용을 믿고, 모르는 항목은 "모름"으로 둔다.
말소리가 있어도 말한 내용은 옮기지 않는다. 사람 이름, 상표, 곡 제목은 쓰지 않는다. 어떤 장면에 어울리는지는 쓰지 않는다.

[분류] {kind} / {label}
[원본 제목] {title}
[원본 태그] {tags}
[올린 사람의 설명] {note}
[자동 분석이 들은 소리] {heard}
[자동 분석] {attrs}
[길이] {duration:.1f}초

다음 JSON 하나만 답한다.
{{
 "matches": 이 소리가 분류 "{label}"에 맞으면 true,
 "title": "짧은 한국어 제목, 20자 이내. 장소를 알면 앞에 쓰고 끝은 소리 이름으로 맺는다 (예: 마트 계산대 병 부딪히는 소리, 패스트푸드점 주문 소리)",
 "source": "소리를 내는 것 (예: 유리병 여러 개, 사람들 수십 명, 지붕에 떨어지는 빗방울)",
 "texture": "어떻게 들리는지 구체적으로: 크기, 높낮이, 리듬, 질감 (예: 높은 땡그랑 소리가 불규칙하게 짧게 난다)",
 "continuous": 처음부터 끝까지 비슷하게 계속 깔리면 true, 한 번이나 몇 번 짧게 나면 false,
 "place": "소리로 알 수 있는 장소를 흔히 쓰는 낱말로 (예: 마트 계산대, 비 오는 골목, 키오스크 앞, 모름)",
 "indoor": "실내" 또는 "바깥" 또는 "모름",
 "time": "아침" 또는 "낮" 또는 "저녁" 또는 "밤" 또는 "모름",
 "weather": "비" 또는 "눈" 또는 "바람" 또는 "맑음" 또는 "모름",
 "mood": "분위기 한 구절 (예: 바쁘고 분주한)",
 "others": "함께 섞여 들리는 다른 소리, 없으면 없음",
 "speech": 사람 말소리가 또렷하게 중심이면 true,
 "voice": 사람 목소리(말소리, 웅성거림, 웃음, 비명, 외침, 아기 울음, 노래)가 배경에서라도 조금이라도 들리면 true,
 "moods": [아래 분위기 코드 중 맞는 것 1~3개],
 "places": ["어울리는 장소 한국어 낱말 0~3개"],
 "quality": 녹음 상태와 배경음으로 쓰기 좋은 정도 1~5
}}
분위기 코드: {moods}
"""


def read_jsonl(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]


def known(v: object) -> str:
    s = str(v or "").strip()
    return "" if s.lower() in UNKNOWN else s


def compose(x: dict, kind: str) -> str:
    """벡터로 만드는 설명. 항목 순서는 콘티의 소리 장면 문장과 같게 한다(들리는 것 → 결 → 이어짐 → 장소 → 시간·날씨 → 분위기)."""
    parts = [f"들리는 소리: {known(x.get('source'))}", f"소리 결: {known(x.get('texture'))}"]
    continuous = x.get("continuous")
    if continuous is None:
        continuous = kind == "AMBIENCE"
    if kind == "MUSIC":
        parts.append("이어짐: 계속 흐르는 배경음악")
    else:
        parts.append("이어짐: 계속 깔리는 배경음" if continuous else "이어짐: 한 번 나는 효과음")
    place = known(x.get("place"))
    indoor = known(x.get("indoor"))
    if place or indoor:
        parts.append(f"장소: {', '.join(p for p in (place, indoor) if p)}")
    for label, field in (("시간대", "time"), ("날씨", "weather"), ("분위기", "mood")):
        if known(x.get(field)):
            parts.append(f"{label}: {known(x.get(field))}")
    if known(x.get("others")):
        parts.append(f"함께 들리는 소리: {known(x.get('others'))}")
    return ". ".join(p for p in parts if not p.endswith(": ")) + "."


def evidence(c: dict, clap: dict | None) -> dict[str, object]:
    cat = CATS.get(c["category"])
    attrs = ""
    heard = ""
    if clap:
        heard = ", ".join(clap.get("heard", [])[:5])
        a = clap.get("attributes", {})
        attrs = ", ".join(f"{k}: {a[k]}" for k in ("place", "time") if k in a)
    return {
        "kind": KIND.get(c["kind"], c["kind"]),
        "label": cat.label if cat else c["category"],
        "title": c.get("title_original", ""),
        "tags": ", ".join(c.get("tags", [])[:15]),
        "note": str(c.get("note") or "").replace("\n", " ")[:500],
        "heard": heard or "없음",
        "attrs": attrs or "없음",
        "duration": float(c.get("duration") or 0),
        "moods": ", ".join(f"{k}({v[0]})" for k, v in MOODS.items()),
    }


# macOS 13 Metal에서 llama-server가 GGML_ASSERT(buf_dst)로 죽는다. 그때는 OLLAMA_NUM_GPU=0으로 CPU에서 돌린다
OPTIONS: dict[str, object] = {"temperature": 0.2, "num_ctx": 4096}
if os.environ.get("OLLAMA_NUM_GPU"):
    OPTIONS["num_gpu"] = int(os.environ["OLLAMA_NUM_GPU"])


def ask(url: str, model: str, prompt: str, keep_alive: object = 0) -> dict:
    # keep_alive=0: 생성이 끝나면 바로 모델을 내려 운영 그림 작업에 VRAM을 돌려준다.
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "think": False,
            "keep_alive": keep_alive,
            "options": OPTIONS,
        }
    ).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/api/generate", body, {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(json.loads(r.read())["response"])


def unload(url: str, model: str) -> None:
    """Ollama에서 모델을 즉시 내린다(프롬프트 없이 keep_alive=0). VRAM을 운영 작업에 양보."""
    body = json.dumps({"model": model, "keep_alive": 0}).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/api/generate", body, {"Content-Type": "application/json"}
    )
    with contextlib.suppress(Exception), urllib.request.urlopen(req, timeout=30):
        pass


def describe(c: dict, x: dict, model: str) -> dict:
    moods = [m for m in x.get("moods", []) if m in MOODS][:3]
    places = [str(p).strip() for p in x.get("places", []) if str(p).strip()][:3]
    quality = int(x.get("quality") or 3)
    # 배경음으로 분류됐는데 한 번 나는 소리면 계속 틀 수 없어 뺀다(와인병 부딪히는 소리)
    one_shot = c["kind"] == "AMBIENCE" and x.get("continuous") is False
    # 음악의 노랫소리는 말소리로 보지 않는다(노래가 든 곡은 앞 차수에서 이미 뺐다)
    speech = bool(x.get("speech")) and c["kind"] != "MUSIC"
    # 배경음·효과음은 뒤에서 사람 목소리가 조금만 들려도 뺀다(만화 대사와 겹친다). 음악은 앞 차수 기준(노래 든 곡 제외)을 따른다
    tags = {str(t).lower() for t in c.get("tags", [])} | {
        str(t).lower() for t in c.get("labels", [])
    }
    tags |= set(re.findall(r"[a-z_]+", str(c.get("title_original", "")).lower()))
    voice = (
        c["kind"] != "MUSIC"
        and c["category"] not in VOICE_OK
        and (bool(x.get("voice")) or bool(VOICE_TAGS & tags))
    )
    exclude = (not x.get("matches", True)) or speech or voice or quality <= 2 or one_shot
    return {
        "key": key(c),
        "model": model,
        "title": str(x.get("title") or c.get("title_original", ""))[:60],
        "description": compose(x, c["kind"]),
        "moods": moods,
        "places": places,
        "exclude": exclude,
        "category": c["category"],
        "fields": {
            k: x.get(k)
            for k in (
                "source",
                "texture",
                "continuous",
                "place",
                "indoor",
                "time",
                "weather",
                "mood",
                "others",
                "voice",
                "quality",
            )
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--clap", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--only", type=Path, help="이 설명 파일에서 빼지 않은 key만 다시 쓴다(기존 소리 다시 쓰기)"
    )
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument(
        "--keep-alive",
        default="0",
        help="Ollama keep_alive. 0이면 생성 후 바로 모델을 내려 VRAM을 비운다(기본)",
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="라이브 GPU 워커가 작업 중이면 비는 동안만 생성한다(GPU PC 공용 시). 워커 로그가 없으면 무효",
    )
    a = ap.parse_args()
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    keep_alive: object = 0 if str(a.keep_alive) in ("0", "0s") else a.keep_alive

    cands = read_jsonl(a.candidates)
    if a.only:
        keep = {d["key"] for d in read_jsonl(a.only) if not d.get("exclude")}
        cands = [c for c in cands if key(c) in keep]
    claps = {x["key"]: x for x in read_jsonl(a.clap)}
    done = {d["key"] for d in read_jsonl(a.out)}
    todo = [c for c in cands if key(c) not in done]
    print(f"{len(todo)}개를 쓴다 (이미 {len(done)}개)", flush=True)
    lock = Lock()
    count = {"ok": 0, "fail": 0}

    def one(c: dict) -> None:
        if a.gate:
            # 운영 그림 작업이 돌면 비는 동안만. 기다리기 시작할 때 모델을 내려 VRAM을 양보한다
            wait_until_free(on_wait=lambda: unload(url, a.model))
        prompt = PROMPT.format(**evidence(c, claps.get(key(c))))
        try:
            d = describe(c, ask(url, a.model, prompt, keep_alive), a.model)
        except Exception as e:  # 한 개 실패로 멈추지 않는다. 다시 돌리면 이어 쓴다
            with lock:
                count["fail"] += 1
                print(f"fail {key(c)} {e}", flush=True)
            return
        with lock, open(a.out, "a", encoding="utf-8") as f:
            f.write(jsonline(d) + "\n")
            count["ok"] += 1
            if count["ok"] % 200 == 0:
                print(f"{count['ok']}개 썼다", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(one, todo))
    print(f"끝: {count['ok']}개, 실패 {count['fail']}개")


if __name__ == "__main__":
    main()
