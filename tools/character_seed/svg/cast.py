"""캐릭터 1000명 생성기. 캐릭터 한 명(정체성)을 7개 테마 x 6가지 표정으로 그린다.

정체성(테마와 상관없음): 성별, 나이, 피부, 얼굴형, 눈, 머리 색·길이, 수염, 안경, 체형, 대표 색, 직업, 동물 모습.
테마마다 같은 사람을 그 세계의 옷·머리·소품으로 옮긴다 (예: 의사 → 조선 의원/의녀 → 무협 신의).
그림책 테마는 어린 시절 모습, 동물 테마는 정해 둔 동물 모습으로 그린다.

  python cast.py --count 1000 --out out_cast [--only 0-9] [--themes DAILY,JOSEON] [--expr neutral,happy]
  → out_cast/cast.json, out_cast/svg/<id>/<THEME>_<expr>.svg
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import characters as C
import jobs  # noqa: F401  (직업 옷·소품을 C.EXTRA_*에 등록한다)
from characters import CX, CY, RX, RY, TORSO

THEMES = ["DAILY", "ANIMALS", "WATERCOLOR", "PICTURE_BOOK", "NEWSPAPER", "JOSEON", "WUXIA"]
EXPRS = ["neutral", "happy", "sad", "angry", "surprised", "embarrassed"]
EXPR_KO = {"neutral": "기본", "happy": "웃음", "sad": "울음", "angry": "화남", "surprised": "놀람", "embarrassed": "당황"}
ANIMALS = ["rabbit", "bear", "fox", "panda", "cat", "deer", "dog", "squirrel"]

# ---------- 새 소품 ----------


def A(name):
    def deco(fn):
        C.EXTRA_ACC[name] = fn
        return fn
    return deco


@A("book")
def _book(p, skin):
    return (f'<path d="M 200 640 L 290 620 L 300 740 L 210 760 Z" fill="{p.c("red")}" {p.st(4)}/>'
            f'<path d="M 290 620 L 340 640 L 350 760 L 300 740 Z" fill="{p.c("cream")}" {p.st(4)}/>')


@A("scroll")
def _scroll(p, skin):
    return (f'<rect x="210" y="620" width="120" height="120" fill="{p.c("cream")}" {p.st(4)}/>'
            f'<rect x="196" y="610" width="148" height="16" rx="8" fill="{p.c("brown")}" {p.st(3)}/>'
            f'<rect x="196" y="734" width="148" height="16" rx="8" fill="{p.c("brown")}" {p.st(3)}/>'
            + "".join(f'<line x1="{x}" y1="640" x2="{x}" y2="720" {p.st(3)}/>' for x in (240, 270, 300)))


@A("brush")
def _brush(p, skin):
    return (f'<line x1="230" y1="760" x2="300" y2="600" stroke="{p.c("brown")}" stroke-width="10"/>'
            f'<path d="M 296 606 L 316 560 L 306 612 Z" fill="#222" {p.st(2)}/>')


@A("hoe")
def _hoe(p, skin):
    return (f'<line x1="560" y1="768" x2="620" y2="300" stroke="{p.c("brown")}" stroke-width="14"/>'
            f'<path d="M 600 300 L 680 320 L 676 350 L 610 334 Z" fill="#999" {p.st(3)}/>')


@A("hammer")
def _hammer(p, skin):
    return (f'<line x1="220" y1="760" x2="280" y2="610" stroke="{p.c("brown")}" stroke-width="14"/>'
            f'<rect x="240" y="580" width="90" height="40" rx="4" fill="#888" {p.st(3)} transform="rotate(22 285 600)"/>')


@A("guitar")
def _guitar(p, skin):
    return (f'<ellipse cx="300" cy="690" rx="80" ry="64" fill="{p.c("orange")}" {p.st(4)}/>'
            f'<circle cx="300" cy="690" r="20" fill="#3a2a22"/>'
            f'<line x1="330" y1="660" x2="520" y2="520" stroke="{p.c("dark brown")}" stroke-width="22"/>'
            f'<line x1="330" y1="660" x2="520" y2="520" {p.st(2)}/>')


@A("ball")
def _ball(p, skin):
    return (f'<circle cx="250" cy="680" r="60" fill="{p.c("orange")}" {p.st(4)}/>'
            f'<path d="M 190 680 L 310 680 M 250 620 L 250 740 M 208 638 Q 250 680 208 722 M 292 638 Q 250 680 292 722" fill="none" {p.st(3)}/>')


@A("headphones")
def _headphones(p, skin):
    return (f'<path d="M 256 300 Q 256 140 {CX} 140 Q 512 140 512 300" fill="none" stroke="#333" stroke-width="16"/>'
            f'<rect x="236" y="280" width="40" height="70" rx="14" fill="{p.c("red")}" {p.st(3)}/>'
            f'<rect x="492" y="280" width="40" height="70" rx="14" fill="{p.c("red")}" {p.st(3)}/>')


@A("goggles")
def _goggles(p, skin):
    return (f'<path d="M 262 250 Q {CX} 230 506 250" fill="none" stroke="#333" stroke-width="12"/>'
            f'<rect x="318" y="226" width="56" height="40" rx="14" fill="#bfe3ff" {p.st(4)}/>'
            f'<rect x="394" y="226" width="56" height="40" rx="14" fill="#bfe3ff" {p.st(4)}/>')


@A("flask")
def _flask(p, skin):
    return (f'<path d="M 240 600 L 270 600 L 270 640 L 310 730 L 200 730 L 240 640 Z" fill="#e8f6ff" {p.st(4)}/>'
            f'<path d="M 222 690 L 288 690 L 310 730 L 200 730 Z" fill="{p.c("green")}"/>')


@A("teacup")
def _teacup(p, skin):
    return (f'<path d="M 220 660 Q 260 740 300 660 Z" fill="{p.c("jade")}" {p.st(4)}/>'
            f'<ellipse cx="260" cy="740" rx="50" ry="10" fill="{p.c("cream")}" {p.st(3)}/>')


@A("cleaver")
def _cleaver(p, skin):
    return (f'<rect x="200" y="600" width="90" height="70" fill="#cfd6db" {p.st(4)}/>'
            f'<rect x="280" y="620" width="70" height="20" rx="6" fill="{p.c("dark brown")}" {p.st(3)}/>')


@A("straw hat")
def _straw(p, skin):
    return (f'<ellipse cx="{CX}" cy="200" rx="220" ry="40" fill="#e3c77a" {p.st()}/>'
            f'<path d="M 300 200 Q 300 110 {CX} 106 Q 468 110 468 200 Z" fill="#e3c77a" {p.st()}/>'
            f'<rect x="300" y="176" width="168" height="18" fill="{p.c("red")}"/>')


@A("tie")
def _tie(p, skin):
    return f'<path d="M {CX} 500 L 368 520 L {CX} 640 L 400 520 Z" fill="{p.c("crimson")}" {p.st(3)}/>'


@A("lanyard")
def _lanyard(p, skin):
    return (f'<path d="M 340 480 L {CX} 600 L 428 480" fill="none" stroke="{p.c("blue")}" stroke-width="6"/>'
            f'<rect x="356" y="600" width="56" height="70" rx="4" fill="#fff" {p.st(3)}/>')


@A("whisk")
def _whisk(p, skin):
    return (f'<line x1="230" y1="760" x2="270" y2="620" stroke="{p.c("brown")}" stroke-width="10"/>'
            + "".join(f'<path d="M 270 620 Q {250 + i * 10} 560 {230 + i * 18} 520" fill="none" stroke="#f4f4f4" stroke-width="5"/>' for i in range(5)))


@A("towel")
def _towel(p, skin):
    return f'<path d="M 300 476 L 280 640 L 320 640 L 330 480 Z" fill="#fff" {p.st(3)}/>'


@A("beard")
def _beard(p, skin):
    return ""  # 색이 필요해 face에서 그린다


def suit(p, c, skin):
    s = p.st()
    return (f'<path d="{TORSO}" fill="{c}" {s}/>'
            f'<path d="M 330 476 L {CX} 600 L 438 476 Z" fill="{p.c("white")}" {s}/>'
            f'<path d="M 320 474 L 360 640 L 330 660 L 296 490 Z M 448 474 L 408 640 L 438 660 L 472 490 Z" fill="{c}" {s}/>'
            f'<path d="M {CX} 510 L 370 528 L {CX} 640 L 398 528 Z" fill="{p.c("crimson")}" {p.st(3)}/>')


def jersey(p, c, skin):
    s = p.st()
    return (f'<path d="{TORSO}" fill="{c}" {s}/>'
            f'<path d="M 330 476 L {CX} 530 L 438 476" fill="none" stroke="#fff" stroke-width="12"/>'
            f'<path d="M 196 600 L 240 600 L 236 768 M 572 600 L 528 600 L 532 768" fill="none" stroke="#fff" stroke-width="10"/>'
            f'<text x="{CX}" y="680" text-anchor="middle" font-family="Arial Black,Arial" font-weight="900" font-size="96" fill="#fff" stroke="#222" stroke-width="3">{getattr(p, "number", 7)}</text>')


def apron_shop(p, c, skin):
    return jobs.apron_over("teal")(p, c, skin)


C.EXTRA_CLOTHES.update({"suit": suit, "jersey": jersey, "apron": apron_shop})

# ---------- 직업: 현대 / 조선 / 무협 ----------
# 각 항목: (옷, [소품·모자], 한국어 직업 이름, 영어 이름). 옷이 None이면 평상복을 쓴다. f_/m_는 성별 전용.

CASUAL = ["hoodie", "cardigan", "sweater", "flannel", "shirt", "jacket", "raincoat", "striped", "blouse", "dress", "overalls"]
CASUAL_M = ["hoodie", "cardigan", "sweater", "flannel", "shirt", "jacket", "raincoat", "striped", "overalls"]

JOBS = {
    "casual": dict(ko="", modern=(None, []), joseon=(None, []), joseon_ko="", wuxia=(None, ["long sword"]), wuxia_ko="협객", en=""),
    "student": dict(ko="학생", en="student", modern=("hoodie", ["book"]), joseon=(None, ["book"]), joseon_ko="유생",
                    wuxia=(None, ["long sword"]), wuxia_ko="문파 제자"),
    "office": dict(ko="회사원", en="office worker", modern=("suit", ["lanyard"]), joseon=(None, ["gat", "scroll"]), joseon_ko="선비",
                   f_joseon=(None, ["folding fan"]), f_joseon_ko="규수", wuxia=(None, ["folding fan"]), wuxia_ko="문사"),
    "doctor": dict(ko="의사", en="doctor", modern=("doctor", ["stethoscope"]), joseon=(None, ["herb basket"]), joseon_ko="의원",
                   f_joseon=("uinyeo", ["garima", "herb basket"]), f_joseon_ko="의녀", wuxia=(None, ["herb hat", "gourd"]), wuxia_ko="신의"),
    "nurse": dict(ko="간호사", en="nurse", modern=("doctor", ["headband white"]), joseon=("uinyeo", ["garima", "herb basket"]), joseon_ko="의녀",
                  m_joseon=(None, ["herb basket"]), m_joseon_ko="의원 제자", wuxia=(None, ["gourd"]), wuxia_ko="의방 제자"),
    "chef": dict(ko="요리사", en="chef", modern=("chef", ["chef hat"]), joseon=(None, ["headband white", "cleaver"]), joseon_ko="숙수",
                 f_joseon=("jumo", ["jumo scarf"]), f_joseon_ko="주모", wuxia=(None, ["headband white", "cleaver"]), wuxia_ko="객잔 주방장"),
    "firefighter": dict(ko="소방관", en="firefighter", modern=("fire", ["fire helmet"]), joseon=("podol", ["jeollip"]), joseon_ko="멸화군",
                        wuxia=(None, ["broadsword"]), wuxia_ko="무사"),
    "police": dict(ko="경찰관", en="police officer", modern=("police", ["police cap"]), joseon=("podol", ["jeollip"]), joseon_ko="포졸",
                   wuxia=(None, ["broadsword"]), wuxia_ko="포쾌"),
    "detective": dict(ko="형사", en="detective", modern=("trench", ["fedora"]), joseon=("podol", ["jeollip"]), joseon_ko="포도부장",
                      f_joseon=(None, ["dagger"]), f_joseon_ko="다모", wuxia=(None, ["folding fan"]), wuxia_ko="명탐"),
    "teacher": dict(ko="선생님", en="teacher", modern=("cardigan", ["glasses", "book"]), joseon=(None, ["gat", "book"]), joseon_ko="훈장",
                    f_joseon=(None, ["book"]), f_joseon_ko="글 선생", wuxia=(None, ["whisk"]), wuxia_ko="장문인"),
    "farmer": dict(ko="농부", en="farmer", modern=("overalls", ["straw hat", "hoe"]), joseon=(None, ["straw hat", "hoe"]), joseon_ko="농부",
                   wuxia=(None, ["straw hat", "hoe"]), wuxia_ko="농부"),
    "barista": dict(ko="바리스타", en="barista", modern=("barista", ["coffee"]), joseon=(None, ["teacup"]), joseon_ko="다모(차 끓이는 이)",
                    wuxia=(None, ["teacup"]), wuxia_ko="찻집 주인"),
    "florist": dict(ko="꽃집 주인", en="florist", modern=("florist", ["bouquet"]), joseon=(None, ["bouquet"]), joseon_ko="꽃 장수",
                    wuxia=(None, ["bouquet"]), wuxia_ko="화원지기"),
    "painter": dict(ko="화가", en="painter", modern=("painter", ["beret red", "palette"]), joseon=(None, ["brush"]), joseon_ko="화원",
                    wuxia=(None, ["brush"]), wuxia_ko="화사"),
    "mail": dict(ko="우체부", en="mail carrier", modern=("postal", ["cap"]), joseon=(None, ["jeollip", "scroll"]), joseon_ko="파발꾼",
                 wuxia=(None, ["broadsword"]), wuxia_ko="표국 표사"),
    "mechanic": dict(ko="정비공", en="mechanic", modern=("coverall", ["mech cap", "wrench"]), joseon=(None, ["headband white", "hammer"]),
                     joseon_ko="대장장이", wuxia=(None, ["hammer"]), wuxia_ko="대장장이"),
    "reporter": dict(ko="기자", en="reporter", modern=("shirt", ["fedora", "notepad"]), joseon=(None, ["brush", "scroll"]), joseon_ko="서리",
                     wuxia=(None, ["scroll"]), wuxia_ko="정보상"),
    "waiter": dict(ko="식당 종업원", en="waiter", modern=("shirt", ["bowtie", "tray"]), f_modern=("waitress", ["headband white", "tray"]),
                   joseon=(None, ["towel", "tray"]), joseon_ko="주막 일꾼", f_joseon=("jumo", ["jumo scarf", "tray"]), f_joseon_ko="주막 일꾼",
                   wuxia=(None, ["towel"]), wuxia_ko="점소이"),
    "astronaut": dict(ko="우주비행사", en="astronaut", modern=("astronaut", ["space helmet"]), joseon=(None, ["gat", "scroll"]),
                      joseon_ko="관상감 천문학자", f_joseon=(None, ["scroll"]), f_joseon_ko="별을 보는 이", wuxia=(None, ["whisk", "gourd"]), wuxia_ko="도사"),
    "musician": dict(ko="음악가", en="musician", modern=("jacket", ["guitar"]), joseon=(None, ["bamboo flute"]), joseon_ko="악사",
                     wuxia=(None, ["bamboo flute"]), wuxia_ko="악사"),
    "athlete": dict(ko="운동선수", en="athlete", modern=("jersey", ["ball"]), joseon=(None, ["headband white"]), joseon_ko="씨름꾼",
                    f_joseon=(None, ["headband white"]), f_joseon_ko="그네 장사", wuxia=("monk", ["bald"]), wuxia_ko="소림 무승",
                    f_wuxia=(None, ["long sword"]), f_wuxia_ko="아미파 검객"),
    "programmer": dict(ko="개발자", en="programmer", modern=("hoodie", ["headphones", "glasses"]), joseon=(None, ["book"]), joseon_ko="책 읽는 선비",
                       wuxia=(None, ["folding fan", "scroll"]), wuxia_ko="책사"),
    "shopkeeper": dict(ko="가게 주인", en="shopkeeper", modern=("apron", ["abacus"]), joseon=("peddler", ["paeraengi"]), joseon_ko="보부상",
                       f_joseon=("jumo", ["jumo scarf"]), f_joseon_ko="주모", wuxia=("innkeeper", ["shop cap", "abacus"]), wuxia_ko="객잔 주인"),
    "scientist": dict(ko="과학자", en="scientist", modern=("doctor", ["goggles", "flask"]), joseon=(None, ["gat", "scroll"]), joseon_ko="관상감 관원",
                      f_joseon=(None, ["scroll"]), f_joseon_ko="관상감 관원", wuxia=(None, ["whisk", "gourd"]), wuxia_ko="도사"),
    "fighter": dict(ko="무술인", en="martial artist", modern=("jersey", ["headband white"]), joseon=(None, ["long sword"]), joseon_ko="무관",
                    wuxia=("assassin", ["mask", "dagger"]), wuxia_ko="자객"),
}
JOB_WEIGHTS = {"casual": 8}

# ---------- 정체성 ----------

SURNAMES = list("김이박최정강조윤장임한오서신권황안송류홍전고문양손배백허남심노하곽성차주우구민나진지엄채원천방공현")
SYL1 = list("가나다라마바사아자차카타파하서지민수하도시유예주윤은채현태준재승우진연소다나보")
SYL2 = list("영희진수민호준현우빈아연은서원윤훈율린별솔람음결담희나라")

HAIR_COLORS = ["black", "black", "black", "dark brown", "dark brown", "brown", "auburn", "blonde"]
CLOTH_COLORS = ["red", "blue", "green", "yellow", "purple", "pink", "orange", "teal", "navy", "olive", "sky blue", "crimson",
                "jade", "brown", "indigo", "cream", "white", "gray"]
F_HAIR = ["long", "bob", "ponytail", "bun", "wavy", "straight", "curly", "short"]
M_HAIR = ["short", "messy", "curly", "buzz", "crew", "short", "long"]


def make_identity(i: int, rnd: random.Random, used: set[str]) -> dict:
    g = rnd.choice("fm")
    age = rnd.choices(["child", "young", "middle", "old"], [12, 48, 25, 15])[0]
    while True:
        name = rnd.choice(SYL1) + rnd.choice(SYL2)
        if rnd.random() < 0.35:
            name = rnd.choice(SURNAMES) + name
        if name not in used:
            used.add(name)
            break
    jobs_pool = list(JOBS)
    weights = [JOB_WEIGHTS.get(j, 1) for j in jobs_pool]
    job = "student" if age == "child" and rnd.random() < 0.5 else rnd.choices(jobs_pool, weights)[0]
    if age == "child" and job not in ("student", "casual", "athlete", "musician", "painter", "astronaut", "chef", "doctor", "firefighter", "police"):
        job = "casual"
    hair = rnd.choice(F_HAIR if g == "f" else M_HAIR)
    hair_color = rnd.choice(HAIR_COLORS)
    if age == "old":
        hair_color = rnd.choice(["gray", "white", "gray"])
        hair = rnd.choice(["bun", "curly", "short"] if g == "f" else ["short", "crew", "curly"])
    elif age == "middle" and rnd.random() < 0.3:
        hair_color = "gray"
    c1, c2 = rnd.sample(CLOTH_COLORS, 2)
    ident = dict(
        id=f"c{i:04d}", name=name, gender=g, age=age, job=job,
        skin=rnd.choice(["light", "light", "warm", "warm", "tan"]),
        face=rnd.choice(["oval", "round", "long"]),
        eyes=rnd.choice(["round", "lash", "dot", "sharp"] if g == "f" else ["round", "dot", "sharp", "smile"]),
        mouth=rnd.choice(["smile", "smile", "flat", "smirk", "open"]),
        hair=hair, hair_color=hair_color,
        beard=(rnd.choice(["none", "mustache", "beard", "goatee"]) if g == "m" and age in ("middle", "old") else
               (rnd.choice(["none", "none", "none", "mustache"]) if g == "m" and age == "young" else "none")),
        glasses=rnd.random() < (0.35 if age in ("middle", "old") else 0.15),
        build=rnd.choices(["slim", "normal", "round"], [3, 5, 2])[0],
        color=c1, color2=c2,
        casual=rnd.choice(CASUAL if g == "f" else CASUAL_M),
        animal=rnd.choice(ANIMALS),
        number=rnd.randint(1, 99),
        wuxia_hair=rnd.choice(["wuxia_long", "wuxia_high"]),
    )
    return ident


# ---------- 테마별로 옮기기 ----------

GARMENT_EN = {"hoodie": "hoodie", "cardigan": "cardigan", "sweater": "sweater", "flannel": "flannel shirt", "shirt": "shirt",
              "jacket": "jacket", "raincoat": "raincoat", "striped": "striped shirt", "blouse": "blouse", "dress": "dress",
              "overalls": "overalls", "suit": "suit", "doctor": "white doctor coat", "chef": "white chef jacket",
              "fire": "firefighter coat", "police": "police uniform", "trench": "trench coat", "barista": "barista apron",
              "florist": "florist apron", "painter": "painter apron", "postal": "postal uniform", "coverall": "work coverall",
              "waitress": "waitress uniform", "astronaut": "space suit", "jersey": "sports jersey", "apron": "shop apron"}
HAIR_EN = {"long": "long {c} hair", "bob": "{c} bob", "ponytail": "{c} ponytail", "bun": "{c} hair bun", "wavy": "wavy {c} hair",
           "straight": "straight {c} hair", "curly": "curly {c} hair", "short": "short {c} hair", "messy": "messy {c} hair",
           "buzz": "{c} buzz cut", "crew": "{c} crew cut", "pigtails": "{c} pigtails"}
PROP_EN = {"glasses": "round glasses", "fedora": "fedora hat", "beret red": "beret", "cap": "cap", "mech cap": "work cap",
           "police cap": "police cap", "fire helmet": "fire helmet", "chef hat": "chef hat", "headphones": "headphones",
           "straw hat": "straw hat", "gat": "black gat", "jeollip": "black jeollip hat", "paeraengi": "straw hat",
           "jumo scarf": "white headscarf", "garima": "black head cloth", "headband white": "white headband",
           "long sword": "long sword", "broadsword": "broadsword", "folding fan": "folding fan", "bamboo flute": "bamboo flute",
           "gourd": "medicine gourd", "abacus": "abacus", "dagger": "dagger", "whisk": "horsetail whisk", "scroll": "scroll",
           "book": "book", "brush": "ink brush", "hoe": "hoe", "hammer": "hammer", "teacup": "teacup", "bouquet": "bouquet",
           "guitar": "guitar", "ball": "ball", "notepad": "notepad", "wrench": "wrench", "tray": "tray", "stethoscope": "stethoscope",
           "cleaver": "cleaver", "space helmet": "space helmet", "mask": "black mask", "bald": "shaved head", "herb hat": "hood",
           "herb basket": "herb basket", "palette": "palette", "coffee": "coffee cup", "lanyard": "id badge", "goggles": "goggles",
           "towel": "towel", "shop cap": "round cap", "flask": "flask", "bowtie": "bow tie", "pearls": "pearl necklace"}
HATS = {"gat", "jeollip", "paeraengi", "jumo scarf", "garima", "straw hat", "fedora", "beret red", "cap", "mech cap", "police cap",
        "fire helmet", "chef hat", "space helmet", "shop cap", "herb hat", "headband white"}


FIXED = {"doctor", "chef", "fire", "police", "postal", "coverall", "waitress", "astronaut", "barista", "florist", "painter", "trench"}


def garment_text(color: str, clothes: str) -> str:
    return GARMENT_EN.get(clothes, clothes) if clothes in FIXED else f"{color} {GARMENT_EN.get(clothes, clothes)}"


def job_for(ident: dict, era: str) -> tuple[str | None, list[str], str]:
    j = JOBS[ident["job"]]
    g = ident["gender"]
    key = {"modern": "modern", "joseon": "joseon", "wuxia": "wuxia"}[era]
    spec = j.get(f"{g}_{key}") or j[key]
    ko = j["ko"] if era == "modern" else (j.get(f"{g}_{key}_ko") or j.get(f"{key}_ko") or "")
    return spec[0], list(spec[1]), ko


def noun(ident: dict, theme: str) -> str:
    g, age = ident["gender"], ident["age"]
    if theme == "PICTURE_BOOK" or age == "child":
        return "little girl" if g == "f" else "little boy"
    return {"young": "young {}", "middle": "middle-aged {}", "old": "old {}"}[age].format("woman" if g == "f" else "man")


def render_spec(ident: dict, theme: str) -> tuple[dict, str, str]:
    """(그리기 사양, look 문구, 이 테마의 직업 이름)."""
    g, age = ident["gender"], ident["age"]
    child = theme == "PICTURE_BOOK" or age == "child"
    hair, hc = ident["hair"], ident["hair_color"]
    if child:
        hc = ident["hair_color"] if ident["hair_color"] not in ("gray", "white") else "black"
        if g == "f" and hair in ("bun", "straight"):
            hair = "pigtails"
    d = dict(skin=ident["skin"], eyes=ident["eyes"], mouth=ident["mouth"], hair_color=hc, color=ident["color"], acc=[], under=[])
    if theme == "ANIMALS":
        clothes, acc, ko = job_for(ident, "modern")
        clothes = clothes or ident["casual"]
        acc = [a for a in acc if a not in ("glasses",)]
        look = f'{ident["animal"]}, {garment_text(ident["color"], clothes)}'
        if acc and acc[0] in HATS:
            look = f'{ident["animal"]}, {PROP_EN[acc[0]]}'
        return dict(animal=ident["animal"], clothes=clothes, color=ident["color"], extra=acc), look, ko
    if theme in ("JOSEON", "WUXIA"):
        era = theme.lower()
        clothes, acc, ko = job_for(ident, era)
        if theme == "JOSEON":
            if clothes is None:
                clothes = "hanbok_w" if g == "f" else "hanbok_m"
            if g == "f":
                hair = "braid" if child or age == "young" else "chignon"
            else:
                hair = "braid" if child else "topknot"
            if child:
                acc = [a for a in acc if a not in ("gat",)]
            garment = "chima" if g == "f" else "jeogori"
            hat = next((a for a in acc if a in HATS), None)
            hair_word = PROP_EN[hat] if hat else ("braided hair" if hair == "braid" else "chignon bun" if hair == "chignon" else "topknot")
            look = f'{"woman" if g == "f" else "man"} in hanbok, {hair_word}, {ident["color"]} {garment}'
            if child:
                look = f'{"girl" if g == "f" else "boy"} in hanbok, {hair_word}, {ident["color"]} {garment}'
        else:
            if clothes is None:
                clothes = "robe"
            hair = ident["wuxia_hair"] if g == "m" else "wuxia_long"
            if clothes in ("monk",):
                hair = "none"
            if clothes == "assassin":
                hair = "none"
            prop = next((a for a in acc if a not in HATS and a not in ("bald", "mask")), None)
            look = f'{"swordswoman" if g == "f" else "swordsman"}, {ident["color"]} robe' + (f", {PROP_EN[prop]}" if prop else "")
        d.update(hair=hair, clothes=clothes, acc=acc)
    else:
        clothes, acc, ko = job_for(ident, "modern")
        if child and ident["job"] not in ("student", "casual"):
            pass  # 꼬마 직업 (그림책의 꼬마 의사 등)
        clothes = clothes or ident["casual"]
        if g == "m" and clothes in ("blouse", "dress"):
            clothes = "shirt"
        if ident["glasses"] and "glasses" not in acc and "space helmet" not in acc and "goggles" not in acc:
            acc.append("glasses")
        d.update(hair=hair, clothes=clothes, acc=acc)
        hat = next((a for a in acc if a in HATS), None)
        hair_text = HAIR_EN.get(hair, "{c} hair").format(c=hc)
        if theme == "NEWSPAPER":
            hair_text = HAIR_EN.get(hair, "{c} hair").format(c="").replace("  ", " ").strip()
            prop = next((a for a in acc if a not in HATS), None) or hat
            look = f"{noun(ident, theme)}, {hair_text}, {PROP_EN[prop] if prop else GARMENT_EN.get(clothes, clothes)}"
        else:
            second = PROP_EN[hat] if hat else hair_text
            look = f"{noun(ident, theme)}, {second}, {garment_text(ident['color'], clothes)}"
    d["acc"] = [a for a in d["acc"] if not (a == "glasses" and theme in ("JOSEON", "WUXIA"))]
    return d, look, ko


# ---------- 얼굴과 표정 ----------


def features(p, ident, expr, ey, my, brow_col, animal=False):
    st = p.st
    o = []
    base_eyes = ident["eyes"] if not animal else "round"
    ex = 46 if animal else 44
    eyes = {"happy": "smile", "sad": "cry", "angry": "angry", "surprised": "wide", "embarrassed": "swirl"}.get(expr, base_eyes)
    for side in (-1, 1):
        x = CX + side * ex
        if eyes == "round":
            o.append(f'<ellipse cx="{x}" cy="{ey}" rx="13" ry="17" fill="#2a2320"/><circle cx="{x + 4}" cy="{ey - 6}" r="5" fill="#fff"/>')
        elif eyes == "lash":
            o.append(f'<ellipse cx="{x}" cy="{ey}" rx="13" ry="17" fill="#2a2320"/><circle cx="{x + 4}" cy="{ey - 6}" r="5" fill="#fff"/>'
                     f'<path d="M {x - 15} {ey - 14} Q {x} {ey - 26} {x + 15} {ey - 14} L {x + side * 22} {ey - 22}" fill="none" {st(4)}/>')
        elif eyes == "smile":
            o.append(f'<path d="M {x - 16} {ey + 4} Q {x} {ey - 16} {x + 16} {ey + 4}" fill="none" {st(6)}/>')
        elif eyes == "sharp":
            o.append(f'<path d="M {x - 18} {ey + 2} Q {x} {ey - 12} {x + 18} {ey - 2 * side} Q {x} {ey + 8} {x - 18} {ey + 2} Z" fill="#2a2320"/>'
                     f'<circle cx="{x + 2}" cy="{ey - 2}" r="3" fill="#fff"/>')
        elif eyes == "dot":
            o.append(f'<circle cx="{x}" cy="{ey}" r="9" fill="#2a2320"/>')
        elif eyes == "cry":  # 꼭 감은 눈 + 눈물 줄기
            o.append(f'<path d="M {x - 16} {ey - 2} Q {x} {ey + 10} {x + 16} {ey - 2}" fill="none" {st(6)}/>'
                     f'<path d="M {x + side * 4} {ey + 8} Q {x + side * 10} {ey + 60} {x + side * 2} {ey + 110}" fill="none" stroke="#6cc3f0" stroke-width="12" stroke-linecap="round" opacity="0.9"/>')
        elif eyes == "angry":
            o.append(f'<ellipse cx="{x}" cy="{ey + 4}" rx="13" ry="13" fill="#2a2320"/>'
                     f'<path d="M {x - 18} {ey - 14} L {x + 18} {ey - 14} L {x + 18} {ey - 2 - side * 8} Z" fill="{"#f8dcc4" if not animal else "none"}" opacity="0"/>')
        elif eyes == "wide":
            o.append(f'<circle cx="{x}" cy="{ey}" r="20" fill="#fff" {st(4)}/><circle cx="{x}" cy="{ey}" r="7" fill="#2a2320"/>')
        elif eyes == "swirl":
            o.append(f'<path d="M {x - 14} {ey - 10} L {x + 4 * side} {ey} L {x - 14} {ey + 10}" fill="none" {st(5)}/>'
                     if side < 0 else f'<path d="M {x + 14} {ey - 10} L {x - 4} {ey} L {x + 14} {ey + 10}" fill="none" {st(5)}/>')
        # 눈썹
        by = ey - (34 if eyes != "sharp" else 26)
        if expr == "sad":
            pts = (by + 2, by - 10)  # 바깥 낮고 안쪽 높게
            o.append(f'<path d="M {x + side * 18} {pts[0] + 6} L {x - side * 16} {pts[1]}" fill="none" stroke="{brow_col}" stroke-width="7" stroke-linecap="round"/>')
        elif expr == "angry":
            o.append(f'<path d="M {x + side * 18} {by - 6} L {x - side * 14} {by + 10}" fill="none" stroke="{brow_col}" stroke-width="9" stroke-linecap="round"/>')
        elif expr == "surprised":
            o.append(f'<path d="M {x - 18} {by - 10} Q {x} {by - 24} {x + 18} {by - 10}" fill="none" stroke="{brow_col}" stroke-width="7" stroke-linecap="round"/>')
        elif not animal:
            tilt = -8 if eyes == "sharp" else 0
            o.append(f'<path d="M {x - 18} {by + (tilt if side < 0 else 0) + 2} Q {x} {by - 8} {x + 18} {by + (tilt if side > 0 else 0) + 2}" '
                     f'fill="none" stroke="{brow_col}" stroke-width="7" stroke-linecap="round"/>')
    # 코
    if animal:
        o.append(f'<ellipse cx="{CX}" cy="{CY + 52}" rx="16" ry="11" fill="#3a2a26"/>')
    else:
        o.append(f'<path d="M {CX - 4} {CY + 50} Q {CX} {CY + 58} {CX + 5} {CY + 52}" fill="none" {st(3)}/>')
    # 입
    mouth = {"happy": "laugh", "sad": "wail", "angry": "growl", "surprised": "o", "embarrassed": "wavy"}.get(expr, ident["mouth"] if not animal else "cat")
    if mouth == "smile":
        o.append(f'<path d="M {CX - 26} {my} Q {CX} {my + 24} {CX + 26} {my}" fill="none" {st(5)}/>')
    elif mouth == "open":
        o.append(f'<path d="M {CX - 24} {my - 2} Q {CX} {my + 34} {CX + 24} {my - 2} Z" fill="#b5463f" {st(4)}/>')
    elif mouth == "flat":
        o.append(f'<path d="M {CX - 18} {my + 4} Q {CX} {my + 10} {CX + 18} {my + 4}" fill="none" {st(5)}/>')
    elif mouth == "smirk":
        o.append(f'<path d="M {CX - 20} {my + 6} Q {CX + 6} {my + 14} {CX + 24} {my - 2}" fill="none" {st(5)}/>')
    elif mouth == "cat":
        o.append(f'<path d="M {CX} {CY + 62} L {CX} {CY + 74} M {CX - 22} {CY + 80} Q {CX - 10} {CY + 92} {CX} {CY + 74} Q {CX + 10} {CY + 92} {CX + 22} {CY + 80}" fill="none" {st(4)}/>')
    elif mouth == "laugh":
        o.append(f'<path d="M {CX - 32} {my - 6} Q {CX} {my + 46} {CX + 32} {my - 6} Z" fill="#b5463f" {st(4)}/>'
                 f'<path d="M {CX - 14} {my + 22} Q {CX} {my + 12} {CX + 14} {my + 22} Q {CX} {my + 32} {CX - 14} {my + 22} Z" fill="#f08a8a"/>')
    elif mouth == "wail":
        o.append(f'<path d="M {CX - 28} {my + 26} Q {CX} {my - 14} {CX + 28} {my + 26} Q {CX} {my + 36} {CX - 28} {my + 26} Z" fill="#b5463f" {st(4)}/>')
    elif mouth == "growl":
        o.append(f'<path d="M {CX - 28} {my + 4} L {CX + 28} {my + 4} L {CX + 22} {my + 24} L {CX - 22} {my + 24} Z" fill="#fff" {st(4)}/>'
                 f'<line x1="{CX - 26}" y1="{my + 14}" x2="{CX + 26}" y2="{my + 14}" {st(3)}/>')
    elif mouth == "o":
        o.append(f'<ellipse cx="{CX}" cy="{my + 8}" rx="14" ry="20" fill="#7a2a26" {st(4)}/>')
    elif mouth == "wavy":
        o.append(f'<path d="M {CX - 28} {my + 6} q 7 -10 14 0 t 14 0 t 14 0 t 14 0" fill="none" {st(4)}/>')
    # 볼과 기호
    gray = p.t.get("gray")
    blush_op = {"happy": 0.6, "embarrassed": 0.85, "angry": 0.0}.get(expr, 0.45)
    if not gray and blush_op:
        o.append(f'<ellipse cx="{CX - 74}" cy="{CY + 62}" rx="22" ry="12" fill="#f08a8a" opacity="{blush_op}"/>'
                 f'<ellipse cx="{CX + 74}" cy="{CY + 62}" rx="22" ry="12" fill="#f08a8a" opacity="{blush_op}"/>')
    if expr == "embarrassed":
        o.append("".join(f'<line x1="{CX + s * 60 + k}" y1="{CY + 52}" x2="{CX + s * 52 + k}" y2="{CY + 72}" {st(3)}/>' for s in (-1, 1) for k in (-14, 0, 14)))
        o.append(f'<path d="M 520 200 Q 540 240 528 256 Q 510 262 506 244 Q 506 226 520 200 Z" fill="#9fdcff" {st(3)}/>')
    if expr == "angry":
        o.append(f'<g transform="translate(500 190)"><path d="M -18 -4 Q -4 -4 -4 -18 M 4 -18 Q 4 -4 18 -4 M 18 4 Q 4 4 4 18 M -4 18 Q -4 4 -18 4" '
                 f'fill="none" stroke="{"#d23c3c" if not gray else "#111"}" stroke-width="7" stroke-linecap="round"/></g>')
    if expr == "surprised":
        o.append(f'<path d="M 540 150 L 548 210 M 566 160 L 556 214 M 520 160 L 536 212" {st(5)}/>')
    if expr == "sad":
        o.append(f'<path d="M {CX - 40} {CY + 130} q 10 20 0 28 q -10 -8 0 -28 Z" fill="#6cc3f0" opacity="0.9"/>')
    return "".join(o)


def head_and_face(p, ident, expr, skin, d):
    st = p.st()
    sx, sy = {"oval": (1, 1), "round": (1.07, 0.97), "long": (0.94, 1.05)}[ident["face"]]
    if ident["build"] == "round":
        sx *= 1.05
    o = [f'<g transform="translate({CX} {CY}) scale({sx} {sy}) translate({-CX} {-CY})">',
         f'<ellipse cx="{CX - RX + 2}" cy="{CY + 20}" rx="18" ry="26" fill="{skin}" {st}/>',
         f'<ellipse cx="{CX + RX - 2}" cy="{CY + 20}" rx="18" ry="26" fill="{skin}" {st}/>',
         f'<ellipse cx="{CX}" cy="{CY}" rx="{RX}" ry="{RY}" fill="{skin}" {st}/>']
    age = ident["age"] if d.get("child") is not True else "child"
    if age == "old":
        o.append(f'<path d="M {CX - 50} {CY - 70} q 50 -12 100 0 M {CX - 40} {CY - 52} q 40 -10 80 0" fill="none" {p.st(3)} opacity="0.7"/>'
                 f'<path d="M {CX - 92} {CY + 10} l -14 -6 M {CX - 92} {CY + 24} l -14 4 M {CX + 92} {CY + 10} l 14 -6 M {CX + 92} {CY + 24} l 14 4" {p.st(3)} opacity="0.7"/>'
                 f'<path d="M {CX - 48} {CY + 66} q -8 20 4 40 M {CX + 48} {CY + 66} q 8 20 -4 40" fill="none" {p.st(3)} opacity="0.6"/>')
    elif age == "middle":
        o.append(f'<path d="M {CX - 46} {CY + 70} q -6 16 2 30 M {CX + 46} {CY + 70} q 6 16 -2 30" fill="none" {p.st(3)} opacity="0.45"/>')
    brow = p.c(d["hair_color"])
    o.append(features(p, ident, expr, CY + 20, CY + 82, brow))
    beard = ident["beard"] if d.get("child") is not True else "none"
    hcol = p.c(d["hair_color"])
    if beard == "mustache":
        o.append(f'<path d="M {CX - 40} {CY + 74} Q {CX - 20} {CY + 60} {CX} {CY + 70} Q {CX + 20} {CY + 60} {CX + 40} {CY + 74} Q {CX + 20} {CY + 70} {CX} {CY + 76} Q {CX - 20} {CY + 70} {CX - 40} {CY + 74} Z" fill="{hcol}" {p.st(3)}/>')
    elif beard == "beard":
        o.append(f'<path d="M {CX - 100} {CY + 40} Q {CX - 90} {CY + 150} {CX} {CY + 170} Q {CX + 90} {CY + 150} {CX + 100} {CY + 40} '
                 f'Q {CX + 80} {CY + 110} {CX + 40} {CY + 110} Q {CX} {CY + 74} {CX - 40} {CY + 110} Q {CX - 80} {CY + 110} {CX - 100} {CY + 40} Z" fill="{hcol}" {p.st(3)}/>')
    elif beard == "goatee":
        o.append(f'<path d="M {CX - 24} {CY + 112} Q {CX} {CY + 170} {CX + 24} {CY + 112} Z" fill="{hcol}" {p.st(3)}/>'
                 f'<path d="M {CX - 34} {CY + 74} Q {CX} {CY + 62} {CX + 34} {CY + 74}" fill="none" stroke="{hcol}" stroke-width="8" stroke-linecap="round"/>')
    o.append(C.hair_front(p, d["hair"], d["hair_color"]))
    o.append("</g>")
    return "".join(o)


def person_svg(theme, ident, expr):
    d, look, ko = render_spec(ident, theme)
    p = C.Pen(theme)
    p.number = ident["number"]
    child = theme == "PICTURE_BOOK" or ident["age"] == "child"
    d["child"] = child
    skin = p.c(C.SKIN[d["skin"]])
    bx = {"slim": 0.9, "normal": 1.0, "round": 1.14}[ident["build"]] * (0.84 if child else 1.0)
    body = [f'<g transform="translate({CX} 0) scale({bx} 1) translate({-CX} 0)">',
            C.hair_back(p, d["hair"], d["hair_color"]), C.neck(p, skin), C.clothes(p, d["clothes"], d["color"], skin), "</g>"]
    body.append(head_and_face(p, ident, expr, skin, d))
    for a in d["acc"]:
        body.append(C.accessory(p, a, skin))
    filt = p.t.get("filt")
    g = f'<g filter="url(#{filt})">' if filt else "<g>"
    parts = [C.background(p), g + "".join(body) + "</g>"]
    return C.svg(p, "".join(parts)), look, ko


def animal_svg(ident, expr):
    d, look, ko = render_spec(ident, "ANIMALS")
    p = C.Pen("ANIMALS")
    p.number = ident["number"]
    d["face_svg"] = features(p, ident, expr, CY + 14, CY + 74, "#3a2a26", animal=True)
    return C.animal(d), look, ko


def build(count: int, seed: int = 4) -> list[dict]:
    rnd = random.Random(seed)
    used: set[str] = set()
    return [make_identity(i, rnd, used) for i in range(count)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--out", default="out_cast")
    ap.add_argument("--only", help="예: 0-9")
    ap.add_argument("--themes", default=",".join(THEMES))
    ap.add_argument("--expr", default=",".join(EXPRS))
    args = ap.parse_args()
    out = Path(args.out)
    cast = build(args.count)
    lo, hi = (0, len(cast) - 1)
    if args.only:
        lo, hi = (int(x) for x in args.only.split("-"))
    themes = args.themes.split(",")
    exprs = args.expr.split(",")
    meta = []
    for ident in cast[lo : hi + 1]:
        entry = dict(ident)
        entry["themes"] = {}
        for t in themes:
            for e in exprs:
                svg, look, ko = animal_svg(ident, e) if t == "ANIMALS" else person_svg(t, ident, e)
                f = out / "svg" / ident["id"] / f"{t}_{e}.svg"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(svg, encoding="utf-8")
                entry["themes"][t] = dict(look=look, job=ko)
        meta.append(entry)
    (out / "cast.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(meta), "명")


if __name__ == "__main__":
    main()
