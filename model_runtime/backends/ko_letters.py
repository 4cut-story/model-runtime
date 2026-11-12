"""대사의 영어 약어를 한글 읽기로 바꾼다.

GPT-SoVITS는 한국어로 읽을 때 "AI", "GPU" 같은 로마자를 엉뚱하게 읽는다.

낱말처럼 읽는 약어(NASA → 나사)와 자주 나오는 이름(YouTube → 유튜브)은 WORDS로 읽고,
나머지 대문자 묶음은 알파벳 이름을 한 자씩 이어 읽는다(GPU → 지피유, A씨 → 에이씨).
소문자가 섞인 낯선 낱말은 그대로 둔다.
"""

from __future__ import annotations

import re

LETTERS = {
    "A": "에이",
    "B": "비",
    "C": "씨",
    "D": "디",
    "E": "이",
    "F": "에프",
    "G": "지",
    "H": "에이치",
    "I": "아이",
    "J": "제이",
    "K": "케이",
    "L": "엘",
    "M": "엠",
    "N": "엔",
    "O": "오",
    "P": "피",
    "Q": "큐",
    "R": "알",
    "S": "에스",
    "T": "티",
    "U": "유",
    "V": "브이",
    "W": "더블유",
    "X": "엑스",
    "Y": "와이",
    "Z": "제트",
}

# 한 자씩 읽지 않고 낱말로 읽는 것. 긴 것부터 맞춘다
WORDS = {
    "ChatGPT": "챗지피티",
    "YouTube": "유튜브",
    "iPhone": "아이폰",
    "KOSDAQ": "코스닥",
    "KOSPI": "코스피",
    "UNESCO": "유네스코",
    "UNICEF": "유니세프",
    "COVID": "코비드",
    "NASA": "나사",
    "NATO": "나토",
    "OPEC": "오펙",
    "FIFA": "피파",
    "ASEAN": "아세안",
    "APEC": "에이펙",
    "OECD": "오이시디",
}

_WORD = re.compile(
    r"(?<![A-Za-z])("
    + "|".join(sorted(map(re.escape, WORDS), key=len, reverse=True))
    + r")(?![a-z])"
)
# 앞뒤에 로마자 소문자가 붙지 않은 대문자 1~6자 (AI, GPU, A씨, 5G)
_CAPS = re.compile(r"(?<![A-Za-z])[A-Z]{1,6}(?![A-Za-z])")


def spell(text: str) -> str:
    """대사 속 영어 약어를 한글 읽기로 바꾼다. 없으면 그대로."""
    text = _WORD.sub(lambda m: WORDS[m.group(1)], text)
    return _CAPS.sub(lambda m: "".join(LETTERS[c] for c in m.group(0)), text)
