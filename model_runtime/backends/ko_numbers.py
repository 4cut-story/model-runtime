"""대사의 숫자를 한글 읽기로 바꾼다. GPT-SoVITS는 아라비아 숫자를 제대로 읽지 못한다.

"260조"는 "이백육십조", "3명"은 "세 명", "2.5%"는 "이 점 오 퍼센트"처럼 읽는다.
세는 말(명·개·살 등) 앞의 1~99는 고유어로, 나머지는 한자어로 읽는다.
"""

from __future__ import annotations

import re

DIGITS = "영일이삼사오육칠팔구"
SMALL = ("", "십", "백", "천")
LARGE = ("", "만", "억", "조", "경")

NATIVE_ONES = ("", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉")
NATIVE_TENS = ("", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔")

# 고유어로 세는 말. "20살"은 "스무 살", "3시"는 "세 시"다. 분·초·년·월·일·원은 한자어로 읽는다
COUNTERS = (
    "명",
    "개(?!월)",
    "살",
    "시간",
    "시",
    "마리",
    "번째",
    "번",
    "잔",
    "병",
    "권",
    "장",
    "곳",
    "가지",
    "사람",
    "달",
    "그릇",
    "켤레",
    "벌",
)

UNITS = {
    "%": " 퍼센트",
    "km": " 킬로미터",
    "kg": " 킬로그램",
    "cm": " 센티미터",
    "mm": " 밀리미터",
    "m": " 미터",
    "g": " 그램",
    "℃": " 도",
    "°C": " 도",
    "$": " 달러",
}

NUMBER = re.compile(
    r"(?P<dollar>\$)?(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<frac>\d+))?"
    r"(?P<unit>%|km|kg|cm|mm|°C|℃|m(?![a-zA-Z])|g(?![a-zA-Z]))?"
    r"(?P<space>\s?)(?P<counter>" + "|".join(COUNTERS) + r")?"
)
MONTHS = ((re.compile(r"(?<![\d.,])6월"), "유월"), (re.compile(r"(?<![\d.,])10월"), "시월"))


def sino(n: int) -> str:
    """한자어 읽기. 0은 영, 10은 십(일십이 아니다), 10000은 만."""
    if n == 0:
        return "영"
    groups: list[str] = []
    k = 0
    while n > 0:
        n, part = divmod(n, 10000)
        if part:
            text = _under_10000(part)
            if k > 0 and part == 1:
                text = ""
            groups.append(text + LARGE[k] if k < len(LARGE) else text)
        k += 1
    return "".join(reversed(groups))


def _under_10000(n: int) -> str:
    out = []
    for i in range(3, -1, -1):
        d = (n // 10**i) % 10
        if d == 0:
            continue
        out.append(("" if d == 1 and i > 0 else DIGITS[d]) + SMALL[i])
    return "".join(out)


def native(n: int) -> str:
    """고유어 읽기(1~99). 세는 말 앞이라 "하나"가 아니라 "한", "스물"이 아니라 "스무"."""
    tens, ones = divmod(n, 10)
    if ones == 0 and tens == 2:
        return "스무"
    return NATIVE_TENS[tens] + NATIVE_ONES[ones]


def _read(m: re.Match[str]) -> str:
    digits = m.group("int").replace(",", "")
    frac = m.group("frac")
    unit = m.group("unit")
    counter = m.group("counter")
    value = int(digits)
    if len(digits) > 1 and digits.startswith("0"):
        # 전화번호·코드처럼 0으로 시작하면 한 자리씩 읽는다
        text = "".join(DIGITS[int(d)] if d != "0" else "공" for d in digits)
    elif counter and not frac and not unit and 0 < value < 100:
        text = native(value)
    else:
        text = sino(value)
    if frac:
        text += " 점 " + "".join(DIGITS[int(d)] for d in frac)
    if unit:
        text += UNITS[unit]
    if m.group("dollar"):
        text += UNITS["$"]
    if counter:
        text += " " + counter
    else:
        text += m.group("space")
    return text


def normalize(text: str) -> str:
    """대사 속 숫자를 한글 읽기로 바꾼다. 숫자가 없으면 그대로."""
    for pattern, word in MONTHS:
        text = pattern.sub(word, text)
    return NUMBER.sub(_read, text)
