"""대사를 읽기 좋게 다듬는다. 숫자·약어 바꾸기(ko_numbers, ko_letters) 앞에 한 번 거친다.

ko_numbers가 시각·점수·범위·분수·전화번호를 읽고, 여기서는 그 밖에 남는 것을 말로 먼저 풀어 둔다.
- 날짜 2026.11.12 → 2026년 11월 12일
- 승부가 붙은 두 자리 점수 10:12로 졌어 → 10 대 12 (시각으로 읽지 않게)
- ₩5,000 → 5,000원, 3.5%p → 3.5%포인트
- 영하 -3도 → 영하 3도, 그 밖의 -3 → 마이너스 3
- ㅋㅋ → 크크, ㅎㅎ → 흐흐, ㅠㅠ는 지움, 말줄임표 ... → 쉼표(줄 끝이면 마침표)
"""

from __future__ import annotations

import re

_DATE = re.compile(r"(?<![\d.])((?:19|20)\d{2})[./-](\d{1,2})[./-](\d{1,2})\.?(?![\d.])")
_RESULT = r"(?=[^\d]{0,8}?(?:승|패|무승부|이겼|졌|비겼|역전|앞서|뒤지))"
_SCORE = re.compile(r"(?<![\d:])(\d{1,2})\s*:\s*(\d{1,2})(?![\d:])" + _RESULT)
_WON = re.compile(r"₩\s*(\d[\d,]*)")
_BELOW_ZERO = re.compile(r"(?<![\w)])-\s*(?=\d[\d.]*\s*(?:도|℃|°))")
_MINUS = re.compile(r"(?<![\w)])-(?=\d)")


def tidy(text: str) -> str:
    """날짜·기호가 붙은 숫자와 채팅 말투를 읽을 수 있는 말로 바꾼 대사."""
    t = _DATE.sub(lambda m: f"{m.group(1)}년 {int(m.group(2))}월 {int(m.group(3))}일", text)
    t = _SCORE.sub(lambda m: f"{m.group(1)} 대 {m.group(2)}", t)
    t = _WON.sub(lambda m: f"{m.group(1)}원", t)
    t = t.replace("%p", "%포인트")
    t = _BELOW_ZERO.sub("영하 ", t)
    t = _MINUS.sub("마이너스 ", t)
    t = re.sub(r"ㅋ{2,}", lambda m: "크" * min(len(m.group(0)), 4), t)
    t = re.sub(r"ㅎ{2,}", lambda m: "흐" * min(len(m.group(0)), 3), t)
    t = re.sub(r"[ㅠㅜ]{2,}", "", t)
    t = re.sub(r"(?:\.{2,}|…+)(?=\s*\S)", ", ", t)
    t = re.sub(r"(?:\.{2,}|…+)\s*$", ".", t)
    t = re.sub(r",\s*([,.?!])", r"\1", t)
    return re.sub(r"[ \t]{2,}", " ", t).strip()
