"""대사 다듬기: 날짜·기호가 붙은 숫자와 채팅 말투를 읽을 수 있는 말로 (대사 읽기 전처리)."""

import pytest

from model_runtime.backends.ko_letters import spell
from model_runtime.backends.ko_numbers import normalize
from model_runtime.backends.ko_text import tidy


@pytest.mark.parametrize(
    ("text", "tidied"),
    [
        ("2026.11.12 기준", "2026년 11월 12일 기준"),
        ("10:12로 졌어", "10 대 12로 졌어"),
        ("10:30에 보자", "10:30에 보자"),
        ("₩5,000이야", "5,000원이야"),
        ("3.5%p 올랐다", "3.5%포인트 올랐다"),
        ("-3도까지 내려간대", "영하 3도까지 내려간대"),
        ("점수는 -3이야", "점수는 마이너스 3이야"),
        ("ㅋㅋㅋ 진짜?", "크크크 진짜?"),
        ("ㅎㅎ 고마워 ㅠㅠ", "흐흐 고마워"),
        ("그게... 아니야", "그게, 아니야"),
        ("몰라…", "몰라."),
        ("숫자가 없는 대사", "숫자가 없는 대사"),
    ],
)
def test_날짜와_기호가_붙은_숫자와_채팅_말투를_말로_풀어_쓴다(text: str, tidied: str) -> None:
    assert tidy(text) == tidied


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("2026.11.12 기준", "이천이십육년 십일월 십이일 기준"),
        ("10:12로 졌어", "십 대 십이로 졌어"),
        ("-3도까지 내려간대", "영하 삼도까지 내려간대"),
        ("₩5,000이야", "오천원이야"),
        ("회의는 10:30에 시작해.", "회의는 열 시 삼십 분에 시작해."),
    ],
)
def test_다듬은_뒤_숫자와_약어까지_한글로_읽는다(text: str, spoken: str) -> None:
    assert normalize(spell(tidy(text))) == spoken
