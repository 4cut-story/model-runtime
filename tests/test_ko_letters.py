import pytest

from model_runtime.backends.ko_letters import spell
from model_runtime.backends.ko_numbers import normalize


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("AI가 다 해", "에이아이가 다 해"),
        ("GPU 값이 올랐대", "지피유 값이 올랐대"),
        ("CEO랑 CFO가 왔어", "씨이오랑 씨에프오가 왔어"),
        ("A씨가 그랬어", "에이씨가 그랬어"),
        ("NASA 발표 봤어?", "나사 발표 봤어?"),
        ("KOSPI가 떨어졌어", "코스피가 떨어졌어"),
        ("ChatGPT한테 물어봐", "챗지피티한테 물어봐"),
        ("YouTube에 올렸어", "유튜브에 올렸어"),
        ("OECD 평균", "오이시디 평균"),
        ("Apple은 그대로", "Apple은 그대로"),
        ("CEOs는 그대로", "CEOs는 그대로"),
        ("영어 없음", "영어 없음"),
    ],
)
def test_대사의_영어_약어를_한글_읽기로_바꾼다(text: str, spoken: str) -> None:
    assert spell(text) == spoken


def test_약어를_읽은_뒤_숫자도_읽는다() -> None:
    assert normalize(spell("5G 요금이 3만 원")) == "오지 요금이 삼만 원"
    assert normalize(spell("3D 안경")) == "삼디 안경"
