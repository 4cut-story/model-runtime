import pytest

from model_runtime.backends.ko_numbers import native, normalize, sino


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("260조 원이래", "이백육십조 원이래"),
        ("1조 5000억", "일조 오천억"),
        ("1,234,000원", "백이십삼만사천원"),
        ("10만 원", "십만 원"),
        ("2.5% 올랐어", "이 점 오 퍼센트 올랐어"),
        ("3명이 왔어", "세 명이 왔어"),
        ("20살이야", "스무 살이야"),
        ("오후 3시 15분", "오후 세 시 십오분"),
        ("11번째", "열한 번째"),
        ("2개월 뒤", "이개월 뒤"),
        ("20대 청년", "이십대 청년"),
        ("2026년 6월 10일", "이천이십육년 유월 십일"),
        ("10월", "시월"),
        ("$5", "오 달러"),
        ("010", "공일공"),
        ("숫자 없음", "숫자 없음"),
    ],
)
def test_대사의_숫자를_한글_읽기로_바꾼다(text: str, spoken: str) -> None:
    assert normalize(text) == spoken


def test_한자어와_고유어_읽기() -> None:
    assert sino(0) == "영"
    assert sino(10) == "십"
    assert sino(10000) == "만"
    assert sino(100_000_000) == "억"
    assert sino(21_0000_0005) == "이십일억오"
    assert native(1) == "한"
    assert native(99) == "아흔아홉"
