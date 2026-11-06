"""긴 SDXL 프롬프트 조각 나누기 (FR-006: 장면 문장이 잘리지 않게)."""

from model_runtime.backends.long_prompt import CHUNK, frame, split_ids

COMMA = 267


def test_FR006_75토큰_이하는_한_조각이다() -> None:
    ids = list(range(1000, 1075))
    assert split_ids(ids, COMMA) == [ids]


def test_FR006_75토큰을_넘으면_쉼표_뒤에서_자른다() -> None:
    ids = [*range(1000, 1070), COMMA, *range(2000, 2020)]
    parts = split_ids(ids, COMMA)
    assert parts[0][-1] == COMMA and len(parts[0]) == 71
    assert parts[1] == list(range(2000, 2020))
    assert [t for p in parts for t in p] == ids


def test_FR006_쉼표가_멀면_75토큰에서_자르고_모든_조각이_75토큰_이하다() -> None:
    ids = [COMMA, *range(1000, 1200)]
    parts = split_ids(ids, COMMA)
    assert [len(p) for p in parts] == [75, 75, 51]
    assert all(len(p) <= CHUNK for p in parts)
    assert [t for p in parts for t in p] == ids


def test_FR006_조각은_시작_끝_토큰을_붙여_77토큰으로_채운다() -> None:
    framed = frame([5, 6], bos=1, eos=2, pad=0)
    assert len(framed) == CHUNK + 2
    assert framed[:4] == [1, 5, 6, 2] and set(framed[4:]) == {0}
