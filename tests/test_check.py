"""그린 뒤 인물 확인 (FR-006: 아이·어른, 성별, 동물 종이 콘티와 같게)."""

from model_runtime.backends.check import (
    Check,
    Question,
    better,
    box,
    parse,
    questions,
    score,
)


def test_FR006_확인_항목을_읽고_모르는_자리와_빈_항목은_뺀다() -> None:
    raw = [
        {"region": "left", "gender": "F", "age": "CHILD"},
        {"region": "right", "animal": "rabbit"},
        {"region": "top", "gender": "M"},
        {"region": "center"},
        {"region": "full", "gender": "X", "age": "OLD"},
        "bad",
    ]
    assert parse(raw) == [
        Check("left", "F", "CHILD"),
        Check("right", animal="rabbit"),
        Check("full", None, "OLD"),
    ]
    assert parse(None) == []


def test_FR006_사람은_나이대와_성별을_동물은_종을_묻는다() -> None:
    qs = questions([Check("left", "F", "CHILD"), Check("right", "M", "OLD")])
    assert qs[0] == Question("left", ("a cartoon person", "an empty background with nobody"), 0)
    assert qs[1] == Question(
        "left", ("a cartoon child", "a cartoon adult", "a cartoon elderly person"), 0
    )
    assert qs[2] == Question("left", ("a cartoon girl", "a cartoon boy"), 0)
    assert qs[4].answer == 2
    assert qs[5] == Question("right", ("a cartoon woman", "a cartoon man"), 1)
    animals = questions([Check("left", animal="bear"), Check("right", animal="rabbit")])
    assert animals[0].labels == ("a cartoon bear", "an empty background with nobody")
    assert animals[1].labels[:2] == ("a cartoon bear", "a cartoon rabbit")
    assert animals[1].labels[-1] == "a cartoon human person"
    assert animals[3].labels[:2] == ("a cartoon rabbit", "a cartoon bear")
    assert len(set(animals[3].labels)) == len(animals[3].labels)
    solo = questions([Check("center", "F", "OLD")])
    assert [q.labels[0] for q in solo] == ["a cartoon child", "a cartoon woman"]


def test_FR006_자리별로_잘라_본다() -> None:
    assert box("left", 768, 768) == (0, 0, 384, 768)
    assert box("right", 768, 768) == (384, 0, 768, 768)
    assert box("full", 768, 768) == (0, 0, 768, 768)


def test_FR006_틀린_수가_적고_확률이_높은_그림을_고른다() -> None:
    qs = [Question("left", ("a", "b"), 0), Question("left", ("a", "b", "c"), 2)]
    assert score([[0.9, 0.1], [0.2, 0.3, 0.5]], qs) == (0, 0.7)
    assert score([[0.4, 0.6], [0.5, 0.1, 0.4]], qs)[0] == 2
    assert score([], []) == (0, 1.0)
    assert better((0, 0.5), (1, 0.9))
    assert better((1, 0.8), (1, 0.7))
    assert not better((1, 0.7), (1, 0.7))
