"""인물별 자리 프롬프트 (FR-006: 두 명 컷에서 인물이 섞이지 않게)."""

from model_runtime.backends.regional import RegionPrompt, columns, grid, parse


def test_FR006_자리_프롬프트는_두_개_이상일_때만_쓴다() -> None:
    raw = [
        {"region": "left", "prompt": "young woman, yellow raincoat"},
        {"region": "right", "prompt": " young man, navy coat "},
        {"region": "top", "prompt": "x"},
        {"region": "center", "prompt": ""},
        "bad",
    ]
    assert parse(raw) == [
        RegionPrompt("left", "young woman, yellow raincoat"),
        RegionPrompt("right", "young man, navy coat"),
    ]
    assert parse(raw[:1]) == []
    assert parse(None) == []


def test_FR006_토큰_수로_잠재_공간_크기를_구한다() -> None:
    assert grid(96 * 96, 768, 768) == (96, 96)
    assert grid(48 * 48, 768, 768) == (48, 48)
    assert grid(24 * 24, 768, 768) == (24, 24)
    assert grid(64 * 48, 384 * 2, 512 * 2) == (64, 48)
    assert grid(77, 768, 768) is None
    assert grid(0, 768, 768) is None


def test_FR006_자리별_가로_열() -> None:
    assert columns("left", 96) == (0, 48)
    assert columns("right", 96) == (48, 96)
    assert columns("center", 24) == (5, 19)
