from model_runtime.backends.check import Check
from model_runtime.backends.pose import ADULT, CHILD, figures, render


def test_FR006_두_사람_컷은_자리마다_뼈대를_하나씩_두고_아이는_작게_그린다() -> None:
    figs = figures([Check("left", "M", "OLD"), Check("right", "F", "CHILD")])
    assert [f.x for f in figs] == [0.28, 0.72]
    assert figs[0].height == ADULT
    assert figs[1].height == ADULT * CHILD
    assert figs[1].top > figs[0].top
    # 발은 같은 바닥에 둔다
    assert round(figs[0].top + figs[0].height * 0.94, 6) == round(
        figs[1].top + figs[1].height * 0.94, 6
    )


def test_FR006_동물이나_자리가_겹치거나_인물이_없으면_뼈대를_그리지_않는다() -> None:
    assert figures([]) == []
    assert figures([Check("left", animal="bear"), Check("right", "F")]) == []
    assert figures([Check("left", "F"), Check("left", "M")]) == []
    assert figures([Check("full", "F"), Check("left", "M")]) == []
    assert render([], 64, 64) is None


def test_FR006_뼈대_그림은_검은_바탕에_자리에_맞춰_그린다() -> None:
    img = render([Check("left", "F", "ADULT"), Check("right", "M", "ADULT")], 256, 256)
    assert img is not None and img.size == (256, 256)
    assert img.getpixel((5, 5)) == (0, 0, 0)
    left = img.crop((0, 0, 128, 256)).getbbox()
    right = img.crop((128, 0, 256, 256)).getbbox()
    assert left is not None and right is not None
    solo = render([Check("center", "F", "ADULT")], 256, 256)
    assert solo is not None
    box = solo.getbbox()
    assert box is not None and box[0] > 40 and box[2] < 216
