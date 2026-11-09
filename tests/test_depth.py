import io

from PIL import Image

from model_runtime.backends.depth import normalized_png


def test_FR048_깊이_지도는_가장_먼_곳_0_가장_가까운_곳_255로_늘린다() -> None:
    depth = Image.new("L", (4, 1))
    depth.putdata([50, 60, 70, 100])
    out = Image.open(io.BytesIO(normalized_png(depth)))
    assert out.size == (4, 1)
    assert list(out.getdata()) == [0, 51, 102, 255]


def test_FR048_평평한_깊이_지도는_그대로_둔다() -> None:
    depth = Image.new("L", (2, 2), 80)
    out = Image.open(io.BytesIO(normalized_png(depth)))
    assert set(out.getdata()) == {80}
