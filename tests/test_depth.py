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


def _png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def test_FR049_가까운_곳은_첫_그림을_먼_곳은_새_그림을_쓴다() -> None:
    from model_runtime.backends.depth import keep_near

    base = Image.new("RGB", (60, 4), (255, 0, 0))
    frame = Image.new("RGB", (60, 4), (0, 0, 255))
    depth = Image.new("L", (60, 4), 0)
    depth.paste(255, (0, 0, 30, 4))
    out = Image.open(io.BytesIO(keep_near(_png(base), _png(frame), _png(depth))))
    assert out.getpixel((2, 2)) == (255, 0, 0)
    assert out.getpixel((57, 2)) == (0, 0, 255)


def test_FR049_크기가_다른_새_그림은_첫_그림_크기로_맞춘다() -> None:
    from model_runtime.backends.depth import keep_near

    base = Image.new("RGB", (8, 4), (255, 0, 0))
    frame = Image.new("RGB", (16, 8), (0, 0, 255))
    depth = Image.new("L", (4, 2), 0)
    out = Image.open(io.BytesIO(keep_near(_png(base), _png(frame), _png(depth))))
    assert out.size == (8, 4)
    assert out.getpixel((4, 2)) == (0, 0, 255)
