"""녹음 중 선행 전사(FR-015)의 구간 경계 처리."""

from typing import Any

from model_runtime import segments


def seg(*words: tuple[float, float, str]) -> dict[str, Any]:
    ws = [{"word": w, "start": s, "end": e} for s, e, w in words]
    return {
        "start": ws[0]["start"],
        "end": ws[-1]["end"],
        "text": " ".join(w for *_, w in words),
        "words": ws,
    }


def test_구간_시각에_오프셋을_더하고_정렬_못한_단어는_앞_단어_시각을_쓴다() -> None:
    raw = [
        {
            "start": 1.0,
            "end": 3.0,
            "text": "가 나",
            "words": [{"word": "가", "start": 1.0, "end": 1.5}, {"word": "나"}],
        }
    ]
    out = segments.shift(raw, 100.0)
    assert out[0]["start"] == 101.0
    assert out[0]["words"][1]["start"] == 101.0
    assert out[0]["words"][1]["end"] == 101.0


def test_경계는_끝_5초에서_15초_전_사이_가장_큰_빈틈의_가운데() -> None:
    s = [seg((80.0, 80.5, "가"), (85.0, 85.5, "나"), (88.0, 88.4, "다"), (89.0, 89.5, "라"))]
    # 끝 100초 → 범위 [85, 95]. 85.5~88.0 빈틈(2.5초)이 가장 크다
    assert segments.choose_cut(s, 100.0) == (85.5 + 88.0) / 2


def test_범위에_단어가_모자라면_범위_가운데에서_자른다() -> None:
    assert segments.choose_cut([seg((10.0, 10.5, "가"))], 100.0) == 90.0


def test_경계_사이에서_시작한_단어만_남기고_잘린_세그먼트는_글을_다시_만든다() -> None:
    whole = seg((10.0, 10.5, "가"), (11.0, 11.5, "나"))
    cut = seg((19.0, 19.5, "다"), (21.0, 21.5, "라"))
    out = segments.keep_between([whole, cut], 10.0, 20.0)
    assert out[0] is whole
    assert out[1]["text"] == "다"
    assert out[1]["end"] == 19.5
    assert segments.keep_between([whole], 12.0, 20.0) == []


def test_단어가_없는_세그먼트는_시작_시각으로_고른다() -> None:
    bare = {"start": 5.0, "end": 6.0, "text": "음", "words": []}
    assert segments.keep_between([bare], 0.0, 10.0) == [bare]
    assert segments.keep_between([bare], 6.0, 10.0) == []


def test_출력_형식은_화자_없으면_SPEAKER_00() -> None:
    out = segments.to_output([{"start": 1, "end": 2, "text": " 가 "}])
    assert out == [{"speaker": "SPEAKER_00", "start": 1.0, "end": 2.0, "text": "가"}]
