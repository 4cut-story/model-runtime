"""guidance > 1(CFG)이면 IP-Adapter embedding을 [negative; positive] 두 벌로 넘긴다 (회차 21)."""

import pytest

torch = pytest.importorskip("torch")

from model_runtime.backends.sdxl import EMBED_SHAPE, SdxlBackend  # noqa: E402


def backend(guidance: float) -> SdxlBackend:
    return SdxlBackend("unused", guidance=guidance)


def test_guidance가_1_이하면_embedding을_그대로_넘긴다() -> None:
    e = torch.ones(EMBED_SHAPE)
    assert backend(0.0)._with_negative(e) is e


def test_guidance가_1보다_크면_앞에_0_embedding을_붙인다() -> None:
    e = torch.ones(EMBED_SHAPE)
    out = backend(2.0)._with_negative(e)
    assert tuple(out.shape) == (2, *EMBED_SHAPE[1:])
    negative, positive = out.chunk(2)
    assert torch.count_nonzero(negative) == 0
    assert torch.equal(positive, e)
