"""라이브 컷 깊이 지도 (docs specs/003-live-mode FR-048).

넓은 그림 한 장에서 Depth Anything V2(작은 모델, 약 25M 파라미터)로 깊이를 추정해
같은 크기의 흑백 PNG(밝을수록 가깝다)로 돌려준다. 화면이 이것으로 2.5D 패럴랙스를 그린다.
처음 쓸 때 Hugging Face에서 모델(약 100MB)을 받고, 한 번 올린 뒤에는 VRAM에 둔다.
"""

from __future__ import annotations

import io
import os
from typing import Any

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("DEPTH_MODEL", DEFAULT_MODEL)
        self._pipe: Any = None

    def _pipeline(self) -> Any:
        if self._pipe is None:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1
            self._pipe = pipeline("depth-estimation", model=self.model, device=device)
        return self._pipe

    def estimate(self, png: bytes) -> bytes:
        from PIL import Image

        image = Image.open(io.BytesIO(png)).convert("RGB")
        result = self._pipeline()(image)
        depth = result["depth"].convert("L").resize(image.size)
        return normalized_png(depth)

    def unload(self) -> None:
        self._pipe = None


def normalized_png(depth: Any) -> bytes:
    """흑백 그림을 가장 먼 곳 0, 가장 가까운 곳 255로 늘려 PNG로 만든다."""
    lo, hi = depth.getextrema()
    if hi > lo:
        scale = 255.0 / (hi - lo)
        depth = depth.point(lambda v: round((v - lo) * scale))
    out = io.BytesIO()
    depth.save(out, format="PNG")
    return out.getvalue()


# 깊이 지도에서 이만큼(0~255) 밝으면 앞쪽(인물)으로 보고 원래 그림을 그대로 둔다.
# 사이 값은 부드럽게 섞는다.
NEAR_FROM = 150
NEAR_TO = 190
FEATHER = 4


def keep_near(base: bytes, frame: bytes, depth: bytes) -> bytes:
    """움직임 그림에서 앞쪽(인물)은 첫 그림 그대로 두고 뒤쪽(배경)만 새 그림을 쓴다.

    배경만 바꿔 다시 그려도 얼굴·옷이 조금씩 달라져 번갈아 보면 인물이 흔들린다. 깊이 지도(밝을수록
    가깝다)로 가까운 곳을 가려 첫 그림을 덮는다 (docs specs/003-live-mode FR-049).
    """
    from PIL import Image, ImageFilter

    first = Image.open(io.BytesIO(base)).convert("RGB")
    moved = Image.open(io.BytesIO(frame)).convert("RGB")
    if moved.size != first.size:
        moved = moved.resize(first.size)
    span = max(1, NEAR_TO - NEAR_FROM)
    mask = (
        Image.open(io.BytesIO(depth))
        .convert("L")
        .resize(first.size)
        .point(lambda v: max(0, min(255, round((v - NEAR_FROM) * 255 / span))))
        .filter(ImageFilter.GaussianBlur(FEATHER))
    )
    out = io.BytesIO()
    Image.composite(first, moved, mask).save(out, format="PNG")
    return out.getvalue()
