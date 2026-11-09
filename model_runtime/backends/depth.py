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
