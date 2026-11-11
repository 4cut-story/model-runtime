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


def match_tone(base: bytes, frame: bytes) -> bytes:
    """움직임 그림의 밝기·대비를 첫 그림에 맞춘다 (FR-049).

    광원이 있는 장면(벽난로·등불)에서 edit가 화면 전체를 밝히거나 색을 바래게 그리면, 번갈아 볼 때
    그림이 깜빡이고 어긋남 검사에서 빠진다. 채널마다 평균·표준편차를 첫 그림에 맞춰 전체 톤만
    되돌리고, 빗줄기·불꽃 같은 국소 변화는 남긴다.
    """
    from PIL import Image, ImageStat

    first = Image.open(io.BytesIO(base)).convert("RGB")
    moved = Image.open(io.BytesIO(frame)).convert("RGB")
    if moved.size != first.size:
        moved = moved.resize(first.size)
    a, b = ImageStat.Stat(first), ImageStat.Stat(moved)
    bands = []
    for band, mean_a, std_a, mean_b, std_b in zip(
        moved.split(), a.mean, a.stddev, b.mean, b.stddev, strict=True
    ):
        scale = min(2.0, max(0.5, std_a / std_b)) if std_b > 0 else 1.0
        bands.append(
            band.point(
                lambda v, s=scale, ma=mean_a, mb=mean_b: max(0, min(255, round((v - mb) * s + ma)))
            )
        )
    out = io.BytesIO()
    Image.merge("RGB", bands).save(out, format="PNG")
    return out.getvalue()


#: 넓은 그림이 컷 그림보다 이만큼 어두우면 밝힌다(평균 밝기 비).
LIFT_BELOW = 0.8

#: 밝힐 때 쓰는 감마의 하한. 더 내리면 색이 바래 밤 분위기가 사라진다.
LIFT_MIN_GAMMA = 0.55


def lift_dark(base: bytes, wide: bytes) -> bytes:
    """넓은 그림이 컷 그림보다 많이 어두우면 감마로 밝힌다 (docs specs/003-live-mode FR-030).

    밤 실내 컷에서 넓은 그림이 컷 그림(평균 밝기 약 150)과 달리 거의 검푸른 방(약 35)으로
    그려져 라이브 모드가 너무 어두웠다. 평균 밝기가 컷 그림의 LIFT_BELOW배보다 낮으면
    그 밝기에 닿도록 감마를 낮추되 LIFT_MIN_GAMMA 아래로는 내리지 않는다(밤 분위기는 남긴다).
    밝기만 올리고 색 비율과 그림 크기는 그대로다. 어둡지 않으면 받은 그림을 그대로 돌려준다.
    """
    from PIL import Image, ImageStat

    first = Image.open(io.BytesIO(base)).convert("L")
    image = Image.open(io.BytesIO(wide)).convert("RGB")
    target = ImageStat.Stat(first).mean[0] * LIFT_BELOW
    gray = image.convert("L")
    if ImageStat.Stat(gray).mean[0] >= target:
        return wide

    def lut(gamma: float) -> list[int]:
        return [round(255 * ((v / 255) ** gamma)) for v in range(256)]

    lo, hi = LIFT_MIN_GAMMA, 1.0
    if ImageStat.Stat(gray.point(lut(lo))).mean[0] <= target:
        gamma = lo
    else:
        for _ in range(16):
            mid = (lo + hi) / 2
            if ImageStat.Stat(gray.point(lut(mid))).mean[0] < target:
                hi = mid
            else:
                lo = mid
        gamma = lo
    out = io.BytesIO()
    image.point(lut(gamma) * 3).save(out, format="PNG")
    return out.getvalue()
