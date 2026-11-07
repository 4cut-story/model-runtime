"""인물별 자리 프롬프트 (두 명 이상 나오는 컷).

프롬프트 한 줄에 두 사람 외모를 함께 적으면 SDXL이 둘을 섞어서, 여자가 옆 남자처럼 그려지고
참조 그림이 없는 인물은 빠지고 옷 색이 옆 사람에게 번졌다(회차 12·13 love-07, family-09, funny-12).
그래서 cross-attention마다 전체 프롬프트 결과에 더해 자리(왼쪽·오른쪽·가운데)별 인물 프롬프트 결과를
구하고, 그 자리 안에서만 섞는다.

회차 14에서 처음 걸었을 때는 두 사람이 한 사람으로 섞였다. 자리 프롬프트마다 "solo"를 넣고, 0.8로 첫
단계부터 걸어서 구도가 잡히기 전에 두 자리가 저마다 화면 가운데 한 사람을 그리려 했기 때문으로 본다.
그래서 구도는 전체 프롬프트가 먼저 정하게 앞 단계(START_STEP 전)에는 걸지 않고, 비율도 낮췄다.
"solo"는 conversation-service가 넣지 않는다.

이 파일은 GPU 없이 확인할 수 있는 계산만 둔다. 어텐션에 끼우는 부분은 sdxl.py에 있다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# 자리 안에서 인물 프롬프트 결과를 섞는 비율. 1이면 자리 안은 인물 프롬프트만 본다.
WEIGHT = 0.5

# 이 단계부터 섞는다(0부터 셈). 8 step Lightning에서 앞 두 단계가 구도를 정한다.
START_STEP = 2

# 자리 상자 (가로 시작, 가로 끝) 비율. 세로는 전체다. IP-Adapter 마스크와 같은 자리다.
BOXES = {
    "left": (0.0, 0.5),
    "right": (0.5, 1.0),
    "center": (0.2, 0.8),
}


@dataclass(frozen=True)
class RegionPrompt:
    """자리 하나와 그 자리 인물의 프롬프트."""

    region: str
    prompt: str


def parse(raw: Any) -> list[RegionPrompt]:
    """작업 입력 images[].regions → 쓸 수 있는 자리 프롬프트. 두 개 미만이면 쓰지 않는다."""
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        region = item.get("region")
        prompt = item.get("prompt")
        if region in BOXES and isinstance(prompt, str) and prompt.strip():
            out.append(RegionPrompt(region, prompt.strip()))
    return out if len(out) >= 2 else []


def grid(seq: int, width: int, height: int) -> tuple[int, int] | None:
    """어텐션 토큰 수 → 잠재 공간 (세로, 가로).

    UNet 단계마다 크기가 절반씩 준다. 맞지 않으면 None.
    """
    h0, w0 = height // 8, width // 8
    if seq <= 0 or h0 <= 0 or w0 <= 0:
        return None
    factor = math.sqrt(h0 * w0 / seq)
    h = round(h0 / factor)
    w = round(w0 / factor)
    if h * w != seq:
        return None
    return h, w


def columns(region: str, w: int) -> tuple[int, int]:
    """자리 → 가로 열 범위 [시작, 끝)."""
    start, end = BOXES[region]
    return round(start * w), round(end * w)


def active(step: int) -> bool:
    """이 단계에서 자리 프롬프트를 섞는가."""
    return step >= START_STEP
