"""컷의 인물 수·자리·키를 뼈대 그림(OpenPose)으로 그린다.

회차 16~18에서 그린 뒤 확인해 다시 그려도 두 사람 컷에서 한 사람이 빠지거나, 컷 안에 4칸 격자가
생기거나, 아이와 어른의 키가 뒤바뀌는 그림이 남았다. 시드만 바꾼 후보 셋이 모두 틀리면 고를 수 없다.
그래서 그리기 전에 컷마다 인물 수만큼 서 있는 뼈대를 자리(왼쪽·오른쪽·가운데)와 키(아이는 작게)에
맞춰 그려 T2I-Adapter로 넣는다. 자리와 나이는 인물 확인과 같은 images[].checks에서 읽는다.

이 파일은 GPU 없이 확인할 수 있게 PIL로만 그린다.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageDraw

from .check import Check

# 서 있는 사람 하나의 18점(OpenPose COCO 순서).
# 가로 = 몸 키 대비 가운데에서 떨어진 정도, 세로 = 0 머리 끝 ~ 1 발
# 그림의 왼쪽이 인물의 오른쪽이다
BODY = (
    (0.0, 0.08),  # 0 코
    (0.0, 0.17),  # 1 목
    (-0.11, 0.18),  # 2 오른 어깨
    (-0.14, 0.33),  # 3 오른 팔꿈치
    (-0.13, 0.46),  # 4 오른 손목
    (0.11, 0.18),  # 5 왼 어깨
    (0.14, 0.33),  # 6 왼 팔꿈치
    (0.13, 0.46),  # 7 왼 손목
    (-0.06, 0.50),  # 8 오른 엉덩이
    (-0.07, 0.72),  # 9 오른 무릎
    (-0.07, 0.94),  # 10 오른 발목
    (0.06, 0.50),  # 11 왼 엉덩이
    (0.07, 0.72),  # 12 왼 무릎
    (0.07, 0.94),  # 13 왼 발목
    (-0.025, 0.065),  # 14 오른 눈
    (0.025, 0.065),  # 15 왼 눈
    (-0.05, 0.075),  # 16 오른 귀
    (0.05, 0.075),  # 17 왼 귀
)
# OpenPose 뼈대 연결과 색 (controlnet_aux와 같은 값)
LIMBS = (
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
)  # fmt: skip
COLORS = (
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
    (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
)  # fmt: skip

# 자리별 몸 가운데 (그림 폭 대비)
CENTERS = {"left": 0.28, "right": 0.72, "center": 0.5, "full": 0.5}
# 어른 몸 키(그림 높이 대비)와 머리 끝 높이. 무릎 아래는 그림 밖으로 나가 얼굴이 작아지지 않게 한다
ADULT = 1.2
TOP = 0.1
# 아이 몸 키 비율. 발은 어른과 같은 바닥에 둔다
CHILD = 0.7


@dataclass(frozen=True)
class Figure:
    """뼈대 하나: 몸 가운데(폭 대비), 머리 끝(높이 대비), 몸 키(높이 대비)."""

    x: float
    top: float
    height: float


def figures(checks: list[Check]) -> list[Figure]:
    """확인할 인물 → 뼈대. 동물이 있거나, 인물이 없거나 넷 이상이거나, 자리가 겹치면 빈 목록."""
    if not checks or len(checks) > 3 or any(c.animal for c in checks):
        return []
    regions = [c.region for c in checks]
    if len(set(regions)) != len(regions) or (len(checks) > 1 and "full" in regions):
        return []
    ground = TOP + ADULT * BODY[10][1]
    out = []
    for c in checks:
        height = ADULT * (CHILD if c.age == "CHILD" else 1.0)
        out.append(Figure(CENTERS[c.region], ground - height * BODY[10][1], height))
    return out


def render(checks: list[Check], width: int, height: int) -> Image.Image | None:
    """뼈대 그림(검은 바탕). 그릴 수 없으면 None."""
    figs = figures(checks)
    if not figs:
        return None
    img = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    stick = max(2, round(height / 128))
    for f in figs:
        points = [
            ((f.x + dx * f.height * height / width) * width, (f.top + dy * f.height) * height)
            for dx, dy in BODY
        ]
        for i, (a, b) in enumerate(LIMBS):
            color = tuple(round(v * 0.6) for v in COLORS[i])
            draw.line([points[a], points[b]], fill=color, width=stick * 2)
        for i, (x, y) in enumerate(points):
            draw.ellipse([x - stick, y - stick, x + stick, y + stick], fill=COLORS[i])
    return img
