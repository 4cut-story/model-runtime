"""그린 뒤 인물 확인 (성별·나이·동물).

회차 14·15의 심각 오류 대부분은 아이가 어른으로, 여자가 남자로, 토끼가 다른 동물로 그려진 것이었다.
conversation-service가 컷마다 인물의 자리와 성별·나이·동물(images[].checks)을 보내면, 워커는 그린
그림의 그 자리를 잘라 CLIP으로 맞는 말을 고르게 하고, 틀리면 시드를 바꿔 다시 그린다.

이 파일은 GPU 없이 확인할 수 있는 부분(어떤 말들 가운데 고를지, 맞았는지)만 둔다.
CLIP을 부르는 부분은 sdxl.py에 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .regional import BOXES

# 다시 그리는 최대 횟수 (처음 그림 빼고)
RETRIES = 2
# 다시 그릴 때 시드에 더하는 값. 콘티의 컷 시드 간격(10)과 겹치지 않게 크게 둔다
SEED_STEP = 1_000_003
# 기대한 말의 확률이 이보다 낮으면 틀린 것으로 본다
THRESHOLD = 0.5
# 인물 수 질문이 틀리면 이만큼 틀린 것으로 센다. 사람이 빠진 그림은 성별·나이가 맞아도 쓸 수 없다
COUNT_WEIGHT = 3
NUMBERS = {2: "two", 3: "three", 4: "four"}

ANIMALS = (
    "bear",
    "rabbit",
    "cat",
    "dog",
    "fox",
    "panda",
    "pig",
    "tiger",
    "lion",
    "mouse",
    "duck",
    "penguin",
    "hamster",
    "squirrel",
    "raccoon",
    "otter",
    "seal",
)


@dataclass(frozen=True)
class Check:
    """확인할 인물 하나: 자리와 성별(F/M)·나이(CHILD/YOUNG/ADULT/MIDDLE/OLD)·동물."""

    region: str
    gender: str | None = None
    age: str | None = None
    animal: str | None = None


@dataclass(frozen=True)
class Question:
    """한 자리에 묻는 것 하나: 고를 말들과 그중 맞는 말의 번호."""

    region: str
    labels: tuple[str, ...]
    answer: int
    weight: int = 1


def parse(raw: Any) -> list[Check]:
    """작업 입력 images[].checks → 확인할 인물. 모르는 자리나 빈 항목은 뺀다."""
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict) or item.get("region") not in (*BOXES, "full"):
            continue
        gender = item.get("gender") if item.get("gender") in ("F", "M") else None
        age = item.get("age") if isinstance(item.get("age"), str) else None
        animal = item.get("animal") if isinstance(item.get("animal"), str) else None
        if gender or age or animal:
            out.append(Check(item["region"], gender, age, animal))
    return out


def _age_group(age: str | None) -> int | None:
    return {"CHILD": 0, "YOUNG": 1, "ADULT": 1, "MIDDLE": 1, "OLD": 2}.get(age or "")


def questions(checks: list[Check]) -> list[Question]:
    """인물마다 물을 것.

    동물은 종(컷에 나오는 다른 동물과 사람 가운데)을, 사람은 나이대와 성별을 따로 묻는다.
    두 명 이상 나오는 컷은 그림 전체에 인물이 몇인지도 묻는다. 회차 16에서 성별·나이만 물었더니
    한 사람만 그린 그림이 골라져 두 사람 컷에서 한 사람이 빠졌다. 회차 17에서 왼쪽·오른쪽 반을
    잘라 "사람이 있는가"를 물었지만, 가운데 선 한 사람이 두 반에 다 걸려 빠진 그림을 못 가렸다.
    회차 14~16의 두 사람 컷 39개로 재 보니 그림 전체에 "two / one" 을 묻는 쪽이 훨씬 잘 가렸다
    (빠진 그림 20개 가운데 18개를 가림, 반 자르기는 13개).
    """
    animals = [c.animal for c in checks if c.animal]
    out = []
    if len(checks) in NUMBERS:
        many = f"{NUMBERS[len(checks)]} cartoon characters"
        out.append(Question("full", (many, "one cartoon character alone"), 0, COUNT_WEIGHT))
    for c in checks:
        if c.animal:
            others = [a for a in ANIMALS if a != c.animal and a not in animals]
            choices = [c.animal, *[a for a in animals if a != c.animal], *others[:4]]
            labels = tuple(f"a cartoon {a}" for a in dict.fromkeys(choices))
            out.append(Question(c.region, (*labels, "a cartoon human person"), 0))
            continue
        group = _age_group(c.age)
        if group is not None:
            labels = ("a cartoon child", "a cartoon adult", "a cartoon elderly person")
            out.append(Question(c.region, labels, group))
        if c.gender:
            pair = (
                ("a cartoon girl", "a cartoon boy")
                if group == 0
                else ("a cartoon woman", "a cartoon man")
            )
            out.append(Question(c.region, pair, 0 if c.gender == "F" else 1))
    return out


def box(region: str, width: int, height: int) -> tuple[int, int, int, int]:
    """자리 → 잘라 볼 상자 (왼쪽, 위, 오른쪽, 아래)."""
    if region == "full":
        return 0, 0, width, height
    start, end = BOXES[region]
    return round(start * width), 0, round(end * width), height


def score(probs: list[list[float]], qs: list[Question]) -> tuple[int, float]:
    """질문별 확률 → (틀린 수(무게를 곱해), 맞는 말 확률 평균).

    다시 그린 그림 가운데 틀린 수가 적고 평균이 높은 것을 고른다.
    """
    wrong = 0
    total = 0.0
    for p, q in zip(probs, qs, strict=True):
        right = p[q.answer]
        total += right
        if right < THRESHOLD or right < max(p):
            wrong += q.weight
    return wrong, total / len(qs) if qs else 1.0


def better(a: tuple[int, float], b: tuple[int, float]) -> bool:
    """a가 b보다 나은 그림인가."""
    return a[0] < b[0] or (a[0] == b[0] and a[1] > b[1])
