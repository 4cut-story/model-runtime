"""IP-Adapter step별 강도 스케줄 (GPU 없이 테스트할 수 있게 sdxl.py에서 뺐다).

IP-Adapter Plus는 얼굴뿐 아니라 머리 모양·옷까지 참조 그림을 따른다.
처음부터 강하게 걸면 참조 그림의 구도(정면 상반신, 회색 배경)를 베껴 장면이 사라진다.
구도는 앞쪽 step에서 정해지므로 앞쪽은 약하게(early),
뒤쪽은 캐릭터 강도에 boost를 더해 강하게 건다.
GPU PC IP-Adapter 벤치(2026-11-05): Plus 1~4 step 0.2 → 5~8 step 0.8이
구도는 프롬프트대로, 정체성(얼굴·머리·안경·옷)은 가장 잘 유지됐다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IpSchedule:
    """early_steps가 0이면 스케줄 없이 캐릭터 강도 그대로 모든 step에 건다."""

    early_scale: float = 0.2
    early_steps: int = 4
    late_boost: float = 0.2

    def scales(self, scales: list[float], step: int) -> list[float]:
        """step(0부터)에서 쓸 참조별 강도. 0~1로 자른다."""
        if step < self.early_steps:
            return [_clip(min(self.early_scale, s)) for s in scales]
        boost = self.late_boost if self.early_steps > 0 else 0.0
        return [_clip(s + boost) if s > 0 else 0.0 for s in scales]

    @property
    def active(self) -> bool:
        return self.early_steps > 0


def _clip(value: float) -> float:
    return min(max(value, 0.0), 1.0)
