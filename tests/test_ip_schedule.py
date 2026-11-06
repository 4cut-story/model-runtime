"""IP-Adapter step별 강도 스케줄 (REQ-018: 두 캐릭터 일관성)."""

from model_runtime.backends.ip_schedule import IpSchedule


def test_REQ018_앞쪽_step은_약하게_뒤쪽_step은_캐릭터_강도에_boost를_더해_건다() -> None:
    s = IpSchedule(early_scale=0.2, early_steps=4, late_boost=0.2)
    assert s.active
    assert s.scales([0.6, 0.7], 0) == [0.2, 0.2]
    assert s.scales([0.6, 0.7], 3) == [0.2, 0.2]
    late = s.scales([0.6, 0.7], 4)
    assert [round(v, 6) for v in late] == [0.8, 0.9]


def test_REQ018_강도는_0에서_1_사이이고_캐릭터_강도보다_앞쪽이_세지_않다() -> None:
    s = IpSchedule(early_scale=0.2, early_steps=4, late_boost=0.2)
    assert s.scales([0.1], 0) == [0.1]
    assert s.scales([0.95], 7) == [1.0]
    # 강도 0(참조 없음)은 뒤쪽에서도 0
    assert s.scales([0.0], 7) == [0.0]


def test_REQ018_early_steps가_0이면_스케줄_없이_캐릭터_강도를_그대로_쓴다() -> None:
    s = IpSchedule(early_steps=0)
    assert not s.active
    assert s.scales([0.6], 0) == [0.6]
    assert s.scales([0.6], 7) == [0.6]
