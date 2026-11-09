"""라이브 GPU 워커(운영 그림·전사·LLM 작업)에 GPU를 양보하기 위한 게이트.

워커 로그(기본 D:\\4cut\\logs\\worker.log)를 읽어 지금 작업을 처리 중인지 본다.
워커는 할 일이 없을 때만 `gpu-jobs/claim`에 204로 폴링하고, 작업을 받으면
`claim 200 OK` 뒤 `작업 시작: job=N`을 찍고 끝나면 `작업 완료/실패: job=N`을 찍는다.
그래서 로그 끝에서 거꾸로 올라가며 가장 최근의 결정적 줄로 판정한다.

describe_local.py(Ollama)와 embed_clips.py가 각 항목 전에 wait_until_free()를 불러,
운영 작업이 돌고 있으면 비울 때까지 기다린다. 운영 서비스가 항상 우선이다.
"""

from __future__ import annotations

import contextlib
import os
import time
from collections.abc import Callable
from pathlib import Path

WORKER_LOG = Path(os.environ.get("WORKER_LOG", r"D:\4cut\logs\worker.log"))


def _tail(log: Path, nbytes: int = 65536) -> list[str]:
    try:
        with open(log, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode("utf-8", "ignore").splitlines()
    except OSError:
        return []


def worker_busy(log: Path = WORKER_LOG) -> bool:
    """운영 워커가 지금 작업을 처리 중이면 True. 로그를 못 읽으면 안전하게 True로 본다."""
    if not log.exists():
        # 워커 로그가 없는 곳(예: 맥)에서는 게이트가 무효가 되어 그냥 진행한다
        return False
    lines = _tail(log)
    if not lines:
        return False
    for ln in reversed(lines):
        if "작업 시작" in ln:
            return True
        if "claim" in ln and "200 OK" in ln:
            return True
        if "claim" in ln and "204" in ln:
            return False
        if "작업 완료" in ln or "작업 실패" in ln:
            return False
    return False


def wait_until_free(
    on_wait: Callable[[], None] | None = None,
    poll: float = 3.0,
    log: Path = WORKER_LOG,
) -> None:
    """운영 작업이 끝날 때까지 블록한다. 처음 기다리기 시작할 때 on_wait를 한 번 부른다
    (예: Ollama 모델을 내려 VRAM을 비우기)."""
    called = False
    while worker_busy(log):
        if on_wait is not None and not called:
            with contextlib.suppress(Exception):
                on_wait()
            called = True
        time.sleep(poll)
