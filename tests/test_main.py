from pathlib import Path

import pytest

from model_runtime import __main__ as entry
from model_runtime.backends.fake import FakeBackends
from model_runtime.config import Settings


def settings(backend: str) -> Settings:
    return Settings(api_url="https://x", token="t", worker_id="w", backend=backend)


def test_가짜_백엔드를_고른다() -> None:
    assert isinstance(entry.build_backends(settings("fake")), FakeBackends)
    with pytest.raises(ValueError, match="WORKER_BACKEND"):
        entry.build_backends(settings("cloud"))


def test_main은_설정을_읽고_워커를_돌린다(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "GPU_WORKER_API_URL=https://4cut.example\nGPU_WORKER_TOKEN=t\n", encoding="utf-8"
    )
    monkeypatch.delenv("GPU_WORKER_API_URL", raising=False)
    monkeypatch.delenv("GPU_WORKER_TOKEN", raising=False)
    ran: list[bool] = []
    monkeypatch.setattr(entry.Worker, "run", lambda self, max_polls=None: ran.append(True))
    assert entry.main() == 0
    assert ran == [True]
