"""`python -m recommenditos.api`: what the launcher sets before uvicorn starts."""

import os

import pytest

from recommenditos.api import __main__ as launcher


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """The uvicorn command lines `main` would have run, with nothing started."""
    runs: list[list[str]] = []
    monkeypatch.setattr(launcher, "prepare_multiprocess_dir", lambda: None)
    monkeypatch.setattr(launcher.uvicorn.main, "main", lambda args, prog_name: runs.append(args))
    return runs


@pytest.mark.req("NFR-02", "NFR-03")
def test_nfr02_the_launcher_gives_lightgbm_one_thread(
    started: list[list[str]], monkeypatch: pytest.MonkeyPatch
):
    """Workers inherit `OMP_NUM_THREADS=1`, so a prediction does not spread over every core."""
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    launcher.main(["--port", "8000"])
    assert os.environ["OMP_NUM_THREADS"] == launcher.OMP_THREADS == "1"
    assert started[0][-2:] == ["--port", "8000"]


@pytest.mark.req("NFR-02")
def test_nfr02_the_launcher_keeps_a_thread_count_the_environment_sets(
    started: list[list[str]], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    launcher.main([])
    assert os.environ["OMP_NUM_THREADS"] == "2"
