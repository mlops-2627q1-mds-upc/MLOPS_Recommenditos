"""The collector of `req` markers that the requirement matrix uses, in any environment.

tools/requirement_matrix.py collects the root project's tests in its own process
with `MarkerCollector`. The monitoring jobs' tests (monitoring/) import numpy 1.26,
pandas 2 and Alibi Detect, which the root environment cannot hold, so the matrix
runs this file in their environment as a script:

    python tools/req_markers.py <pyproject.toml> <rootdir> <tests dir> <output.json>

which collects, without running a test, and writes the markers it saw as JSON.
It imports nothing but pytest and the standard library, so that it runs in both.
"""

import contextlib
import io
import json
from pathlib import Path
import sys

import pytest

#: The marker a test uses to declare the requirements it verifies.
MARKER = "req"


class MarkerCollector:
    """Records the `req` markers pytest sees, without running a single test."""

    def __init__(self) -> None:
        self.marked: dict[str, tuple[str, ...]] = {}
        self.empty: list[str] = []

    def pytest_collection_modifyitems(self, items) -> None:
        for item in items:
            markers = list(item.iter_markers(name=MARKER))
            ids = tuple(dict.fromkeys(str(arg) for marker in markers for arg in marker.args))
            # A parametrised test is one test in the matrix: twelve rows of the
            # same node ID with a different `[case]` suffix say nothing extra.
            test_id = item.nodeid.partition("[")[0]
            if markers and not ids:
                # Recorded rather than ignored. `@pytest.mark.req()` reads like a
                # record of coverage and is none, and the per-test check in
                # tests/conftest.py cannot see it on a test that never runs, so
                # skipping it here left it caught by nothing at all.
                if test_id not in self.empty:
                    self.empty.append(test_id)
                continue
            if not ids:
                continue
            self.marked.setdefault(test_id, ids)


def collect_arguments(config: Path, rootdir: Path, tests_dir: Path) -> list[str]:
    """pytest's arguments for a collection that reads the markers and runs nothing.

    The configuration and the root are passed explicitly, so the matrix builds
    the same from any working directory, and nothing is cached because collecting
    is not a test run.

    `-o addopts=` drops the project's own `addopts` for this run, which is what
    keeps a collection from pretending to be one. Inheriting them made the
    collect-only run rewrite `reports/junit.xml` as a `tests="0"` document, so
    building the matrix after `make test` destroyed the record of the 114 tests
    that had just passed. The reports belong to the real test run; this run
    produces none.
    """
    return [
        "--collect-only",
        "-q",
        "-o",
        "addopts=",
        "-p",
        "no:cacheprovider",
        "-c",
        str(config),
        "--rootdir",
        str(rootdir),
        str(tests_dir),
    ]


def main(argv: list[str]) -> int:
    config, rootdir, tests_dir, output = (Path(arg) for arg in argv)
    collector = MarkerCollector()
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        outcome = pytest.main(collect_arguments(config, rootdir, tests_dir), plugins=[collector])
    if outcome != pytest.ExitCode.OK:
        print(captured.getvalue())
        return int(outcome)
    output.write_text(
        json.dumps({"marked": collector.marked, "empty": collector.empty}), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
