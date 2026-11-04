"""Fixture samples and a fixture prediction log for the drift job's tests, and for CI's stack.

`write_fixture_samples` builds a `data/monitoring/` out of the six control and
six ES listings of tests/fixtures/: the two replay files, the manifest, and a
drift reference computed from the control listings with `drift_features`, as
the `monitoring_samples` stage computes the real one. CI's integration job
cannot `dvc pull` the real samples, so it bakes these into the monitoring
image:

    uv run --directory monitoring --locked python -m tests.samples ../data/monitoring

`PredictionLog` writes a prediction log with the API's tables
(`recommenditos/api/prediction_log.py`): the answers the drift job reads, and
the feedback it computes the error and coverage from.
"""

import ast
from collections.abc import Mapping
from datetime import datetime, timedelta
import gzip
import json
from pathlib import Path
import shutil
import sqlite3
import sys
from typing import Any
import uuid

from recommenditos_monitoring.drift_features import drift_features

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_bodies(name: str) -> list[dict[str, Any]]:
    lines = (FIXTURES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["body"] for line in lines if line.strip()]


def write_fixture_samples(target: Path, reference_copies: int = 10) -> Path:
    """A samples directory from the fixtures; the reference repeats the control listings."""
    target.mkdir(parents=True, exist_ok=True)
    for name in ("control", "es"):
        source = FIXTURES / f"{name}.jsonl"
        (target / f"{name}.jsonl.gz").write_bytes(gzip.compress(source.read_bytes(), mtime=0))
    shutil.copy(FIXTURES / "manifest.json", target / "manifest.json")
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    reference = drift_features(
        fixture_bodies("control") * reference_copies, manifest["reference_date"]
    )
    reference.to_parquet(target / "reference.parquet", index=False)
    return target


#: The API's prediction log module, whose schema the fixture log uses; the
#: monitoring environment cannot import it, so its text is parsed.
API_LOG_MODULE = Path(__file__).parents[2] / "recommenditos" / "api" / "prediction_log.py"


def api_schema() -> str:
    """The API's own `_SCHEMA`, so that the fixture log cannot drift from the real one."""
    module = ast.parse(API_LOG_MODULE.read_text(encoding="utf-8"))
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and [getattr(target, "id", None) for target in node.targets] == ["_SCHEMA"]
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    raise LookupError(f"{API_LOG_MODULE} has no _SCHEMA string")


_SCHEMA = api_schema()


class PredictionLog:
    """A prediction log in WAL mode, written the way the API writes it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._connection = sqlite3.connect(path, isolation_level=None)
        self._connection.execute("PRAGMA journal_mode = WAL")
        # No sync per commit: a test's log need not survive a power cut.
        self._connection.execute("PRAGMA synchronous = OFF")
        # As the API's store: the rotation deletes an answer's feedback with it.
        self._connection.execute("PRAGMA foreign_keys = ON")
        if self._connection.execute("SELECT count(*) FROM sqlite_master").fetchone() == (0,):
            self._connection.executescript(_SCHEMA)

    def add(  # noqa: PLR0913 - one keyword per column a test varies
        self,
        inputs: Mapping[str, Any],
        answered_at: datetime,
        *,
        endpoint: str = "/v1/predict",
        client: str | None = "replay",
        inputs_sent_only: int | None = 1,
        model_version: str = "v1",
        outputs: Mapping[str, Any] | None = None,
    ) -> int:
        cursor = self._connection.execute(
            "INSERT INTO predictions (request_id, endpoint, answered_at, latency_ms, "
            "model_variant, model_version, inputs, outputs, warnings, client, inputs_sent_only) "
            "VALUES (?, ?, ?, 12.5, 'lgbm-basic', ?, ?, ?, '[]', ?, ?)",
            (
                str(uuid.uuid4()),
                endpoint,
                answered_at.isoformat(timespec="milliseconds"),
                model_version,
                json.dumps(inputs),
                json.dumps(outputs or {}),
                client,
                inputs_sent_only,
            ),
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def add_many(
        self, bodies: list[dict[str, Any]], start: datetime, step: timedelta, **kwargs: Any
    ) -> list[int]:
        return [self.add(body, start + i * step, **kwargs) for i, body in enumerate(bodies)]

    def add_feedback(self, seq: int, observed_price_eur: float, received_at: datetime) -> None:
        """The observed price of the answer `seq`, as `POST /v1/feedback` stores it."""
        self._connection.execute(
            "INSERT INTO feedback (request_id, observed_price_eur, received_at) "
            "SELECT request_id, ?, ? FROM predictions WHERE seq = ?",
            (observed_price_eur, received_at.isoformat(timespec="milliseconds"), seq),
        )

    def rotate(self, up_to_seq: int) -> None:
        """The API's rotation: the oldest answers go, with their feedback."""
        self._connection.execute("DELETE FROM predictions WHERE seq <= ?", (up_to_seq,))

    def close(self) -> None:
        self._connection.close()


if __name__ == "__main__":
    print(write_fixture_samples(Path(sys.argv[1])))
