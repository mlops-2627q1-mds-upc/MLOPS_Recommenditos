"""NFR-11 on the real samples, with the real detectors, through the job's own pass (EDN-14).

The listings take the path of M6's traffic: the replay's body with
`registration_date` shifted by the whole months to its send time
(`replay.request_body`), logged as the client sent it (#191, no validation in
between), and read back by `run_once`, which computes each age at
`as_of_for(answered_at)`, cuts windows of 1,000 and tests them against the
10,000-listing reference. The sends start an hour before the 8th's month
boundary, so that the rule is tested where the two shifts meet.

- 20 windows of 1,000 control listings, drawn from `train` at random apart
  from the reference: at most 1 is flagged.
- The first 1,000 `ES` listings: the window is flagged and names at least three
  drifted features besides `country_code`.

The samples come from `dvc pull data/monitoring` (the `monitoring_samples`
stage). Without them the test skips, and fails where
RECOMMENDITOS_REQUIRE_MONITORING_SAMPLES=1 says they must be there: in CD's
publish job, which pulls them, so that no release ships whose drift job misses
NFR-11. Pull requests cannot pull them (the DagsHub credentials are main's
only, EDN-91), so the Tests job of a pull request skips it and says so.
"""

from datetime import UTC, datetime, timedelta
import gzip
import json
import os
from pathlib import Path
from typing import Any

import pytest
from samples import PredictionLog

from recommenditos_monitoring.job import WINDOW_SIZE, Paths, finding, run_once
from recommenditos_monitoring.replay import request_body
from recommenditos_monitoring.store import Store

SAMPLES = Path(
    os.environ.get(
        "RECOMMENDITOS_MONITORING_DATA_DIR",
        Path(__file__).resolve().parents[2] / "data" / "monitoring",
    )
)
REQUIRED = os.environ.get("RECOMMENDITOS_REQUIRE_MONITORING_SAMPLES") == "1"
NEEDED = ("manifest.json", "reference.parquet", "control.jsonl.gz", "es.jsonl.gz")

CONTROL_WINDOWS = 20
#: An hour before 2026-12-08 00:00 UTC, when the 13th month after 2025-11-08 completes.
FIRST_SEND = datetime(2026, 12, 7, 23, 0, tzinfo=UTC)
#: The replay's default rate, 5 listings/s, and an answer 40 ms after the send.
INTERVAL, LATENCY = timedelta(seconds=0.2), timedelta(milliseconds=40)


def _samples_or_skip() -> None:
    missing = [name for name in NEEDED if not (SAMPLES / name).is_file()]
    if not missing:
        return
    message = f"the monitoring samples are not in {SAMPLES} (no {', '.join(missing)}): "
    if REQUIRED:
        pytest.fail(message + "RECOMMENDITOS_REQUIRE_MONITORING_SAMPLES=1 needs them")
    pytest.skip(message + "run `dvc pull data/monitoring` to test NFR-11")


def _bodies(name: str, count: int) -> list[dict[str, Any]]:
    bodies = []
    with gzip.open(SAMPLES / f"{name}.jsonl.gz", "rt", encoding="utf-8") as lines:
        for line in lines:
            bodies.append(json.loads(line)["body"])
            if len(bodies) == count:
                break
    assert len(bodies) == count, f"{name} holds {len(bodies)} listings, not {count}"
    return bodies


@pytest.fixture(scope="module")
def windows(tmp_path_factory: pytest.TempPathFactory) -> Store:
    _samples_or_skip()
    root = tmp_path_factory.mktemp("nfr11")
    reference_date = json.loads((SAMPLES / "manifest.json").read_text())["reference_date"]
    bodies = _bodies("control", CONTROL_WINDOWS * WINDOW_SIZE) + _bodies("es", WINDOW_SIZE)
    log = PredictionLog(root / "prediction-log.sqlite3")
    for i, body in enumerate(bodies):
        sent = FIRST_SEND + i * INTERVAL
        log.add(request_body(body, sent, reference_date), sent + LATENCY)
    log.close()
    paths = Paths(log=log.path, db=root / "monitor.db", data_dir=SAMPLES, textfile_dir=None)

    outcome = run_once(paths)

    assert len(outcome.closed) == CONTROL_WINDOWS + 1
    assert outcome.open_requests == 0
    return Store.open(paths.db)


@pytest.mark.req("NFR-11")
def test_at_most_one_of_20_control_windows_is_flagged(windows):
    control = windows.windows()[:CONTROL_WINDOWS]
    assert all(w.status == "tested" and w.requests == WINDOW_SIZE for w in control)
    assert all(w.reference_rows == 10_000 for w in control)
    flagged = [w for w in control if w.flagged]

    assert len(flagged) <= 1, "\n".join(finding(w, windows.features(w.window_id)) for w in flagged)


@pytest.mark.req("NFR-11", "FR-14")
def test_the_first_es_window_is_flagged_with_three_features_besides_the_market(windows):
    es = windows.windows()[CONTROL_WINDOWS]
    features = windows.features(es.window_id)
    drifted = {f.feature for f in features if f.drifted}
    report = finding(es, features)

    assert es.es_share == 1.0
    assert es.flagged, report
    assert "country_code" in drifted, report
    assert len(drifted - {"country_code"}) >= 3, report
    # What the store gives the dashboard and the retraining step for the window.
    assert es.first_answered_at < es.last_answered_at
    assert {f.feature for f in features if f.p_value < es.threshold} == drifted
