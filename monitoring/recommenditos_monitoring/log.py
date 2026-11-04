"""Reading the API's prediction log the way the drift job compares it (FR-13, #191).

The log is the API's SQLite file (`recommenditos/api/prediction_log.py`), which
its workers keep writing while the job reads. The job opens it read-only: it
cannot change a row, and the API owns the schema. WAL still needs the
directory writable for its shared-memory file, even for a reader
(docs/docs/api.md#the-records), so the volume is mounted read-write all the same.

A row is one the drift job compares when it is

- marked `inputs_sent_only = 1`: its `inputs` hold only the fields the client
  sent (#191); an older row also holds the defaults of the fields left out,
  which would look like real values;
- from no marked client or from the replay: the web page, the smoke checks and
  the load test are test traffic (`client`, #269);
- an answer of a priced endpoint, `/v1/predict` or `/v1/price-range`: those are
  the listings to value, and the ones feedback can refer to;
- a request with all ten fields FR-01 requires. `/v1/price-range` requires only
  `make`, so a range request may lack the others, and a window of such requests
  would show missing fields the reference never has.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any

#: The fields FR-01 requires (`ValuationRequest`); every reference listing has them.
REQUIRED_FIELDS: tuple[str, ...] = (
    "make",
    "model",
    "registration_date",
    "mileage_km_raw",
    "power_kw",
    "fuel_category",
    "transmission",
    "country_code",
    "body_type",
    "seller_type",
)

#: The answers with a price (prediction_log.PRICED_ENDPOINTS).
PRICED_ENDPOINTS: tuple[str, ...] = ("/v1/predict", "/v1/price-range")

#: How long the job waits for a worker's write before a read gives up.
BUSY_TIMEOUT_S = 5.0

_ROWS = """
SELECT seq, request_id, endpoint, answered_at, model_variant, model_version, inputs
FROM predictions
WHERE seq > ?
  AND inputs_sent_only = 1
  AND (client IS NULL OR client = 'replay')
  AND endpoint IN (?, ?)
ORDER BY seq
"""


class LogError(Exception):
    """The prediction log cannot be read."""


@dataclass(frozen=True)
class LogRow:
    """One answer of the log the drift job compares."""

    seq: int
    request_id: str
    endpoint: str
    answered_at: str
    model_variant: str
    model_version: str
    inputs: Mapping[str, Any]

    @property
    def has_required_fields(self) -> bool:
        return all(self.inputs.get(name) is not None for name in REQUIRED_FIELDS)


def connect_read_only(path: Path) -> sqlite3.Connection:
    """A read-only connection to the log; it fails when there is no log to read."""
    if not path.is_file():
        raise LogError(f"there is no prediction log at {path}; has the API started?")
    uri = f"{path.resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=BUSY_TIMEOUT_S)


def rows_after(connection: sqlite3.Connection, seq: int) -> Iterator[LogRow]:
    """The rows the job compares with a `seq` above `seq`, in log order, one at a time."""
    try:
        cursor = connection.execute(_ROWS, (seq, *PRICED_ENDPOINTS))
    except sqlite3.OperationalError as error:
        raise LogError(f"the prediction log cannot be read: {error}") from error
    for row in cursor:
        yield LogRow(
            seq=row[0],
            request_id=row[1],
            endpoint=row[2],
            answered_at=row[3],
            model_variant=row[4],
            model_version=row[5],
            inputs=json.loads(row[6]),
        )


def last_seq(connection: sqlite3.Connection) -> int:
    """The highest `seq` of the log, 0 for an empty one."""
    (value,) = connection.execute("SELECT coalesce(max(seq), 0) FROM predictions").fetchone()
    return int(value)
