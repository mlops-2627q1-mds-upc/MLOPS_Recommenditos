"""Write the load test's request bodies from real held-out listings (issue #143).

    uv run python -m tools.loadtest.bodies            (make loadtest-bodies)

Samples listings of the processed test split, which the model never saw, and
writes each as a `/v1/predict` body with every field the listing has, one JSON
object per line. The Locust file cuts each body down to what an endpoint
accepts. A listing without one of the ten fields FR-01 requires, or with a
value the request schema refuses, is not a valid request and is left out.

The test split holds no `ES` listings (EDN-03). M6's replay of them is a
different run: `--split data/processed/holdout_es.parquet`.

Needs `uv run dvc pull data/processed/test.parquet`. The output is small and
standard JSON, so it can be copied to wherever Locust runs, the VM included,
which needs neither DVC nor the data.
"""

import argparse
import ast
from collections.abc import Iterator
import json
import math
from pathlib import Path
import re
from typing import Any

import pandas as pd
from pydantic import ValidationError

from recommenditos.api.schemas.predict import ValuationRequest
from recommenditos.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

DEFAULT_SPLIT = PROCESSED_DATA_DIR / "test.parquet"
DEFAULT_OUTPUT = INTERIM_DATA_DIR / "loadtest" / "bodies.jsonl"
DEFAULT_COUNT = 2000
DEFAULT_SEED = 143

_EQUIPMENT = (
    "equipment_comfort",
    "equipment_entertainment",
    "equipment_extra",
    "equipment_safety",
)
_INTEGER = {"nr_prev_owners", "gears", "nr_seats", "nr_doors", "cylinders", "cylinders_volume_cc"}


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _weight_kg(text: str) -> int | None:
    """`'2,355 kg'` as 2355, the way a client sends the kerb weight."""
    digits = re.sub(r"[^0-9]", "", text)
    return int(digits) if digits else None


def to_body(listing: dict[str, Any]) -> dict[str, Any] | None:
    """The `/v1/predict` body of one processed listing, or None if it is no valid request."""
    body: dict[str, Any] = {}
    for field in ValuationRequest.model_fields:
        value: Any = listing.get(field)
        if _missing(value):
            continue
        if field == "registration_date":
            value = pd.Timestamp(value).strftime("%Y-%m")
        elif field in _EQUIPMENT:
            value = ast.literal_eval(value) if isinstance(value, str) else list(value)
        elif field == "weight_kg":
            value = _weight_kg(str(value))
        elif field in _INTEGER:
            value = int(value)
        elif hasattr(value, "item"):
            value = value.item()
        if value is not None:
            body[field] = value
    try:
        ValuationRequest.model_validate(body)
    except ValidationError:
        return None
    return body


def sample_bodies(split: Path, count: int, seed: int) -> Iterator[dict[str, Any]]:
    """Up to `count` valid bodies from a random sample of the listings in `split`."""
    frame = pd.read_parquet(split).sample(frac=1.0, random_state=seed)
    produced = 0
    for listing in frame.to_dict(orient="records"):
        body = to_body(listing)
        if body is not None:
            yield body
            produced += 1
            if produced == count:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    bodies = list(sample_bodies(args.split, args.count, args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(body, ensure_ascii=False) + "\n" for body in bodies)
    args.output.write_text(lines, encoding="utf-8", newline="\n")
    print(f"{len(bodies)} bodies from {args.split} written to {args.output}")


if __name__ == "__main__":
    main()
