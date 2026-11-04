"""Write the load test's request bodies from real held-out listings (issue #143).

    uv run python -m tools.loadtest.bodies            (make loadtest-bodies)

Samples listings of the processed test split, which the model never saw, and
writes each as a `/v1/predict` body with every field the listing has, one JSON
object per line. The mapping is `recommenditos.data.request_body.to_body`, which
M6's replay samples share. The Locust file cuts each body down to what an
endpoint accepts. A listing without one of the ten fields FR-01 requires, with a
value the request schema refuses, or with a category the served bundle does not
accept (FR-03, FR-04) is not a valid request and is left out, so that every
error the load test counts is the API's.

The test split holds no `ES` listings (EDN-03). M6's replay of them is a
different run, from the versioned samples of the `monitoring_samples` stage
(`data/monitoring/`).

Needs `uv run dvc pull data/processed/test.parquet models/lgbm-basic`, or the
bundle the target serves with `--model`. The output is small and
standard JSON, so it can be copied to wherever Locust runs, the VM included,
which needs neither DVC nor the data.
"""

import argparse
from collections.abc import Iterator
import json
from pathlib import Path
from typing import Any

import pandas as pd

from recommenditos.api.serving import ServedModel
from recommenditos.config import INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from recommenditos.data.request_body import to_body

DEFAULT_SPLIT = PROCESSED_DATA_DIR / "test.parquet"
DEFAULT_OUTPUT = INTERIM_DATA_DIR / "loadtest" / "bodies.jsonl"
DEFAULT_MODEL = MODELS_DIR / "lgbm-basic"
DEFAULT_COUNT = 2000
DEFAULT_SEED = 143


def sample_bodies(
    split: Path, served: ServedModel, count: int, seed: int
) -> Iterator[dict[str, Any]]:
    """Up to `count` bodies `served` accepts, from a random sample of the listings in `split`."""
    frame = pd.read_parquet(split).sample(frac=1.0, random_state=seed)
    produced = 0
    for listing in frame.to_dict(orient="records"):
        body = to_body(listing)
        if body is not None and not served.faults(body):
            yield body
            produced += 1
            if produced == count:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    bodies = list(sample_bodies(args.split, ServedModel.load(args.model), args.count, args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(body, ensure_ascii=False) + "\n" for body in bodies)
    args.output.write_text(lines, encoding="utf-8", newline="\n")
    print(f"{len(bodies)} bodies from {args.split} written to {args.output}")


if __name__ == "__main__":
    main()
