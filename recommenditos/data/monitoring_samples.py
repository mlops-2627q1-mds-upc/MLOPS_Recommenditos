"""`monitoring_samples` stage: the versioned samples M6's monitoring builds on.

M6 has no real users (EDN-13), so the traffic that makes drift, error and
coverage visible is a replay of listings (#227), and the drift job (#228)
compares each window of logged requests with a training reference (NFR-11).
Both read the samples this stage writes to `data/monitoring/`, so a run can be
reproduced from `dvc.lock` instead of from an ad-hoc draw (NFR-06):

- `reference`: `monitoring.reference_size` listings of `train`, the drift
  reference.
- `control`: `monitoring.control_size` further listings of `train`, none of
  them in the reference. Drawn at random from the same rows, so a control
  window is i.i.d. with the reference: EDN-14 measured that a seller-grouped
  control, which is what `test` is, gets flagged as drift. The model was fitted
  on these listings, so the error and coverage the replay measures on them are
  in-sample and optimistic; they show that the pipeline works, not how good the
  model is.
- `es`: every listing of the `ES` holdout (EDN-03), the drift scenario.

Only listings the API accepts are drawn: a supported make (FR-04) and a body
that `ValuationRequest` and `PartialListing` both validate, which needs every
field FR-01 requires. Nothing else reaches the prediction log, so a reference
holding such rows would compare every window the drift job reads with listings
that can never appear in it, such as the 1 % of supported training listings
without `power_kw`.

Each sample is written twice:

- `<name>.jsonl.gz`: one JSON object per line, `{"listing_id", "observed_price_eur",
  "body"}`, where `body` is the request as a client sends it (`request_body.to_body`):
  a missing value is left out, never sent as `null`. `registration_date` is the
  listing's own; the replay shifts it to the send date.
- `<name>.parquet`: the same listings as the drift job compares them, the 23
  features EDN-14 validated NFR-11 with (`DRIFT_FEATURES`), derived from the body
  by `drift_features` with the age on `reference_date`.

The rows are in a random order, so every run of consecutive rows is a random
sample of its set and a replay can stop after any window. `manifest.json` says
how many listings each source had and why the others were left out, and which
features are numeric and which categorical. "The monitoring samples" in
docs/docs/pipeline.md describes the files for the jobs that read them.
"""

from collections.abc import Iterable, Mapping, Sequence
import gzip
import json
from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
from pydantic import ValidationError
import typer

from recommenditos.api.schemas.partial_listing import PartialListing
from recommenditos.config import DATA_DIR, PARAMS_FILE, PROCESSED_DATA_DIR
from recommenditos.data.build_features import age_years, read_supported_makes
from recommenditos.data.request_body import to_body
from recommenditos.data.split_data import HOLDOUT_NAME
from recommenditos.pipeline import load_params, read_frame, write_frame, write_json
from recommenditos.schema import PROCESSED_SCHEMA, Column, Schema

app = typer.Typer()

#: Where the stage writes. Not under `data/processed/`, because a file there
#: would sit beside the frames `features` and `comparables` read; and defined
#: here rather than in `recommenditos/config.py`, which every stage depends on.
MONITORING_DATA_DIR = DATA_DIR / "monitoring"

#: The sample names and the `split` frame each is drawn from.
SAMPLES: dict[str, str] = {"reference": "train", "control": "train", "es": HOLDOUT_NAME}

MANIFEST_FILE = "manifest.json"

#: The numeric features EDN-14 tested, compared on their non-missing values (KS).
NUMERIC: tuple[str, ...] = (
    "age_years",
    "mileage_km_raw",
    "power_kw",
    "nr_prev_owners",
    "gears",
    "cylinders_volume_cc",
    "nr_seats",
    "nr_doors",
)
#: The categorical features EDN-14 tested (chi-square). `model` is not one: its
#: 349 levels leave well under one expected count per cell in a window of 1,000.
CATEGORICAL: tuple[str, ...] = (
    "make",
    "body_type",
    "fuel_category",
    "transmission",
    "drive_train",
    "seller_type",
    "country_code",
)
#: Whether each numeric feature was sent, tested as a categorical, because a KS
#: test on the non-missing values cannot see a change in how often one is missing.
MISSING_INDICATORS: tuple[str, ...] = tuple(f"{name}__missing" for name in NUMERIC)
#: The 23 features of NFR-11, in EDN-14's order.
DRIFT_FEATURES: tuple[str, ...] = NUMERIC + CATEGORICAL + MISSING_INDICATORS

#: The value of a categorical feature the request did not send.
MISSING = "__MISSING__"
#: The two values of a missing indicator.
ABSENT, PRESENT = "missing", "present"

SAMPLE_SCHEMA = Schema(
    name="monitoring-sample",
    columns=(
        Column("listing_id", "str", False, "`<split frame>:<row position in it>`."),
        Column("observed_price_eur", "float64", False, "The listing's asking price in EUR."),
        *(
            Column(name, "float64", True, "Numeric drift feature; null where not sent.")
            for name in NUMERIC
        ),
        *(
            Column(name, "str", False, f"Categorical drift feature; `{MISSING}` where not sent.")
            for name in CATEGORICAL
        ),
        *(
            Column(name, "str", False, f"`{ABSENT}` or `{PRESENT}`.")
            for name in MISSING_INDICATORS
        ),
    ),
)


def accepted_body(listing: Mapping[str, Any], supported: frozenset[str]) -> dict[str, Any] | None:
    """The body a client sends for `listing`, or None if the API would refuse it."""
    body = to_body(dict(listing))
    if body is None or body["make"] not in supported:
        return None
    try:
        PartialListing.model_validate(body)
    except ValidationError:
        return None
    return body


def drift_features(bodies: Iterable[Mapping[str, Any]], reference_date: str) -> pd.DataFrame:
    """The drift features of request bodies, as the reference holds them.

    A field the body does not carry is null for a numeric feature and `MISSING`
    for a categorical one. The age is computed from `registration_date` on
    `reference_date` by the function training and serving use.
    """
    frame = pd.DataFrame.from_records(list(bodies))
    rows = len(frame)
    out = pd.DataFrame(index=range(rows))

    def column(name: str) -> pd.Series:
        if name in frame:
            return frame[name].reset_index(drop=True)
        return pd.Series([None] * rows, dtype="object")

    registered = pd.to_datetime(column("registration_date"), format="%Y-%m")
    out["age_years"] = age_years(registered, reference_date).astype("float64")
    for name in NUMERIC[1:]:
        out[name] = pd.to_numeric(column(name)).astype("float64")
    for name in CATEGORICAL:
        out[name] = column(name).astype("str").where(column(name).notna(), MISSING)
    for name in NUMERIC:
        out[f"{name}__missing"] = np.where(out[name].isna(), ABSENT, PRESENT)
    return out.astype({name: "str" for name in CATEGORICAL + MISSING_INDICATORS})


def accepted(
    frame: pd.DataFrame, source: str, supported: frozenset[str]
) -> tuple[list[str], list[float], list[dict[str, Any]], dict[str, int]]:
    """The listings of `frame` the API accepts, their ids and prices, and the dropped counts."""
    ids, prices, bodies = [], [], []
    dropped = {"unsupported_make": 0, "invalid_request": 0}
    for position, listing in enumerate(frame.to_dict(orient="records")):
        if listing["make"] not in supported:
            dropped["unsupported_make"] += 1
            continue
        body = accepted_body(listing, supported)
        if body is None:
            dropped["invalid_request"] += 1
            continue
        ids.append(f"{source}:{position}")
        prices.append(float(listing["price"]))
        bodies.append(body)
    return ids, prices, bodies, dropped


def write_replay(
    path: Path, ids: Sequence[str], prices: Sequence[float], bodies: Sequence[dict]
) -> None:
    """Write the gzipped JSON lines, the same bytes on every run and platform.

    `mtime=0` and no file name keep the gzip header free of anything but the
    content, since DVC hashes the file byte for byte.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        for listing_id, price, body in zip(ids, prices, bodies, strict=True):
            record = {"listing_id": listing_id, "observed_price_eur": price, "body": body}
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            gz.write(line.encode("utf-8"))


def read_replay(path: Path) -> list[dict[str, Any]]:
    """The records of a replay file, in order."""
    with gzip.open(path, "rt", encoding="utf-8") as lines:
        return [json.loads(line) for line in lines]


@app.command()
def main(
    input_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Path = MONITORING_DATA_DIR,
    params_path: Path = PARAMS_FILE,
):
    params = load_params(params_path)
    sizes = {
        "reference": int(params["monitoring"]["reference_size"]),
        "control": int(params["monitoring"]["control_size"]),
    }
    reference_date = params["reference_date"]
    supported = frozenset(read_supported_makes(input_dir))
    rng = np.random.default_rng(params["seed"])

    sources: dict[str, Any] = {}
    pools = {}
    for source in sorted(set(SAMPLES.values())):
        frame = read_frame(input_dir / f"{source}.parquet", PROCESSED_SCHEMA)
        ids, prices, bodies, dropped = accepted(frame, source, supported)
        pools[source] = (ids, prices, bodies)
        sources[source] = {"rows": len(frame), "accepted": len(ids), "dropped": dropped}

    train_ids = pools["train"][0]
    needed = sizes["reference"] + sizes["control"]
    if len(train_ids) < needed:
        raise ValueError(
            f"`train` has {len(train_ids):,} listings the API accepts, fewer than the "
            f"{needed:,} that `monitoring.reference_size` and `monitoring.control_size` ask for"
        )
    # One permutation cut in two, so reference and control cannot share a listing.
    order = rng.permutation(len(train_ids))
    picks = {
        "reference": order[: sizes["reference"]],
        "control": order[sizes["reference"] : needed],
        "es": rng.permutation(len(pools[HOLDOUT_NAME][0])),
    }

    written = {}
    for name, source in SAMPLES.items():
        ids, prices, bodies = pools[source]
        chosen = [int(i) for i in picks[name]]
        sample_ids = [ids[i] for i in chosen]
        sample_prices = [prices[i] for i in chosen]
        sample_bodies = [bodies[i] for i in chosen]
        write_replay(output_dir / f"{name}.jsonl.gz", sample_ids, sample_prices, sample_bodies)
        features = drift_features(sample_bodies, reference_date)
        features.insert(0, "observed_price_eur", sample_prices)
        features.insert(0, "listing_id", sample_ids)
        write_frame(features, output_dir / f"{name}.parquet", SAMPLE_SCHEMA)
        written[name] = {"source": source, "rows": len(chosen)}

    write_json(
        {
            "seed": params["seed"],
            "reference_date": reference_date,
            "samples": written,
            "sources": sources,
            "features": {
                "numeric": list(NUMERIC),
                "categorical": list(CATEGORICAL),
                "missing_indicators": list(MISSING_INDICATORS),
            },
            "missing_category": MISSING,
        },
        output_dir / MANIFEST_FILE,
    )
    for name, entry in written.items():
        logger.info(f"{name}: {entry['rows']:,} listings of `{entry['source']}`.")
    for source, entry in sources.items():
        logger.info(
            f"`{source}`: {entry['accepted']:,} of {entry['rows']:,} listings accepted by the "
            f"API; left out {entry['dropped']}."
        )


if __name__ == "__main__":
    app()
