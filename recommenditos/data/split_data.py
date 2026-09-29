"""`split` stage: hold Spain out, then split the rest grouped by seller.

STUB. It writes every artefact the rest of the pipeline reads, with the two
rules that must never silently regress already in place: no `ES` row leaves the
holdout, and no `seller_group_id` appears in two sets.

Issue #35 replaces the split itself and adds the tests. What it still owes:
the supported-make list is written here as an empty list rather than computed
(EDN-05 counts makes with at least `min_listings_per_make` listings after the
holdout and before the split), and the realised ratios are only approximate,
because whole sellers are assigned at a time and sellers differ in size.

The bucketing below is deterministic from `seed` and `seller_group_id` alone,
so it does not depend on row order and reproduces on a clean clone (NFR-06).
"""

import hashlib
import json
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from recommenditos.config import INTERIM_DATA_DIR, PARAMS_FILE, PROCESSED_DATA_DIR
from recommenditos.pipeline import load_params, read_frame, write_frame
from recommenditos.schema import INTERIM_SCHEMA, PROCESSED_SCHEMA

#: The sets `split` writes, besides the country holdout. Downstream stages
#: iterate this, so adding a set is a change here and in params.yaml only.
SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "calibration", "test")

#: Resolution of the group hash. A group's bucket is its hash modulo this,
#: so the ratios are honoured to within one bucket.
_BUCKETS = 10_000

app = typer.Typer()


def assign_split(groups: pd.Series, ratios: dict[str, float], seed: int) -> pd.Series:
    """Map each group key to one of `SPLIT_NAMES`, deterministically.

    Hashing the group rather than shuffling the rows means the assignment
    depends on the seed and the key alone: the same seller lands in the same
    set whatever order the frame arrives in, and no seller is ever split.
    """
    edges: list[tuple[str, float]] = []
    cumulative = 0.0
    for name in SPLIT_NAMES:
        cumulative += ratios[name]
        edges.append((name, cumulative))

    def bucket(key: str) -> str:
        digest = hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()
        position = int(digest[:8], 16) % _BUCKETS / _BUCKETS
        for name, edge in edges:
            if position < edge:
                return name
        return SPLIT_NAMES[-1]

    return groups.map(bucket)


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "listings.parquet",
    output_dir: Path = PROCESSED_DATA_DIR,
    params_path: Path = PARAMS_FILE,
):
    params = load_params(params_path)
    split_params = params["split"]
    frame = read_frame(input_path, INTERIM_SCHEMA)

    # 1. The holdout leaves first, so nothing below can see an `ES` row (EDN-03).
    is_holdout = frame["country_code"] == split_params["holdout_country"]
    holdout, remaining = frame[is_holdout], frame[~is_holdout]
    write_frame(holdout, output_dir / "holdout_es.parquet", PROCESSED_SCHEMA)

    # 2. The supported makes are an artefact, never a hard-coded list: the API
    #    reads them from the model metadata (FR-04).
    logger.warning(
        "STUB: the supported-make list is written empty. Issue #35 computes the makes "
        f"with at least {split_params['min_listings_per_make']} listings."
    )
    makes_path = output_dir / "supported_makes.json"
    makes_path.parent.mkdir(parents=True, exist_ok=True)
    makes_path.write_text(json.dumps({"supported_makes": []}, indent=2) + "\n", encoding="utf-8")

    # 3. Whole sellers at a time, so no seller appears in two sets (EDN-14).
    assigned = assign_split(
        remaining[split_params["group_key"]], split_params["ratios"], params["seed"]
    )
    for name in SPLIT_NAMES:
        subset = remaining[assigned == name]
        write_frame(subset, output_dir / f"{name}.parquet", PROCESSED_SCHEMA)
        logger.info(f"{name}: {len(subset):,} rows ({len(subset) / max(len(remaining), 1):.1%}).")


if __name__ == "__main__":
    app()
