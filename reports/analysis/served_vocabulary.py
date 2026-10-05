"""The feature space over every training row against over the served training rows (EDN-67).

Issue #63 moves the supported-make filter in front of the vocabulary: the `features` stage now
restricts every frame to the makes `split` lists before it decides which levels and columns exist.
This script measures what that changes, by building the vocabulary both ways from the same
`split` output with the stage's own `build_vocabulary`, so its numbers cannot drift from what the
stage does:

- the `make` and `model` levels, and every other categorical's;
- the equipment items that clear `features.equipment_min_frequency`, against the distinct items
  the rows hold at all;
- the `model_version` levels above the floor and the share of training rows they cover, at the
  configured floor and at 0.001, the two values `params.yaml` quotes;
- the rows each frame keeps.

It reads the `split` artefacts under `data/processed/`, not the raw CSV, so it needs `dvc pull`
(or a `dvc repro` up to `split`) and nothing else. Like `split_gate.py` it imports the stage, so
run it from the repository root. The `features` block it measures is read from `params.yaml` and
printed, so the output says which values it was measured at.
"""

from collections import Counter
import json
from pathlib import Path

import pandas as pd

from recommenditos.data.build_features import (
    FEATURE_INPUTS,
    _items,
    build_vocabulary,
    model_version,
    read_supported_makes,
)
from recommenditos.pipeline import load_params, read_frame
from recommenditos.schema import PROCESSED_SCHEMA

PROCESSED = Path("data/processed")
ALTERNATIVE_TRIM_FLOOR = 0.001

params = load_params(Path("params.yaml"))
feature_params = params["features"]
supported = read_supported_makes(PROCESSED)
columns = feature_params["sets"]["extended"]
print("features block:", json.dumps({k: v for k, v in feature_params.items() if k != "sets"}))
print(f"supported makes ({len(supported)}): {', '.join(supported)}")

print("\nrows per frame, all against served")
for name in FEATURE_INPUTS:
    makes = pd.read_parquet(PROCESSED / f"{name}.parquet", columns=["make"])["make"]
    kept = int(makes.isin(supported).sum())
    print(f"  {name:12s} {len(makes):>7,} -> {kept:>7,}  ({len(makes) - kept:,} removed)")

train = read_frame(PROCESSED / "train.parquet", PROCESSED_SCHEMA)
populations = {"all training rows": train, "served training rows": train[train["make"].isin(supported)]}

for label, frame in populations.items():
    print(f"\n== {label}: {len(frame):,}")
    vocabulary = build_vocabulary(frame, columns, feature_params)
    print("  categorical levels:", {column: len(levels) for column, levels in vocabulary.categories.items()})
    print("  make levels:", ", ".join(vocabulary.categories["make"]))
    total_items = 0
    for source, items in vocabulary.equipment.items():
        distinct: Counter[str] = Counter()
        for value in frame[source].dropna():
            distinct.update(set(_items(value)))
        total_items += len(distinct)
        print(f"  {source}: {len(items)} of {len(distinct)} items get a column")
    print(f"  equipment columns: {vocabulary.n_equipment_features} of {total_items} items")
    trims = model_version(frame["model_version"], tokens=feature_params["model_version_tokens"])
    for floor in (feature_params["model_version_min_frequency"], ALTERNATIVE_TRIM_FLOOR):
        counts = trims.value_counts()
        levels = counts[counts > floor * len(trims)]
        print(
            f"  model_version at floor {floor}: {len(levels)} levels covering "
            f"{trims.isin(levels.index).mean():.1%} of the rows"
        )
    # The extended set's columns, less the four equipment lists, plus the multi-hot columns built
    # from them, plus the two targets every matrix carries (`price` and `log_price`).
    print(f"  extended matrix columns: {len(columns) - len(vocabulary.equipment) + vocabulary.n_equipment_features + 2}")
