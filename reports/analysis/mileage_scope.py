"""What dropping the listings outside the accepted mileage range would change (#25).

The interim suite's mileage rule fails on the real snapshot: three cleaned
listings carry an odometer reading above `validate.mileage_max_km`, the bound
FR-03 refuses at serving time. One way to make the rule hard is a preprocessing
rule that drops them, as EDN-22 does for registration dates. This measures what
that would cost, by running the real `split`, `features`, `train` and `evaluate`
stages on the interim frame without those rows, in a temporary directory, and
comparing the result with the committed `metrics.json`.

Run from the repository root after `dvc pull`, with tracking off:

    MLFLOW_TRACKING_URI= uv run python reports/analysis/mileage_scope.py \
        > reports/analysis/mileage_scope_results.txt
"""

import json
from pathlib import Path
import tempfile

import pandas as pd

from recommenditos.config import INTERIM_DATA_DIR, METRICS_FILE, PARAMS_FILE, PROCESSED_DATA_DIR
from recommenditos.data import build_features, split_data
from recommenditos.data.split_data import assign_split
from recommenditos.modeling import evaluate, train
from recommenditos.pipeline import load_params

params = load_params(PARAMS_FILE)
bounds = params["validate"]
interim = pd.read_parquet(INTERIM_DATA_DIR / "listings.parquet")
mileage = interim["mileage_km_raw"]
outside = mileage.notna() & ~mileage.between(bounds["mileage_min_km"], bounds["mileage_max_km"])
supported = {
    entry["make"]
    for entry in json.loads((PROCESSED_DATA_DIR / "supported_makes.json").read_text())[
        "supported_makes"
    ]
}

print(f"interim rows: {len(interim):,}, outside {bounds['mileage_min_km']:,} to "
      f"{bounds['mileage_max_km']:,} km: {int(outside.sum())}")
rows = interim.loc[outside, ["make", "model", "registration_date", "mileage_km_raw", "price",
                              "country_code", "seller_group_id"]]
rows = rows.assign(
    split=assign_split(rows["seller_group_id"], params["split"]["ratios"], params["seed"]).values,
    supported_make=rows["make"].isin(supported).values,
).drop(columns="seller_group_id")
print(rows.to_string())

with tempfile.TemporaryDirectory() as scratch:
    root = Path(scratch)
    kept = root / "interim.parquet"
    interim.loc[~outside].to_parquet(kept, index=False)
    processed = root / "processed"
    split_data.main(kept, processed, PARAMS_FILE)
    for feature_set in params["features"]["sets"]:
        build_features.main(feature_set, processed, processed / "features", PARAMS_FILE)
    for variant in params["train"]["variants"]:
        train.main(variant, processed / "features", root / "models", PARAMS_FILE)
    evaluate.main(processed / "features", root / "models", root / "metrics",
                  root / "metrics.json", PARAMS_FILE)
    after = json.loads((root / "metrics.json").read_text())

before = json.loads(METRICS_FILE.read_text())
print("\nvariant         MdAPE before  MdAPE after   within20 before  within20 after  gate")
for name, entry in before["variants"].items():
    new = after["variants"][name]
    print(f"{name:<15} {entry['mdape']:.6f}      {new['mdape']:.6f}      "
          f"{entry['within_20pct']:.6f}         {new['within_20pct']:.6f}        "
          f"{entry['gate_blocked_by']} -> {new['gate_blocked_by']}")
