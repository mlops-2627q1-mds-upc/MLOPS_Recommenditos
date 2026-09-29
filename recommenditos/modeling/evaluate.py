"""`evaluate` stage: the metrics artefacts and the SC-01 to SC-06 gate.

The course demo has no equivalent - it asserts one threshold inside
`tests/test_model.py`. We need artefacts, because NFR-01 gates deployment on
the success criteria and EDN-12 makes promotion a human decision that someone
has to be able to read a result off.

Two outputs:

- `metrics.json`, the DVC `metrics` file: the gate verdict NFR-01 reads, plus
  the headline numbers of each variant. Deliberately narrow, because
  `dvc metrics show` flattens nested JSON into one column per leaf and
  `dvc metrics diff` is what compares two model versions across a merge.
- `reports/metrics/<variant>.json`, the full record per variant, including
  every criterion with its measured value. This is a plain tracked output, and
  it is where issue #39's per-segment table for the report belongs.

STUB. The point metrics of problem-spec section 6 are computed for real; the
criteria that need machinery no stage has yet report `null` rather than a
fabricated pass, so a reader can tell a missing measurement from a met one.
Issue #39 adds the per-segment breakdown (SC-04), the interval coverage
(SC-05) and the input-masking sweep (SC-06), with tests that verify the metric
implementations against hand-computed values and that a deliberately bad model
fails the gate.
"""

import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from recommenditos.config import (
    METRICS_FILE,
    MODELS_DIR,
    PARAMS_FILE,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from recommenditos.pipeline import load_params, read_frame
from recommenditos.schema import feature_schema

#: Every criterion NFR-01 gates on. A variant is deployable only when all six
#: pass, so a `None` here blocks the gate rather than being ignored.
CRITERIA: tuple[str, ...] = ("sc01", "sc02", "sc03", "sc04", "sc05", "sc06")

#: The "close enough" bands problem-spec section 6 reports a share for.
_CLOSE_ENOUGH_BANDS = {"within_10pct": 0.10, "within_20pct": 0.20}

_PERCENT = 100.0

app = typer.Typer()


def point_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """MdAPE, the +/-10 % and +/-20 % shares, MAE and MAPE (problem-spec 6)."""
    relative = (predicted - actual).abs() / actual
    return {
        "mdape": float(relative.median()),
        **{name: float((relative <= band).mean()) for name, band in _CLOSE_ENOUGH_BANDS.items()},
        "mae_eur": float(((predicted - actual).abs()).mean()),
        "mape": float(relative.mean()),
    }


def evaluate_gate(metrics: dict[str, float], baseline_mdape: float | None, criteria: dict) -> dict:
    """Each success criterion as flat `<id>_measured` / `<id>_passed` keys."""
    mdape = metrics["mdape"]
    within = metrics["within_20pct"]
    improvement = None if not baseline_mdape else (baseline_mdape - mdape) / baseline_mdape
    return {
        "sc01_measured": mdape,
        "sc01_passed": mdape <= criteria["sc01_mdape_max"],
        "sc02_measured": within,
        "sc02_passed": within >= criteria["sc02_within_20pct_min"],
        "sc03_measured": improvement,
        "sc03_passed": None
        if improvement is None
        else improvement >= criteria["sc03_mdape_improvement_over_baseline_min"],
        "sc04_measured": None,
        "sc04_passed": None,
        "sc04_note": "per-segment breakdown arrives with issue #39",
        "sc05_measured": None,
        "sc05_passed": None,
        "sc05_note": "needs the UC2 conformal intervals",
        "sc06_measured": None,
        "sc06_passed": None,
        "sc06_note": "input-masking sweep arrives with issue #39",
    }


def _evaluate_variant(variant: str, params: dict, input_dir: Path, models_dir: Path) -> dict:
    model = json.loads((models_dir / variant / "model.json").read_text(encoding="utf-8"))
    feature_set = model["feature_set"]
    schema = feature_schema(
        params["features"]["sets"][feature_set], name=f"features-{feature_set}"
    )
    test = read_frame(input_dir / feature_set / "test.parquet", schema)
    # The stub model is a constant in log space; #37 replaces it with a fitted
    # estimator and this becomes `model.predict(test[features])`.
    predicted = pd.Series(
        np.exp(np.full(len(test), model["constant_log_price"])), index=test.index
    )
    return {
        "variant": variant,
        # What actually produced these numbers, so a committed metrics file
        # cannot be read as a result while it is still only the stub's.
        "estimator": model["estimator"],
        "planned_estimator": model["planned_estimator"],
        "feature_set": feature_set,
        "n_test_rows": len(test),
        **point_metrics(test["price"], predicted),
    }


@app.command()
def main(
    input_dir: Path = PROCESSED_DATA_DIR / "features",
    models_dir: Path = MODELS_DIR,
    metrics_dir: Path = REPORTS_DIR / "metrics",
    summary_path: Path = METRICS_FILE,
    params_path: Path = PARAMS_FILE,
):
    params = load_params(params_path)
    criteria = params["evaluate"]["success_criteria"]

    logger.warning("STUB: SC-04, SC-05 and SC-06 report null until issue #39 lands.")

    measured = {
        variant: _evaluate_variant(variant, params, input_dir, models_dir)
        for variant in params["train"]["variants"]
    }
    baseline_mdape = measured.get(params["evaluate"]["baseline_variant"], {}).get("mdape")

    metrics_dir.mkdir(parents=True, exist_ok=True)
    headline = {}
    for variant, record in measured.items():
        record.update(evaluate_gate(record, baseline_mdape, criteria))
        path = metrics_dir / f"{variant}.json"
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        logger.info(f"{variant}: MdAPE {record['mdape'] * _PERCENT:.1f} % -> {path}.")
        # Three numbers per variant, so `dvc metrics show` stays a table a
        # person can read and `dvc metrics diff` says something useful.
        headline[variant] = {
            "mdape": record["mdape"],
            "within_20pct": record["within_20pct"],
            "gate_passed": all(record[f"{each}_passed"] for each in CRITERIA),
        }

    passing = [name for name, entry in headline.items() if entry["gate_passed"]]
    best = min(measured.values(), key=lambda record: record["mdape"])
    summary = {
        "gate_passed": bool(passing),
        "n_variants": len(measured),
        "n_variants_passing": len(passing),
        "best_variant": best["variant"],
        "best_mdape": best["mdape"],
        "variants": headline,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.success(f"Gate: {summary['n_variants_passing']}/{summary['n_variants']} variants pass.")


if __name__ == "__main__":
    app()
