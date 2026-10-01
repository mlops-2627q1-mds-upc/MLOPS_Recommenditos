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

The stage reports on the test rows the API would answer, which is the makes the
model was fitted on (EDN-48). It takes that list out of the model's own metadata
rather than reading `supported_makes.json` again, so the evaluated population is
by construction the trained population.

Each variant's metrics are appended to the MLflow run `train@<variant>` created,
rather than logged to a run of this stage's own. One run per variant then holds
the hyperparameters, the artefact, the energy figures and the verdict, which is
what makes the four comparable in one table.

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
import pandas as pd
import typer

from recommenditos.config import (
    METRICS_FILE,
    MODELS_DIR,
    PARAMS_FILE,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from recommenditos.data.build_features import FeatureSpace
from recommenditos.modeling.model import Model, load_model
from recommenditos.pipeline import load_params, read_frame
from recommenditos.tracking import resume_run

#: Every criterion NFR-01 gates on. A variant is deployable only when all six
#: pass, so a `None` here blocks the gate rather than being ignored.
CRITERIA: tuple[str, ...] = ("sc01", "sc02", "sc03", "sc04", "sc05", "sc06")

#: The "close enough" bands problem-spec section 6 reports a share for.
_CLOSE_ENOUGH_BANDS = {"within_10pct": 0.10, "within_20pct": 0.20}

_PERCENT = 100.0

app = typer.Typer()


def point_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """MdAPE, the +/-10 % and +/-20 % shares, MAE and MAPE (problem-spec 6).

    The denominator is the absolute actual price: with a signed one a negative
    price would make a 10 % error read as a perfect prediction. Prices cannot
    be negative once the range filter runs, but this function is also what the
    API and the drift job will use, on data no filter has seen.
    """
    if actual.empty:
        raise ValueError("cannot compute metrics on an empty set")
    if (actual == 0).any():
        raise ValueError("cannot compute a percentage error against a price of 0")

    relative = (predicted - actual).abs() / actual.abs()
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
    # `not baseline_mdape` would read a perfect baseline as a missing one.
    improvement = (
        None
        if baseline_mdape is None or baseline_mdape == 0
        else (baseline_mdape - mdape) / baseline_mdape
    )
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


def _evaluate_variant(variant: str, input_dir: Path, models_dir: Path) -> dict:
    model = load_model(models_dir / variant)
    feature_set = model.feature_set
    # The contract travels with the matrices, because the equipment multi-hot
    # columns and the category levels are whatever the training rows decided.
    # `FeatureSpace` because #39's masking sweep rebuilds a frame per masked
    # field and has to cast it back against the same levels.
    space = FeatureSpace.load(input_dir / feature_set, name=f"features-{feature_set}")
    test = read_frame(input_dir / feature_set / "test.parquet", space.schema)
    test = _supported_only(test, model)
    return {
        "variant": variant,
        # What actually produced these numbers, so a committed metrics file
        # cannot be read as a result of something it was not.
        "estimator": model.estimator,
        "feature_set": feature_set,
        "n_test_rows": len(test),
        "mlflow_run_id": model.metadata["mlflow"]["run_id"],
        **point_metrics(test["price"], model.predict_eur(test)),
    }


def _supported_only(test: pd.DataFrame, model: Model) -> pd.DataFrame:
    """The test rows the API would answer, which is the population to report on.

    Taken from the model's own metadata rather than from `supported_makes.json`,
    so the evaluated population is by construction the one the model was fitted
    on (EDN-48) and the two cannot be read from different `split` runs.

    TODO(#39): the per-make segments of SC-04 need the same restriction, or the
    criterion reports a segment the API answers with a 422.
    """
    supported = model.metadata["training"]["supported_makes"]
    kept = test[test["make"].isin(supported)]
    logger.info(
        f"{model.variant}: {len(kept):,} of {len(test):,} test rows are one of the "
        f"{len(supported)} supported make(s)."
    )
    return kept


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
        variant: _evaluate_variant(variant, input_dir, models_dir)
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
        # Appended to the run `train@<variant>` opened, not a run of its own, so
        # the experiment holds one run per variant carrying its hyperparameters,
        # its artefact and its verdict. A context manager per variant because
        # this stage covers all four in one process, and a run left open would
        # collect the next variant's metrics.
        with resume_run(record["mlflow_run_id"]) as run:
            run.log_metrics(
                {
                    "mdape": record["mdape"],
                    "within_10pct": record["within_10pct"],
                    "within_20pct": record["within_20pct"],
                    "mae_eur": record["mae_eur"],
                    "mape": record["mape"],
                    **{f"{each}_measured": record[f"{each}_measured"] for each in CRITERIA},
                }
            )
            run.set_tags({"gate_passed": str(all(record[f"{c}_passed"] for c in CRITERIA))})
        # Three numbers per variant, so `dvc metrics show` stays a table a
        # person can read and `dvc metrics diff` says something useful.
        headline[variant] = {
            "mdape": record["mdape"],
            "within_20pct": record["within_20pct"],
            "gate_passed": all(record[f"{each}_passed"] for each in CRITERIA),
        }

    passing = [name for name, entry in headline.items() if entry["gate_passed"]]
    best = min(measured.values(), key=lambda record: record["mdape"])
    # NFR-01 gates the deployment of *a model*, so the variant this artefact
    # puts forward has to be one that met all six criteria. The lowest MdAPE
    # overall is reported separately, because it is what a reader looks for
    # first and it would be confusing to omit it.
    deployable = min(
        (measured[name] for name in passing), key=lambda record: record["mdape"], default=None
    )
    summary = {
        "gate_passed": bool(passing),
        "n_variants": len(measured),
        "n_variants_passing": len(passing),
        "deployable_variant": None if deployable is None else deployable["variant"],
        "best_variant": best["variant"],
        "best_mdape": best["mdape"],
        "variants": headline,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.success(f"Gate: {summary['n_variants_passing']}/{summary['n_variants']} variants pass.")


if __name__ == "__main__":
    app()
