"""`train` stage: one run of the experiment ladder, once per variant.

STUB. It fits the simplest thing that is a model - a constant predictor at the
median of the training `log_price` - and saves it in a format `evaluate` can
load. That is enough for the whole pipeline to produce real metrics before any
real estimator exists.

`dvc.yaml` iterates `train.variants` in params.yaml with `foreach`, so the
stages are `train@b0`, `train@b1`, `train@lgbm-basic`, `train@lgbm-extended`
and each declares only its own key under `params:`. Changing one variant's
hyperparameters therefore retrains that variant alone.

Issue #37 implements the four estimators and the MLflow tracking on DagsHub:
one experiment, one run per variant, parameters, metrics and the model artefact
logged. Note the demo's warning that each DVC stage becomes its own MLflow run,
so the runs have to be grouped deliberately.
"""

import json
from pathlib import Path

from loguru import logger
import typer

from recommenditos.config import MODELS_DIR, PARAMS_FILE, PROCESSED_DATA_DIR
from recommenditos.data.build_features import FeatureSpace
from recommenditos.pipeline import load_params, read_frame

app = typer.Typer()


@app.command()
def main(
    variant: str = typer.Argument(..., help="a key of train.variants in params.yaml"),
    input_dir: Path = PROCESSED_DATA_DIR / "features",
    output_dir: Path = MODELS_DIR,
    params_path: Path = PARAMS_FILE,
):
    params = load_params(params_path)
    settings = params["train"]["variants"][variant]
    feature_set = settings["feature_set"]
    # The matrix's columns depend on the data - the equipment multi-hot columns
    # and the category levels are whatever the training rows decided - so the
    # contract travels with the matrices instead of being rebuilt from
    # params.yaml here. `FeatureSpace` because #37 needs the vocabulary too.
    space = FeatureSpace.load(input_dir / feature_set, name=f"features-{feature_set}")
    schema = space.schema

    train = read_frame(input_dir / feature_set / "train.parquet", schema)

    logger.warning(
        f"STUB: {variant!r} is fitted as a constant predictor, not as "
        f"{settings['estimator']!r}. Issue #37 implements the estimators and MLflow."
    )
    model = {
        "variant": variant,
        "estimator": "constant_median",
        "planned_estimator": settings["estimator"],
        "feature_set": feature_set,
        # Features and targets as two lists, not one. The matrix carries the
        # label beside the inputs so that one file per split is enough, which
        # means every consumer of this artefact - #37's estimators, #39's
        # masking sweep, the API - has to be told where the boundary is. One
        # list of `schema.names` ends in `price, log_price`, so a consumer that
        # takes it at its word fits the target on itself.
        "features": list(schema.feature_names),
        "targets": list(schema.target_names),
        "seed": params["seed"],
        "constant_log_price": float(train["log_price"].median()),
        "n_training_rows": len(train),
    }

    model_dir = output_dir / variant
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    logger.success(f"Wrote {model_dir / 'model.json'}.")


if __name__ == "__main__":
    app()
