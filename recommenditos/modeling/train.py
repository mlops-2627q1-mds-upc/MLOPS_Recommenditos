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
from recommenditos.pipeline import load_params, read_frame
from recommenditos.schema import feature_schema

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
    schema = feature_schema(
        params["features"]["sets"][feature_set], name=f"features-{feature_set}"
    )

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
        "features": list(schema.names),
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
