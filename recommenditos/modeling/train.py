"""`train` stage: one run of the experiment ladder, once per variant.

`dvc.yaml` iterates `train.variants` in params.yaml with `foreach`, so the
stages are `train@b0`, `train@b1`, `train@lgbm-basic` and `train@lgbm-extended`,
and each declares only its own key under `params:`. Changing one variant's
hyperparameters therefore retrains that variant alone.

What this module does is everything around the fit: read the matrices and their
contract, check that their rows are the makes the API serves, open one MLflow
run, call `fit_variant`, write the bundle and log the run. The estimators
themselves and the predict seam are in `model.py`, because `evaluate` depends on
that file and must not depend on this one.

Three properties are deliberate and each is tested.

**The rows are the served population (EDN-48).** `split` stays a lossless
partition and records which makes cleared EDN-05's threshold; `features` applies
that list to every frame before it builds the vocabulary (EDN-67), so no make
the API answers with a 422 is in the model's training data or in the encoding
it is fitted in. This stage checks that property on what it reads rather than
filtering again, which would only ever matter on matrices from another run and
there would hide the mismatch. The list travels into the bundle, so the API
reads the scope off the model it is serving.

**One variant is one MLflow run.** This stage creates it and records its id in
`model.json`; `evaluate` resumes that run rather than opening its own, so the
experiment holds four runs, each carrying its hyperparameters, its artefact,
issue #38's energy figures and issue #39's test metrics and gate verdict.

**Training works with no credentials and no network.** `optional_run` degrades
to a handle that logs nothing, because the test suite and a fresh clone have to
be able to train. `RECOMMENDITOS_REQUIRE_TRACKING=1` turns that into a failure
for the runs whose numbers are cited.
"""

from pathlib import Path
import shutil
import time

from loguru import logger
import typer

from recommenditos.config import MODELS_DIR, PARAMS_FILE, PROCESSED_DATA_DIR
from recommenditos.data.build_features import (
    FeatureSpace,
    check_served_makes_only,
    read_supported_makes,
)
from recommenditos.modeling.model import Model, TrainingData, fit_variant, load_model
from recommenditos.pipeline import load_params, read_frame
from recommenditos.tracking import Run, optional_run

#: The split a variant is fitted on, and the one early stopping watches. Named
#: rather than inlined because the pair is the whole of what a fit may read: the
#: `calibration` split is held for the conformal intervals of UC2 and `test` for
#: `evaluate`, and a stage that reads either would invalidate both.
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"

#: Where the bundle lands inside the MLflow run, so the UI shows `model/` rather
#: than four loose files at the run's root.
ARTIFACT_PATH = "model"

app = typer.Typer()


def _fit(
    variant: str, settings: dict, data: TrainingData, *, seed: int, num_threads: int
) -> tuple[Model, float]:
    """The measured section: the fit, and nothing else.

    TODO(#38): wrap exactly this call with CodeCarbon's `EmissionsTracker` and
    return the emissions alongside the model. Nothing else in this module
    allocates compute worth measuring, which is what makes "only the fit is
    measured" true rather than approximate, and the two log calls in `main` that
    the energy metrics and the hardware parameters belong to are marked there.
    """
    started = time.perf_counter()
    model = fit_variant(variant, settings, data, seed=seed, num_threads=num_threads)
    return model, time.perf_counter() - started


def read_matrices(
    input_dir: Path, feature_set: str, supported_makes: tuple[str, ...]
) -> TrainingData:
    """The two matrices a fit may read, checked to hold only the supported makes.

    The contract comes from the artefact beside the matrices rather than from
    `features.sets`: the equipment multi-hot columns and every category's levels
    are whatever the training rows decided, so a contract rebuilt from
    params.yaml would not describe the file it is validating.

    Why the model is fitted on the supported makes at all (FR-04, EDN-05,
    EDN-48): fitting on a make the product will not serve does not inflate a
    metric - rare makes are harder, so a pooled figure over them is if anything
    pessimistic - but it makes the reported population a different one from the
    served population, and a number about cars nobody can ask about is not a
    number about the product. It would also loosen the one external check the
    numbers have: problem-spec section 8's reference values come from a run
    whose documented scope is section 2's, supported makes included. That run's
    code is not in the repository and it split 80/20 rather than 60/10/10/20, so
    the model card reads the agreement with it as a cross-check, not as a
    like-for-like comparison.

    `features` applies the list (EDN-67); this function only checks it, for both
    frames, because early stopping watches the validation split and a validation
    set holding makes the API refuses would stop the fit on one population and
    report a number about another.
    """
    space = FeatureSpace.load(input_dir / feature_set, name=f"features-{feature_set}")
    frames = {}
    for split in (TRAIN_SPLIT, VALIDATION_SPLIT):
        frames[split] = read_frame(input_dir / feature_set / f"{split}.parquet", space.schema)
        check_served_makes_only(frames[split], supported_makes, name=split)
    return TrainingData(
        space=space,
        train=frames[TRAIN_SPLIT],
        validation=frames[VALIDATION_SPLIT],
        supported_makes=supported_makes,
    )


def _logged_params(
    variant: str, settings: dict, data: TrainingData, *, seed: int, num_threads: int
) -> dict:
    """What MLflow records about how this run was configured (NFR-14).

    Every params.yaml key `dvc.yaml` declares for this stage but one, plus the
    variant's name and the shape of the data the keys were applied to. A key is
    logged under its own name, without the `train.variants.<variant>.` path that
    differs between variants, so the runs of the ladder share their columns, and
    a hyperparameter under its estimator's name, so LightGBM's `num_leaves` and
    the Ridge's `alpha` never share one. The exception is `train.mlflow_experiment`,
    which is recorded as what it is, the experiment the run belongs to, rather
    than repeated as a parameter of every run in it.

    `num_threads` is logged although the trees do not depend on it, because
    LightGBM writes it into `booster.txt` (EDN-53), so it changes the artefact
    and DVC reruns the stage for it.

    The stage logs its own parameters because only it knows which keys it
    declared; the commit and the data version are the same question for every
    run and `tracked_run` answers them.
    `tests/test_model.py::test_a_run_records_every_parameter_its_stage_declares`
    reads the declared keys out of `dvc.yaml` and compares them with a real run.
    """
    return {
        "variant": variant,
        "estimator": settings["estimator"],
        "feature_set": settings["feature_set"],
        "seed": seed,
        "num_threads": num_threads,
        "n_features": len(data.space.schema.feature_names),
        "n_train_rows": len(data.train),
        "n_validation_rows": len(data.validation),
        "n_supported_makes": len(data.supported_makes),
        **{f"{settings['estimator']}.{key}": value for key, value in settings["params"].items()},
    }


def _log_the_run(
    run: Run, model: Model, *, fit_seconds: float, params: dict, directory: Path
) -> None:
    """Everything this stage tells MLflow about the run it just finished.

    No metric in euros. `train` must not touch the test set, and a train or
    validation MdAPE would be a second implementation of the metric beside
    `evaluate.point_metrics`, so one run could carry two numbers that disagree.
    The absolute error in log space is what early stopping reads, so it is the
    honest thing for this stage to report.
    """
    training = model.metadata["training"]
    run.set_tags(
        {
            "variant": model.variant,
            "estimator": model.estimator,
            "feature_set": model.feature_set,
            "dvc_stage": f"train@{model.variant}",
        }
    )
    run.log_params(params)
    run.log_metrics(
        {
            "train_l1_log_price": training["train_l1_log_price"],
            "validation_l1_log_price": training["validation_l1_log_price"],
            "best_iteration": training.get("best_iteration"),
            "fit_seconds": fit_seconds,
            # TODO(#38): the energy metrics of the fit belong in this call.
        }
    )
    run.log_artifacts(directory, artifact_path=ARTIFACT_PATH)


@app.command()
def main(
    variant: str = typer.Argument(..., help="a key of train.variants in params.yaml"),
    input_dir: Path = PROCESSED_DATA_DIR / "features",
    output_dir: Path = MODELS_DIR,
    params_path: Path = PARAMS_FILE,
):
    params = load_params(params_path)
    settings = params["train"]["variants"][variant]
    seed = params["seed"]
    num_threads = params["train"]["num_threads"]

    # Beside the split frames rather than beside the matrices, because that is
    # where `split` writes it and `features` reads it. Derived from `input_dir`
    # so that a test - and a `dvc repro` of a stage whose paths were redirected -
    # cannot read the make list of one run against the matrices of another.
    supported_makes = read_supported_makes(input_dir.parent)
    data = read_matrices(input_dir, settings["feature_set"], supported_makes)

    with optional_run(params["train"]["mlflow_experiment"], variant) as run:
        model, fit_seconds = _fit(variant, settings, data, seed=seed, num_threads=num_threads)
        model.metadata["mlflow"] = {
            "experiment": params["train"]["mlflow_experiment"],
            "tracking_mode": run.mode,
            # What `evaluate` resumes, so the test metrics and the gate verdict
            # land in the run that already holds the hyperparameters.
            "run_id": run.run_id,
        }
        directory = output_dir / variant
        _replace_directory(directory)
        model.save(directory)
        logged = _logged_params(variant, settings, data, seed=seed, num_threads=num_threads)
        # TODO(#38): the hardware parameters of the measurement belong in `logged`.
        _log_the_run(run, model, fit_seconds=fit_seconds, params=logged, directory=directory)

    training = model.metadata["training"]
    logger.success(
        f"{variant}: fitted {settings['estimator']!r} on {training['n_train_rows']:,} rows in "
        f"{fit_seconds:.2f} s, validation L1(log price) "
        f"{training['validation_l1_log_price']:.4f}."
    )
    # The load, not the fitted object: what `evaluate` and the API will use is
    # what came off the disk, and a bundle that cannot be read back is a failed
    # training run even though the fit succeeded.
    load_model(directory)


def _replace_directory(directory: Path) -> None:
    """Start the bundle empty, so no file of a previous fit survives into it.

    `dvc repro` removes a stage's outputs before running it, but `train.main` is
    also called directly - by the tests, and by anyone re-running one variant by
    hand - and then a variant switched from `lightgbm` to `ridge` would leave its
    `booster.txt` beside the new `pipeline.joblib`. `load_model` dispatches on the
    record, so it would ignore the stale file; DVC would not, and the directory's
    hash would depend on which fits a machine happened to have run.
    """
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)


if __name__ == "__main__":
    app()
