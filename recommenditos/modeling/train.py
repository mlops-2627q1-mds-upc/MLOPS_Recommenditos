"""`train` stage: one run of the experiment ladder, once per variant.

`dvc.yaml` iterates `train.variants` in params.yaml with `foreach`, so the
stages are `train@b0`, `train@b1`, `train@lgbm-basic` and `train@lgbm-extended`,
and each declares only its own key under `params:`. Changing one variant's
hyperparameters therefore retrains that variant alone.

What this module does is everything around the fit: read the matrices and their
contract, check that their rows are the makes the API serves, open one MLflow
run, call `fit_with_intervals`, write the bundle and log the run. The estimators
themselves and the predict seam are in `model.py`, because `evaluate` depends on
that file and must not depend on this one.

Four properties are deliberate and each is tested.

**The rows are the served population (EDN-05).** `split` stays a lossless
partition and records which makes cleared EDN-05's threshold; `features` applies
that list to every frame before it builds the vocabulary, so no make
the API answers with a 422 is in the model's training data or in the encoding
it is fitted in. This stage checks that property on what it reads rather than
filtering again, which would only ever matter on matrices from another run and
there would hide the mismatch. The list travels into the bundle, so the API
reads the scope off the model it is serving.

**One variant is one MLflow run.** This stage creates it and records its id in
`model.json`; `evaluate` resumes that run rather than opening its own, so the
experiment holds four runs, each carrying its hyperparameters, its artefact,
the energy and emissions of its fit and issue #39's test metrics and gate
verdict.

**Only the fit is measured (NFR-10, issue #38).** `energy.measure` wraps
`fit_with_intervals` and nothing else, and CodeCarbon's row for that fit is written to
`reports/emissions/<variant>.csv`. What the figure is - an estimate from the
CPU's TDP and the process's CPU time, not a meter reading, on every machine we
train on - is `recommenditos/modeling/energy.py`'s to say, and the model card says why
it is measured the way it is.

**The price range is fitted with its point model (EDN-85).** A variant whose
block names `intervals` also fits UC2's quantile models and calibrates them with
MAPIE on the `calibration` split (`intervals.py`), inside the same measured call
and the same run, and `calibration` is read only for such a variant.

**Training works with no credentials and no network.** `optional_run` degrades
to a handle that logs nothing, because the test suite and a fresh clone have to
be able to train. `RECOMMENDITOS_REQUIRE_TRACKING=1` turns that into a failure
for the runs whose numbers are cited.
"""

import os

# OpenBLAS reads its kernel from this variable once, when numpy loads it, so
# this comes before any import that loads numpy. `check_openblas_kernel` in
# model.py says why `b1` needs it and fails the fit if it did not take effect.
# A value the caller set wins.
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")

from dataclasses import replace
from pathlib import Path
import shutil

from loguru import logger
import pandas as pd
import typer

from recommenditos.config import MODELS_DIR, PARAMS_FILE, PROCESSED_DATA_DIR
from recommenditos.data.build_features import (
    FeatureSpace,
    check_served_makes_only,
    read_supported_makes,
)
from recommenditos.modeling.energy import (
    EMISSIONS_DIR,
    EnergySettings,
    FitMeasurement,
    measure,
)
from recommenditos.modeling.intervals import CALIBRATION_SPLIT, fit_with_intervals
from recommenditos.modeling.model import (
    ESTIMATORS,
    Model,
    RequestContract,
    TrainingData,
    blanked_copy_counts,
    interval_settings,
    load_model,
)
from recommenditos.pipeline import load_params, read_frame
from recommenditos.tracking import Run, optional_run

#: The split a variant is fitted on, and the one early stopping watches. Named
#: rather than inlined because, with `calibration`, they are the whole of what a
#: fit may read: `calibration` is read only to calibrate the conformal intervals
#: of UC2 (`read_calibration`) and `test` is `evaluate`'s alone.
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"

#: Where the bundle lands inside the MLflow run, so the UI shows `model/` rather
#: than four loose files at the run's root.
ARTIFACT_PATH = "model"

app = typer.Typer()


def read_matrices(
    input_dir: Path,
    feature_set: str,
    supported_makes: tuple[str, ...],
    *,
    request: RequestContract | None = None,
) -> TrainingData:
    """The two matrices a fit may read, checked to hold only the supported makes.

    `request` travels with them, untouched: it is what tells a variant that adds
    blanked copies of its training rows which fields a copy may omit (EDN-82),
    and a caller that fits no such variant may leave it out.

    The contract comes from the artefact beside the matrices rather than from
    `features.sets`: the equipment multi-hot columns and every category's levels
    are whatever the training rows decided, so a contract rebuilt from
    params.yaml would not describe the file it is validating.

    Why the model is fitted on the supported makes at all (FR-04, EDN-05):
    fitting on a make the product will not serve does not inflate a
    metric - rare makes are harder, so a pooled figure over them is if anything
    pessimistic - but it makes the reported population a different one from the
    served population, and a number about cars nobody can ask about is not a
    number about the product. It would also loosen the one external check the
    numbers have: problem-spec section 8's reference values come from a run
    whose documented scope is section 2's, supported makes included. That run's
    code is not in the repository and it split 80/20 rather than 60/10/10/20, so
    the model card reads the agreement with it as a cross-check, not as a
    like-for-like comparison.

    `features` applies the list; this function only checks it, for both
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
        request=request,
    )


def read_calibration(
    input_dir: Path, feature_set: str, supported_makes: tuple[str, ...]
) -> pd.DataFrame:
    """The `calibration` matrix, checked like the other two, for the interval calibration.

    Its own function, called only for a variant that names `intervals`, so that
    no point model and no early stopping can come to read it.
    """
    space = FeatureSpace.load(input_dir / feature_set, name=f"features-{feature_set}")
    frame = read_frame(input_dir / feature_set / f"{CALIBRATION_SPLIT}.parquet", space.schema)
    check_served_makes_only(frame, supported_makes, name=CALIBRATION_SPLIT)
    return frame


def logged_params(variant: str, data: TrainingData, params: dict) -> dict:
    """What MLflow records about how this run was configured (NFR-14).

    Every params.yaml key `dvc.yaml` declares for this stage but one, plus the
    variant's name and the shape of the data the keys were applied to. A key is
    logged under its own name, without the `train.variants.<variant>.` path that
    differs between variants, so the runs of the ladder share their columns, and
    a hyperparameter under its estimator's name, so LightGBM's `num_leaves` and
    the Ridge's `alpha` never share one. The exception is `train.mlflow_experiment`,
    which is recorded as what it is, the experiment the run belongs to, rather
    than repeated as a parameter of every run in it.

    `n_train_rows` is the listings the variant was fitted on. A variant that adds
    blanked copies of them (EDN-82) logs how many under `blanked_copies.*`, and
    the rows its estimator saw are `n_fit_rows` in `model.json`.

    `num_threads` is logged although the trees do not depend on it, because
    LightGBM writes it into `booster.txt`, so it changes the artefact
    and DVC reruns the stage for it. `train.energy` is logged as `energy.*`, as it
    stands in params.yaml, because it decides what the run's energy figure means.

    The stage logs its own parameters because only it knows which keys it
    declared; the commit and the hashes of the stage's inputs are the same
    question for every run, and `tracked_run` answers them
    (`recommenditos/provenance.py`).
    `tests/test_model.py::test_a_run_records_every_parameter_its_stage_declares`
    reads the declared keys out of `dvc.yaml` and compares them with a real run.

    Public because the hyperparameter search (`recommenditos/modeling/tune.py`)
    logs each of its fits under the same names, from a copy of `params` with the
    point's values in it, so a search run and a pipeline run share their columns.
    """
    settings = params["train"]["variants"][variant]
    request = RequestContract.from_params(params)
    return {
        "variant": variant,
        "estimator": settings["estimator"],
        "feature_set": settings["feature_set"],
        "seed": params["seed"],
        "num_threads": params["train"]["num_threads"],
        "n_features": len(data.space.schema.feature_names),
        "n_train_rows": len(data.train),
        "n_validation_rows": len(data.validation),
        "n_supported_makes": len(data.supported_makes),
        # The copies of its training rows the fit added (EDN-82), at zero for a
        # variant that names none, so every run of the ladder says which it is.
        **{
            f"blanked_copies.{kind}": count
            for kind, count in blanked_copy_counts(variant, settings).items()
        },
        # The request contract, under the keys `dvc.yaml` declares it by. It
        # decides which fields a blanked copy omits, so it can change the model.
        "request.required_input_fields": list(request.required_fields),
        "request.price_range_required_input_fields": list(request.price_range_required_fields),
        **{
            f"request.input_field_to_feature.{field}": feature
            for field, feature in request.field_to_feature.items()
        },
        # The optional keys at the value the fit used, so a run whose params block
        # leaves `min_child_samples` out still records the 20 it was fitted with,
        # and a pipeline run lines up column for column with a search run (EDN-73).
        **{
            f"{settings['estimator']}.{key}": value
            for key, value in {
                **ESTIMATORS[settings["estimator"]].optional_hyperparameters,
                **settings["params"],
            }.items()
        },
        **{f"energy.{key}": value for key, value in params["train"]["energy"].items()},
    }


def interval_params(variant: str, params: dict) -> dict:
    """The `intervals` block of a variant as MLflow parameters, empty when it names none.

    Apart from `logged_params` because the hyperparameter search logs its fits
    through that function too, and a search fit is the point model alone: it
    fits no quantile model and must not carry the settings of one.
    """
    chosen = interval_settings(variant, params["train"]["variants"][variant])
    if chosen is None:
        return {}
    return {
        "intervals.confidence_level": chosen["confidence_level"],
        **{
            f"intervals.blanked_copies.{kind}": count
            for kind, count in chosen["blanked_copies"].items()
        },
    }


def interval_metrics(model: Model) -> dict:
    """The calibration of the price range as MLflow metrics, empty without one.

    The two offsets MAPIE computed, in log price, the trees each quantile model
    kept, the calibration rows they were computed on, and the seconds the price
    range took of the measured fit. `evaluate` adds the
    test coverage to the same run.
    """
    record = model.metadata.get("intervals")
    if not record:
        return {}
    return {
        "interval_offset_lower": record["offsets"]["lower"],
        "interval_offset_upper": record["offsets"]["upper"],
        **{
            f"interval_best_iteration_{name}": float(trees)
            for name, trees in record["best_iteration"].items()
        },
        "n_calibration_listings": float(record["calibration"]["n_listings"]),
        "n_calibration_rows": float(record["calibration"]["n_rows"]),
        "interval_fit_seconds": record["seconds"],
    }


def _log_the_run(
    run: Run, model: Model, *, cost: FitMeasurement, params: dict, directory: Path
) -> None:
    """Everything this stage tells MLflow about the run it just finished.

    The fit's record, and the bundle under `model/`. The bundle is this stage's
    alone: the hyperparameter search logs its fits through `log_fit` too, and
    uploads none, because a point that is not adopted is never served.
    """
    log_fit(run, model, cost=cost, params=params, tags={"dvc_stage": f"train@{model.variant}"})
    metrics = interval_metrics(model)
    if metrics:
        run.log_metrics(metrics)
    run.log_artifacts(directory, artifact_path=ARTIFACT_PATH)


def log_fit(
    run: Run, model: Model, *, cost: FitMeasurement, params: dict, tags: dict | None = None
) -> None:
    """The tags, parameters and metrics of one measured fit, without its bundle.

    No metric in euros. `train` must not touch the test set, and a train or
    validation MdAPE would be a second implementation of the metric beside
    `evaluate.point_metrics`, so one run could carry two numbers that disagree.
    The absolute error in log space is what early stopping reads, so it is the
    honest thing for this stage to report.

    The energy figures are metrics of this run and the way they were produced is
    its parameters, so `evaluate`'s test metrics land beside them in the same row
    of the experiment's table (EDN-55).
    """
    training = model.metadata["training"]
    early_stopped = training.get("early_stopped")
    run.set_tags(
        {
            "variant": model.variant,
            "estimator": model.estimator,
            "feature_set": model.feature_set,
            **(tags or {}),
        }
    )
    run.log_params(params)
    run.log_metrics(
        {
            "train_l1_log_price": training["train_l1_log_price"],
            "validation_l1_log_price": training["validation_l1_log_price"],
            "best_iteration": training.get("best_iteration"),
            "boosting_rounds": training.get("boosting_rounds"),
            # 1 or 0 rather than a tag, so `metrics.early_stopped = 0` finds every
            # run whose `n_estimators` budget ended the fit, beside the other metrics.
            "early_stopped": None if early_stopped is None else float(early_stopped),
            **cost.metrics(),
        }
    )


@app.command()
def main(
    variant: str = typer.Argument(..., help="a key of train.variants in params.yaml"),
    input_dir: Path = PROCESSED_DATA_DIR / "features",
    output_dir: Path = MODELS_DIR,
    params_path: Path = PARAMS_FILE,
    energy_dir: Path = EMISSIONS_DIR,
):
    params = load_params(params_path)
    settings = params["train"]["variants"][variant]
    seed = params["seed"]
    num_threads = params["train"]["num_threads"]
    # Before the matrices are read, so a misconfigured measurement fails the
    # stage before any work is done rather than after the fit it was to measure.
    energy = EnergySettings.from_params(params["train"]["energy"])

    # Beside the split frames rather than beside the matrices, because that is
    # where `split` writes it and `features` reads it. Derived from `input_dir`
    # so that a test - and a `dvc repro` of a stage whose paths were redirected -
    # cannot read the make list of one run against the matrices of another.
    supported_makes = read_supported_makes(input_dir.parent)
    # The contract goes with the matrices whether or not this variant adds blanked
    # copies of its training rows (EDN-82): which fields a copy may omit is FR-01's
    # to say, and a fit that needs to know and was not told refuses to guess.
    data = read_matrices(
        input_dir,
        settings["feature_set"],
        supported_makes,
        request=RequestContract.from_params(params),
    )
    # Only a variant with a price range reads the calibration split (EDN-85).
    if interval_settings(variant, settings) is not None:
        data = replace(
            data,
            calibration=read_calibration(input_dir, settings["feature_set"], supported_makes),
        )

    with optional_run(params["train"]["mlflow_experiment"], variant) as run:
        # The measured section: the fit, and nothing else. `fit_with_intervals` is
        # the one call in this module that allocates compute worth measuring: the
        # point model, and for a variant with a price range its quantile models and
        # their calibration. That is what makes "only the fit is measured" true
        # rather than approximate - the
        # matrices are read above, and the MLflow round trips and the bundle are
        # written below, all outside the tracker. One expression, so nothing can
        # drift into the measured section by being written on the next line.
        model, cost = measure(
            lambda: fit_with_intervals(
                variant, settings, data, seed=seed, num_threads=num_threads
            ),
            name=variant,
            settings=energy,
            output_file=energy_dir / f"{variant}.csv",
        )
        # What only this stage knows about the fit, kept with the model the way
        # the MLflow run id is. The run id is the key into the emissions
        # CSV, which is how the energy comparison joins a figure to the model it
        # was measured on instead of trusting that the two files are from one run.
        model.metadata["fit"] = {
            "seconds": cost.fit_seconds,
            "cpu_seconds": cost.fit_cpu_seconds,
            "codecarbon_run_id": cost.run_id,
            "cpu_power_method": cost.cpu_power_method,
            "cpu_tdp_w": cost.cpu_tdp_w,
        }
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
        logged = {
            **logged_params(variant, data, params),
            **interval_params(variant, params),
            **cost.params(),
        }
        _log_the_run(run, model, cost=cost, params=logged, directory=directory)

    training = model.metadata["training"]
    copies = training["n_fit_rows"] - training["n_train_rows"]
    logger.success(
        f"{variant}: fitted {settings['estimator']!r} on {training['n_train_rows']:,} rows"
        + (f" and {copies:,} blanked copies of them" if copies else "")
        + f" in {cost.fit_seconds:.2f} s, validation L1(log price) "
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
