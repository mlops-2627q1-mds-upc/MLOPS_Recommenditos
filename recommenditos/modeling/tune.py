"""The hyperparameter search: a grid of measured, tracked fits, decided on validation (EDN-78).

The decision step of the protocol in `docs/docs/pipeline.md`, "Tuning a
hyperparameter", which EDN-73 wrote down and EDN-78 extends. A sweep is one
variant and a small grid, and it ends in one of two verdicts: keep the
reference configuration, or adopt one point of the grid. What it never does is
change `params.yaml`: adopting a value is a reviewed edit followed by a
`dvc repro`, which is what reads the test split, once, for the configuration
that was chosen.

**What a sweep fits.** Every point of the grid, plus the reference - the
variant as `params.yaml` has it, with `--fix` applied - whether or not the grid
names it, because every difference is taken against it. Each fit goes through
the `train` stage's own `read_matrices` and `fit_variant`, so it is the stage's
fit, with one change: `n_estimators` is raised to a ceiling far past the curve,
20,000 by default as in EDN-73, so early stopping decides every point and no
point is judged at a budget that cut it off. A point the ceiling did end is
reported and cannot be adopted.

**What it reads.** The `train` and `validation` matrices, and the processed
validation frame for its seller groups. Never `test` or `calibration`: a choice
made on the rows that later judge it would make the gate's numbers an optimistic
estimate of it, and `tests/test_tune.py` runs a sweep with both files deleted.

**How it decides.** By `validation_l1_log_price`, the metric early stopping
reads and the one whose bootstrap intervals are narrow enough to resolve
anything on this data (EDN-73). Each point's per-row errors are compared with
the reference's by a paired bootstrap resampled by seller group, because the
split is grouped by seller (EDN-14), with the same draws for every point. The
intervals are simultaneous: the max-statistic bootstrap widens them until all
of them hold together in 95 % of the draws, because the luckiest of eleven
points no better than the reference clears a per-point interval in up to about
one sweep in four. A point is adopted only when its simultaneous interval lies
entirely below zero, and when several do, the one with the lowest validation
L1. Validation MdAPE is reported beside it, through `evaluate`'s own
`point_metrics` and `relative_error`, and decides nothing.

**What it records (issue #88).** Every fit is measured by CodeCarbon exactly as
`train` measures one, and logged as its own MLflow run in the experiment
`recommenditos-price-tuning`, tagged `sweep`, `sweep_role=point`, `point`,
`is_reference` and `variant` as it opens,
with its hyperparameters under the names `train` logs them by, its validation
metrics, its fit time and its energy. No bundle is uploaded: a point that is not
adopted is never served, and one that is gets refitted by `dvc repro`. One more
run, `sweep_role=summary`, carries the totals of the sweep and its verdict, and
the same goes to `reports/tuning/<sweep>.json` and `.txt`.

The search's energy is kept apart from the product's, in
`reports/tuning/emissions/<sweep>.csv`. Not in `reports/emissions/`: each
variant's CSV there is the record of the fit that produced its model, which
`train@<variant>` owns, and the directory as a whole is a dependency of the
`compare-energy` stage, so a search writing into it would make that stage stale.
NFR-10's 15-minute budget is about retraining the product, and the search's cost
scales with its grid instead (EDN-78).

**Provenance.** The runs carry the commit and `git_dirty` like every run the
tracking seam opens. A sweep that reads the matrices at their default path also
carries the `train.deps.<path>` hashes of `train@<variant>`'s inputs, which name
the matrices and the supported-make list it was fitted on. They do not cover
`data/processed/validation.parquet`, which the sweep reads for the seller groups;
the commit's `dvc.lock` and `dvc pull` restore that file. One pointed elsewhere,
as every test is, carries none, as a stage called outside DVC does
(`recommenditos/provenance.py`).
The sweep writes its files into a temporary directory and moves them into
`reports/tuning/` only after its last run has closed, because an untracked file
counts as a change and would tag every later run of the sweep `git_dirty`. For
the same reason a sweep's files are committed before the next sweep is run.

It runs outside `dvc repro`, from the repository root after `dvc pull`, one
sweep per variant and grid; `docs/docs/pipeline.md` gives the commands.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Annotated

from loguru import logger
import mlflow
import numpy as np
import typer
import yaml

from recommenditos.config import PARAMS_FILE, PROCESSED_DATA_DIR, REPORTS_DIR
from recommenditos.data.build_features import read_supported_makes
from recommenditos.modeling.energy import EnergySettings, FitMeasurement, measure
from recommenditos.modeling.evaluate import point_metrics, relative_error
from recommenditos.modeling.model import ESTIMATORS, Model, TrainingData, fit_variant
from recommenditos.modeling.train import (
    VALIDATION_SPLIT,
    log_fit,
    logged_params,
    read_matrices,
)
from recommenditos.pipeline import load_params, read_frame, write_json
from recommenditos.provenance import input_tags
from recommenditos.schema import PROCESSED_SCHEMA
from recommenditos.tracking import (
    REQUIRE_TRACKING_ENV_VAR,
    REQUIRED_ENV_VARS,
    UNCONFIGURED_TRACKING_URI,
    optional_run,
)

#: The MLflow experiment every sweep logs to, so `recommenditos-price` holds only
#: the pipeline's runs (EDN-73).
TUNING_EXPERIMENT = "recommenditos-price-tuning"

#: Where a sweep writes its results, and its energy record beneath that.
TUNING_DIR = REPORTS_DIR / "tuning"
EMISSIONS_SUBDIR = "emissions"

#: The `n_estimators` every point is fitted with: far past where early stopping
#: lands on the real snapshot (round 4,066 at the slowest point EDN-73 fitted),
#: so the curve decides every point rather than the budget.
SEARCH_CEILING = 20_000

#: Bootstrap draws, and the joint coverage of the simultaneous intervals. At
#: 2,000 draws the 95th percentile of the max statistic has 100 draws beyond it.
BOOTSTRAP_DRAWS = 2_000
CONFIDENCE = 0.95

#: The keys a grid may not sweep, because the search sets them itself: the
#: ceiling above, and the patience that makes it a ceiling.
SEARCH_OWNED_KEYS = frozenset({"n_estimators", "early_stopping_rounds"})

#: A sweep id names files and a tag, so it is kept to characters every file
#: system and MLflow's search syntax take as they are.
_SWEEP_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

_PERCENT = 100
_WH_PER_KWH = 1000.0

#: One point of a grid: its hyperparameters in the grid's order, as pairs so that
#: a point can key a dict and be sorted.
type Point = tuple[tuple[str, object], ...]

#: A statistic of per-row values under bootstrap weights (one weight per row).
type Statistic = Callable[[np.ndarray, np.ndarray], float]

app = typer.Typer()


class TuningError(RuntimeError):
    """The sweep was asked for something it cannot answer honestly."""


# --------------------------------------------------------------------------
# The statistics: a paired bootstrap by seller group, and the simultaneous
# intervals. `reports/analysis/tuning_sweep.py` imports these, so EDN-73's
# evidence and every later sweep are computed by one implementation.
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Bootstrap:
    """How the differences are resampled: by these groups, with this seed."""

    #: The seller group of every validation row, in the matrix's row order.
    groups: np.ndarray
    seed: int
    draws: int = BOOTSTRAP_DRAWS
    confidence: float = CONFIDENCE


def mean_error(values: np.ndarray, weight: np.ndarray) -> float:
    """The weighted mean: validation L1 under one bootstrap draw."""
    return float((weight * values).sum() / weight.sum())


def median_error(values: np.ndarray, weight: np.ndarray) -> float:
    """The weighted median, by repeating each row its weight's number of times.

    Validation MdAPE under one bootstrap draw. Repeating rather than
    interpolating a weighted quantile, so a draw's median is exactly the median
    of the rows the draw contains, which is what the gate's MdAPE is.
    """
    return float(np.median(np.repeat(values, weight)))


def paired_bootstrap[K](
    per_row: Mapping[K, np.ndarray], reference: K, statistic: Statistic, bootstrap: Bootstrap
) -> tuple[list[K], np.ndarray]:
    """Each key's statistic minus the reference's, under every resampling of the groups.

    Paired, by seller group: a draw resamples the groups with replacement, which
    gives every row the number of times its group was drawn as a weight, and
    every key is scored on the same weights, so the comparisons share their
    noise. Returns the keys other than the reference, in `per_row`'s order, and
    a draws x keys array of differences.

    The random stream is consumed by one `integers` call per draw whatever the
    statistic, so two calls with one seed - one for L1 and one for MdAPE - see
    exactly the draws EDN-73's single loop saw.
    """
    rng = np.random.default_rng(bootstrap.seed)
    unique, inverse = np.unique(bootstrap.groups, return_inverse=True)
    others = [key for key in per_row if key != reference]
    differences = np.empty((bootstrap.draws, len(others)))
    for draw in range(bootstrap.draws):
        drawn = rng.integers(0, len(unique), size=len(unique))
        weight = np.bincount(drawn, minlength=len(unique))[inverse]
        baseline = statistic(per_row[reference], weight)
        for index, key in enumerate(others):
            differences[draw, index] = statistic(per_row[key], weight) - baseline
    return others, differences


def simultaneous(
    draws: np.ndarray, observed: np.ndarray, confidence: float = CONFIDENCE
) -> tuple[float, np.ndarray]:
    """The max-statistic multiplier and each point's half-width at `confidence` jointly.

    Each point's difference is standardised by its own bootstrap spread, the
    largest standardised deviation across the points is taken per draw, and its
    `confidence` percentile replaces a single interval's 1.96. With one point
    that percentile is about 1.96 itself; with many it is larger.

    A point whose difference does not vary across the draws - predictions
    identical to the reference's - deviates by nothing and gets a zero
    half-width, rather than a division by zero.
    """
    spread = draws.std(axis=0, ddof=1)
    standardised = np.divide(
        np.abs(draws - observed),
        spread,
        out=np.zeros_like(draws, dtype="float64"),
        where=spread > 0,
    )
    largest = np.max(standardised, axis=1)
    multiplier = float(np.percentile(largest, confidence * _PERCENT))
    return multiplier, multiplier * spread


@dataclass(frozen=True)
class Comparison:
    """One point against the reference, on one metric."""

    point: Point
    difference: float
    #: The per-point interval at the same confidence, and the share of draws in
    #: which the point was better: what one comparison alone would say.
    per_point: tuple[float, float]
    share_better: float
    #: The interval that holds jointly with every other point's. The only one
    #: the verdict reads.
    simultaneous: tuple[float, float]

    @property
    def resolved_better(self) -> bool:
        return self.simultaneous[1] < 0


def compare(
    per_row: Mapping[Point, np.ndarray],
    reference: Point,
    statistic: Statistic,
    plain: Callable[[np.ndarray], float],
    bootstrap: Bootstrap,
) -> tuple[float | None, list[Comparison]]:
    """Every point against the reference: the multiplier, and one comparison each.

    `plain` is the statistic on the rows as they are, the observed value the
    intervals are centred on. The multiplier is `None` when the grid holds no
    point but the reference, so there is nothing to compare.
    """
    others, differences = paired_bootstrap(per_row, reference, statistic, bootstrap)
    if not others:
        return None, []
    observed = np.array([plain(per_row[key]) - plain(per_row[reference]) for key in others])
    multiplier, half = simultaneous(differences, observed, bootstrap.confidence)
    tail = (1 - bootstrap.confidence) / 2 * _PERCENT
    comparisons = []
    for index, key in enumerate(others):
        low, high = np.percentile(differences[:, index], [tail, _PERCENT - tail])
        comparisons.append(
            Comparison(
                point=key,
                difference=float(observed[index]),
                per_point=(float(low), float(high)),
                share_better=float(np.mean(differences[:, index] < 0)),
                simultaneous=(
                    float(observed[index] - half[index]),
                    float(observed[index] + half[index]),
                ),
            )
        )
    return multiplier, comparisons


def validation_seller_groups(
    processed_dir: Path, data: TrainingData, supported_makes: tuple[str, ...]
) -> np.ndarray:
    """The seller group of every validation row, to resample them as the split drew them.

    The matrix does not carry `seller_group_id`, so it is joined back from the
    split frame `features` read: the matrix keeps that frame's served rows in
    their order (EDN-67), which the row count and the prices check, so a matrix
    from another run fails here rather than resampling the wrong groups.
    """
    frame = read_frame(processed_dir / f"{VALIDATION_SPLIT}.parquet", PROCESSED_SCHEMA)
    served = frame[frame["make"].isin(supported_makes)].reset_index(drop=True)
    if len(served) != len(data.validation) or not np.allclose(
        served["price"].to_numpy(dtype="float64"),
        data.validation["price"].to_numpy(dtype="float64"),
    ):
        raise TuningError(
            f"the validation frame in {processed_dir} does not line up with the validation "
            f"matrix ({len(served):,} served rows against {len(data.validation):,}), so the "
            f"seller groups cannot be joined back to it. Are the two from one run?"
        )
    return served["seller_group_id"].to_numpy()


# --------------------------------------------------------------------------
# What a sweep is
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Sweep:
    """One variant, one grid, and the values every point of it shares."""

    #: The sweep's id: the `sweep` tag of its runs and the name of its files.
    name: str
    variant: str
    #: The values to try, by hyperparameter, in the order the points vary.
    grid: Mapping[str, Sequence]
    #: Set on every point, the reference included: how the second grid of
    #: EDN-78 holds `learning_rate` and `num_leaves` at the first grid's winner.
    fixed: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchPaths:
    """Where a sweep reads and writes, every one redirectable as `train.main`'s are."""

    input_dir: Path = PROCESSED_DATA_DIR / "features"
    params_path: Path = PARAMS_FILE
    output_dir: Path = TUNING_DIR

    def results(self, sweep: str) -> Path:
        return self.output_dir / f"{sweep}.json"

    def report(self, sweep: str) -> Path:
        return self.output_dir / f"{sweep}.txt"

    def emissions(self, sweep: str) -> Path:
        return self.output_dir / EMISSIONS_SUBDIR / f"{sweep}.csv"


@dataclass(frozen=True)
class Protocol:
    """The knobs of the protocol itself, at the values EDN-73 and EDN-78 decided."""

    ceiling: int = SEARCH_CEILING
    draws: int = BOOTSTRAP_DRAWS
    confidence: float = CONFIDENCE
    experiment: str = TUNING_EXPERIMENT


def parse_assignments(values: Sequence[str], *, single: bool) -> dict[str, list]:
    """`key=v1,v2` options as `{key: [v1, v2]}`, each value read as YAML.

    As YAML so that `63` is an int and `0.025` a float, the types `params.yaml`
    gives the same values. `single` is for `--fix`, which takes one value a key.
    """
    parsed: dict[str, list] = {}
    for value in values:
        key, sign, listed = value.partition("=")
        key = key.strip()
        if not sign or not key or not listed.strip():
            raise TuningError(f"{value!r} is not of the form key=value[,value...].")
        if key in parsed:
            raise TuningError(f"{key} is given twice.")
        items = [yaml.safe_load(item) for item in listed.split(",")]
        # A hyperparameter is a number. YAML reads `1e-1` as text, which would then
        # fail to sort against the floats of the grid, so it is refused here.
        if not all(isinstance(item, int | float) and not isinstance(item, bool) for item in items):
            raise TuningError(f"{key}={listed} holds a value that is not a plain number.")
        if single and len(items) != 1:
            raise TuningError(f"{key} is fixed to one value, not {len(items)}.")
        if len(set(items)) != len(items):
            raise TuningError(f"{key} lists a value twice: {listed}.")
        parsed[key] = items
    return parsed


def point_label(point: Point) -> str:
    """A point as text, for run names, tags and the results file."""
    return ";".join(f"{key}={value}" for key, value in point)


def grid_points(grid: Mapping[str, Sequence]) -> list[Point]:
    """Every combination of the grid's values, keys in the grid's order."""
    return [tuple(zip(grid, values, strict=True)) for values in product(*grid.values())]


def _check_keys(keys: Sequence[str], estimator: type[Model], what: str) -> None:
    readable = set(estimator.hyperparameters) | set(estimator.optional_hyperparameters)
    owned = sorted(set(keys) & SEARCH_OWNED_KEYS)
    if owned:
        raise TuningError(
            f"{what} names {', '.join(owned)}, which the search sets itself: every point is "
            f"fitted at a ceiling far past the curve, so early stopping decides its trees."
        )
    unknown = sorted(set(keys) - readable)
    if unknown:
        raise TuningError(
            f"{what} names {', '.join(unknown)}, which {estimator.estimator_name!r} does not "
            f"read. It reads {', '.join(sorted(readable))}."
        )


# --------------------------------------------------------------------------
# Running a sweep
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Fit:
    """What one point's fit produced, beyond its per-row errors."""

    point: Point
    reference: bool
    run_id: str | None
    training: dict
    validation_mdape: float
    cost: FitMeasurement

    @property
    def budget_bound(self) -> bool:
        """The ceiling ended the fit, so the point is not judged at its own best."""
        return self.training.get("early_stopped") is False

    def record(self) -> dict:
        return {
            "point": dict(self.point),
            "label": point_label(self.point),
            "reference": self.reference,
            "mlflow_run_id": self.run_id,
            "best_iteration": self.training.get("best_iteration"),
            "boosting_rounds": self.training.get("boosting_rounds"),
            "early_stopped": self.training.get("early_stopped"),
            "train_l1_log_price": self.training["train_l1_log_price"],
            "validation_l1_log_price": self.training["validation_l1_log_price"],
            "validation_mdape": self.validation_mdape,
            "fit_seconds": self.cost.fit_seconds,
            "fit_cpu_seconds": self.cost.fit_cpu_seconds,
            "energy_kwh": float(self.cost.record["energy_consumed"]),
            "emissions_kg_co2eq": float(self.cost.record["emissions"]),
            "codecarbon_run_id": self.cost.run_id,
        }


@dataclass(frozen=True, eq=False)
class _Context:
    """Everything the points of one sweep share, read and checked once."""

    sweep: Sweep
    paths: SearchPaths
    protocol: Protocol
    params: dict
    estimator: type[Model]
    #: The configuration every point starts from: the variant as committed, the
    #: optional keys at their defaults, and what the sweep fixes.
    base: dict
    reference: Point
    points: list[Point]
    energy: EnergySettings
    data: TrainingData
    bootstrap: Bootstrap
    tags: dict
    #: Where the sweep writes while it runs: a temporary directory, so the files
    #: it produces are not in the working tree while its runs are open. Every run
    #: is tagged with `git_dirty`, and a CSV the sweep itself had just written
    #: would mark every run after the first one dirty (EDN-74).
    staged: SearchPaths | None = None


def search(
    sweep: Sweep, paths: SearchPaths | None = None, protocol: Protocol | None = None
) -> dict:
    """Run one sweep and return its results document, as written to `<sweep>.json`."""
    paths = paths or SearchPaths()
    context = _prepare(sweep, paths, protocol or Protocol())
    with tempfile.TemporaryDirectory() as staging:
        staged = replace(paths, output_dir=Path(staging))
        document = _run(replace(context, staged=staged))
        # Moved into the tree only now, once no run of the sweep is open any more.
        for written, final in (
            (staged.emissions(sweep.name), paths.emissions(sweep.name)),
            (staged.results(sweep.name), paths.results(sweep.name)),
            (staged.report(sweep.name), paths.report(sweep.name)),
        ):
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(written, final)
    logger.success(
        f"sweep {sweep.name}: {document['verdict']['text']} Wrote {paths.results(sweep.name)}."
    )
    return document


def _run(context: _Context) -> dict:
    """Fit every point, decide, and write the sweep's record into the staging directory."""
    sweep, staged = context.sweep, context.staged
    errors: dict[Point, np.ndarray] = {}
    relative: dict[Point, np.ndarray] = {}
    fits: list[Fit] = []
    for point in context.points:
        fit, errors[point], relative[point] = _fit_point(context, point)
        fits.append(fit)

    reference, bootstrap = context.reference, context.bootstrap
    l1 = compare(errors, reference, mean_error, np.mean, bootstrap)
    mdape = compare(relative, reference, median_error, np.median, bootstrap)
    # A point the ceiling cut off is not judged at its own best, so it is reported
    # and never adopted; among the rest, the lowest validation L1 that resolves.
    bound = {fit.point for fit in fits if fit.budget_bound}
    candidates = [row for row in l1[1] if row.resolved_better and row.point not in bound]
    adopted = min(candidates, key=lambda row: row.difference).point if candidates else None

    document = _document(context, fits, l1, mdape, adopted)
    write_json(document, staged.results(sweep.name))
    staged.report(sweep.name).write_text(_report(document), encoding="utf-8", newline="\n")
    _log_summary(context, document)
    return document


def _prepare(sweep: Sweep, paths: SearchPaths, protocol: Protocol) -> _Context:
    """Check the request, then read what every point needs, before any fit runs."""
    if not _SWEEP_ID.fullmatch(sweep.name):
        raise TuningError(f"the sweep id {sweep.name!r} must match {_SWEEP_ID.pattern}.")
    if not sweep.grid:
        raise TuningError("the grid is empty; name at least one hyperparameter to sweep.")
    written = (paths.results(sweep.name), paths.report(sweep.name), paths.emissions(sweep.name))
    existing = [str(path) for path in written if path.exists()]
    if existing:
        raise TuningError(
            f"sweep {sweep.name!r} has already written {', '.join(existing)}. A sweep's "
            f"record is evidence: give a new sweep its own id rather than mixing two in a file."
        )
    params = load_params(paths.params_path)
    if sweep.variant not in params["train"]["variants"]:
        raise TuningError(f"{sweep.variant!r} is not a key of train.variants in params.yaml.")
    settings = params["train"]["variants"][sweep.variant]
    estimator = ESTIMATORS[settings["estimator"]]
    _check_ceiling(protocol.ceiling, settings["params"])
    _check_keys(list(sweep.grid), estimator, "the grid")
    _check_keys(list(sweep.fixed), estimator, "the fixed values")
    both = sorted(set(sweep.grid) & set(sweep.fixed))
    if both:
        raise TuningError(f"{', '.join(both)} is both swept and fixed.")
    _refuse_a_tracked_sweep_id(sweep.name, protocol.experiment)
    # Before the matrices are read, as in `train`, so a misconfigured measurement
    # fails before any work is done.
    energy = EnergySettings.from_params(params["train"]["energy"])

    base = {**estimator.optional_hyperparameters, **settings["params"], **sweep.fixed}
    reference: Point = tuple((key, base[key]) for key in sweep.grid)
    points = sorted({*grid_points(sweep.grid), reference}, key=_sort_key)

    supported = read_supported_makes(paths.input_dir.parent)
    data = read_matrices(paths.input_dir, settings["feature_set"], supported)
    groups = validation_seller_groups(paths.input_dir.parent, data, supported)
    tags = {"sweep": sweep.name}
    if paths.input_dir.resolve() == (PROCESSED_DATA_DIR / "features").resolve():
        tags |= input_tags(f"train@{sweep.variant}")
    logger.info(
        f"sweep {sweep.name}: {sweep.variant}, {len(points)} point(s) at a ceiling of "
        f"{protocol.ceiling:,} trees, reference {point_label(reference)}; validation "
        f"{len(groups):,} rows in {len(np.unique(groups)):,} seller groups."
    )
    return _Context(
        sweep=sweep,
        paths=paths,
        protocol=protocol,
        params=params,
        estimator=estimator,
        base=base,
        reference=reference,
        points=points,
        energy=energy,
        data=data,
        bootstrap=Bootstrap(groups, params["seed"], protocol.draws, protocol.confidence),
        tags=tags,
    )


def _check_ceiling(ceiling: int, committed: dict) -> None:
    """A ceiling the curve can stop under: whole trees, and more than the patience."""
    patience = committed.get("early_stopping_rounds", 0)
    if isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling <= patience:
        raise TuningError(
            f"the ceiling must be a whole number of trees above the variant's "
            f"early_stopping_rounds of {patience}, so early stopping can end every fit; "
            f"it is {ceiling!r}."
        )


def _refuse_a_tracked_sweep_id(name: str, experiment: str) -> None:
    """Refuse an id the experiment already holds runs for.

    A sweep's files reach `reports/tuning/` only once it has finished, so one that
    was interrupted - Ctrl-C, or a failed request under
    `RECOMMENDITOS_REQUIRE_TRACKING=1` - leaves no file behind but does leave runs
    in MLflow. Rerun under the same id, it would put a second run for a point
    beside the first, and nothing would say which belongs to the record. Checked
    only where tracking is configured, since otherwise no run is logged either; a
    server that cannot be asked is a failure only when tracking is required, as
    it is for `optional_run`.
    """
    configured = all(
        os.environ.get(variable, "") not in {"", UNCONFIGURED_TRACKING_URI}
        for variable in REQUIRED_ENV_VARS
    )
    if not configured:
        return
    try:
        client = mlflow.MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
        found = client.get_experiment_by_name(experiment)
        runs = (
            []
            if found is None
            else client.search_runs([found.experiment_id], f"tags.sweep = '{name}'", max_results=1)
        )
    except Exception as error:  # any failure to reach the server
        if os.environ.get(REQUIRE_TRACKING_ENV_VAR, "") not in {"", "0"}:
            raise
        logger.warning(f"could not check MLflow for earlier runs of sweep {name!r}: {error}")
        return
    if runs:
        raise TuningError(
            f"the MLflow experiment {experiment!r} already holds runs of sweep {name!r}, from "
            f"an earlier attempt that did not finish or whose files were removed. Run this "
            f"sweep under a fresh id; the earlier runs stay as the record of that attempt."
        )


def _sort_key(point: Point) -> tuple:
    return tuple(value for _, value in point)


def _fit_point(context: _Context, point: Point) -> tuple[Fit, np.ndarray, np.ndarray]:
    """Fit one point, measured and tracked, and return it with its per-row errors."""
    sweep, params, data = context.sweep, context.params, context.data
    hyperparameters = {**context.base, **dict(point)}
    if "n_estimators" in context.estimator.hyperparameters:
        hyperparameters["n_estimators"] = context.protocol.ceiling
    settings = {**params["train"]["variants"][sweep.variant], "params": hyperparameters}
    # `params` with this point's settings in it, so `logged_params` names every
    # key as `train` does and a search run shares its columns with a pipeline run.
    variants = {**params["train"]["variants"], sweep.variant: settings}
    point_params = {**params, "train": {**params["train"], "variants": variants}}
    label = point_label(point)
    is_reference = point == context.reference

    tags = {
        **context.tags,
        "sweep_role": "point",
        "point": label,
        "is_reference": str(is_reference).lower(),
        "variant": sweep.variant,
    }
    with optional_run(context.protocol.experiment, f"{sweep.name}-{sweep.variant}-{label}") as run:
        # Before the fit, so the run of a fit that fails or is interrupted still
        # names its sweep and point.
        run.set_tags(tags)
        # The measured section is the fit and nothing else, as in `train`.
        model, cost = measure(
            lambda: fit_variant(
                sweep.variant,
                settings,
                data,
                seed=params["seed"],
                num_threads=params["train"]["num_threads"],
            ),
            name=f"{sweep.name} {sweep.variant} {label}",
            settings=context.energy,
            output_file=context.staged.emissions(sweep.name),
        )
        predicted = model.predict_eur(data.validation)
        # Through `evaluate`'s own implementation, so this is the gate's metric on
        # validation rows rather than a second definition of it.
        mdape = point_metrics(data.validation["price"], predicted)["mdape"]
        log_fit(
            run,
            model,
            cost=cost,
            params={**logged_params(sweep.variant, data, point_params), **cost.params()},
            tags=tags,
        )
        run.log_metrics({"validation_mdape": mdape})

    log_price = data.validation["log_price"].to_numpy(dtype="float64")
    errors = np.abs(model.predict_log_price(data.validation).to_numpy(dtype="float64") - log_price)
    relative = relative_error(data.validation["price"], predicted).to_numpy(dtype="float64")
    fit = Fit(point, is_reference, run.run_id, model.metadata["training"], mdape, cost)
    logger.info(
        f"{label}: validation L1 {fit.training['validation_l1_log_price']:.5f}, MdAPE "
        f"{mdape:.4%}, best round {fit.training.get('best_iteration')}, "
        f"{cost.fit_seconds:.1f} s" + ("  (the ceiling ended it)" if fit.budget_bound else "")
    )
    return fit, errors, relative


def _document(
    context: _Context,
    fits: list[Fit],
    l1: tuple[float | None, list[Comparison]],
    mdape: tuple[float | None, list[Comparison]],
    adopted: Point | None,
) -> dict:
    """The results of a sweep as one JSON-ready document."""
    sweep, bootstrap = context.sweep, context.bootstrap
    reference_fit = next(fit for fit in fits if fit.reference)

    def comparisons(result: tuple[float | None, list[Comparison]]) -> dict:
        multiplier, rows = result
        return {
            "simultaneous_multiplier": multiplier,
            "points": [
                {
                    "label": point_label(row.point),
                    "difference": row.difference,
                    "per_point_interval": list(row.per_point),
                    "share_of_draws_better": row.share_better,
                    "simultaneous_interval": list(row.simultaneous),
                    "resolved_better": row.resolved_better,
                }
                for row in rows
            ],
        }

    confidence = f"{bootstrap.confidence * _PERCENT:g} %"
    if adopted is None:
        text = (
            f"keep the reference {point_label(context.reference)}: no point's simultaneous "
            f"{confidence} interval of the validation L1 difference lies below zero."
        )
    else:
        text = (
            f"adopt {point_label(adopted)}: its simultaneous {confidence} interval of the "
            f"validation L1 difference lies below zero."
        )
    return {
        "sweep": sweep.name,
        "variant": sweep.variant,
        "tags": context.tags,
        "grid": {key: list(values) for key, values in sweep.grid.items()},
        "fixed": dict(sweep.fixed),
        "reference": point_label(context.reference),
        "ceiling": context.protocol.ceiling,
        "selection_metric": "validation_l1_log_price",
        "bootstrap": {
            "draws": bootstrap.draws,
            "confidence": bootstrap.confidence,
            "seed": bootstrap.seed,
            "resampled_by": "seller_group_id",
            "validation_rows": len(bootstrap.groups),
            "seller_groups": len(np.unique(bootstrap.groups)),
        },
        "fits": [fit.record() for fit in fits],
        "budget_bound": [point_label(fit.point) for fit in fits if fit.budget_bound],
        "validation_l1": comparisons(l1),
        "validation_mdape": comparisons(mdape),
        # Summed over the measurements rather than read back from the CSV, so the
        # totals are this sweep's fits exactly.
        "totals": {
            "points": len(fits),
            "fit_seconds": sum(fit.cost.fit_seconds for fit in fits),
            "fit_cpu_seconds": sum(fit.cost.fit_cpu_seconds for fit in fits),
            "energy_kwh": sum(float(fit.cost.record["energy_consumed"]) for fit in fits),
            "emissions_kg_co2eq": sum(float(fit.cost.record["emissions"]) for fit in fits),
        },
        "verdict": {
            "adopted": None if adopted is None else point_label(adopted),
            "adopted_point": None if adopted is None else dict(adopted),
            "reference_validation_l1_log_price": (
                reference_fit.training["validation_l1_log_price"]
            ),
            "text": text,
        },
    }


def _report(document: dict) -> str:
    """The document as a table a person reads, in the shape of EDN-73's results file."""
    bootstrap, totals = document["bootstrap"], document["totals"]
    lines = [
        f"# recommenditos.modeling.tune, sweep {document['sweep']} ({document['variant']})",
        f"grid {document['grid']}, fixed {document['fixed']}, reference {document['reference']}",
        (
            f"ceiling {document['ceiling']:,}, seed {bootstrap['seed']}, {bootstrap['draws']:,} "
            f"draws over {bootstrap['seller_groups']:,} seller groups of "
            f"{bootstrap['validation_rows']:,} validation rows"
        ),
        "",
        (
            f"{'point':<40}{'best round':>11}{'rounds run':>12}{'fit s':>8}{'energy Wh':>11}"
            f"{'val L1(log)':>13}{'val MdAPE':>11}"
        ),
    ]
    for fit in document["fits"]:
        lines.append(
            f"{fit['label']:<40}{fit['best_iteration'] or 0:>11,}"
            f"{fit['boosting_rounds'] or 0:>12,}{fit['fit_seconds']:>8.1f}"
            f"{fit['energy_kwh'] * _WH_PER_KWH:>11.4f}"
            f"{fit['validation_l1_log_price']:>13.5f}{fit['validation_mdape']:>11.4%}"
            + ("  (reference)" if fit["reference"] else "")
            + ("  (ceiling bound)" if fit["early_stopped"] is False else "")
        )
    for metric, form in (("validation_l1", "+.5f"), ("validation_mdape", "+.3%")):
        block = document[metric]
        lines += [
            "",
            (
                f"{metric}, point minus reference: difference [per-point interval] share of "
                f"draws better | simultaneous interval"
            ),
        ]
        for row in block["points"]:
            low, high = row["per_point_interval"]
            s_low, s_high = row["simultaneous_interval"]
            lines.append(
                f"  {row['label']:<40}{row['difference']:{form}} [{low:{form}}, {high:{form}}] "
                f"{row['share_of_draws_better']:6.1%} | [{s_low:{form}}, {s_high:{form}}]"
            )
        if block["simultaneous_multiplier"] is not None:
            lines.append(
                f"  simultaneous multiplier {block['simultaneous_multiplier']:.2f}, against "
                f"1.96 for one interval"
            )
    lines += [
        "",
        (
            f"search totals: {totals['points']} fits, {totals['fit_seconds']:.1f} s of fitting "
            f"({totals['fit_cpu_seconds']:.1f} CPU-s), "
            f"{totals['energy_kwh'] * _WH_PER_KWH:.4f} Wh, "
            f"{totals['emissions_kg_co2eq'] * 1e6:.2f} mg CO2eq"
        ),
        f"verdict: {document['verdict']['text']}",
    ]
    return "\n".join(lines) + "\n"


def _log_summary(context: _Context, document: dict) -> None:
    """The sweep's own run: its totals, its verdict, and its two results files."""
    sweep, totals, verdict = context.sweep, document["totals"], document["verdict"]
    run_name = f"{sweep.name}-{sweep.variant}-summary"
    with optional_run(context.protocol.experiment, run_name) as run:
        run.set_tags(
            {
                **context.tags,
                "sweep_role": "summary",
                "variant": sweep.variant,
                "verdict": "keep" if verdict["adopted"] is None else "adopt",
                "adopted": verdict["adopted"] or "none",
                "reference_point": document["reference"],
            }
        )
        run.log_params(
            {
                "variant": sweep.variant,
                "seed": document["bootstrap"]["seed"],
                "ceiling": document["ceiling"],
                "bootstrap.draws": document["bootstrap"]["draws"],
                "bootstrap.confidence": document["bootstrap"]["confidence"],
                **{f"grid.{key}": str(values) for key, values in document["grid"].items()},
                **{f"fixed.{key}": value for key, value in document["fixed"].items()},
            }
        )
        run.log_metrics(
            {
                "n_points": float(totals["points"]),
                "total_fit_seconds": totals["fit_seconds"],
                "total_fit_cpu_seconds": totals["fit_cpu_seconds"],
                "total_energy_kwh": totals["energy_kwh"],
                "total_emissions_kg_co2eq": totals["emissions_kg_co2eq"],
                "simultaneous_multiplier": document["validation_l1"]["simultaneous_multiplier"],
                "reference_validation_l1_log_price": verdict["reference_validation_l1_log_price"],
                "best_validation_l1_log_price": min(
                    fit["validation_l1_log_price"] for fit in document["fits"]
                ),
                "adopted": float(verdict["adopted"] is not None),
            }
        )
        with tempfile.TemporaryDirectory() as staging:
            for path in (context.staged.results(sweep.name), context.staged.report(sweep.name)):
                shutil.copy(path, Path(staging) / path.name)
            run.log_artifacts(Path(staging), artifact_path="results")


@app.command()
def main(
    variant: Annotated[str, typer.Argument(help="a key of train.variants in params.yaml")],
    grid: Annotated[
        list[str],
        typer.Option("--grid", help="key=v1,v2,... to sweep; repeat it for each hyperparameter"),
    ],
    sweep: Annotated[str, typer.Option("--sweep", help="this sweep's id: its tag and file name")],
    fix: Annotated[
        list[str] | None,
        typer.Option("--fix", help="key=value set on every point, the reference included"),
    ] = None,
    ceiling: int = SEARCH_CEILING,
):
    """Run one sweep on the default paths, as `docs/docs/pipeline.md` describes."""
    try:
        fixed = parse_assignments(fix or [], single=True)
        search(
            Sweep(
                name=sweep,
                variant=variant,
                grid=parse_assignments(grid, single=False),
                fixed={key: values[0] for key, values in fixed.items()},
            ),
            protocol=Protocol(ceiling=ceiling),
        )
    except TuningError as error:
        # Logged rather than raised, for the reason `tracking.main` gives: a
        # misnamed key is something to read, not a traceback to debug.
        logger.error(str(error))
        raise typer.Exit(1) from error


if __name__ == "__main__":
    app()
