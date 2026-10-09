"""The hyperparameter search of issue #88 (EDN-78), and the optional `min_child_samples`.

Two halves. The statistics - the paired bootstrap by seller group and the
max-statistic simultaneous intervals - are checked on per-row errors built by
hand, where the right verdict is known: a point better on every row is adopted,
points that differ from the reference only by noise are not, and the multiplier
is a single interval's 1.96 for one comparison and grows with more. The sweep
itself runs on the synthetic fixture, where its numbers mean nothing and its
behaviour is what is checked: what it reads, what it measures, and what it
records in MLflow, its only store.

Like `tests/test_model.py`, nothing here reaches DagsHub: tracking is switched
off for every test, and a test that runs a sweep points MLflow at a SQLite store
under `tmp_path` through the `store` fixture.
"""

import json
from pathlib import Path
import shutil

import mlflow
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from recommenditos.config import PARAMS_FILE, REPORTS_DIR
from recommenditos.data import build_features, preprocess, split_data
from recommenditos.data.build_features import read_supported_makes
from recommenditos.modeling import tune
from recommenditos.modeling.energy import read_record
from recommenditos.modeling.model import ESTIMATORS, LightGBMModel, ModelError, fit_variant
from recommenditos.modeling.train import read_matrices
from recommenditos.modeling.tune import (
    ARTIFACT_PATH,
    EMISSIONS_FILE,
    REPORT_FILE,
    RESULTS_FILE,
    TUNING_EXPERIMENT,
    Bootstrap,
    Protocol,
    SearchPaths,
    Sweep,
    TuningError,
    compare,
    mean_error,
    parse_assignments,
    search,
    simultaneous,
)
from recommenditos.pipeline import load_params
from recommenditos.tracking import REQUIRE_TRACKING_ENV_VAR, REQUIRED_ENV_VARS

#: Small enough that a sweep on the fixture takes seconds, large enough that the
#: max statistic's 95th percentile is not decided by a handful of draws.
DRAWS = 400

#: Far past where early stopping lands on the fixture, so every fit there is
#: decided by its curve, as the real search's 20,000 is on the real data.
CEILING = 3_000


@pytest.fixture(autouse=True)
def _no_tracking_server(monkeypatch):
    """Tracking off, whatever `.env` the developer's machine has.

    A sweep logs a run per fit, and a test that inherited real credentials
    would put its synthetic runs into the team's `recommenditos-price-tuning`.
    """
    for name in (*REQUIRED_ENV_VARS, REQUIRE_TRACKING_ENV_VAR):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def store(tmp_path: Path, monkeypatch) -> str:
    """A local SQLite tracking store, configured the way `.env` configures DagsHub.

    A local store puts its artifacts in `./mlruns`, relative to the working
    directory, so the test runs from `tmp_path`, as `tests/test_model.py`'s do.
    """
    before = mlflow.get_tracking_uri()
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "someone")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "a-token")
    monkeypatch.chdir(tmp_path)
    yield uri
    mlflow.set_tracking_uri(before)


def _listing(directory: Path) -> list[str]:
    return sorted(str(path.relative_to(directory)) for path in directory.rglob("*"))


@pytest.fixture(scope="session")
def processed(tmp_path_factory, _generated_frame: pd.DataFrame) -> Path:
    """The processed splits and the `basic` matrices of the synthetic fixture."""
    root = tmp_path_factory.mktemp("tune")
    raw = root / "raw" / "listings.parquet"
    raw.parent.mkdir(parents=True)
    _generated_frame.to_parquet(raw, index=False)
    processed = root / "processed"
    preprocess.main(raw, root / "interim" / "listings.parquet", PARAMS_FILE)
    split_data.main(root / "interim" / "listings.parquet", processed, PARAMS_FILE)
    build_features.main("basic", processed, processed / "features", PARAMS_FILE)
    return processed


def _paths(processed: Path) -> SearchPaths:
    return SearchPaths(input_dir=processed / "features", params_path=PARAMS_FILE)


def _protocol(**overrides) -> Protocol:
    return Protocol(**{"ceiling": CEILING, "draws": DRAWS, **overrides})


# --------------------------------------------------------------------------
# The statistics, on errors whose verdict is known
# --------------------------------------------------------------------------


def _reference_errors(rows: int = 4_000, groups: int = 400) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    return np.abs(rng.normal(0, 1, rows)), rng.integers(0, groups, rows)


def _noisy(reference: np.ndarray, seed: int) -> np.ndarray:
    """Errors that differ from `reference` by noise alone, with no shift."""
    return reference + np.random.default_rng(seed).normal(0, 0.3, len(reference))


def test_one_comparison_gets_a_single_intervals_multiplier():
    """With one point the max statistic is one statistic, so its 95th percentile is about 1.96."""
    reference, groups = _reference_errors()
    per_row = {(("k", 0),): reference, (("k", 1),): _noisy(reference, 1)}

    multiplier, _ = compare(
        per_row, (("k", 0),), mean_error, np.mean, Bootstrap(groups, seed=3, draws=2_000)
    )

    assert multiplier == pytest.approx(1.96, abs=0.15)


def test_many_comparisons_widen_every_interval():
    """The luckiest of many points is luckier than any one, so the multiplier grows past 1.96."""
    reference, groups = _reference_errors()
    per_row = {(("k", 0),): reference} | {
        (("k", seed),): _noisy(reference, seed) for seed in range(1, 12)
    }

    multiplier, rows = compare(
        per_row, (("k", 0),), mean_error, np.mean, Bootstrap(groups, seed=3, draws=2_000)
    )

    assert multiplier > 1.96 + 0.3
    for row in rows:
        per_point_width = row.per_point[1] - row.per_point[0]
        assert row.simultaneous[1] - row.simultaneous[0] > per_point_width


def test_a_point_better_on_every_row_is_adopted_and_noise_is_not():
    """The protocol's two failure modes, both on one grid.

    A point whose error is 10 % lower on every row is resolved better; eleven
    points that differ from the reference by noise alone are not, although some
    of them look better by a per-point reading.
    """
    reference, groups = _reference_errors()
    better = (("k", 99),)
    per_row = {(("k", 0),): reference, better: reference * 0.9} | {
        (("k", seed),): _noisy(reference, seed) for seed in range(1, 12)
    }

    _, rows = compare(per_row, (("k", 0),), mean_error, np.mean, Bootstrap(groups, 3, DRAWS))

    resolved = [row.point for row in rows if row.resolved_better]
    assert resolved == [better]


def test_a_point_identical_to_the_reference_gets_a_zero_width_not_a_division_by_zero():
    draws = np.column_stack([np.zeros(100), np.random.default_rng(1).normal(0, 1, 100)])

    multiplier, half = simultaneous(draws, np.array([0.0, 0.0]))

    assert np.isfinite(multiplier)
    assert half[0] == 0


# --------------------------------------------------------------------------
# A sweep on the fixture
# --------------------------------------------------------------------------


def _summary_files(client: mlflow.MlflowClient, run_id: str, into: Path) -> dict[str, Path]:
    """The summary run's `results/` artifacts, downloaded, by file name."""
    local = Path(client.download_artifacts(run_id, ARTIFACT_PATH, str(into)))
    return {path.name: path for path in local.iterdir()}


def _runs(store: str, sweep: str) -> list:
    client = mlflow.MlflowClient(tracking_uri=store)
    experiment = client.get_experiment_by_name(TUNING_EXPERIMENT)
    return client.search_runs([experiment.experiment_id], f"tags.sweep = '{sweep}'")


def test_a_sweep_never_reads_the_test_or_calibration_split(
    processed: Path, tmp_path: Path, store: str, monkeypatch
):
    """The choice is made on validation rows, so the gate's later reading of test stays honest.

    Both ways a read could happen are closed: the files are gone from the copy
    the sweep reads, and a read of either name anywhere fails the test.
    """
    copy = tmp_path / "processed"
    shutil.copytree(processed, copy)
    for split in ("test", "calibration"):
        for path in copy.rglob(f"{split}.parquet"):
            path.unlink()
    read_parquet = pd.read_parquet

    def guarded(path, *args, **kwargs):
        assert Path(path).stem not in {"test", "calibration"}, f"the sweep read {path}"
        return read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", guarded)

    document = search(
        Sweep("no-test", "lgbm-basic", {"learning_rate": [0.1]}), _paths(copy), _protocol()
    )

    assert document["verdict"]["text"]


def test_the_reference_is_fitted_even_when_the_grid_leaves_it_out(processed: Path, store: str):
    """Every difference is taken against it, so it is never optional."""
    committed = load_params(PARAMS_FILE)["train"]["variants"]["lgbm-basic"]["params"]

    document = search(
        Sweep("reference", "lgbm-basic", {"learning_rate": [0.1]}), _paths(processed), _protocol()
    )

    labels = {fit["label"]: fit["reference"] for fit in document["fits"]}
    assert labels == {
        f"learning_rate={committed['learning_rate']}": True,
        "learning_rate=0.1": False,
    }
    assert document["reference"] == f"learning_rate={committed['learning_rate']}"
    assert all(fit["early_stopped"] for fit in document["fits"]), "a fit hit the ceiling"


@pytest.mark.req("NFR-10")
def test_the_summary_run_carries_the_record_the_table_and_one_energy_row_per_fit(
    processed: Path, tmp_path: Path, store: str
):
    """MLflow is a sweep's only store (EDN-78), so everything it found is an artifact there.

    The second grid of EDN-78 is the case run here: `min_child_samples` swept with
    `learning_rate` fixed, so the reference is the fixed value at LightGBM's
    default of 20, a key `params.yaml` does not even name. Nothing is written
    into the repository's `reports/`, which the variants' energy CSVs share.
    """
    reports_before = _listing(REPORTS_DIR)

    document = search(
        Sweep(
            "second-grid",
            "lgbm-basic",
            {"min_child_samples": [5, 20, 50]},
            fixed={"learning_rate": 0.1},
        ),
        _paths(processed),
        _protocol(),
    )

    assert _listing(REPORTS_DIR) == reports_before
    assert document["reference"] == "min_child_samples=20"
    assert document["fixed"] == {"learning_rate": 0.1}
    (summary,) = [
        run for run in _runs(store, "second-grid") if run.data.tags["sweep_role"] == "summary"
    ]
    client = mlflow.MlflowClient(tracking_uri=store)
    files = _summary_files(client, summary.info.run_id, tmp_path / "downloaded")
    assert set(files) == {RESULTS_FILE, REPORT_FILE, EMISSIONS_FILE}
    assert json.loads(files[RESULTS_FILE].read_text("utf-8")) == document
    assert "verdict: " in files[REPORT_FILE].read_text("utf-8")
    rows = pd.read_csv(files[EMISSIONS_FILE], dtype={"run_id": "str"})
    assert len(rows) == len(document["fits"]) == 3
    for fit in document["fits"]:
        recorded = read_record(files[EMISSIONS_FILE], fit["codecarbon_run_id"])
        assert fit["energy_kwh"] == recorded["energy_consumed"]
    assert document["totals"]["energy_kwh"] == pytest.approx(rows["energy_consumed"].sum())
    assert document["totals"]["fit_seconds"] == pytest.approx(
        sum(fit["fit_seconds"] for fit in document["fits"])
    )


@pytest.mark.req("NFR-10", "NFR-14")
def test_every_fit_of_a_sweep_is_a_tagged_run_with_its_metric_fit_time_and_energy(
    processed: Path, store: str
):
    """Issue #88's second acceptance criterion, read back from MLflow.

    A run per point, tagged with the sweep, carrying the validation metric, the
    fit time and the energy, and no model bundle; and one summary run whose
    total is the points' energy summed.
    """
    document = search(
        Sweep("tracked", "lgbm-basic", {"learning_rate": [0.05, 0.1], "num_leaves": [15]}),
        _paths(processed),
        _protocol(),
    )
    client = mlflow.MlflowClient(tracking_uri=store)
    runs = _runs(store, "tracked")
    points = [run for run in runs if run.data.tags["sweep_role"] == "point"]
    summaries = [run for run in runs if run.data.tags["sweep_role"] == "summary"]
    artefacts = {
        run.info.run_id: [entry.path for entry in client.list_artifacts(run.info.run_id)]
        for run in runs
    }

    assert len(points) == len(document["fits"]) == 3
    assert {run.info.run_id for run in points} == {
        fit["mlflow_run_id"] for fit in document["fits"]
    }
    for run in points:
        assert run.data.tags["variant"] == "lgbm-basic"
        assert run.data.tags["is_reference"] in {"true", "false"}
        assert {"validation_l1_log_price", "validation_mdape", "fit_seconds", "energy_kwh"} <= set(
            run.data.metrics
        )
        assert run.data.params["lightgbm.n_estimators"] == str(CEILING)
        assert run.data.params["lightgbm.min_child_samples"] == "20"
        assert "git_commit" in run.data.tags
        # Pointed at matrices under `tmp_path`, the runs name no input hashes:
        # those of `dvc.yaml` would describe files these fits never read.
        assert not [name for name in run.data.tags if ".deps." in name]
        assert artefacts[run.info.run_id] == []
    assert {run.data.tags["point"] for run in points} == {fit["label"] for fit in document["fits"]}
    (summary,) = summaries
    assert summary.data.tags["verdict"] in {"keep", "adopt"}
    assert summary.data.tags["reference_point"] == document["reference"]
    assert summary.data.metrics["total_energy_kwh"] == pytest.approx(
        sum(run.data.metrics["energy_kwh"] for run in points)
    )
    assert artefacts[summary.info.run_id] == [ARTIFACT_PATH]


def test_a_sweep_without_tracking_is_refused_before_any_fit(processed: Path, monkeypatch):
    """With MLflow the only store, a sweep without it would fit every point and keep nothing."""
    calls = []
    monkeypatch.setattr(tune, "fit_variant", lambda *args, **kwargs: calls.append(1))

    with pytest.raises(TuningError, match="not configured"):
        search(
            Sweep("untracked", "lgbm-basic", {"learning_rate": [0.1]}),
            _paths(processed),
            _protocol(),
        )

    assert calls == []


def test_the_command_line_reports_a_sweep_without_tracking_and_exits_1():
    result = CliRunner().invoke(
        tune.app, ["lgbm-basic", "--grid", "learning_rate=0.1", "--sweep", "untracked"]
    )

    assert result.exit_code == 1


@pytest.mark.parametrize(
    ("grid", "fixed", "message"),
    [
        ({"n_estimators": [100, 200]}, {}, "sets itself"),
        ({"early_stopping_rounds": [10]}, {}, "sets itself"),
        ({"max_depth": [3]}, {}, "does not read"),
        ({"learning_rate": [0.1]}, {"learning_rate": 0.05}, "both swept and fixed"),
        ({}, {}, "grid is empty"),
    ],
)
def test_a_sweep_that_cannot_be_judged_is_refused_before_any_fit(
    grid: dict, fixed: dict, message: str, tmp_path: Path
):
    """Refused while reading the request, so the matrices are never even opened."""
    paths = SearchPaths(input_dir=tmp_path / "nowhere")

    with pytest.raises(TuningError, match=message):
        search(Sweep("refused", "lgbm-basic", grid, fixed), paths, _protocol())


@pytest.mark.req("NFR-14")
def test_an_interrupted_sweep_leaves_attributable_runs_and_its_id_is_not_reused(
    processed: Path, store: str, monkeypatch
):
    """An interrupted sweep uploads no record, but its runs are in MLflow: tagged, and counted.

    The second point's fit is interrupted. Its run must still name its sweep and
    point, and a rerun under the same id must be refused rather than put a second
    run for the first point beside the first one's.
    """
    calls = []
    fit = tune.fit_variant

    def interrupted_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return fit(*args, **kwargs)

    monkeypatch.setattr(tune, "fit_variant", interrupted_second)
    sweep = Sweep("crashed", "lgbm-basic", {"learning_rate": [0.05, 0.1]})

    with pytest.raises(KeyboardInterrupt):
        search(sweep, _paths(processed), _protocol())
    runs = _runs(store, "crashed")
    with pytest.raises(TuningError, match="fresh id"):
        search(sweep, _paths(processed), _protocol())

    assert len(runs) == len(_runs(store, "crashed")) == 2
    assert {run.data.tags["point"] for run in runs} == {"learning_rate=0.05", "learning_rate=0.1"}
    assert {run.data.tags["sweep_role"] for run in runs} == {"point"}
    assert {run.info.status for run in runs} == {"FINISHED", "FAILED"}


def test_a_finished_sweeps_id_is_not_reused(processed: Path, store: str):
    """A second run under the same id would put two records under one tag."""
    sweep = Sweep("again", "lgbm-basic", {"learning_rate": [0.1]})
    search(sweep, _paths(processed), _protocol())

    with pytest.raises(TuningError, match="fresh id"):
        search(sweep, _paths(processed), _protocol())


@pytest.mark.parametrize("ceiling", [0, -5, 50, 2.5])
def test_a_ceiling_early_stopping_cannot_stop_under_is_refused(ceiling, tmp_path: Path):
    """Refused before any run opens: at or below the patience the ceiling decides every fit."""
    with pytest.raises(TuningError, match="ceiling"):
        search(
            Sweep("refused", "lgbm-basic", {"learning_rate": [0.1]}),
            SearchPaths(input_dir=tmp_path / "nowhere"),
            _protocol(ceiling=ceiling),
        )


def test_the_command_line_parses_a_grid_as_params_yaml_types():
    assert parse_assignments(["learning_rate=0.025,0.05", "num_leaves=31,63"], single=False) == {
        "learning_rate": [0.025, 0.05],
        "num_leaves": [31, 63],
    }
    with pytest.raises(TuningError, match="one value"):
        parse_assignments(["learning_rate=0.025,0.05"], single=True)
    with pytest.raises(TuningError, match="twice"):
        parse_assignments(["num_leaves=31,31"], single=False)
    with pytest.raises(TuningError, match="not a plain number"):
        parse_assignments(["learning_rate=1e-1"], single=False)


def test_an_unknown_variant_is_refused_by_name(tmp_path: Path):
    with pytest.raises(TuningError, match="lgbm-nothing"):
        search(
            Sweep("refused", "lgbm-nothing", {"learning_rate": [0.1]}),
            SearchPaths(input_dir=tmp_path / "nowhere"),
            _protocol(),
        )


def test_the_command_line_reports_a_refused_sweep_and_exits_1():
    result = CliRunner().invoke(
        tune.app, ["lgbm-basic", "--grid", "n_estimators=10,20", "--sweep", "refused"]
    )

    assert result.exit_code == 1


# --------------------------------------------------------------------------
# `min_child_samples`, optional at LightGBM's default
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def basic_data(processed: Path):
    return read_matrices(processed / "features", "basic", read_supported_makes(processed))


def _lightgbm(**params) -> dict:
    committed = load_params(PARAMS_FILE)["train"]["variants"]["lgbm-basic"]
    return {**committed, "params": {**committed["params"], **params}}


def test_leaving_min_child_samples_out_fits_exactly_the_trees_its_default_fits(basic_data):
    """So `params.yaml` and `dvc.lock` need not change until a search adopts another value."""
    implicit = fit_variant("lgbm-basic", _lightgbm(), basic_data, seed=1, num_threads=1)
    explicit = fit_variant(
        "lgbm-basic", _lightgbm(min_child_samples=20), basic_data, seed=1, num_threads=1
    )

    assert "min_child_samples" not in implicit.metadata["params"]
    assert implicit.booster.model_to_string() == explicit.booster.model_to_string()
    assert "[min_data_in_leaf: 20]" in implicit.booster.model_to_string()


def test_a_min_child_samples_value_reaches_the_booster(basic_data):
    model = fit_variant(
        "lgbm-basic", _lightgbm(min_child_samples=5), basic_data, seed=1, num_threads=1
    )

    assert "[min_data_in_leaf: 5]" in model.booster.model_to_string()
    assert model.metadata["params"]["min_child_samples"] == 5


def test_an_unknown_lightgbm_key_is_still_refused_and_the_optional_one_is_named(basic_data):
    with pytest.raises(ModelError, match="optionally min_child_samples"):
        fit_variant("lgbm-basic", _lightgbm(min_leaf=5), basic_data, seed=1, num_threads=1)


def test_only_lightgbm_has_an_optional_key():
    """Ridge and the baseline still read exactly the keys their params blocks set."""
    assert {name: dict(cls.optional_hyperparameters) for name, cls in ESTIMATORS.items()} == {
        "median_baseline": {},
        "ridge": {},
        LightGBMModel.estimator_name: {"min_child_samples": 20},
    }
