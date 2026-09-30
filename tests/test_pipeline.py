"""The pipeline end to end, and the wiring in `dvc.yaml`.

CI has no DVC and no DagsHub credentials, so `dvc repro` cannot be the gate
there. These tests run the same stage functions in the same order against the
synthetic fixture in a temporary directory, which covers the logic, and read
`dvc.yaml` to check that the DAG still describes the modules that exist.

`dvc repro` on a clean clone remains the manual check of NFR-06.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from tests.conftest import PII_COLUMNS
import yaml

from recommenditos.config import PARAMS_FILE, PROJ_ROOT
from recommenditos.data import (
    build_features,
    download_raw_dataset,
    gx_context_configuration,
    preprocess,
    split_data,
    validate_data,
)
from recommenditos.data.split_data import SPLIT_NAMES
from recommenditos.modeling import evaluate, train
from recommenditos.schema import (
    INTERIM_SCHEMA,
    PROCESSED_SCHEMA,
    RAW_SCHEMA,
    SchemaError,
    feature_schema,
)

DVC_FILE = PROJ_ROOT / "dvc.yaml"

#: Stage name to the module that implements it, as EDN-28 lays the package out.
STAGE_MODULES = {
    "download": "recommenditos.data.download_raw_dataset",
    "preprocess": "recommenditos.data.preprocess",
    "configure_gx": "recommenditos.data.gx_context_configuration",
    "validate-data": "recommenditos.data.validate_data",
    "split": "recommenditos.data.split_data",
    "features": "recommenditos.data.build_features",
    "train": "recommenditos.modeling.train",
    "evaluate": "recommenditos.modeling.evaluate",
}


@pytest.fixture(scope="session")
def dvc_stages() -> dict:
    return yaml.safe_load(DVC_FILE.read_text(encoding="utf-8"))["stages"]


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory, _generated_frame, params) -> dict:
    """Every stage after `download`, run in order on the fixture.

    Session-scoped, because it is the expensive fixture and every assertion
    below reads the same run.
    """
    root = tmp_path_factory.mktemp("pipeline")
    raw_path = root / "raw" / "listings.parquet"
    raw_path.parent.mkdir(parents=True)
    _generated_frame.to_parquet(raw_path, index=False)

    interim_path = root / "interim" / "listings.parquet"
    processed = root / "processed"
    features = processed / "features"
    models = root / "models"
    metrics_dir = root / "metrics"
    summary_path = root / "metrics.json"
    validation_path = root / "data-validation" / "summary.json"

    preprocess.main(raw_path, interim_path, PARAMS_FILE)
    # Every path is passed explicitly. A stage's defaults point into the real
    # repository, so a test that relies on them writes its result over the
    # pipeline's - which is exactly how a test run's validation summary once
    # ended up committed.
    validate_data.main(raw_path, interim_path, validation_path)
    split_data.main(interim_path, processed, PARAMS_FILE)
    for feature_set in params["features"]["sets"]:
        build_features.main(feature_set, processed, features, PARAMS_FILE)
    for variant in params["train"]["variants"]:
        train.main(variant, features, models, PARAMS_FILE)
    evaluate.main(features, models, metrics_dir, summary_path, PARAMS_FILE)

    return {
        "raw": raw_path,
        "interim": interim_path,
        "processed": processed,
        "features": features,
        "models": models,
        "metrics_dir": metrics_dir,
        "summary": summary_path,
        "validation": validation_path,
    }


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def test_the_pipeline_runs_end_to_end_without_data_or_credentials(pipeline):
    # No `req("NFR-06")` marker: NFR-06 is about `dvc repro` on a clean clone
    # reproducing the same splits and metrics within a tolerance, and about every
    # MLflow run recording its commit, data version and parameters. This asserts
    # that two files exist. Its evidence is the manual re-run before each delivery
    # that the specification names.
    assert pipeline["interim"].exists()
    assert pipeline["summary"].exists()


def test_every_stage_writes_a_contract_valid_frame(pipeline, params):
    INTERIM_SCHEMA.validate(pd.read_parquet(pipeline["interim"]))
    for name in (*SPLIT_NAMES, "holdout_es"):
        PROCESSED_SCHEMA.validate(pd.read_parquet(pipeline["processed"] / f"{name}.parquet"))
    for feature_set, columns in params["features"]["sets"].items():
        schema = feature_schema(columns, name=feature_set)
        for name in (*SPLIT_NAMES, "holdout_es"):
            schema.validate(
                pd.read_parquet(pipeline["features"] / feature_set / f"{name}.parquet")
            )


@pytest.mark.req("NFR-08")
def test_no_pii_column_survives_preprocessing(pipeline):
    interim = pd.read_parquet(pipeline["interim"])
    assert set(PII_COLUMNS).isdisjoint(interim.columns)


@pytest.mark.req("NFR-08")
def test_no_pii_column_reaches_a_feature_matrix(pipeline, params):
    for feature_set in params["features"]["sets"]:
        matrix = pd.read_parquet(pipeline["features"] / feature_set / "train.parquet")
        assert set(PII_COLUMNS).isdisjoint(matrix.columns)


def test_the_holdout_takes_every_spanish_listing_and_no_other(pipeline, params):
    holdout_country = params["split"]["holdout_country"]
    holdout = pd.read_parquet(pipeline["processed"] / "holdout_es.parquet")
    assert (holdout["country_code"] == holdout_country).all()

    for name in SPLIT_NAMES:
        subset = pd.read_parquet(pipeline["processed"] / f"{name}.parquet")
        assert not (subset["country_code"] == holdout_country).any()


def test_no_seller_appears_in_two_sets(pipeline, params):
    key = params["split"]["group_key"]
    seen: dict[str, str] = {}
    for name in SPLIT_NAMES:
        subset = pd.read_parquet(pipeline["processed"] / f"{name}.parquet")
        for group in subset[key].unique():
            assert seen.setdefault(group, name) == name, f"{group} is in {seen[group]} and {name}"


def test_the_split_loses_no_row(pipeline):
    interim = len(pd.read_parquet(pipeline["interim"]))
    parts = sum(
        len(pd.read_parquet(pipeline["processed"] / f"{name}.parquet"))
        for name in (*SPLIT_NAMES, "holdout_es")
    )
    assert parts == interim


def test_age_is_derived_and_missing_where_the_date_is(pipeline):
    matrix = pd.read_parquet(pipeline["features"] / "basic" / "train.parquet")
    assert "age_years" in matrix.columns
    assert "registration_date" not in matrix.columns


def test_a_model_is_written_for_every_variant(pipeline, params):
    for variant in params["train"]["variants"]:
        model = json.loads((pipeline["models"] / variant / "model.json").read_text())
        assert model["variant"] == variant
        assert model["feature_set"] == params["train"]["variants"][variant]["feature_set"]


def test_the_gate_reports_every_success_criterion(pipeline, params):
    # No `req("NFR-01")` marker: this asserts the shape of the metrics record, not
    # the rule NFR-01 states, which is that a model missing a criterion is not
    # released. The two tests below and the null-is-not-a-pass test carry that.
    for variant in params["train"]["variants"]:
        record = json.loads((pipeline["metrics_dir"] / f"{variant}.json").read_text())
        for criterion in evaluate.CRITERIA:
            assert f"{criterion}_measured" in record
            assert f"{criterion}_passed" in record


@pytest.mark.req("NFR-01")
def test_a_model_that_misses_the_criteria_does_not_pass_the_gate(pipeline):
    # The stub trains a constant predictor, so the gate must say no. A gate
    # that passes a model this bad would be worse than no gate.
    summary = json.loads(pipeline["summary"].read_text())
    assert summary["gate_passed"] is False
    assert summary["n_variants_passing"] == 0
    # Nothing is put forward for deployment while nothing passes.
    assert summary["deployable_variant"] is None


@pytest.mark.req("NFR-01")
def test_the_gate_discriminates_rather_than_rejecting_everything(params):
    # The test above cannot tell a working gate from one that says no to
    # everything, because SC-04 to SC-06 are null until issue #39 lands. This
    # one checks the criteria that are implemented actually distinguish.
    criteria = params["evaluate"]["success_criteria"]
    good = {"mdape": 0.067, "within_20pct": 0.906}
    bad = {"mdape": 0.547, "within_20pct": 0.161}

    passed = evaluate.evaluate_gate(good, baseline_mdape=0.119, criteria=criteria)
    failed = evaluate.evaluate_gate(bad, baseline_mdape=0.119, criteria=criteria)

    assert (passed["sc01_passed"], passed["sc02_passed"], passed["sc03_passed"]) == (
        True,
        True,
        True,
    )
    assert (failed["sc01_passed"], failed["sc02_passed"], failed["sc03_passed"]) == (
        False,
        False,
        False,
    )


@pytest.mark.req("NFR-01")
def test_an_unmeasured_criterion_is_null_rather_than_a_pass(pipeline):
    # NFR-01 is "only a model meeting every SC-01 to SC-06 is released", and SC-04
    # to SC-06 have no thresholds until issue #39 lands. This is the test that
    # keeps the word "every" honest in the meantime: an unmeasured criterion is
    # never recorded as met, so it can never be the reason a model is released.
    record = json.loads((pipeline["metrics_dir"] / "b0.json").read_text())
    for criterion in ("sc04", "sc05", "sc06"):
        assert record[f"{criterion}_passed"] is None


def test_validate_data_fails_the_stage_on_a_broken_frame(pipeline, tmp_path):
    # The course demo only logs the failure count, so a `dvc repro` there can
    # produce a model from data that never passed validation.
    broken = pd.read_parquet(pipeline["interim"]).drop(columns=["log_price"])
    broken_path = tmp_path / "broken.parquet"
    broken.to_parquet(broken_path, index=False)

    with pytest.raises(SchemaError):
        validate_data.main(pipeline["raw"], broken_path, tmp_path / "summary.json")

    # The summary is written before the raise, so a failed run leaves a record
    # of which artefact broke rather than only a traceback in the DVC log.
    written = json.loads((tmp_path / "summary.json").read_text())
    assert written["interim"]["passed"] is False


# --------------------------------------------------------------------------
# download (#33) and configure_gx (#25)
# --------------------------------------------------------------------------


def _params_override(tmp_path: Path, params: dict, **blocks) -> Path:
    """A copy of params.yaml with some keys of some blocks changed.

    Every stage takes its params file as an argument for exactly this reason:
    a test can vary a parameter without monkeypatching anything, which is what
    the stage tickets need to test their rules against the fixture.
    """
    changed = {**params, **{key: {**params[key], **value} for key, value in blocks.items()}}
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    return path


def test_download_writes_a_contract_valid_raw_frame(tmp_path, params):
    # The synthetic source, pinned here rather than inherited from params.yaml,
    # so that the flip to `zenodo` (#57) does not turn this test into a 548 MB
    # download. The real source is covered in test_download.py.
    output = tmp_path / "listings.parquet"
    download_raw_dataset.main(
        output,
        _params_override(
            tmp_path, params, download={"source": download_raw_dataset.SYNTHETIC, "rows": 60}
        ),
    )

    RAW_SCHEMA.validate(pd.read_parquet(output))


def test_download_refuses_a_source_it_does_not_implement(tmp_path, params):
    # Failing loudly beats silently producing synthetic data when someone
    # mistypes the parameter or names a source nobody has implemented.
    with pytest.raises(NotImplementedError, match="kaggle"):
        download_raw_dataset.main(
            tmp_path / "listings.parquet",
            _params_override(tmp_path, params, download={"source": "kaggle"}),
        )


def test_the_download_source_is_one_the_stage_implements(params):
    # Not pinned to `synthetic`: issue #57 flips this to `zenodo` together with
    # the lock refresh, and a tripwire that turns the suite red would just teach
    # its author to edit a test. What must hold is that the value names a source
    # the stage knows, which is what the stage exports `SOURCES` for.
    assert params["download"]["source"] in download_raw_dataset.SOURCES


def test_configure_gx_is_runnable():
    gx_context_configuration.main()


# --------------------------------------------------------------------------
# The wiring
# --------------------------------------------------------------------------


def test_dvc_yaml_defines_exactly_the_stages_the_package_implements(dvc_stages):
    assert set(dvc_stages) == set(STAGE_MODULES)


def test_every_stage_runs_its_own_module(dvc_stages):
    for stage, module in STAGE_MODULES.items():
        definition = dvc_stages[stage]
        body = definition.get("do", definition)
        assert f"-m {module}" in body["cmd"], stage


def test_every_stage_declares_its_own_module_as_a_dependency(dvc_stages):
    for stage, module in STAGE_MODULES.items():
        definition = dvc_stages[stage]
        body = definition.get("do", definition)
        path = module.replace(".", "/") + ".py"
        assert path in body["deps"], f"{stage} does not depend on {path}"


def test_every_declared_parameter_exists(dvc_stages, params):
    for stage, definition in dvc_stages.items():
        body = definition.get("do", definition)
        for declared in body.get("params", []):
            for key in _resolve(declared, definition, params):
                # A sentinel, not `is not None`: a parameter whose value is
                # legitimately 0, false or null is still declared.
                assert _lookup(params, key) is not _MISSING, (
                    f"{stage} reads a missing param: {key}"
                )


def test_every_foreach_iterates_a_real_parameter(dvc_stages, params):
    for stage, definition in dvc_stages.items():
        if "foreach" not in definition:
            continue
        key = definition["foreach"].removeprefix("${").removesuffix("}")
        assert isinstance(_lookup(params, key), dict), (
            f"{stage} iterates {key}, which must be a mapping so its stages are named "
            f"after the item rather than its position"
        )


def test_no_parameter_is_dead(dvc_stages, params):
    """Every key in params.yaml is declared by some stage.

    The reverse direction is checked above. This one catches the failure that
    actually happens: a key is added for a stage ticket, nobody declares it,
    and changing it reruns nothing - which quietly breaks the property EDN-30
    exists for.
    """
    declared: set[str] = set()
    for definition in dvc_stages.values():
        body = definition.get("do", definition)
        for entry in body.get("params", []):
            for key in _resolve(entry, definition, params):
                # Declaring `split` covers `split.ratios.train`, and declaring
                # `features.sets.basic` covers nothing above it.
                declared.add(key)

    undeclared = sorted(
        key
        for key in _leaf_groups(params)
        if not any(key == each or key.startswith(each + ".") for each in declared)
    )
    assert not undeclared, f"no stage declares: {', '.join(undeclared)}"


def _leaf_groups(params: dict, prefix: str = "") -> list[str]:
    """Every top-level key, and every second-level key under it."""
    keys = []
    for key, value in params.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict) and not prefix:
            keys.extend(_leaf_groups(value, f"{dotted}."))
        else:
            keys.append(dotted)
    return keys


def test_the_supported_make_list_has_the_shape_the_api_reads(pipeline):
    # FR-04 rejects a make outside this list and echoes it back, so the API
    # and the split stage have to agree on its shape. Fixing it here means
    # neither invents one.
    written = json.loads((pipeline["processed"] / "supported_makes.json").read_text())

    assert set(written) == {"supported_makes"}
    assert isinstance(written["supported_makes"], list)


def test_the_gate_is_in_the_graph_not_beside_it(dvc_stages):
    # The course demo's validate-data declares no outs, so nothing depends on
    # it and `dvc repro split` walks past the gate. Ours must not.
    summary = "reports/data-validation/summary.json"
    declared = [entry for entry in dvc_stages["validate-data"]["outs"]]
    assert any(summary in str(entry) for entry in declared)
    assert summary in dvc_stages["split"]["deps"]


def test_the_metrics_artefact_is_declared_so_dvc_metrics_diff_works(dvc_stages):
    declared = dvc_stages["evaluate"]["metrics"]
    assert any("metrics.json" in entry for entry in declared)


def _resolve(declared: str, definition: dict, params: dict) -> list[str]:
    """Expand a `${key}` or `${item.x}` placeholder in a params declaration."""
    if "${" not in declared:
        return [declared]
    foreach = _lookup(params, definition["foreach"].removeprefix("${").removesuffix("}"))
    resolved = []
    for key, item in foreach.items():
        expanded = declared.replace("${key}", key)
        # A foreach item is a mapping for `train.variants` and a plain list for
        # `features.sets`; only the former offers `${item.<field>}`.
        fields = item.items() if isinstance(item, dict) else ()
        for field, value in fields:
            expanded = expanded.replace("${item." + field + "}", str(value))
        resolved.append(expanded)
    return resolved


_MISSING = object()


def _lookup(params: dict, dotted: str):
    current = params
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


def test_the_params_file_dvc_reads_is_the_one_the_tests_read():
    assert PARAMS_FILE == PROJ_ROOT / "params.yaml"
    assert Path(PARAMS_FILE).exists()
