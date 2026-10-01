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
from tests.conftest import PII_COLUMNS, params_override
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
    TARGET_NAMES,
    SchemaError,
    load_feature_schema,
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

    # The fixture *is* what `download.source: synthetic` produces, so the stages
    # are run against a params file that says so rather than against the
    # project's, which names the real snapshot. Otherwise the metrics artefacts
    # this run writes would carry `data_source: zenodo` over generated rows -
    # the exact mislabel that field exists to make impossible.
    params_path = params_override(
        root, params, download={"source": download_raw_dataset.SYNTHETIC}
    )

    interim_path = root / "interim" / "listings.parquet"
    processed = root / "processed"
    features = processed / "features"
    models = root / "models"
    metrics_dir = root / "metrics"
    summary_path = root / "metrics.json"
    validation_path = root / "data-validation" / "summary.json"

    preprocess.main(raw_path, interim_path, params_path)
    # Every path is passed explicitly. A stage's defaults point into the real
    # repository, so a test that relies on them writes its result over the
    # pipeline's - which is exactly how a test run's validation summary once
    # ended up committed.
    validate_data.main(raw_path, interim_path, validation_path)
    split_data.main(interim_path, processed, params_path)
    for feature_set in params["features"]["sets"]:
        build_features.main(feature_set, processed, features, params_path)
    for variant in params["train"]["variants"]:
        train.main(variant, features, models, params_path)
    evaluate.main(features, models, metrics_dir, summary_path, params_path)

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
    for feature_set in params["features"]["sets"]:
        # Not `feature_schema` of the params list: the matrices carry the
        # equipment multi-hot columns too, so the contract to check them against
        # is the one their stage wrote beside them.
        schema = load_feature_schema(pipeline["features"] / feature_set, name=feature_set)
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


# Seller disjointness and the lossless partition are asserted in
# tests/test_split.py, which owns the stage: `test_no_seller_group_appears_in_two_sets`
# was body-for-body this file's copy, and `test_the_five_artefacts_partition_the_interim_frame`
# compares the rebuilt frame row by row instead of summing counts, so it also
# catches a row duplicated into one set and dropped from another.


def test_age_is_derived_and_missing_where_the_date_is(pipeline):
    matrix = pd.read_parquet(pipeline["features"] / "basic" / "train.parquet")
    assert "age_years" in matrix.columns
    assert "registration_date" not in matrix.columns


def test_a_model_is_written_for_every_variant(pipeline, params):
    for variant in params["train"]["variants"]:
        model = json.loads((pipeline["models"] / variant / "model.json").read_text())
        assert model["variant"] == variant
        assert model["feature_set"] == params["train"]["variants"][variant]["feature_set"]


def test_the_model_artefact_never_lists_the_target_among_its_features(pipeline, params):
    # `models/<variant>/model.json` is what #37 extends, what #39 reads and what
    # the API loads, so a `features` list ending in `price, log_price` asserts
    # that the target is an input. The matrix carries the label beside the
    # inputs, so the artefact has to say where the boundary is; 164 entries is
    # well past the point where anybody re-reads the list.
    for variant in params["train"]["variants"]:
        model = json.loads((pipeline["models"] / variant / "model.json").read_text())
        schema = load_feature_schema(
            pipeline["features"] / model["feature_set"], name=model["feature_set"]
        )

        assert set(model["features"]).isdisjoint(TARGET_NAMES), variant
        assert model["targets"] == list(TARGET_NAMES), variant
        # Together they are the matrix, so nothing is silently unaccounted for.
        assert model["features"] + model["targets"] == list(schema.names), variant


def test_the_reported_population_is_the_served_population(pipeline, params):
    # EDN-48: the model is fitted on the makes the API answers, so the metrics
    # have to be computed over those rows and no others. Otherwise the numbers
    # describe a population the product does not serve and cannot be compared
    # with the reference values of problem-spec section 8, which are all
    # post-filter. Asserted here rather than in tests/test_model.py because it is
    # a property of the two stages agreeing, not of one model.
    supported = {
        entry["make"]
        for entry in json.loads((pipeline["processed"] / "supported_makes.json").read_text())[
            "supported_makes"
        ]
    }
    for variant in params["train"]["variants"]:
        model = json.loads((pipeline["models"] / variant / "model.json").read_text())
        record = json.loads((pipeline["metrics_dir"] / f"{variant}.json").read_text())
        test = pd.read_parquet(pipeline["features"] / model["feature_set"] / "test.parquet")
        expected = int(test["make"].isin(supported).sum())
        assert 0 < expected < len(test), (
            "the fixture has to hold both supported and unsupported makes, "
            "or this test cannot fail"
        )
        assert record["n_test_rows"] == expected, variant


def test_the_gate_reports_every_success_criterion(pipeline, params):
    # No `req("NFR-01")` marker: this asserts the shape of the metrics record, not
    # the rule NFR-01 states, which is that a model missing a criterion is not
    # released. The two tests below and the null-is-not-a-pass test carry that.
    for variant in params["train"]["variants"]:
        record = json.loads((pipeline["metrics_dir"] / f"{variant}.json").read_text())
        for criterion in evaluate.CRITERIA:
            assert f"{criterion}_measured" in record
            assert f"{criterion}_passed" in record


def test_the_metrics_artefacts_say_which_data_and_which_estimator_produced_them(pipeline, params):
    # No `req` marker: this asserts the shape of the artefacts, not a requirement.
    # It exists because of a real defect rather than for completeness. Before the
    # `evaluate` stage was implemented, `metrics.json` carried one MdAPE under all
    # four variant names, computed by a constant-median stub on generated rows,
    # and nothing in the file said either of those things - so `dvc metrics show`
    # read as a model result. This run is on the stand-in, and these are the two
    # fields that have to say so.
    summary = json.loads(pipeline["summary"].read_text())
    assert summary["data_source"] == download_raw_dataset.SYNTHETIC

    for variant, settings in params["train"]["variants"].items():
        record = json.loads((pipeline["metrics_dir"] / f"{variant}.json").read_text())
        assert record["data_source"] == download_raw_dataset.SYNTHETIC, variant
        # Read out of the bundle rather than out of params, because what the
        # metrics have to name is the estimator that answered. The stub era is
        # the case that makes the distinction real: it wrote `constant_median`
        # beside the `median_baseline` the variant had asked for.
        fitted = json.loads((pipeline["models"] / variant / "model.json").read_text())["estimator"]
        assert record["estimator"] == fitted, variant
        assert summary["variants"][variant]["estimator"] == fitted, variant
        assert fitted == settings["estimator"], variant


@pytest.mark.req("NFR-01")
def test_a_model_that_misses_the_criteria_does_not_pass_the_gate(pipeline):
    # SC-05 has no measurement until the UC2 intervals exist, and a criterion that
    # was not measured is never a pass, so nothing can be released yet whatever
    # the fitted estimators score. NFR-01's word is "every", and this is what
    # keeps it honest.
    #
    # That the gate discriminates rather than simply refusing everything is
    # asserted in tests/test_model.py::test_the_gate_flips_every_measured_criterion_together,
    # which drives the same reference values through all five measurable criteria
    # instead of the three this file used to cover.
    summary = json.loads(pipeline["summary"].read_text())
    assert summary["gate_passed"] is False
    assert summary["n_variants_passing"] == 0
    # Nothing is put forward for deployment while nothing passes.
    assert summary["deployable_variant"] is None
    # And the artefact says which criteria are outstanding, so a reader can tell
    # "0 of 4" caused by the model from "0 of 4" caused by a missing measurement.
    # Both here: SC-05 needs the UC2 intervals, and on 2,000 fixture rows no
    # segment level reaches the 500-row bar SC-04 needs, so the criterion is not
    # measured either. On a real run only SC-05 is outstanding; the test below
    # lowers the bar so SC-04 produces a verdict on the fixture too.
    assert summary["criteria_not_measured"] == "sc04,sc05"


@pytest.mark.req("NFR-01")
def test_sc05_is_the_one_criterion_no_model_can_be_measured_against_yet(pipeline, params):
    # Narrowed to SC-05 deliberately: SC-04 and SC-06 are measured now, and a test
    # asserting they are `None` would be satisfied by an implementation that
    # returned `None` unconditionally.
    for variant in params["train"]["variants"]:
        record = json.loads((pipeline["metrics_dir"] / f"{variant}.json").read_text())
        assert record["sc05_passed"] is None, variant
        assert record["sc05_status"] == "not_measured", variant
        assert record["sc06_passed"] is not None, variant


@pytest.mark.req("NFR-01")
def test_sc04_is_measured_as_soon_as_a_segment_level_is_large_enough(pipeline, params, tmp_path):
    # On the 2,000-row fixture no segment level reaches the project's 500-row bar,
    # so SC-04 is `None` there whatever the code does - which means a test that
    # only asserted that would pass against `return None`. This run lowers the bar
    # instead, so the fixture produces a real verdict.
    #
    # The counts come from the fixture's own split, so they move with
    # `FIXTURE_ROWS` and `params["seed"]`; the assertion is on the shape and on
    # the largest levels rather than on the whole list.
    criteria = params["evaluate"]["success_criteria"]
    evaluate.main(
        pipeline["features"],
        pipeline["models"],
        tmp_path / "metrics",
        tmp_path / "metrics.json",
        params_override(
            tmp_path,
            params,
            evaluate={"success_criteria": {**criteria, "sc04_min_segment_rows": 50}},
        ),
    )
    record = json.loads((tmp_path / "metrics" / "b0.json").read_text())
    qualifying = [row for row in record["segments"] if row["counts_toward_sc04"]]

    assert record["sc04_measured"] is not None
    assert record["sc04_passed"] in (True, False)
    assert record["sc04_n_qualifying"] == len(qualifying) > 0
    assert record["sc04_min_segment_rows"] == 50
    assert record["sc04_worst_segment"]["mdape"] == record["sc04_measured"]
    assert all(row["n"] >= 50 for row in qualifying)


def test_the_report_tables_carry_the_columns_the_report_is_written_against(pipeline):
    # The header of each CSV, asserted as a line rather than through pandas, which
    # reads by name and would not notice a reordering. A LaTeX table and a
    # spreadsheet both read these files positionally as often as by name, so the
    # order is part of what the report is written against.
    segments = (pipeline["metrics_dir"] / evaluate.SEGMENTS_FILE).read_text().splitlines()
    masked = (pipeline["metrics_dir"] / evaluate.MASKED_INPUTS_FILE).read_text().splitlines()

    assert segments[0] == (
        "variant,segment,level,n,mdape,within_10pct,within_20pct,mae_eur,mape,"
        "counts_toward_sc04,excluded_because"
    )
    assert masked[0] == (
        "variant,criterion,field,n_columns_masked,mdape,mdape_ratio,threshold,passed"
    )


def test_the_report_tables_carry_every_variant_and_exclude_the_price_buckets(pipeline, params):
    # The two CSVs the report and its LaTeX tables read. The price buckets have to
    # be *present* and excluded rather than absent, because the report cites them
    # as the diagnostic that shows why they may not gate.
    segments = pd.read_csv(pipeline["metrics_dir"] / evaluate.SEGMENTS_FILE)
    masked = pd.read_csv(pipeline["metrics_dir"] / evaluate.MASKED_INPUTS_FILE)
    variants = list(params["train"]["variants"])

    assert list(segments["variant"].unique()) == variants
    assert list(masked["variant"].unique()) == variants
    price_buckets = segments[segments["segment"] == "price_bucket"]
    assert not price_buckets.empty
    assert not price_buckets["counts_toward_sc04"].any()
    assert (price_buckets["excluded_because"] == evaluate.EXCLUDED_TARGET_CONDITIONED).all()
    # Every variant's sweep ends in the all-at-once scenario and carries P1.
    assert (masked["field"] == evaluate.ALL_OPTIONAL_FIELDS).sum() == len(variants)
    assert (masked["criterion"] == "sc05").sum() == len(variants)


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


def test_download_writes_a_contract_valid_raw_frame(tmp_path, params):
    # The synthetic source, pinned here rather than inherited from params.yaml,
    # which names the real snapshot, so that this test is not a 548 MB download.
    # The real source is covered in test_download.py.
    output = tmp_path / "listings.parquet"
    download_raw_dataset.main(
        output,
        params_override(
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
            params_override(tmp_path, params, download={"source": "kaggle"}),
        )


def test_the_download_source_is_one_the_stage_implements(params):
    # Not pinned to either value: the project's source has already changed once,
    # and a tripwire that turns the suite red on the next change would just teach
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
            if "${key}" in entry:
                # Except the mapping a `foreach` iterates: it walks every key of
                # it, so `features.sets.${key}` declares all of `features.sets`,
                # spread over one stage per feature set. Without this, the only
                # way to keep the mapping covered is a stage declaring the whole
                # of it, which then reruns for a change to a set it does not use.
                declared.add(entry.split(".${key}")[0])

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
    # neither invents one. The two provenance keys are part of that shape: the
    # threshold and the population the counts were taken over, without which an
    # entry's count cannot be read at all (issue #35).
    written = json.loads((pipeline["processed"] / "supported_makes.json").read_text())

    assert set(written) == {"min_listings_per_make", "counted_over_rows", "supported_makes"}
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
