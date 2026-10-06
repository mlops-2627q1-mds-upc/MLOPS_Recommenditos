"""The data gate: `configure_gx` builds the suites, `validate-data` runs them (#25).

Everything runs against the synthetic fixture, in a context built in a
temporary directory, so the suite needs neither the raw data nor the `gx/`
store `dvc pull` brings. That is also the strongest test of `configure_gx`
there is: every assertion below runs against a store it built from nothing.

The rules are checked from both sides. A rule that passes the fixture proves
little on its own, because an expectation Great Expectations cannot evaluate
also fails rather than errors, and one that is evaluated on the wrong thing can
pass anything. So each rule has a frame that breaks it, and the test asserts the
break is caught - and, for a tolerant rule, that the tolerance is where
`params.yaml` puts it.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from tests import conftest
from tests.conftest import PII_COLUMNS

from recommenditos.config import PARAMS_FILE
from recommenditos.data import gx_context_configuration as configure_gx
from recommenditos.data import validate_data
from recommenditos.data.gx_context_configuration import (
    BOUNDED_NULLABLE_COLUMNS,
    CONTRACT,
    INTERIM,
    KIND,
    RAW,
    RAW_AS_VALIDATED,
    RULE,
)
from recommenditos.data.preprocess import main as preprocess_main
from recommenditos.data.validate_data import (
    DataValidationError,
    failures,
    load_context,
    validate_frame,
)
from recommenditos.data.validate_data import main as validate_main
from recommenditos.schema import INTERIM_SCHEMA, RAW_SCHEMA

DATE_RULE = ("expect_column_values_to_be_between", "registration_date")
MILEAGE_RULE = ("expect_column_values_to_be_between", "mileage_km_raw")
PRICE_RULE = ("expect_column_values_to_be_between", "price")
COLUMN_LIST = "expect_table_columns_to_match_ordered_list"
NOT_NULL = "expect_column_values_to_not_be_null"
ROW_COUNT = "expect_table_row_count_to_be_between"

#: The columns the raw contract fills and the interim contract lets be null,
#: which the interim suite asserts preprocessing keeps filled.
KEPT_FILLED = (
    "make",
    "body_type",
    "equipment_comfort",
    "equipment_entertainment",
    "equipment_extra",
    "equipment_safety",
)

#: A registration date after the 2025-11-08 reference date, as the raw file
#: writes one. 137 of the 164 such dates in the snapshot are in January 2026.
LATE = "2026-01-01"


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict[str, Path]:
    """A context built by the real stage, with its outputs redirected."""
    root = tmp_path_factory.mktemp("gx")
    paths = {
        "root": root,
        "context": root / "gx",
        "results": root / "data-validation" / "results",
        "docs": root / "data-validation" / "data-docs",
    }
    configure_gx.main(paths["context"], paths["results"], paths["docs"], PARAMS_FILE)
    return paths


@pytest.fixture(scope="module")
def context(built):
    return load_context(built["context"])


@pytest.fixture(scope="module")
def frames(_generated_frame, tmp_path_factory) -> dict:
    """The fixture as `download` and `preprocess` would have written it."""
    root = tmp_path_factory.mktemp("frames")
    raw_path = root / "raw.parquet"
    _generated_frame.to_parquet(raw_path, index=False)
    interim_path = root / "interim.parquet"
    preprocess_main(raw_path, interim_path, PARAMS_FILE)
    return {
        "raw_path": raw_path,
        "interim_path": interim_path,
        "raw": pd.read_parquet(raw_path),
        "interim": pd.read_parquet(interim_path),
    }


@pytest.fixture(scope="module")
def stage(built, frames) -> dict:
    """The stage run once on the fixture, and the summary it wrote."""
    summary_path = built["root"] / "data-validation" / "summary.json"
    returned = validate_main(
        frames["raw_path"], frames["interim_path"], summary_path, built["context"]
    )
    return {"returned": returned, "path": summary_path}


def rule(entry: dict, expectation: str, column: str) -> dict:
    """The one rule of a frame's summary entry with this expectation and column."""
    (found,) = [
        each
        for each in entry["rules"]
        if each["expectation"] == expectation and each.get("column") == column
    ]
    return found


def _tree_digest(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(directory)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------
# The stage on the fixture
# --------------------------------------------------------------------------


def test_both_frames_pass_their_contract_and_their_suite(stage):
    for name in (RAW, INTERIM):
        entry = stage["returned"][name]
        assert entry["passed"], failures(entry)
        assert entry["contract"]["failed"] == []
        assert all(each["success"] for each in entry["rules"])


def test_the_summary_on_disk_is_what_the_stage_returned(stage):
    assert json.loads(stage["path"].read_text(encoding="utf-8")) == stage["returned"]


def test_the_summary_carries_no_run_id_or_timestamp(stage, built, frames, tmp_path):
    # It is git-tracked, so a rerun on unchanged data has to write the same
    # bytes: otherwise every `dvc repro` would be a diff, and `split`, which
    # depends on the file, would rerun for nothing.
    again = tmp_path / "summary.json"
    validate_main(frames["raw_path"], frames["interim_path"], again, built["context"])

    assert again.read_bytes() == stage["path"].read_bytes()


def test_validation_renders_the_data_docs_outside_the_context(stage, built):
    index = built["docs"] / "index.html"
    assert index.exists()
    pages = {
        path.parent.parent.parent.name for path in built["docs"].glob("validations/**/*.html")
    }
    assert pages == {RAW, INTERIM}
    assert not list(built["context"].glob("**/*.html"))


# --------------------------------------------------------------------------
# EDN-22: registration dates after the reference date
# --------------------------------------------------------------------------


def test_the_raw_suite_flags_the_late_registration_and_tolerates_it(stage, raw_frame, params):
    # The fixture holds exactly one, so the count says the rule is evaluated
    # on the parsed date rather than on the text, which it cannot be.
    late = pd.to_datetime(raw_frame["registration_date"]) > pd.Timestamp(params["reference_date"])
    found = rule(stage["returned"][RAW], *DATE_RULE)

    assert late.sum() == 1
    assert found["unexpected_count"] == 1
    assert found["success"]
    assert found["mostly"] == params["validate"]["raw_mostly"]


def test_the_raw_tolerance_is_the_one_params_yaml_sets(context, raw_frame, params):
    # Just inside and just outside `validate.raw_mostly`, so a tolerance that
    # drifted away from the parameter, in either direction, fails here. Counted
    # from a frame with no late registration at all, so the fixture's own one
    # does not shift the count.
    reference = pd.Timestamp(params["reference_date"])
    dates = pd.to_datetime(raw_frame["registration_date"])
    on_time = raw_frame.assign(
        registration_date=raw_frame["registration_date"].mask(
            dates > reference, reference.strftime("%Y-%m-%d")
        )
    )
    dated = on_time.index[on_time["registration_date"].notna()]
    tolerated = int((1 - params["validate"]["raw_mostly"]) * len(dated))

    def with_late(count: int) -> pd.DataFrame:
        frame = on_time.copy()
        frame.loc[dated[:count], "registration_date"] = LATE
        return frame

    assert tolerated > 0
    assert validate_frame(context, RAW, with_late(tolerated))["passed"]
    beyond = validate_frame(context, RAW, with_late(tolerated + 1))
    assert not beyond["passed"]
    assert not rule(beyond, *DATE_RULE)["success"]


def test_a_late_registration_in_the_cleaned_frame_fails_the_stage(built, frames, tmp_path):
    # The hard half of EDN-22, through the stage itself: a failed expectation
    # fails `validate-data`, which is what stops `split` and everything after
    # it. The course demo only logs the failure count.
    interim = frames["interim"].copy()
    interim.loc[interim.index[0], "registration_date"] = pd.Timestamp(LATE)
    broken = tmp_path / "interim.parquet"
    interim.to_parquet(broken, index=False)
    summary_path = tmp_path / "summary.json"

    with pytest.raises(DataValidationError, match="registration_date"):
        validate_main(frames["raw_path"], broken, summary_path, built["context"])

    # The summary is written before the raise, so the failed run leaves a
    # record of which rule broke and how often.
    written = json.loads(summary_path.read_text(encoding="utf-8"))
    assert written[RAW]["passed"]
    assert not written[INTERIM]["passed"]
    assert rule(written[INTERIM], *DATE_RULE)["unexpected_count"] == 1


# --------------------------------------------------------------------------
# The ranges
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("bound", "offset"), [("price_min_eur", -0.01), ("price_max_eur", 0.01)])
def test_a_price_outside_the_training_range_fails_the_cleaned_suite(
    context, frames, params, bound, offset
):
    # One cent outside. The fixture already carries a listing exactly on each
    # bound and passes, so together the two say the range is inclusive.
    interim = frames["interim"].copy()
    interim.loc[interim.index[0], "price"] = params["preprocess"][bound] + offset

    found = rule(validate_frame(context, INTERIM, interim), *PRICE_RULE)
    assert not found["success"]
    assert found["unexpected_count"] == 1


def test_the_raw_suite_does_not_hold_the_published_file_to_the_training_range(stage, frames):
    # The raw file runs from 1 EUR to 13.5M EUR and the fixture reproduces both
    # ends; the range is a filter preprocessing applies, so the raw frame breaks
    # it by design and the raw suite has no price rule to break.
    assert (frames["raw"]["price"] < 500).any()
    assert not [each for each in stage["returned"][RAW]["rules"] if each.get("column") == "price"]


@pytest.mark.parametrize("name", [RAW, INTERIM])
def test_the_mileage_rule_tolerates_a_few_readings_and_no_more(context, frames, params, name):
    # A check, not a filter: preprocessing keeps the snapshot's three readings
    # above the range, so the rule must pass them on both frames and still fail
    # a systematic break. The tolerance is a share, so the count it allows is
    # taken from the frame rather than from the snapshot's three.
    frame = frames[name]
    read = frame.index[frame["mileage_km_raw"].notna()]
    tolerated = int((1 - params["validate"]["mileage_mostly"]) * len(read))

    def beyond(count: int) -> pd.DataFrame:
        broken = frame.copy()
        broken.loc[read[:count], "mileage_km_raw"] = params["validate"]["mileage_max_km"] + 1
        return broken

    assert tolerated > 0
    inside = rule(validate_frame(context, name, beyond(tolerated)), *MILEAGE_RULE)
    assert inside["success"]
    assert inside["unexpected_count"] == tolerated
    assert not rule(validate_frame(context, name, beyond(tolerated + 1)), *MILEAGE_RULE)["success"]


# --------------------------------------------------------------------------
# The contract, and what the suite adds to it
# --------------------------------------------------------------------------


@pytest.mark.req("NFR-08")
@pytest.mark.parametrize("column", PII_COLUMNS)
def test_the_cleaned_suite_refuses_a_pii_column(context, frames, column):
    # NFR-08 names this suite as its evidence on the processed data. The
    # contract keeps the columns out by selecting, so this is the independent
    # check that a frame which somehow carried one would not pass the gate.
    found = validate_frame(context, INTERIM, frames["interim"].assign(**{column: "x"}))

    assert not found["passed"]
    assert [each for each in found["contract"]["failed"] if each.startswith(COLUMN_LIST)]


def test_the_cleaned_suite_refuses_is_used(context, frames):
    # EDN-23 excludes it as a feature: `offer_type` carries the same
    # information reliably, and a `False` in it means "not asserted".
    found = validate_frame(context, INTERIM, frames["interim"].assign(is_used=True))
    assert not found["passed"]


def test_the_interim_suite_keeps_filled_exactly_the_columns_the_raw_contract_fills(context):
    # Written out rather than derived, because the suite derives the list from
    # the two contracts: a contract change that silently dropped a column from
    # it would pass a derived assertion. Hard, unlike the two fill rates of the
    # bounded columns, which carry `validate.filled_mostly`.
    kept = {
        e.column: e.mostly
        for e in context.suites.get(INTERIM).expectations
        if e.meta.get(KIND) == RULE
        and e.expectation_type == NOT_NULL
        and e.column not in BOUNDED_NULLABLE_COLUMNS
    }

    assert sorted(kept) == sorted(KEPT_FILLED)
    assert set(kept.values()) == {1}


@pytest.mark.parametrize("column", KEPT_FILLED)
def test_a_gap_preprocessing_introduces_is_reported_by_the_interim_suite(context, frames, column):
    # The two layers, side by side: the interim contract lets these columns be
    # null, so this rule is what reports a gap in the cleaned frame. It is a
    # check on preprocessing, not on the scrape: the raw contract declares them
    # filled, so a gap in a future scrape already fails `download` with a
    # contract error and never reaches this suite.
    interim = frames["interim"].copy()
    interim.loc[interim.index[0], column] = None
    INTERIM_SCHEMA.validate(interim)

    found = validate_frame(context, INTERIM, interim)
    assert not found["passed"]
    assert rule(found, NOT_NULL, column)["unexpected_count"] == 1


@pytest.mark.parametrize("flag", ["has_full_service_history", "non_smoking", "is_rental"])
def test_a_condition_flag_must_be_a_boolean(context, raw_frame, flag):
    # Checked by Great Expectations itself, not only by the contract: the frame
    # goes to the suite directly, past the read that would have refused it.
    found = validate_frame(context, RAW, raw_frame.assign(**{flag: raw_frame[flag].astype("str")}))
    assert not found["passed"]


def test_no_rule_holds_is_used_to_offer_type(context, raw_frame):
    # EDN-23: `is_used = False` under `offer_type = U` is "not asserted", in
    # 18,108 of the snapshot's used passenger-car rows. A rule that they agree
    # would fail the published file by design, so there is none to fail.
    assert ((raw_frame["offer_type"] == "U") & ~raw_frame["is_used"]).any()
    assert validate_frame(context, RAW, raw_frame.assign(is_used=False))["passed"]


def test_the_suites_check_exactly_the_columns_the_contract_declares_non_null(context):
    # The generated not-null checks, one per non-nullable column and no more, so
    # a loop that dropped them, or one that checked a nullable column, fails.
    for name, schema in ((RAW, RAW_AS_VALIDATED), (INTERIM, INTERIM_SCHEMA)):
        contract = [
            e for e in context.suites.get(name).expectations if e.meta.get(KIND) == CONTRACT
        ]
        checked = sorted(e.column for e in contract if e.expectation_type == NOT_NULL)

        assert checked
        assert checked == sorted(column.name for column in schema.columns if not column.nullable)


def test_the_suites_are_written_from_the_contracts(context):
    for name, schema in ((RAW, RAW_AS_VALIDATED), (INTERIM, INTERIM_SCHEMA)):
        suite = context.suites.get(name)
        contract = [e for e in suite.expectations if e.meta.get(KIND) == CONTRACT]
        (columns,) = [e for e in contract if e.expectation_type == COLUMN_LIST]
        types = {e.column: e.type_ for e in contract if e.expectation_type.endswith("of_type")}

        assert columns.column_list == list(schema.names)
        assert types == schema.dtypes


def test_the_raw_suite_sees_the_published_file_with_only_the_date_parsed():
    # Great Expectations cannot bound a text date, so this one column is cast
    # by the interim contract's own cast. Nothing else may differ, or the raw
    # suite would stop describing the published file.
    assert RAW_AS_VALIDATED.names == RAW_SCHEMA.names
    differs = [
        name
        for name in RAW_SCHEMA.names
        if RAW_AS_VALIDATED.column(name) != RAW_SCHEMA.column(name)
    ]
    assert differs == ["registration_date"]
    assert RAW_AS_VALIDATED.column("registration_date").dtype == (
        INTERIM_SCHEMA.column("registration_date").dtype
    )


# --------------------------------------------------------------------------
# Vacuous passes: a rule that holds because there is nothing to check
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", [RAW, INTERIM])
def test_an_empty_frame_fails_its_suite(context, frames, params, name):
    # Every column rule holds over no rows at all, so without a row count an
    # empty frame would pass every expectation of its suite.
    found = validate_frame(context, name, frames[name].iloc[0:0])
    (count,) = [each for each in found["rules"] if each["expectation"] == ROW_COUNT]

    assert not found["passed"]
    assert not count["success"]
    assert count["min_value"] == params["validate"]["min_rows"]


@pytest.mark.parametrize("name", [RAW, INTERIM])
@pytest.mark.parametrize("column", ["registration_date", "mileage_km_raw"])
def test_a_bounded_column_that_empties_out_fails_its_suite(context, frames, params, name, column):
    # Great Expectations skips missing values, so a range over a column with no
    # values left holds vacuously. The column is emptied without changing its
    # dtype, so the contract cannot be what catches it.
    frame = frames[name].copy()
    frame[column] = frame[column].where(pd.Series(False, index=frame.index))
    found = validate_frame(context, name, frame)
    filled = rule(found, NOT_NULL, column)

    assert rule(found, "expect_column_values_to_be_between", column)["success"]
    assert not filled["success"]
    assert filled["mostly"] == params["validate"]["filled_mostly"]
    assert not found["passed"]


def test_a_suite_with_no_expectations_is_refused(tmp_path, frames):
    # An empty suite passes any frame, so it is refused rather than run.
    context_dir = tmp_path / "gx"
    configure_gx.main(context_dir, tmp_path / "results", tmp_path / "docs", PARAMS_FILE)
    suite = load_context(context_dir).suites.get(RAW)
    for expectation in list(suite.expectations):
        suite.delete_expectation(expectation)
    suite.save()

    with pytest.raises(DataValidationError, match="no expectations"):
        validate_frame(load_context(context_dir), RAW, frames["raw"])


def test_a_context_validates_against_its_own_suites_when_another_was_loaded_since(
    context, frames, tmp_path
):
    # Great Expectations resolves a checkpoint through the context loaded last
    # in the process. Without `validate_frame` making its own context current,
    # this run would read the other store and fail its freshness check.
    configure_gx.main(tmp_path / "gx", tmp_path / "results", tmp_path / "docs", PARAMS_FILE)
    load_context(tmp_path / "gx")

    assert validate_frame(context, RAW, frames["raw"])["passed"]


def test_a_run_that_returns_fewer_results_than_the_suite_holds_is_refused(frames, monkeypatch):
    # A result Great Expectations could not compute still comes back, as a
    # failure; one that does not come back at all would read as a pass. A stub
    # stands in for the context, because no real run can be made to drop one.
    validation = SimpleNamespace(success=True, results=[object(), object()])
    checkpoint = SimpleNamespace(run=lambda **_: SimpleNamespace(run_results={"run": validation}))
    suite = SimpleNamespace(expectations=[object(), object(), object()])
    context = SimpleNamespace(
        checkpoints=SimpleNamespace(get=lambda _: checkpoint),
        suites=SimpleNamespace(get=lambda _: suite),
    )

    # The stub must not become Great Expectations' current project for the
    # tests that run after this one.
    monkeypatch.setattr(validate_data.project_manager, "set_project", lambda _: None)

    with pytest.raises(DataValidationError, match="3 expectations but 2 results"):
        validate_frame(context, INTERIM, frames["interim"])


def test_a_rule_great_expectations_could_not_evaluate_says_why(context, frames):
    # Great Expectations turns an exception into a failed result, which on its
    # own reads like a data problem. A date column written as text makes the
    # date bound raise, and the summary has to say that it did.
    interim = frames["interim"]
    found = validate_frame(
        context,
        INTERIM,
        interim.assign(registration_date=interim["registration_date"].astype("str")),
    )
    entry = rule(found, "expect_column_values_to_be_between", "registration_date")

    assert not entry["success"]
    assert "error" in entry
    assert "not evaluated" in " ".join(failures(found))


# --------------------------------------------------------------------------
# The context as a stage output
# --------------------------------------------------------------------------


def test_running_the_suites_leaves_the_context_unchanged(context, built, frames):
    # `gx/` is `configure_gx`'s DVC output. If a checkpoint run wrote into it,
    # `dvc status` would report that stage changed after every validation.
    before = _tree_digest(built["context"])
    validate_frame(context, RAW, frames["raw"])

    assert _tree_digest(built["context"]) == before


def test_the_stored_results_carry_no_fingerprint_of_the_frame(stage, built):
    # Great Expectations hashes the whole frame into each batch's markers, which
    # on the real raw frame takes the stage from 2.2 GB to 4.4 GB of peak memory.
    # `validate_data` switches that off through a module constant of Great
    # Expectations; if an upgrade renames it, the switch silently stops working,
    # and this is where that shows.
    stored = [path.read_text(encoding="utf-8") for path in built["results"].rglob("*.json")]

    assert stored
    assert not [text for text in stored if "pandas_data_fingerprint" in text]


def test_the_context_names_no_path_of_the_machine_that_built_it(built):
    # The demo's store records the absolute path of its author's checkout, so
    # it works nowhere else. Ours is pulled into every clone.
    machine = str(built["root"])
    for path in built["context"].rglob("*"):
        if path.is_file():
            assert machine not in path.read_text(encoding="utf-8", errors="replace"), path


def test_configure_gx_rebuilds_rather_than_accumulates(tmp_path, params):
    # Run twice, as `dvc repro` on a machine that already has the store does
    # when DVC is not the one deleting it first. The second run must neither
    # fail, as the demo's does, nor keep anything the first one wrote.
    context_dir = tmp_path / "gx"
    stray = context_dir / "expectations" / "a_suite_nobody_defines.json"
    for _ in range(2):
        configure_gx.main(context_dir, tmp_path / "results", tmp_path / "docs", PARAMS_FILE)
        stray.write_text("{}", encoding="utf-8")
    configure_gx.main(context_dir, tmp_path / "results", tmp_path / "docs", PARAMS_FILE)

    assert not stray.exists()
    assert sorted(path.stem for path in (context_dir / "expectations").glob("*.json")) == [
        INTERIM,
        RAW,
    ]


def test_configure_gx_refuses_to_delete_what_it_did_not_build(tmp_path):
    keep = tmp_path / "gx" / "notes.txt"
    keep.parent.mkdir()
    keep.write_text("not a Great Expectations context", encoding="utf-8")

    with pytest.raises(FileExistsError):
        configure_gx.main(keep.parent, tmp_path / "results", tmp_path / "docs", PARAMS_FILE)
    assert keep.exists()


def test_the_conftest_guard_watches_the_paths_the_stages_write():
    # The guard repeats these paths rather than importing them, because the
    # import would load Great Expectations into every test run; a copy that
    # drifted would make it watch a directory no stage writes.
    assert conftest.GX_DIR == configure_gx.GX_DIR
    assert conftest.VALIDATION_DIR == configure_gx.VALIDATION_DIR
    assert conftest.VALIDATION_SUMMARY_FILE == configure_gx.VALIDATION_SUMMARY_FILE


def test_a_missing_context_is_reported_rather_than_created(tmp_path):
    # `gx.get_context` creates an empty store wherever it is pointed, so
    # without the check the stage would fail later on a missing checkpoint.
    missing = tmp_path / "gx"
    with pytest.raises(DataValidationError, match="configure_gx"):
        load_context(missing)
    assert not missing.exists()
