"""The `monitoring_samples` stage: the replay's requests and the drift reference (#227, #228).

The stage runs on the `split` frames of the synthetic fixture with sizes that
fixture can fill. What is checked is what the replay and the drift job rely on:
the sizes, that reference and control share no listing, that every body is a
request the API accepts with nothing sent as `null`, that a listing id leads
back to its listing, and that the same seed writes the same bytes.
"""

import ast
from collections.abc import Iterator
from pathlib import Path
import sys

import pandas as pd
import pytest
from tests.conftest import params_override
import yaml

from recommenditos.api.schemas.partial_listing import PartialListing
from recommenditos.api.schemas.predict import ValuationRequest
from recommenditos.config import PROJ_ROOT
from recommenditos.data import drift_features as drift_module
from recommenditos.data import monitoring_samples
from recommenditos.data.build_features import age_years, read_supported_makes
from recommenditos.data.drift_features import (
    ABSENT,
    CATEGORICAL,
    DRIFT_FEATURES,
    MISSING,
    NUMERIC,
    PRESENT,
    as_of_for,
    drift_features,
    months_since_reference,
    shifted_registration,
)
from recommenditos.data.monitoring_samples import MANIFEST_FILE, SAMPLES, read_replay
from recommenditos.data.split_data import HOLDOUT_NAME

REFERENCE_SIZE = 300
CONTROL_SIZE = 400


@pytest.fixture(scope="module")
def processed(matrices: dict) -> Path:
    return Path(matrices["processed"])


def run_stage(processed: Path, root: Path, params: dict, **monitoring) -> Path:
    sizes = {"reference_size": REFERENCE_SIZE, "control_size": CONTROL_SIZE, **monitoring}
    root.mkdir(parents=True, exist_ok=True)
    params_path = params_override(root, params, monitoring=sizes)
    output = root / "monitoring"
    monitoring_samples.main(processed, output, params_path)
    return output


@pytest.fixture(scope="module")
def samples(processed: Path, params: dict, tmp_path_factory) -> Path:
    return run_stage(processed, tmp_path_factory.mktemp("samples"), params)


def replay(samples: Path, name: str) -> list[dict]:
    return read_replay(samples / f"{name}.jsonl.gz")


def nulls(value: object) -> Iterator[object]:
    """Every `None` anywhere inside a JSON value."""
    if value is None:
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from nulls(item)
    elif isinstance(value, list):
        for item in value:
            yield from nulls(item)


def test_each_sample_has_its_size_and_the_es_sample_every_accepted_holdout_listing(samples):
    manifest = yaml.safe_load((samples / MANIFEST_FILE).read_text(encoding="utf-8"))
    es_accepted = manifest["sources"][HOLDOUT_NAME]["accepted"]

    assert es_accepted > 0, "the fixture's holdout gives no accepted listing, so ES checks nothing"
    expected = {"reference": REFERENCE_SIZE, "control": CONTROL_SIZE, "es": es_accepted}
    for name, size in expected.items():
        assert len(replay(samples, name)) == size
        assert len(pd.read_parquet(samples / f"{name}.parquet")) == size
        assert manifest["samples"][name] == {"source": SAMPLES[name], "rows": size}


def test_reference_and_control_share_no_listing_and_come_from_train(samples):
    ids = {name: [r["listing_id"] for r in replay(samples, name)] for name in SAMPLES}

    assert not set(ids["reference"]) & set(ids["control"])
    for name, listing_ids in ids.items():
        assert len(set(listing_ids)) == len(listing_ids), f"{name} holds a listing twice"
        assert all(i.startswith(f"{SAMPLES[name]}:") for i in listing_ids)


@pytest.mark.parametrize("name", list(SAMPLES))
def test_every_body_is_a_request_the_api_accepts_with_nothing_sent_as_null(
    samples, processed, name
):
    supported = set(read_supported_makes(processed))
    for record in replay(samples, name):
        body = record["body"]
        ValuationRequest.model_validate(body)
        PartialListing.model_validate(body)
        assert body["make"] in supported
        assert not list(nulls(body)), f"{record['listing_id']} sends a null"
        assert record["observed_price_eur"] > 0


@pytest.mark.parametrize("name", list(SAMPLES))
def test_a_listing_id_leads_back_to_its_listing_and_price(samples, processed, name):
    frames = {
        source: pd.read_parquet(processed / f"{source}.parquet")
        for source in set(SAMPLES.values())
    }
    for record in replay(samples, name):
        source, position = record["listing_id"].split(":")
        listing = frames[source].iloc[int(position)]
        assert listing["price"] == record["observed_price_eur"]
        assert listing["make"] == record["body"]["make"]
        assert (
            listing["registration_date"].strftime("%Y-%m") == record["body"]["registration_date"]
        )


@pytest.mark.parametrize("name", list(SAMPLES))
def test_the_parquet_holds_the_drift_features_of_the_same_listings_in_the_same_order(
    samples, params, name
):
    records = replay(samples, name)
    frame = pd.read_parquet(samples / f"{name}.parquet")

    assert list(frame.columns) == ["listing_id", "observed_price_eur", *DRIFT_FEATURES]
    assert len(DRIFT_FEATURES) == 23 and "model" not in DRIFT_FEATURES
    assert frame["listing_id"].tolist() == [r["listing_id"] for r in records]
    expected = drift_features([r["body"] for r in records], params["reference_date"])
    pd.testing.assert_frame_equal(frame[list(DRIFT_FEATURES)], expected, check_dtype=False)


def test_drift_features_read_an_unsent_field_as_missing(params):
    sent = {"registration_date": "2020-11", "mileage_km_raw": 1000, "drive_train": "4WD"}
    features = drift_features([sent, {"registration_date": "2020-11"}], params["reference_date"])

    expected_age = age_years(pd.Series([pd.Timestamp("2020-11-01")]), params["reference_date"])
    assert features["age_years"].tolist() == [expected_age[0]] * 2
    assert features["mileage_km_raw"].iloc[0] == 1000 and pd.isna(features["mileage_km_raw"][1])
    assert features["mileage_km_raw__missing"].tolist() == [PRESENT, ABSENT]
    assert features["gears__missing"].tolist() == [ABSENT, ABSENT]
    assert features["drive_train"].tolist() == ["4WD", MISSING]
    for name in CATEGORICAL:
        assert features[name].notna().all()
    for name in NUMERIC:
        assert features[name].dtype == "float64"


@pytest.mark.req("NFR-06")
def test_the_same_seed_writes_the_same_bytes_and_another_seed_another_reference(
    samples, processed, params, tmp_path
):
    again = run_stage(processed, tmp_path / "again", params)
    for path in sorted(samples.iterdir()):
        assert path.read_bytes() == (again / path.name).read_bytes(), path.name

    other = run_stage(processed, tmp_path / "other", {**params, "seed": params["seed"] + 1})
    assert replay(other, "reference") != replay(samples, "reference")


def test_the_stage_refuses_sizes_train_cannot_fill(processed, params, tmp_path):
    with pytest.raises(ValueError, match="fewer than"):
        run_stage(processed, tmp_path, params, control_size=10**6)


def _imports(path: Path) -> set[str]:
    """Every module name `path` imports, by `import x.y` or by `from x.y import z`."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def _imported_modules(path: Path) -> set[Path]:
    """The project modules `path` imports, directly or through other project modules."""
    seen: set[Path] = set()
    pending = [path]
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        seen.add(module)
        for name in _imports(module):
            if name.startswith("recommenditos"):
                target = (PROJ_ROOT / Path(*name.split("."))).with_suffix(".py")
                if target.is_file():
                    pending.append(target)
    return seen


def test_dvc_yaml_declares_every_module_the_stage_runs_and_nothing_downstream_reads_it():
    stages = yaml.safe_load((PROJ_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    stage = stages["monitoring_samples"]
    modules = _imported_modules(PROJ_ROOT / "recommenditos/data/monitoring_samples.py")

    declared = set(stage["deps"])
    missing = {m.relative_to(PROJ_ROOT).as_posix() for m in modules} - declared
    assert not missing, f"the stage runs {sorted(missing)} but does not declare them"
    assert stage["outs"] == ["data/monitoring"]
    for name, other in stages.items():
        deps = other.get("deps", []) or other.get("do", {}).get("deps", [])
        assert not any(dep.startswith("data/monitoring") for dep in deps), name


def test_the_stage_writes_nothing_but_its_samples_and_manifest(samples):
    files = sorted(path.name for path in samples.iterdir())
    expected = sorted(
        [MANIFEST_FILE] + [f"{n}.jsonl.gz" for n in SAMPLES] + [f"{n}.parquet" for n in SAMPLES]
    )
    assert files == expected


def test_the_drift_feature_module_imports_only_numpy_pandas_and_the_standard_library():
    """#228's environment (pandas 2, numpy 1.26) carries a copy and has no `recommenditos`."""
    path = Path(drift_module.__file__)
    roots = {name.split(".")[0] for name in _imports(path)}

    assert roots - {"numpy", "pandas"} <= set(sys.stdlib_module_names), roots


def test_the_inlined_age_is_the_one_training_computes(params):
    bodies = [{"registration_date": month} for month in ("1999-01", "2016-02", "2025-11")]
    registered = pd.to_datetime(pd.Series(["1999-01-01", "2016-02-01", "2025-11-01"]))

    expected = age_years(registered, params["reference_date"])
    actual = drift_features(bodies, params["reference_date"])["age_years"]
    assert actual.tolist() == expected.tolist()


@pytest.mark.parametrize(
    ("when", "months"),
    [
        ("2025-11-08T00:00:00Z", 0),
        ("2026-11-07T23:59:59Z", 11),
        ("2026-11-08T00:00:00Z", 12),
        ("2026-11-08T00:30:00+01:00", 11),
        ("2026-12-31T23:59:59Z", 13),
        ("2027-01-01T00:00:00Z", 13),
    ],
)
def test_a_month_is_counted_at_midnight_utc_on_the_reference_day(params, when, months):
    assert months_since_reference(when, params["reference_date"]) == months
    expected = pd.Timestamp(params["reference_date"]) + pd.DateOffset(months=months)
    assert as_of_for(when, params["reference_date"]) == expected


@pytest.mark.parametrize("name", list(SAMPLES))
@pytest.mark.parametrize(
    # In the past, because a shifted date has to lie before the current month.
    "answered_at",
    ["2026-11-04T09:15:00Z", "2026-03-08T00:00:01Z", "2026-10-07T23:59:59Z"],
)
def test_a_replayed_request_as_logged_has_the_drift_features_of_its_reference_row(
    samples, params, name, answered_at
):
    """Shift, validate, log as #191 does, read back at `as_of_for`: the parquet's row again."""
    reference_date = params["reference_date"]
    months = months_since_reference(answered_at, reference_date)
    logged = []
    for record in replay(samples, name):
        sent = {
            **record["body"],
            "registration_date": shifted_registration(record["body"]["registration_date"], months),
        }
        request = ValuationRequest.model_validate(sent)
        logged.append(request.model_dump(mode="json", exclude_unset=True, exclude_none=True))
    as_of = [as_of_for(answered_at, reference_date)] * len(logged)

    actual = drift_features(logged, as_of)
    expected = pd.read_parquet(samples / f"{name}.parquet")[list(DRIFT_FEATURES)]
    others = [feature for feature in DRIFT_FEATURES if feature != "age_years"]
    pd.testing.assert_frame_equal(actual[others], expected[others], check_dtype=False)
    # The shifted months and the months after the reference date are the same
    # number but not always the same days: February and the 31st make up to three.
    days = (actual["age_years"] - expected["age_years"]) * 365.25
    assert days.round(6).between(-3, 3).all(), days.describe()


def test_no_bodies_give_an_empty_frame_with_every_drift_feature(params):
    features = drift_features([], params["reference_date"])

    assert features.empty and list(features.columns) == list(DRIFT_FEATURES)
    assert drift_features([], []).shape == (0, len(DRIFT_FEATURES))


def _datetime64(dates: list[str]):
    return pd.to_datetime(pd.Series(dates)).to_numpy()


def _utc_series(dates: list[str]) -> pd.Series:
    return pd.Series(pd.to_datetime(dates, utc=True))


@pytest.mark.parametrize(
    "as_of",
    [list, pd.Series, _datetime64, _utc_series],
    ids=["list", "series-of-strings", "numpy-datetime64", "series-of-utc-timestamps"],
)
def test_one_age_date_per_body_may_come_as_any_array_like(as_of):
    bodies = [{"registration_date": "2020-11"}, {"registration_date": "2020-11"}]
    dates = ["2025-11-08", "2026-11-08"]

    ages = drift_features(bodies, as_of(dates))["age_years"].tolist()
    assert ages == [drift_features(bodies[:1], date)["age_years"][0] for date in dates]
    with pytest.raises(ValueError, match="dates for 2 bodies"):
        drift_features(bodies, as_of(dates[:1]))
