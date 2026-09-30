"""The `split` stage: the holdout, the supported-make list and the grouped split.

This is the one stage where a mistake does not fail anything, it just makes every
number we report a little better than the truth. So the invariants are tested
directly against the artefacts the stage writes, rather than inferred from the
pipeline running green: no `ES` row outside the holdout, no seller in two sets,
nothing lost, and ratios that are honoured as far as whole sellers allow.

The tests run the real `preprocess` and `split` functions on the synthetic
fixture in a temporary directory, so they need neither the 548 MB snapshot nor
any credential.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from recommenditos.config import PARAMS_FILE
from recommenditos.data import preprocess, split_data
from recommenditos.data.split_data import (
    HOLDOUT_NAME,
    MAKES_FILE,
    SPLIT_NAMES,
    assign_split,
    check_ratios,
    ratio_tolerances,
    supported_makes,
)
from recommenditos.schema import PROCESSED_SCHEMA

#: Enough seeds that the ratio tolerance is a property of the stage rather than
#: a lucky draw of the one seed params.yaml happens to carry.
SEED_SWEEP = range(1, 26)


@pytest.fixture(scope="module")
def interim_path(tmp_path_factory, _generated_frame) -> Path:
    """The fixture as `preprocess` leaves it, which is what `split` reads."""
    root = tmp_path_factory.mktemp("split-input")
    raw_path = root / "listings.parquet"
    _generated_frame.to_parquet(raw_path, index=False)
    interim = root / "interim.parquet"
    preprocess.main(raw_path, interim, PARAMS_FILE)
    return interim


@pytest.fixture(scope="module")
def interim(interim_path: Path) -> pd.DataFrame:
    return pd.read_parquet(interim_path)


@pytest.fixture(scope="module")
def processed(tmp_path_factory, interim_path: Path) -> Path:
    """One real run of the stage, whose artefacts every assertion below reads."""
    output_dir = tmp_path_factory.mktemp("split-output")
    split_data.main(interim_path, output_dir, PARAMS_FILE)
    return output_dir


@pytest.fixture(scope="module")
def sets(processed: Path) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(processed / f"{name}.parquet")
        for name in (*SPLIT_NAMES, HOLDOUT_NAME)
    }


@pytest.fixture(scope="module")
def pool(interim: pd.DataFrame, params: dict) -> pd.DataFrame:
    """The rows the split divides: everything the holdout did not take."""
    return interim[interim["country_code"] != params["split"]["holdout_country"]]


def _params_with(tmp_path: Path, params: dict, **split_keys) -> Path:
    """A copy of params.yaml with some keys of the `split` block changed."""
    changed = {**params, "split": {**params["split"], **split_keys}}
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# The artefacts
# --------------------------------------------------------------------------


def test_the_stage_writes_five_sets_and_the_make_list(processed: Path):
    for name in (*SPLIT_NAMES, HOLDOUT_NAME):
        assert (processed / f"{name}.parquet").exists(), name
    assert (processed / MAKES_FILE).exists()


# --------------------------------------------------------------------------
# The `ES` holdout (EDN-03)
# --------------------------------------------------------------------------


def test_the_holdout_takes_every_holdout_country_row_and_nothing_else(
    sets: dict[str, pd.DataFrame], interim: pd.DataFrame, params: dict
):
    country = params["split"]["holdout_country"]
    holdout = sets[HOLDOUT_NAME]

    assert len(holdout) > 0, "the fixture has no holdout-country row to hold out"
    assert (holdout["country_code"] == country).all()
    assert len(holdout) == int((interim["country_code"] == country).sum())


def test_no_holdout_country_row_reaches_a_training_set(
    sets: dict[str, pd.DataFrame], params: dict
):
    # The whole of M6 rests on this: an `ES` row in train, validation,
    # calibration or test makes the new-market replay a replay of data the
    # model has already seen, and the drift it is meant to show disappears.
    country = params["split"]["holdout_country"]
    for name in SPLIT_NAMES:
        assert not (sets[name]["country_code"] == country).any(), name


def test_rows_with_no_country_at_all_are_split_rather_than_held_out(
    sets: dict[str, pd.DataFrame], interim: pd.DataFrame
):
    # 15 rows of the real file carry no country. They are not the held-out
    # market, so they belong in the split; a null-safe comparison is what keeps
    # them there instead of in the holdout.
    missing = int(interim["country_code"].isna().sum())
    assert missing > 0, "the fixture has no row without a country"
    assert sum(int(sets[name]["country_code"].isna().sum()) for name in SPLIT_NAMES) == missing
    assert int(sets[HOLDOUT_NAME]["country_code"].isna().sum()) == 0


# --------------------------------------------------------------------------
# The grouped split (EDN-14)
# --------------------------------------------------------------------------


def test_no_seller_group_appears_in_two_sets(sets: dict[str, pd.DataFrame], params: dict):
    key = params["split"]["group_key"]
    seen: dict[str, str] = {}
    for name in SPLIT_NAMES:
        for group in sets[name][key].unique():
            assert seen.setdefault(group, name) == name, f"{group} is in {seen[group]} and {name}"


def test_the_five_artefacts_partition_the_interim_frame(
    sets: dict[str, pd.DataFrame], interim: pd.DataFrame
):
    # Sums alone would accept a row duplicated into one set and dropped from
    # another, which is exactly the shape a filtering bug takes.
    columns = list(PROCESSED_SCHEMA.names)
    rebuilt = pd.concat(sets.values(), ignore_index=True)

    assert len(rebuilt) == len(interim)
    assert rebuilt.sort_values(columns, ignore_index=True).equals(
        interim.sort_values(columns, ignore_index=True)
    )


@pytest.mark.req("NFR-06")
def test_the_assignment_does_not_depend_on_the_row_order(pool: pd.DataFrame, params: dict):
    # The property the hash buys over a shuffle: a reordered frame - a different
    # Parquet row group, a differently ordered upstream stage - gives the same
    # split, so `dvc repro` on a clean clone reproduces it.
    ratios, seed, key = params["split"]["ratios"], params["seed"], params["split"]["group_key"]
    shuffled = pool.sample(frac=1.0, random_state=0)

    straight = dict(zip(pool[key], assign_split(pool[key], ratios, seed), strict=True))
    reordered = dict(zip(shuffled[key], assign_split(shuffled[key], ratios, seed), strict=True))

    assert straight == reordered


@pytest.mark.req("NFR-06")
def test_the_same_seed_gives_the_same_split(pool: pd.DataFrame, params: dict):
    ratios, seed, key = params["split"]["ratios"], params["seed"], params["split"]["group_key"]

    assert assign_split(pool[key], ratios, seed).equals(assign_split(pool[key], ratios, seed))


def test_a_different_seed_moves_sellers_between_sets(pool: pd.DataFrame, params: dict):
    ratios, key = params["split"]["ratios"], params["split"]["group_key"]
    first = assign_split(pool[key], ratios, 1)
    second = assign_split(pool[key], ratios, 2)

    assert not first.equals(second)
    moved = (first != second).mean()
    # A seed that moved one seller in a thousand would pass the inequality above
    # while leaving the two splits effectively the same.
    assert moved > 0.1, f"only {moved:.1%} of the rows changed set"


def test_a_seller_keeps_its_set_when_the_data_grows(pool: pd.DataFrame, params: dict):
    # What makes FR-15's retrain-with-`ES` run comparable to the current model:
    # adding rows must not reshuffle the sellers that were already there, or the
    # before-and-after metrics differ because the test set changed, not the model.
    ratios, seed, key = params["split"]["ratios"], params["seed"], params["split"]["group_key"]
    known = pool[key].drop_duplicates().reset_index(drop=True)
    grown = pd.concat([known, pd.Series([f"new-seller-{index}" for index in range(500)])])

    before = dict(zip(known, assign_split(known, ratios, seed), strict=True))
    after = dict(zip(grown, assign_split(grown, ratios, seed), strict=True))

    assert {group: after[group] for group in before} == before


# --------------------------------------------------------------------------
# Ratio honesty
# --------------------------------------------------------------------------


def test_the_realised_ratios_stay_inside_the_tolerance_the_group_sizes_allow(
    sets: dict[str, pd.DataFrame], pool: pd.DataFrame, params: dict
):
    sizes = {name: len(sets[name]) for name in SPLIT_NAMES}
    group_sizes = pool[params["split"]["group_key"]].value_counts()

    deviations = check_ratios(sizes, group_sizes, params["split"]["ratios"])

    tolerances = ratio_tolerances(group_sizes, params["split"]["ratios"])
    for name in SPLIT_NAMES:
        assert abs(deviations[name]) <= tolerances[name], name
    assert min(sizes.values()) > 0


def test_the_ratios_hold_for_every_seed_and_not_only_for_ours(pool: pd.DataFrame, params: dict):
    # The tolerance has to be a statement about the stage, not about the one
    # seed params.yaml carries, or the next `dvc exp` sweep turns it red.
    ratios, key = params["split"]["ratios"], params["split"]["group_key"]
    group_sizes = pool[key].value_counts()

    for seed in SEED_SWEEP:
        assigned = assign_split(pool[key], ratios, seed)
        sizes = {name: int((assigned == name).sum()) for name in SPLIT_NAMES}
        check_ratios(sizes, group_sizes, ratios)


def test_a_ratio_block_that_does_not_describe_a_split_is_refused(pool: pd.DataFrame, params: dict):
    # Both guards protect the same thing: a params.yaml edit that leaves one set
    # without rows, so the UC2 intervals calibrate on nothing or every success
    # criterion is computed on an empty test set.
    groups = pool[params["split"]["group_key"]]

    with pytest.raises(ValueError, match="no ratio given for calibration"):
        assign_split(groups, {"train": 0.7, "validation": 0.1, "test": 0.2}, params["seed"])

    with pytest.raises(ValueError, match="must sum to 1"):
        assign_split(
            groups,
            {"train": 0.6, "validation": 0.1, "calibration": 0.1, "test": 0.1},
            params["seed"],
        )


def test_an_empty_pool_is_reported_rather_than_divided_by_zero(params: dict):
    # The stage would otherwise fail on a zero division deep inside a share
    # calculation, which says nothing about the interim frame being empty.
    with pytest.raises(ValueError, match="no row was assigned"):
        split_data.realised_shares(dict.fromkeys(SPLIT_NAMES, 0))

    with pytest.raises(ValueError, match="no groups to split"):
        ratio_tolerances(pd.Series([], dtype="int64"), params["split"]["ratios"])


def test_the_tolerance_tightens_as_the_pool_gains_sellers(params: dict):
    ratios = params["split"]["ratios"]
    small = ratio_tolerances(pd.Series([1] * 200), ratios)
    large = ratio_tolerances(pd.Series([1] * 20_000), ratios)

    for name in SPLIT_NAMES:
        assert large[name] < small[name], name


def test_one_dominant_seller_widens_the_tolerance_rather_than_hiding_in_it(params: dict):
    # A seller holding a tenth of the pool cannot be placed neatly by any
    # grouped split. The tolerance has to say so out loud, because a fixed
    # percentage would either fail every run or accept any size at all.
    ratios = params["split"]["ratios"]
    even = ratio_tolerances(pd.Series([1] * 1000), ratios)
    lopsided = ratio_tolerances(pd.Series([100] + [1] * 900), ratios)

    for name in SPLIT_NAMES:
        assert lopsided[name] > even[name], name


def test_a_split_that_misses_its_ratios_fails_instead_of_being_logged(params: dict):
    ratios = params["split"]["ratios"]
    group_sizes = pd.Series([1] * 10_000)
    honest = {"train": 6_000, "validation": 1_000, "calibration": 1_000, "test": 2_000}

    check_ratios(honest, group_sizes, ratios)

    with pytest.raises(ValueError, match="calibration"):
        check_ratios({**honest, "calibration": 0, "train": 7_000}, group_sizes, ratios)


def test_the_realised_test_share_keeps_sc04_measurable(
    sets: dict[str, pd.DataFrame], pool: pd.DataFrame, params: dict
):
    # The configured 20 % is checked in test_data.py; this is the share that was
    # actually realised, which is the one SC-04's 500-row segments depend on.
    share = len(sets["test"]) / len(pool)
    assert share >= 0.15, f"the realised test share is {share:.1%}"


# --------------------------------------------------------------------------
# The supported-make list (EDN-05, FR-04)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def make_list(processed: Path) -> list[dict]:
    return json.loads((processed / MAKES_FILE).read_text(encoding="utf-8"))["supported_makes"]


@pytest.mark.req("FR-04")
def test_every_entry_names_a_make_and_the_count_that_admitted_it(make_list: list[dict]):
    assert make_list, "the fixture produced no supported make"
    for entry in make_list:
        assert set(entry) == {"make", "listings"}
        assert isinstance(entry["make"], str)
        assert isinstance(entry["listings"], int)


@pytest.mark.req("FR-04")
def test_the_list_is_exactly_the_makes_that_reach_the_threshold(
    make_list: list[dict], pool: pd.DataFrame, params: dict
):
    threshold = params["split"]["min_listings_per_make"]
    counts = pool["make"].value_counts()

    assert {entry["make"] for entry in make_list} == set(counts[counts >= threshold].index)
    for entry in make_list:
        assert entry["listings"] == int(counts[entry["make"]])
        assert entry["listings"] >= threshold


def test_the_counts_are_taken_after_the_holdout(
    make_list: list[dict], interim: pd.DataFrame, pool: pd.DataFrame
):
    # EDN-05 counts support on the rows that are actually trained on, so a make
    # kept in scope by its Spanish listings alone would be a make the model has
    # too little data for. The two counts have to differ somewhere, or this
    # test would pass on an implementation that ignored the holdout.
    listed = {entry["make"]: entry["listings"] for entry in make_list}
    after_holdout = pool["make"].value_counts()
    whole_frame = interim["make"].value_counts()

    assert all(listed[make] == int(after_holdout[make]) for make in listed)
    assert any(int(whole_frame[make]) != listed[make] for make in listed)


def test_the_counts_are_taken_before_the_split(
    make_list: list[dict], sets: dict[str, pd.DataFrame]
):
    # Counting after the split would make support depend on which set a seller
    # landed in, so the same data under another seed would support other makes.
    per_set = pd.concat([sets[name]["make"] for name in SPLIT_NAMES]).value_counts()
    for entry in make_list:
        assert entry["listings"] == int(per_set[entry["make"]])


@pytest.mark.req("NFR-06")
def test_the_list_is_ordered_by_support_and_then_by_name(make_list: list[dict]):
    # Deterministic order, so the artefact's hash depends on the counts alone
    # and `dvc repro` does not report a change that is only a reordering.
    keys = [(-entry["listings"], entry["make"]) for entry in make_list]
    assert keys == sorted(keys)


def test_the_stage_keeps_the_rows_of_unsupported_makes(
    make_list: list[dict], sets: dict[str, pd.DataFrame]
):
    # This stage records the scope, it does not enforce it: the rows stay in the
    # sets so that EDN-05's threshold can be revisited without re-running the
    # split, and the stages that build the model input apply the list. If that
    # reading is ever reversed, this test is the one to change first.
    listed = {entry["make"] for entry in make_list}
    present = set(pd.concat([sets[name]["make"] for name in SPLIT_NAMES]))

    assert present - listed, "the fixture has no unsupported make left to keep"


def test_a_threshold_no_make_reaches_fails_instead_of_writing_an_empty_list(
    interim_path: Path, tmp_path: Path, params: dict
):
    # An empty list makes FR-04 reject every request while the rest of the
    # pipeline goes on training happily, so it has to stop the stage.
    with pytest.raises(ValueError, match="supported-make list would be empty"):
        split_data.main(
            interim_path,
            tmp_path / "processed",
            _params_with(tmp_path, params, min_listings_per_make=10**9),
        )


def test_the_threshold_decides_the_list_rather_than_the_code(
    pool: pd.DataFrame, make_list: list[dict]
):
    smallest = min(entry["listings"] for entry in make_list)
    tightened = supported_makes(pool["make"], smallest + 1)
    expected = [entry for entry in make_list if entry["listings"] > smallest]

    assert tightened == expected
    assert len(tightened) < len(make_list)


def test_ties_in_support_are_broken_by_name(params: dict):
    makes = pd.Series(["Zed"] * 300 + ["Alfa"] * 300 + ["Rare"] * 5)

    listed = supported_makes(makes, params["split"]["min_listings_per_make"])

    assert [entry["make"] for entry in listed] == ["Alfa", "Zed"]
