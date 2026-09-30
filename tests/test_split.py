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
    DISPERSION_SIGMAS,
    HOLDOUT_NAME,
    MAKES_FILE,
    MAX_DISPERSION_BOUND,
    MAX_GROUP_SHARE,
    MIN_SHARE_OF_RATIO,
    SPLIT_NAMES,
    GateLimits,
    assign_split,
    check_ratios,
    cross_holdout_sellers,
    dispersion_bounds,
    supported_makes,
)
from recommenditos.schema import PROCESSED_SCHEMA

#: Enough seeds that the gate is a property of the stage rather than a lucky
#: draw of the one seed params.yaml happens to carry.
SEED_SWEEP = range(1, 26)

#: The floor to sweep seeds against on the fixture pool, instead of the
#: project's `MIN_SHARE_OF_RATIO`.
#:
#: The floor is a statement about a pool of the real snapshot's granularity.
#: The fixture pool has 608 seller groups against the snapshot's 28,435, so its
#: shares scatter far more: over 2,000 seeds the smallest share any set reaches
#: is 0.478 of its ratio here against 0.820 on the snapshot
#: (`reports/analysis/split_gate_results.txt`). Sweeping the project's 0.75 here
#: would therefore measure the fixture's size, not the stage's correctness. The
#: project value is exercised on the one real run of the stage, in `processed`.
FIXTURE_MIN_SHARE_OF_RATIO = 0.45


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


def test_the_holdout_is_included_in_the_disjointness_check(
    sets: dict[str, pd.DataFrame], params: dict
):
    # Iterating SPLIT_NAMES alone leaves the holdout out of every disjointness
    # assertion, which is how a seller in the holdout *and* in train went
    # unnoticed: 81 of them in this fixture. The overlap is allowed here, but it
    # has to be exactly the set the stage reports, not whatever happens to be
    # there.
    key = params["split"]["group_key"]
    split_groups = pd.concat([sets[name][key] for name in SPLIT_NAMES])
    reported = cross_holdout_sellers(sets[HOLDOUT_NAME][key], split_groups)

    overlap = set(sets[HOLDOUT_NAME][key].unique()) & set(split_groups.unique())

    assert overlap == reported


def test_the_fixtures_cross_border_sellers_are_counted_rather_than_hidden(
    sets: dict[str, pd.DataFrame], params: dict
):
    # EDN-03 holds out every row of the holdout country, and the holdout is
    # selected per row, so a dealer listing in `ES` and elsewhere is in the
    # holdout and in a split set. On the real snapshot this happens 0 times,
    # because `hash_seller_group` keys a dealer by its company name and no name
    # appears both inside and outside `ES` there. The fixture draws names
    # independently of the country, so it does happen here - and a replay of a
    # holdout that contains dealers the model trained on understates drift
    # (EDN-14), which is a limitation of the fixture worth pinning rather than
    # discovering later.
    key = params["split"]["group_key"]
    split_groups = pd.concat([sets[name][key] for name in SPLIT_NAMES])

    crossing = cross_holdout_sellers(sets[HOLDOUT_NAME][key], split_groups)

    assert len(crossing) == 81, (
        f"the fixture's cross-border seller count moved to {len(crossing)}; if the "
        f"generator or hash_seller_group changed, re-measure it, and if it reached 0 the "
        f"fixture no longer exercises this case at all"
    )


def test_a_seller_only_inside_the_holdout_country_never_reaches_a_split_set(
    sets: dict[str, pd.DataFrame], params: dict
):
    # The half of the conflict that must hold unconditionally: whatever the
    # cross-border dealers do, a seller whose every listing is `ES` is entirely
    # in the holdout, or the drift replay would be missing its own rows.
    key, country = params["split"]["group_key"], params["split"]["holdout_country"]
    holdout = sets[HOLDOUT_NAME]
    split_groups = set(pd.concat([sets[name][key] for name in SPLIT_NAMES]).unique())

    only_es = set(holdout[key].unique()) - split_groups

    assert only_es, "the fixture has no seller that lists only in the holdout country"
    assert not (holdout[holdout[key].isin(only_es)]["country_code"] != country).any()


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
    #
    # The claim holds along this axis only. A change to `seed`, to the ratios or
    # to the order of `SPLIT_NAMES` reshuffles most sellers - reordering the
    # tuple moves 80.0 % of them - so an FR-15 comparison is valid only while
    # those three are untouched, and the test above measures exactly that for
    # the seed.
    ratios, seed, key = params["split"]["ratios"], params["seed"], params["split"]["group_key"]
    known = pool[key].drop_duplicates().reset_index(drop=True)
    grown = pd.concat([known, pd.Series([f"new-seller-{index}" for index in range(500)])])

    before = dict(zip(known, assign_split(known, ratios, seed), strict=True))
    after = dict(zip(grown, assign_split(grown, ratios, seed), strict=True))

    assert {group: after[group] for group in before} == before


# --------------------------------------------------------------------------
# Ratio honesty
# --------------------------------------------------------------------------


def test_every_set_clears_the_floor_and_the_dispersion_bound(
    sets: dict[str, pd.DataFrame], pool: pd.DataFrame, params: dict
):
    # The oracle is independent of the function under test: the floor is a
    # fraction of the configured ratio, which no measurement of the realised
    # data can move.
    ratios = params["split"]["ratios"]
    sizes = {name: len(sets[name]) for name in SPLIT_NAMES}
    group_sizes = pool[params["split"]["group_key"]].value_counts()
    total = sum(sizes.values())

    report = check_ratios(sizes, group_sizes, ratios)

    for name in SPLIT_NAMES:
        assert sizes[name] > 0, name
        assert sizes[name] / total >= ratios[name] * MIN_SHARE_OF_RATIO, name
        assert abs(report["deviations"][name]) <= report["bounds"][name], name
    assert int(group_sizes.max()) / total <= MAX_GROUP_SHARE


def test_the_gate_holds_for_every_seed_and_not_only_for_ours(pool: pd.DataFrame, params: dict):
    # The gate has to be a statement about the stage, not about the one seed
    # params.yaml carries, or the next `dvc exp` sweep turns it red.
    ratios, key = params["split"]["ratios"], params["split"]["group_key"]
    group_sizes = pool[key].value_counts()

    for seed in SEED_SWEEP:
        assigned = assign_split(pool[key], ratios, seed)
        sizes = {name: int((assigned == name).sum()) for name in SPLIT_NAMES}
        check_ratios(
            sizes,
            group_sizes,
            ratios,
            GateLimits(min_share_of_ratio=FIXTURE_MIN_SHARE_OF_RATIO),
        )


def test_the_stage_itself_refuses_a_split_the_gate_rejects(
    interim_path: Path, tmp_path: Path, params: dict
):
    # The gate being right is worth nothing if `main` does not call it.
    # Replacing the `check_ratios(...)` call with a dict of zeros used to
    # survive the whole suite, because every gate test called the function
    # directly with hand-made sizes. This one runs the stage, and it groups by
    # `make` - the mis-set `group_key` that the old bound could not see at all,
    # because a handful of huge groups derived a bound no share could miss.
    with pytest.raises(ValueError, match=r"\[concentration\]"):
        split_data.main(
            interim_path,
            tmp_path / "processed",
            _params_with(tmp_path, params, group_key="make"),
        )


def test_a_rejected_split_writes_no_artefact_at_all(
    interim_path: Path, tmp_path: Path, params: dict
):
    # The gate used to run after all six artefacts were written, so a rejected
    # split had already overwritten the previous run's train.parquet and
    # supported_makes.json before failing.
    output = tmp_path / "processed"

    with pytest.raises(ValueError):
        split_data.main(interim_path, output, _params_with(tmp_path, params, group_key="make"))

    written = sorted(path.name for path in output.glob("*")) if output.exists() else []
    assert not written, f"a rejected split left {written} behind"


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
    # Reached from the library rather than from `main`, which stops at the
    # holdout with a message naming that cause instead. Both guards stay,
    # because a zero division deep inside a share calculation says nothing at
    # all about the frame that caused it.
    with pytest.raises(ValueError, match="no row was assigned"):
        split_data.realised_shares(dict.fromkeys(SPLIT_NAMES, 0))

    with pytest.raises(ValueError, match="no groups to split"):
        dispersion_bounds(pd.Series([], dtype="int64"), params["split"]["ratios"])


def test_a_frame_that_is_all_holdout_country_says_so_rather_than_blaming_the_makes(
    tmp_path: Path, interim: pd.DataFrame, params: dict
):
    # The old message was "no make reaches 300 listings in 0 rows", which sends
    # the reader to the support threshold for a frame that held nothing but
    # `ES`.
    country = params["split"]["holdout_country"]
    all_holdout = interim.copy()
    all_holdout["country_code"] = country
    path = tmp_path / "all-holdout.parquet"
    all_holdout.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="nothing left to split"):
        split_data.main(path, tmp_path / "processed", PARAMS_FILE)


def test_the_dispersion_bound_tightens_as_the_pool_gains_sellers(params: dict):
    ratios = params["split"]["ratios"]
    small = dispersion_bounds(pd.Series([1] * 200), ratios)
    large = dispersion_bounds(pd.Series([1] * 20_000), ratios)

    for name in SPLIT_NAMES:
        assert large[name] < small[name], name


def test_the_dispersion_bound_is_capped_so_concentration_cannot_buy_slack(params: dict):
    # The defect the cap exists for: the bound was derived from the realised
    # group sizes, so it widened exactly when one dealer dominated and the
    # split deserved more suspicion. Uncapped, a single group holding the whole
    # pool derives a 196 pp allowance that no realised share can miss.
    ratios = params["split"]["ratios"]

    lopsided = dispersion_bounds(pd.Series([1_200] + [1] * 8_800), ratios)
    degenerate = dispersion_bounds(pd.Series([10_000]), ratios)

    for name in SPLIT_NAMES:
        assert lopsided[name] <= MAX_DISPERSION_BOUND, name
        assert degenerate[name] == MAX_DISPERSION_BOUND, name


def test_the_gate_accepts_a_split_that_came_out_right(params: dict):
    ratios = params["split"]["ratios"]
    group_sizes = pd.Series([1] * 10_000)
    honest = {"train": 6_000, "validation": 1_000, "calibration": 1_000, "test": 2_000}

    report = check_ratios(honest, group_sizes, ratios)

    assert report["deviations"] == dict.fromkeys(SPLIT_NAMES, 0.0)


def test_an_empty_set_fails_the_gate_however_the_bound_came_out(params: dict):
    # The exploit this closes: with one group holding the whole pool the derived
    # bound reached 196 pp, so three empty sets passed without an error.
    ratios = params["split"]["ratios"]
    all_in_train = {"train": 10_000, "validation": 0, "calibration": 0, "test": 0}

    with pytest.raises(ValueError, match=r"\[non-empty\]") as raised:
        check_ratios(all_in_train, pd.Series([10_000]), ratios)

    assert "validation, calibration, test" in str(raised.value)


def test_a_set_below_the_floor_fails_the_gate_however_wide_the_bound_is(params: dict):
    # A calibration set at 0.58 of its intended size cleared the derived bound
    # on the real snapshot, which is the same failure the PR rejected a fixed
    # 6.8 pp bound for, one factor weaker. The floor is a fraction of the
    # *configured* ratio, so no property of the realised data can widen it.
    ratios = params["split"]["ratios"]
    group_sizes = pd.Series([1] * 10_000)
    shrunk = round(1_000 * 0.58)
    broken = {
        "train": 6_000 + (1_000 - shrunk),
        "validation": 1_000,
        "calibration": shrunk,
        "test": 2_000,
    }

    with pytest.raises(ValueError, match=r"\[floor\].*calibration"):
        check_ratios(broken, group_sizes, ratios)


def test_the_floor_is_a_fraction_of_the_ratio_and_not_of_the_realised_data(params: dict):
    # The property that makes the floor unconditional: the share at which it
    # fires is the same on two pools whose derived bounds differ, because it is
    # a fraction of the configured ratio and of nothing else. The old gate had
    # no such share - it moved with the group sizes, which is how a calibration
    # set at 0.58 of its intended size got through.
    #
    # Both pools are coarse on purpose. On a fine-grained pool the dispersion
    # rule is the tighter of the two and fires first, which is right: a 25 %
    # shortfall among 10,000 singleton sellers really is more surprising than
    # among a few hundred large ones. The floor is what binds at the real
    # snapshot's granularity, which is pinned separately.
    ratios = params["split"]["ratios"]
    at_the_floor = round(10_000 * ratios["calibration"] * MIN_SHARE_OF_RATIO) + 1
    sizes = {
        "train": 6_000 + (1_000 - at_the_floor),
        "validation": 1_000,
        "calibration": at_the_floor,
        "test": 2_000,
    }

    for group_sizes in (pd.Series([1_000] + [1] * 9_000), pd.Series([300] * 10 + [1] * 7_000)):
        check_ratios(sizes, group_sizes, ratios, GateLimits(max_group_share=1.0))
        with pytest.raises(ValueError, match=r"\[floor\]"):
            check_ratios(
                {**sizes, "calibration": at_the_floor - 2, "train": sizes["train"] + 2},
                group_sizes,
                ratios,
                GateLimits(max_group_share=1.0),
            )


def test_a_group_above_the_declared_share_fails_the_gate(params: dict):
    # The rule that catches the two cases the derived bound could not: a dealer
    # dumped into one set, and a `group_key` naming the wrong column. Both look
    # to the bound like a pool that simply allows more slack.
    ratios = params["split"]["ratios"]
    dominant = 1_200
    group_sizes = pd.Series([dominant] + [1] * (10_000 - dominant))
    rest = 10_000 - dominant
    dumped = {
        "train": rest - round(rest * 0.1) - round(rest * 0.2),
        "validation": round(rest * 0.1) + dominant,
        "calibration": round(rest * 0.1),
        "test": round(rest * 0.2),
    }
    dumped["train"] += 10_000 - sum(dumped.values())

    with pytest.raises(ValueError, match=r"\[concentration\].*group_key") as raised:
        check_ratios(dumped, group_sizes, ratios)

    assert "12.0%" in str(raised.value)


def test_the_gate_names_every_rule_that_fired_not_only_the_first(params: dict):
    # A message naming one rule sends the reader after one cause when several
    # are true, and the concentration rule is usually the one that explains the
    # others. An empty set is reported as empty and not also as below the floor,
    # because the second statement adds nothing to the first.
    ratios = params["split"]["ratios"]

    with pytest.raises(ValueError) as degenerate:
        check_ratios(
            {"train": 10_000, "validation": 0, "calibration": 0, "test": 0},
            pd.Series([10_000]),
            ratios,
        )

    message = str(degenerate.value)
    assert "[non-empty]" in message
    assert "[concentration]" in message
    assert "[floor]" not in message

    # A dealer holding a fifth of the pool, dumped into train: nothing is empty,
    # but the shares are wrong and the reason is the concentration.
    with pytest.raises(ValueError) as dumped:
        check_ratios(
            {"train": 7_600, "validation": 800, "calibration": 800, "test": 800},
            pd.Series([2_000] + [1] * 8_000),
            ratios,
        )

    message = str(dumped.value)
    assert "[floor]" in message
    assert "[concentration]" in message


def test_the_gate_constants_are_the_ones_the_measurement_chose(params: dict):
    # One test pins each constant, so a change to a number that a committed
    # measurement justifies cannot pass as a tidy-up. The measurement is
    # reports/analysis/split_gate.py; re-run it before moving any of these.
    assert DISPERSION_SIGMAS == 5.0
    assert MIN_SHARE_OF_RATIO == 0.75
    assert MAX_GROUP_SHARE == 0.05
    assert MAX_DISPERSION_BOUND == 0.15
    # The floor has to bind before the dispersion bound on the real snapshot's
    # granularity, or it is decoration: the dispersion rule would fire first and
    # the floor would never be the reason anything failed.
    real_pool = pd.Series([3_383] + [3] * 33_000)
    bounds = dispersion_bounds(real_pool, params["split"]["ratios"])
    for name in ("validation", "calibration"):
        floor_slack = params["split"]["ratios"][name] * (1 - MIN_SHARE_OF_RATIO)
        assert floor_slack < bounds[name], name


def test_a_null_group_key_is_refused_rather_than_split_by_its_spelling(params: dict):
    # Unreachable through the stage, because the interim contract declares the
    # column non-nullable. Worth a guard anyway: the key is stringified, so
    # `None` and `float("nan")` would land in different sets while the pool the
    # bounds are derived from drops both.
    groups = pd.Series(["a", "b", None, float("nan")], dtype="object")

    with pytest.raises(ValueError, match="carry no group key"):
        assign_split(groups, params["split"]["ratios"], params["seed"])


def test_the_realised_test_share_keeps_sc04_measurable(
    sets: dict[str, pd.DataFrame], pool: pd.DataFrame, params: dict
):
    # The configured 20 % is checked in test_data.py; this is the share that was
    # actually realised, which is the one SC-04's 500-row segments depend on.
    share = len(sets["test"]) / len(pool)
    assert share >= 0.15, f"the realised test share is {share:.1%}"


# --------------------------------------------------------------------------
# The supported-make list (EDN-05)
#
# Deliberately unmarked with `@pytest.mark.req("FR-04")`. FR-04 is the API's
# 422, and the specification names an API test as its route; these tests cover
# the pipeline half, that the list is computed from the data and written where
# the API reads it. The requirement matrix that PR #51 adds has one granularity,
# "covered by a test", so marking them would make FR-04 read as verified while
# the rejection nobody has written yet is what the requirement is about. The
# link is kept in prose instead, and FR-04 stays open until the API test exists.
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def make_document(processed: Path) -> dict:
    return json.loads((processed / MAKES_FILE).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def make_list(make_document: dict) -> list[dict]:
    return make_document["supported_makes"]


def test_every_entry_names_a_make_and_the_count_that_admitted_it(make_list: list[dict]):
    assert make_list, "the fixture produced no supported make"
    for entry in make_list:
        assert set(entry) == {"make", "listings"}
        assert isinstance(entry["make"], str)
        assert isinstance(entry["listings"], int)


def test_the_artefact_carries_the_threshold_and_the_population_it_was_counted_over(
    make_document: dict, pool: pd.DataFrame, params: dict
):
    # Without these a reader sees `{"make": "Volkswagen", "listings": 348}` and
    # cannot tell whether 348 cleared the bar or what it is 348 out of, so the
    # artefact cannot be audited on its own. They cannot drift from the values
    # that were applied, because the same call emits and applies them.
    assert make_document["min_listings_per_make"] == params["split"]["min_listings_per_make"]
    assert make_document["counted_over_rows"] == len(pool)


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
