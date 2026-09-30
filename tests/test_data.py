"""The synthetic fixture, and the paths through the data stages.

The fixture is the reason the stage tickets can be built in parallel, so its
guarantees are tested here rather than assumed. If one of these fails, a test
somewhere else is silently checking nothing.
"""

import pandas as pd
import pytest
from tests.conftest import PII_COLUMNS

from recommenditos import config
from recommenditos.data.preprocess import apply_row_rules, hash_seller_group
from recommenditos.data.split_data import SPLIT_NAMES, assign_split, dispersion_bounds
from recommenditos.data.synthetic import (
    DEDUP_KEY,
    EDGE_CASE_ROWS,
    SNAPSHOT_DATE,
    generate_raw_listings,
)
from recommenditos.schema import RAW_SCHEMA


def test_data_dirs_are_nested_under_project_root():
    assert config.RAW_DATA_DIR == config.DATA_DIR / "raw"
    assert config.INTERIM_DATA_DIR == config.DATA_DIR / "interim"
    assert config.PROCESSED_DATA_DIR == config.DATA_DIR / "processed"
    assert config.EXTERNAL_DATA_DIR == config.DATA_DIR / "external"
    assert config.DATA_DIR == config.PROJ_ROOT / "data"


# --------------------------------------------------------------------------
# The fixture
# --------------------------------------------------------------------------


def test_the_fixture_satisfies_the_raw_contract(raw_frame):
    RAW_SCHEMA.validate(raw_frame)


def test_the_fixture_is_reproducible_from_its_seed():
    assert generate_raw_listings(200, seed=7).equals(generate_raw_listings(200, seed=7))


def test_a_different_seed_gives_a_different_fixture():
    assert not generate_raw_listings(200, seed=7).equals(generate_raw_listings(200, seed=8))


@pytest.mark.parametrize(
    ("description", "predicate"),
    [
        # EDN-22: age would be negative for these.
        (
            "registered after the snapshot",
            lambda f: pd.to_datetime(f.registration_date) > SNAPSHOT_DATE,
        ),
        # EDN-22 again, from the other side: the bound is inclusive, and no body
        # row can show it because they are all registered on the first of a month.
        (
            "registered on the snapshot",
            lambda f: pd.to_datetime(f.registration_date) == SNAPSHOT_DATE,
        ),
        # EDN-03: the drift holdout.
        ("an ES listing", lambda f: f.country_code == "ES"),
        # EDN-05 / FR-04: a make far below any support threshold.
        ("an unsupported make", lambda f: f.make == "Bugatti"),
        # The params price range, at both ends.
        ("a price below the range", lambda f: f.price < 500),
        ("a price above the range", lambda f: f.price > 2_000_000),
        # EDN-04: the used passenger-car scope.
        ("a new car", lambda f: f.offer_type == "N"),
        ("a pre-registered car", lambda f: f.is_preregistered),
        ("a transporter", lambda f: f.vehicle_type == "Transporter"),
        # Age is missing, not zero.
        ("a missing registration date", lambda f: f.registration_date.isna()),
        # The split's group-key fallback.
        ("a private seller", lambda f: f.seller_company_name.isna()),
        # EDN-23: a False in the condition flags is "not asserted", not "no".
        ("a used car flagged is_used=False", lambda f: (f.offer_type == "U") & ~f.is_used),
        # 15 rows of the real file carry no seller and no location at all.
        (
            "a listing with no seller or location",
            lambda f: f.seller_company_name.isna() & f.country_code.isna() & f.city.isna(),
        ),
        # The price range's edges, so inclusive-or-exclusive is testable.
        ("a price exactly on the lower bound", lambda f: f.price == 500),
        ("a price exactly on the upper bound", lambda f: f.price == 2_000_000),
    ],
)
def test_the_fixture_contains(raw_frame, description, predicate):
    assert predicate(raw_frame).any(), f"the fixture has no {description}"


def test_the_fixture_deduplicates_on_the_key_the_pipeline_uses(params):
    # The generator carries its own copy of the key, because it takes no params
    # file and has to build exactly one duplicate. A copy that has drifted would
    # build the fixture for a rule the pipeline no longer has, and the duplicate
    # would be found by neither.
    assert list(DEDUP_KEY) == list(params["preprocess"]["dedup_key"])


def test_the_fixture_contains_exactly_one_duplicate_on_the_dedup_key(raw_frame, params):
    # Exactly one, not "at least one". An earlier version of the generator
    # built every edge row from the same body row, so most of them were
    # duplicates and preprocessing deleted the cases the fixture exists to
    # provide - including the `ES` row and the private seller.
    assert raw_frame.duplicated(subset=params["preprocess"]["dedup_key"]).sum() == 1


@pytest.mark.parametrize(
    ("description", "predicate"),
    [
        ("an ES listing", lambda f: f.country_code == "ES"),
        ("an unsupported make", lambda f: f.make == "Bugatti"),
        ("a private seller", lambda f: f.seller_company_name.isna()),
        ("a listing with no seller or location", lambda f: f.country_code.isna()),
        ("a missing registration date", lambda f: f.registration_date.isna()),
        ("a price on the lower bound", lambda f: f.price == 500),
        ("a price on the upper bound", lambda f: f.price == 2_000_000),
        (
            "a listing registered on the reference date",
            lambda f: pd.to_datetime(f.registration_date) == SNAPSHOT_DATE,
        ),
    ],
)
def test_the_edge_cases_survive_preprocessing(raw_frame, params, description, predicate):
    # Being in the raw frame is not enough: a case the scope, date, price and
    # deduplication rules delete is a case no downstream ticket can test
    # against. These are the ones that must reach the interim frame.
    #
    # The real rules, not a copy of them: while `preprocess` was a stub this
    # test carried its own re-implementation, which could have agreed with the
    # fixture and disagreed with the pipeline. Checked on the edge-case block
    # itself, because the body rows satisfy most of these predicates by chance,
    # which would make the test pass however the rules treated the rows the
    # fixture actually guarantees.
    kept, _ = apply_row_rules(raw_frame, params)
    surviving = kept.loc[kept.index.intersection(raw_frame.tail(EDGE_CASE_ROWS).index)]

    assert predicate(surviving).any(), f"{description} does not survive the preprocessing rules"


def test_the_holdout_case_does_not_swallow_the_others(raw_frame):
    # If every edge row carried the holdout country, `split` would divert the
    # whole block and the train, validation, calibration and test frames would
    # contain none of them.
    edge = raw_frame.tail(EDGE_CASE_ROWS)
    assert (edge["country_code"] == "ES").sum() == 1


def test_the_fixture_keeps_the_real_make_skew(raw_frame):
    # EDN-05's threshold only means something on a skewed distribution, and
    # the real file has four makes at 83 % of the rows.
    shares = raw_frame["make"].value_counts(normalize=True)
    assert shares.iloc[0] > 0.25
    assert shares.head(4).sum() > 0.75


def test_the_fixture_has_sellers_with_several_listings(raw_frame):
    # Otherwise a seller-grouped split would be indistinguishable from a
    # random one and #35's invariant would be untestable.
    counts = raw_frame["seller_company_name"].value_counts()
    assert counts.max() > 1
    assert counts.size < len(raw_frame)


def test_the_fixture_carries_pii_shaped_columns(raw_frame):
    # A "no PII survives preprocessing" test is only meaningful if the input
    # had some to begin with. These values are generated, never real.
    for column in PII_COLUMNS:
        assert raw_frame[column].notna().any()


def test_the_equipment_lists_are_repr_strings_and_never_null(raw_frame):
    for column in (
        "equipment_comfort",
        "equipment_entertainment",
        "equipment_extra",
        "equipment_safety",
    ):
        assert raw_frame[column].notna().all()
        assert raw_frame[column].str.startswith("[").all()


# --------------------------------------------------------------------------
# preprocess (#34)
# --------------------------------------------------------------------------


def test_the_group_key_is_stable_and_carries_no_plaintext_seller_name(raw_frame):
    # Stable, so two runs group the same way (NFR-06), and no value is a name.
    # Not that a name cannot be recovered from a value: the hash is unsalted, so
    # the published source file inverts it, which EDN-37 accepts and discloses
    # rather than the code pretending otherwise.
    hashed = hash_seller_group(raw_frame)

    assert hashed.notna().all()
    assert hashed.equals(hash_seller_group(raw_frame))
    names = set(raw_frame["seller_company_name"].dropna())
    assert names.isdisjoint(set(hashed))


def test_private_sellers_group_by_location_not_into_one_bucket(raw_frame):
    private = raw_frame[raw_frame["seller_company_name"].isna()]
    assert hash_seller_group(private).nunique() > 1


def test_a_seller_with_no_location_still_gets_a_group(raw_frame):
    # The degenerate case: no name and no location. Such listings share one
    # group, which is the safe direction - they cannot then be split across
    # two sets, which is the leak the grouping exists to prevent.
    blank = raw_frame[
        raw_frame["seller_company_name"].isna()
        & raw_frame["country_code"].isna()
        & raw_frame["city"].isna()
    ]
    hashed = hash_seller_group(blank)

    assert len(blank) > 0
    assert hashed.notna().all()
    assert hashed.nunique() == 1


def test_two_listings_from_one_dealer_share_a_group(raw_frame):
    dealers = raw_frame[raw_frame["seller_company_name"].notna()]
    hashed = hash_seller_group(dealers)
    assert hashed.nunique() == dealers["seller_company_name"].nunique()


# --------------------------------------------------------------------------
# split (#35)
# --------------------------------------------------------------------------


def test_the_split_never_puts_a_seller_in_two_sets(raw_frame, params):
    groups = hash_seller_group(raw_frame)
    assigned = assign_split(groups, params["split"]["ratios"], params["seed"])

    per_group = pd.DataFrame({"group": groups, "split": assigned}).groupby("group")["split"]
    assert per_group.nunique().max() == 1


def test_the_split_ratios_are_roughly_honoured(raw_frame, params):
    groups = hash_seller_group(raw_frame)
    ratios = params["split"]["ratios"]
    assigned = assign_split(groups, ratios, params["seed"])
    shares = assigned.value_counts(normalize=True)
    # Two bounds, because they answer different questions and the derived one
    # alone is weaker than what it replaced. `ratio / 2` is an independent
    # oracle: a round number that owes nothing to the code under test, and on
    # this frame it is tighter than the derived bound for validation and
    # calibration (5.00 pp against 6.44 pp), which are the two sets an
    # undersized split hurts most.
    for name in SPLIT_NAMES:
        assert shares[name] > 0
        assert abs(shares[name] - ratios[name]) < ratios[name] / 2

    # And the stage's own bound, so this frame also exercises the gate the
    # stage runs. tests/test_split.py checks it across seeds rather than only
    # for the one params.yaml carries.
    bounds = dispersion_bounds(groups.value_counts(), ratios)
    for name in SPLIT_NAMES:
        assert abs(shares[name] - ratios[name]) <= bounds[name]


def test_the_split_ratios_in_params_sum_to_one(params):
    assert sum(params["split"]["ratios"].values()) == pytest.approx(1.0)


def test_the_test_share_keeps_sc04_measurable(params):
    # Below about 15 % Volvo and then Suzuki fall under SC-04's 500 test rows
    # (reports/analysis/make_support_results.txt), so the criterion would stop
    # checking makes the model card says it checks.
    assert params["split"]["ratios"]["test"] >= 0.15
