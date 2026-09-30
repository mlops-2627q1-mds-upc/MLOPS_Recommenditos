"""The synthetic fixture, and the paths through the data stages.

The fixture is the reason the stage tickets can be built in parallel, so its
guarantees are tested here rather than assumed. If one of these fails, a test
somewhere else is silently checking nothing.
"""

import pandas as pd
import pytest
from tests.conftest import PII_COLUMNS

from recommenditos import config
from recommenditos.data.preprocess import hash_seller_group
from recommenditos.data.split_data import SPLIT_NAMES, assign_split, ratio_tolerances
from recommenditos.data.synthetic import (
    EDGE_CASE_ROWS,
    SNAPSHOT_DATE,
    generate_raw_listings,
)
from recommenditos.schema import RAW_SCHEMA

DEDUP_KEY = [
    "make",
    "model",
    "model_version",
    "mileage_km_raw",
    "registration_date",
    "price",
    "power_kw",
]


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


def test_the_fixture_contains_exactly_one_duplicate_on_the_dedup_key(raw_frame):
    # Exactly one, not "at least one". An earlier version of the generator
    # built every edge row from the same body row, so most of them were
    # duplicates and preprocessing deleted the cases the fixture exists to
    # provide - including the `ES` row and the private seller.
    assert raw_frame.duplicated(subset=DEDUP_KEY).sum() == 1


def _scoped_like_preprocess(frame, params):
    """The rows issue #34's rules leave behind, in the order it specifies."""
    rules = params["preprocess"]
    kept = frame[
        (frame["offer_type"] == rules["offer_type"])
        & ~frame["is_preregistered"]
        & (frame["vehicle_type"] == rules["vehicle_type"])
    ]
    registered = pd.to_datetime(kept["registration_date"])
    kept = kept[registered.isna() | (registered <= pd.Timestamp(params["reference_date"]))]
    kept = kept[
        kept["price"].between(rules["price_min_eur"], rules["price_max_eur"], inclusive="both")
    ]
    return kept.drop_duplicates(subset=rules["dedup_key"], keep="first")


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
    ],
)
def test_the_edge_cases_survive_preprocessing(raw_frame, params, description, predicate):
    # Being in the raw frame is not enough: a case the scope, date, price and
    # deduplication rules delete is a case no downstream ticket can test
    # against. These are the ones that must reach the interim frame.
    assert predicate(_scoped_like_preprocess(raw_frame, params)).any(), (
        f"{description} does not survive the preprocessing rules"
    )


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


def test_the_group_key_is_stable_and_hides_the_seller_name(raw_frame):
    hashed = hash_seller_group(raw_frame)

    assert hashed.notna().all()
    assert hashed.equals(hash_seller_group(raw_frame))
    # The name must not be recoverable from, or visible in, the key.
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


def test_the_same_seed_gives_the_same_split(raw_frame, params):
    groups = hash_seller_group(raw_frame)
    ratios, seed = params["split"]["ratios"], params["seed"]

    assert assign_split(groups, ratios, seed).equals(assign_split(groups, ratios, seed))


def test_a_different_seed_moves_sellers_between_sets(raw_frame, params):
    groups = hash_seller_group(raw_frame)
    ratios = params["split"]["ratios"]

    assert not assign_split(groups, ratios, 1).equals(assign_split(groups, ratios, 2))


def test_the_split_ratios_are_roughly_honoured(raw_frame, params):
    groups = hash_seller_group(raw_frame)
    ratios = params["split"]["ratios"]
    assigned = assign_split(groups, ratios, params["seed"])
    shares = assigned.value_counts(normalize=True)
    # The bound comes from the group sizes rather than from a round number.
    # Whole sellers move at a time, so how far a share can honestly land from
    # its ratio is a property of the data; `ratio_tolerances` derives it, and
    # tests/test_split.py checks that it holds across seeds rather than only
    # for the one params.yaml carries.
    tolerances = ratio_tolerances(groups.value_counts(), ratios)

    for name in SPLIT_NAMES:
        assert shares[name] > 0
        assert abs(shares[name] - ratios[name]) <= tolerances[name]


def test_the_split_ratios_in_params_sum_to_one(params):
    assert sum(params["split"]["ratios"].values()) == pytest.approx(1.0)


def test_the_test_share_keeps_sc04_measurable(params):
    # Below about 15 % Volvo and then Suzuki fall under SC-04's 500 test rows
    # (reports/analysis/make_support_results.txt), so the criterion would stop
    # checking makes the model card says it checks.
    assert params["split"]["ratios"]["test"] >= 0.15
