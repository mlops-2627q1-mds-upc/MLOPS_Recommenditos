"""The `preprocess` stage: one rule at a time, and the order they run in (#34).

Every assertion is against the synthetic fixture, so the suite needs no data
access. The fixture guarantees one row per rule whatever its size, which is
what makes the exact counts below meaningful rather than brittle: they are 3,
1, 2 and 1 at 30 rows and at 2,000.

The real snapshot's funnel is measured by `reports/analysis/preprocess_funnel.py`
and its committed output, so that the report can cite it without reading a
pipeline log; the pipeline logs the same funnel for whatever data it ran on.
"""

import numpy as np
import pandas as pd
import pytest
from tests.conftest import PII_COLUMNS, params_override

from recommenditos.config import PARAMS_FILE
from recommenditos.data.preprocess import (
    INPUT_STEP,
    ROW_RULES,
    apply_row_rules,
    assert_the_contract_excludes,
    derive_group_key_and_drop_pii,
    hash_seller_group,
    seller_group_key,
)
from recommenditos.data.preprocess import main as preprocess_main
from recommenditos.data.synthetic import EDGE_CASE_ROWS, SNAPSHOT_DATE
from recommenditos.schema import INTERIM_SCHEMA, RAW_SCHEMA

#: The rules by the name they report under, so a test below names a rule rather
#: than a position in `ROW_RULES` - the order is what several of them vary.
SCOPE = "used passenger cars"
DATE = "registered by the reference date"
PRICE = "price in the training range"
DEDUP = "deduplicated"

#: One value per deduplication-key column that no generated listing carries, and
#: that every rule above deduplication keeps: the makes and models come from a
#: fixed catalogue, body mileages are multiples of 100, powers are whole numbers,
#: prices are multiples of ten plus at most 15 hundredths, and the earliest
#: generated registration is 300 months before the 2025-11-08 snapshot.
A_VALUE_NO_OTHER_LISTING_HAS = {
    "make": "No Such Make",
    "model": "No Such Model",
    "model_version": "No Such Version",
    "mileage_km_raw": 123_456.0,
    "registration_date": "1999-01-01",
    "price": 1234.56,
    "power_kw": 123.5,
}

#: The columns problem-spec section 4 excludes for a reason other than PII: the
#: four leakage columns and the three identifiers. The PII ones are in
#: `tests.conftest.PII_COLUMNS`, because more than one test module needs them.
LEAKAGE_AND_IDENTIFIER_COLUMNS = (
    "price_net",
    "price_vat_rate",
    "price_tax_deductible",
    "price_negotiable",
    "id",
    "german_hsn_tsn",
)


@pytest.fixture(scope="module")
def stage(_generated_frame, tmp_path_factory):
    """The stage run once end to end, and what it wrote.

    Both paths are explicit: the stage's defaults point into the repository, so
    a test that relied on them would overwrite the pipeline's own artefacts.
    """
    root = tmp_path_factory.mktemp("preprocess")
    raw_path = root / "listings.parquet"
    _generated_frame.to_parquet(raw_path, index=False)
    interim_path = root / "interim.parquet"

    funnel = preprocess_main(raw_path, interim_path, PARAMS_FILE)
    return {"raw": _generated_frame, "interim": pd.read_parquet(interim_path), "funnel": funnel}


@pytest.fixture
def funnel(raw_frame, params):
    return apply_row_rules(raw_frame, params)[1]


@pytest.fixture
def kept(raw_frame, params):
    return apply_row_rules(raw_frame, params)[0]


# --------------------------------------------------------------------------
# Rule 1: the group key is hashed before the PII columns are dropped
# --------------------------------------------------------------------------


def test_the_group_key_is_hashed_from_the_frame_that_still_has_the_seller_name(stage):
    # The order of rules 1 and 2 is not observable from the interim frame's
    # shape - both orders produce the same columns - but it is observable from
    # the values: hashing after the drop could only produce a constant.
    expected = hash_seller_group(stage["raw"])
    written = stage["interim"]["seller_group_id"]

    assert written.notna().all()
    assert set(written) <= set(expected)
    assert written.nunique() > 1


def test_the_group_key_is_the_one_the_split_groups_by(stage, raw_frame, params, kept):
    # The key exists for the split, and for nothing else: no feature set names
    # `seller_group_id`, and the API could not compute it if it wanted to,
    # because `zip` and `city` are not among `evaluate.required_input_fields`.
    # So what has to hold is that the column the split reads is the function's
    # output, row for row and in order, rather than something the stage hashed
    # on its own.
    expected = hash_seller_group(raw_frame).loc[kept.index]

    assert stage["interim"]["seller_group_id"].tolist() == expected.tolist()
    assert params["split"]["group_key"] in stage["interim"].columns


def test_no_two_sellers_share_a_group_id(raw_frame):
    # A collision merges two sellers into one split group, which is the leak the
    # grouping exists to prevent, so the digest needs room: 16 hex characters are
    # 64 bits, where the real file's 30,717 groups collide with a chance near 3
    # in 100 billion. The width is asserted as well as the injectivity, because
    # the fixture has a few hundred groups and would not notice a truncation
    # that already collides once on the real file at 8 characters.
    hashed = hash_seller_group(raw_frame)

    assert hashed.nunique() == seller_group_key(raw_frame).nunique()
    assert (hashed.str.len() == 16).all()


# --------------------------------------------------------------------------
# Rule 2: PII, leakage and identifier columns do not survive
# --------------------------------------------------------------------------


@pytest.mark.req("NFR-08")
def test_no_excluded_column_reaches_the_interim_frame(stage):
    excluded = {*PII_COLUMNS, *LEAKAGE_AND_IDENTIFIER_COLUMNS}
    assert excluded.isdisjoint(stage["interim"].columns)


@pytest.mark.req("NFR-08")
def test_the_interim_contract_names_no_column_the_exclusion_table_excludes():
    # `Schema.conform` keeps the excluded columns out by selecting the
    # contract's columns, so the exclusion table of problem-spec section 4 is
    # enforced structurally - but only for as long as the contract names none
    # of them. This reads that table off the raw contract's own descriptions,
    # which mark each excluded column and each PII column, so a column added to
    # either module without the other fails here rather than in a review.
    marked = tuple(
        column.name
        for column in RAW_SCHEMA.columns
        if "Excluded:" in column.description or "PII" in column.description
    )

    assert set(marked) >= {*PII_COLUMNS, *LEAKAGE_AND_IDENTIFIER_COLUMNS}
    assert set(marked).isdisjoint(INTERIM_SCHEMA.names)


@pytest.mark.req("NFR-08")
def test_the_contract_guard_rejects_a_column_the_contract_keeps():
    # The guard behind the assertion above: `conform` cannot be trusted to hide
    # a column the contract names.
    with pytest.raises(ValueError, match="must not survive"):
        assert_the_contract_excludes(["make", "price"], INTERIM_SCHEMA)

    assert assert_the_contract_excludes(PII_COLUMNS, INTERIM_SCHEMA) is None


@pytest.mark.req("NFR-08")
def test_the_stage_refuses_to_run_when_the_contract_could_leak_pii(raw_path, tmp_path, params):
    # The test above covers the assertion, this one covers the call: without it,
    # deleting the guard from the stage leaves the whole "refuses before it reads
    # anything" mechanism unobserved, and a green suite is then the only evidence
    # anyone looks at. The input exists, so the refusal can only come from the
    # guard, and the output must not: refusing after writing would be no guard.
    output = tmp_path / "interim.parquet"
    leaky = params_override(tmp_path, params, preprocess={"pii_columns": ["make"]})

    with pytest.raises(ValueError, match="must not survive"):
        preprocess_main(raw_path, output, leaky)

    assert not output.exists()


@pytest.mark.req("NFR-08")
def test_the_frame_the_row_rules_see_carries_no_pii(raw_frame, params):
    # The explicit drop, which the contract makes unnecessary for the artefact
    # but not for anything in between: a log line, a traceback or an
    # intermediate artefact someone adds later sees this frame, not the write.
    prepared = derive_group_key_and_drop_pii(raw_frame, params)

    assert set(params["preprocess"]["pii_columns"]).isdisjoint(prepared.columns)
    assert prepared["seller_group_id"].notna().all()


def test_no_row_rule_reads_a_pii_column(raw_frame, params, funnel):
    # Which is what lets the stage drop them before the rules run, instead of
    # relying on the write to be the only thing that ever sees the frame.
    stripped = raw_frame.drop(columns=list(params["preprocess"]["pii_columns"]))

    assert apply_row_rules(stripped, params)[1] == funnel


# --------------------------------------------------------------------------
# Rule 3: used passenger cars (EDN-04)
# --------------------------------------------------------------------------


def test_only_used_passenger_cars_survive_the_scope(kept, params):
    rules = params["preprocess"]

    assert (kept["offer_type"] == rules["offer_type"]).all()
    assert (kept["vehicle_type"] == rules["vehicle_type"]).all()
    assert not kept["is_preregistered"].any()


def test_the_scope_removes_the_new_the_pre_registered_and_the_transporter(funnel, raw_frame):
    # One of each, which is what the fixture guarantees whatever its size.
    assert funnel.removed(SCOPE) == 3
    assert (raw_frame["offer_type"] == "N").sum() == 1
    assert raw_frame["is_preregistered"].sum() == 1
    assert (raw_frame["vehicle_type"] == "Transporter").sum() == 1


def test_a_used_car_whose_is_used_flag_disagrees_is_kept(kept, raw_frame):
    # EDN-23: a `False` there means "not asserted", not "new". It disagrees
    # with `offer_type` in 18,446 of the real file's `U` rows, so reading it as
    # a negative would throw away 16 % of the scope for nothing.
    disagreeing = raw_frame[(raw_frame["offer_type"] == "U") & ~raw_frame["is_used"]]

    assert len(disagreeing) > 0
    assert kept.index.intersection(disagreeing.index).size > 0


def test_is_used_is_not_a_feature(stage):
    assert "is_used" not in stage["interim"].columns


# --------------------------------------------------------------------------
# Rule 4: registered by the reference date (EDN-22)
# --------------------------------------------------------------------------


def test_no_listing_registered_after_the_reference_date_survives(kept, params):
    # Age is the reference date minus this one, so a later date means a
    # negative age and nothing downstream is built for that. Phrased as "none
    # is later" rather than "all are earlier", because a missing date is
    # neither and is deliberately kept.
    assert not (kept["registration_date"] > pd.Timestamp(params["reference_date"])).any()


def test_the_date_rule_removes_the_post_snapshot_listing_and_nothing_else(funnel, raw_frame):
    after = pd.to_datetime(raw_frame["registration_date"]) > SNAPSHOT_DATE

    assert after.sum() == 1
    assert funnel.removed(DATE) == 1


def test_a_listing_registered_on_the_reference_date_survives(kept, raw_frame, params):
    # The bound is inclusive, and only the fixture can show it: every
    # registration date in the snapshot is the first of a month while the
    # reference date is the 8th, so a strict bound would remove nothing there and
    # then drop every listing registered that day in serving, where the
    # reference date is the request date.
    reference = pd.Timestamp(params["reference_date"])
    on_the_bound = pd.to_datetime(raw_frame["registration_date"]) == reference

    assert on_the_bound.sum() == 1
    assert (kept["registration_date"] == reference).any()


def test_a_missing_registration_date_is_kept(kept, raw_frame):
    # Missing age is a case the feature stage and the model handle; a dropped
    # row is one the model never learns from.
    assert raw_frame["registration_date"].isna().any()
    assert kept["registration_date"].isna().any()


# --------------------------------------------------------------------------
# Rule 5: the training price range
# --------------------------------------------------------------------------


def test_only_prices_inside_the_params_range_survive(kept, params):
    rules = params["preprocess"]

    assert kept["price"].min() >= rules["price_min_eur"]
    assert kept["price"].max() <= rules["price_max_eur"]


def test_both_bounds_are_inclusive(kept, params):
    rules = params["preprocess"]

    assert (kept["price"] == rules["price_min_eur"]).any()
    assert (kept["price"] == rules["price_max_eur"]).any()


def test_the_price_rule_removes_one_listing_at_each_end(funnel):
    assert funnel.removed(PRICE) == 2


# --------------------------------------------------------------------------
# Rule 6: deduplication on the 7-column key
# --------------------------------------------------------------------------


def test_no_duplicate_on_the_key_survives(kept, params):
    assert not kept.duplicated(subset=params["preprocess"]["dedup_key"]).any()


def test_the_key_is_the_seven_columns_the_ticket_fixes(params):
    # Measured, not a default: on the real file the seven-column key removes
    # 4,506 rows, the same key without `power_kw` removes 4,542 and without
    # `price` 6,184, so a column dropped from it deletes listings that are not
    # duplicates - and the funnel reports the same shape either way.
    assert tuple(params["preprocess"]["dedup_key"]) == (
        "make",
        "model",
        "model_version",
        "mileage_km_raw",
        "registration_date",
        "price",
        "power_kw",
    )
    assert set(A_VALUE_NO_OTHER_LISTING_HAS) == set(params["preprocess"]["dedup_key"])


@pytest.mark.parametrize("column", sorted(A_VALUE_NO_OTHER_LISTING_HAS))
def test_two_listings_differing_in_one_key_column_are_not_duplicates(
    raw_frame, params, funnel, column
):
    # The test above pins the parameter; this one pins the rule's use of it. A
    # rule that reads six of the seven columns deletes a listing that differs in
    # the seventh, and on the real file that is 36 listings for `power_kw` alone.
    # Every value below passes the rules above deduplication, so the copy can
    # only be lost to this one.
    twin = raw_frame.iloc[[0]].assign(**{column: A_VALUE_NO_OTHER_LISTING_HAS[column]})
    grown = pd.concat([raw_frame, twin], ignore_index=True)

    _, with_twin = apply_row_rules(grown, params)

    assert with_twin.rows(DEDUP) == funnel.rows(DEDUP) + 1
    assert with_twin.removed(DEDUP) == funnel.removed(DEDUP)


def test_which_copy_survives_follows_the_content_and_not_the_row_order(raw_frame, params):
    # Every other test here counts rows, and the count is the same whichever copy
    # of a pair survives, so this one asserts the rows themselves. On the real
    # file the copies differ in equipment, colour, previous owners and the seller
    # they group by: reversing the input moved 2,730 of the 105,405 survivors and
    # shuffling it 1,549, with the row count identical in every case. Now that the
    # stage fetches the published file instead of generating it, a re-publish, a
    # chunked read or an upstream sort would do the same.
    kept_ids = set(apply_row_rules(raw_frame, params)[0]["id"])
    reordered = {
        "reversed": raw_frame[::-1],
        "shuffled": raw_frame.sample(frac=1.0, random_state=params["seed"]),
    }

    for description, frame in reordered.items():
        survivors = apply_row_rules(frame.reset_index(drop=True), params)[0]
        assert set(survivors["id"]) == kept_ids, description


def test_deduplication_removes_the_fixture_duplicate_and_keeps_the_lower_id(
    kept, raw_frame, params
):
    key = params["preprocess"]["dedup_key"]
    pair = raw_frame.index[raw_frame.duplicated(subset=key, keep=False)]
    lower, higher = raw_frame.loc[pair, "id"].sort_values().index

    assert len(pair) == 2
    assert lower in kept.index
    assert higher not in kept.index


@pytest.mark.parametrize(
    ("rule", "disqualify"),
    [
        (SCOPE, lambda pair: pair.assign(offer_type="N")),
        (DATE, lambda pair: pair.assign(registration_date="2026-01-01")),
        (PRICE, lambda pair: pair.assign(price=1.0)),
    ],
)
def test_deduplication_counts_only_the_duplicates_the_rules_above_it_left(
    raw_frame, params, funnel, rule, disqualify
):
    # This is what the rule order buys, and the reason it is load-bearing: rule
    # 6 removes 4,506 rows of the real file rather than the 6,347 duplicates
    # the raw file holds, because the rules above it delete the rest first.
    # EDN-22 measured the same effect from the other side, 27 post-snapshot
    # listings before deduplication and 26 after.
    #
    # A duplicate pair that `rule` disqualifies must therefore cost `rule` two
    # rows and cost deduplication nothing. If deduplication moved above `rule`,
    # it would claim the pair instead and both counts would move.
    pair = disqualify(raw_frame.iloc[[0, 0]])
    grown = pd.concat([raw_frame, pair], ignore_index=True)

    _, with_pair = apply_row_rules(grown, params)

    assert with_pair.removed(rule) == funnel.removed(rule) + 2
    assert with_pair.removed(DEDUP) == funnel.removed(DEDUP)


# --------------------------------------------------------------------------
# Rule 7: the target
# --------------------------------------------------------------------------


def test_the_target_is_the_log_of_the_price(stage):
    interim = stage["interim"]

    assert np.allclose(interim["log_price"], np.log(interim["price"]))
    assert np.isfinite(interim["log_price"]).all()


# --------------------------------------------------------------------------
# The funnel the report cites
# --------------------------------------------------------------------------


def test_the_rules_run_in_the_order_the_ticket_fixes():
    # The order is a contract, not a comment. Only deduplication's position
    # changes the artefact; the other three permutations leave the real file at
    # 105,405 rows either way and still change the per-rule counts the dataset
    # card and reports/analysis/preprocess_funnel_results.txt publish - the date
    # rule first removes 164 instead of 27, the price rule first 77 instead of
    # 75. A rule renamed in the stage also fails here rather than quietly
    # turning the tests that name it into tests of nothing.
    assert [rule.name for rule in ROW_RULES] == [SCOPE, DATE, PRICE, DEDUP]


def test_the_funnel_reports_the_input_and_then_one_line_per_rule(funnel, raw_frame):
    assert [step.rule for step in funnel.steps] == [INPUT_STEP, *(r.name for r in ROW_RULES)]
    assert funnel.rows(INPUT_STEP) == len(raw_frame)
    assert funnel.steps[0].removed is None


def test_every_step_accounts_for_the_rows_the_step_above_it_lost(funnel):
    for above, step in zip(funnel.steps[:-1], funnel.steps[1:], strict=True):
        assert step.removed == above.rows - step.rows
        assert step.rows <= above.rows


def test_the_last_step_is_the_frame_the_stage_writes(funnel, kept, stage):
    assert funnel.rows(ROW_RULES[-1].name) == len(kept) == len(stage["interim"])


def test_the_funnel_renders_as_a_table_the_report_can_quote(funnel):
    lines = funnel.render().splitlines()

    assert len(lines) == len(funnel.steps)
    for step, line in zip(funnel.steps, lines, strict=True):
        assert step.rule in line
        assert f"{step.rows:,}" in line


def test_asking_the_funnel_for_a_rule_it_does_not_have_says_which_it_has(funnel):
    with pytest.raises(KeyError, match=INPUT_STEP):
        funnel.rows("no such rule")
    with pytest.raises(KeyError, match="not a rule"):
        funnel.removed(INPUT_STEP)


# --------------------------------------------------------------------------
# The stage as a whole
# --------------------------------------------------------------------------


def test_the_stage_writes_a_frame_that_satisfies_the_interim_contract(stage):
    INTERIM_SCHEMA.validate(stage["interim"])


def test_the_stage_reports_the_same_funnel_it_wrote(stage):
    assert stage["funnel"].rows(ROW_RULES[-1].name) == len(stage["interim"])


def test_the_rules_keep_every_edge_case_a_later_ticket_has_to_test_against(kept, raw_frame):
    # The fixture's guarantees are only worth something if the rows survive to
    # where they are consumed. Checked on the edge-case block itself: the body
    # rows cover most of these predicates by chance, so a check over the whole
    # frame would pass however the rules treated the guaranteed rows.
    edge = kept.index.intersection(raw_frame.tail(EDGE_CASE_ROWS).index)
    surviving = kept.loc[edge]

    assert (surviving["country_code"] == "ES").any()
    assert (surviving["make"] == "Bugatti").any()
    assert surviving["registration_date"].isna().any()
    assert surviving["country_code"].isna().any()
