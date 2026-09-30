"""`preprocess` stage: raw listings to the interim contract.

The rules run in the order issue #34 fixes, and that order is load-bearing.
Every rule sees only what the rule above it left, so moving one changes what
the others remove: EDN-22 measured 27 future registrations when deduplication
runs after it and 26 when it runs before, and deduplication itself only counts
duplicates among listings that are actually in scope. The order is therefore
expressed once, as `ROW_RULES`, and the stage reports one row count per rule so
the report can cite the funnel rather than a single before-and-after pair.

The supported-make filter is deliberately not here: EDN-05 counts support after
the `ES` holdout, which happens in `split`.

`hash_seller_group` lives here rather than in `split` because its input is
`seller_company_name`, which no later stage can see: the interim contract does
not name it. `split` then groups by the hash alone (`split.group_key`). It is
the split's grouping key and not an anonymisation measure; the function's own
docstring says what it protects and what it does not (EDN-35).
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
import typer

from recommenditos.config import INTERIM_DATA_DIR, PARAMS_FILE, RAW_DATA_DIR
from recommenditos.pipeline import load_params, read_frame, write_frame
from recommenditos.schema import INTERIM_SCHEMA, RAW_SCHEMA, Schema

#: The funnel's first line: the rows the stage was handed, before any rule.
INPUT_STEP = "as read"

app = typer.Typer()


def seller_group_key(frame: pd.DataFrame) -> pd.Series:
    """The plaintext key `hash_seller_group` hashes, one value per listing.

    Public so that a test can count the distinct keys without rebuilding them,
    which is what makes a hash collision observable: two source keys sharing one
    group id merge two sellers into a single split group, and that is the leak
    the grouping exists to prevent.
    """
    private = frame["seller_company_name"].isna()
    location = (
        frame["country_code"].fillna("??").astype("str")
        + "|"
        + frame["zip"].fillna("?????").astype("str")
        + "|"
        + frame["city"].fillna("?").astype("str")
    )
    return frame["seller_company_name"].where(~private, "private|" + location)


def hash_seller_group(frame: pd.DataFrame) -> pd.Series:
    """The split's grouping key, derived before `seller_company_name` is dropped.

    A dealer groups by its company name. A private seller has none, so it
    groups by location instead, which is what problem-spec section 5 means by
    "location for private sellers".

    What the hash gives: no column of the artefact holds a seller name, street,
    zip or city, so nothing downstream - a feature matrix, a model, a log line,
    an API response - can read a seller's identity off it (NFR-08).

    What it does not give: anonymity. The digest is unsalted, so a key always
    hashes to the same value, and the key space is the published source file's
    own columns. A dictionary built from that file inverts all 17,141 dealer ids
    and all 13,576 private-seller ids in about 50 ms, and for a private seller
    that recovers `zip` and `city`, two of the columns `preprocess.pii_columns`
    removes. EDN-35 keeps the hash unsalted and discloses that instead: a pepper
    would make the split irreproducible on a clean clone without the secret, and
    the dealer behind a listing is recoverable from the public file regardless by
    joining on make, model, price, mileage and registration date, which we do
    publish. Truncating the digest neither helps nor hurts there, because a
    dictionary attack does not care how short the output is; 16 hex characters
    are 64 bits, which keeps the chance of an accidental collision over the
    file's 30,717 groups near 3 in 100 billion, while 8 characters already
    collide once on that file.

    Two deliberate over-groupings, both in the safe direction: listings that
    end up in one group cannot be split across two sets, which is the leak the
    grouping exists to prevent.

    - A private seller with no location at all (15 such rows in the raw file)
      falls into one shared group.
    - A dealer groups by its company name alone, so two unrelated dealers of
      the same name in different countries merge. The location is available
      and deliberately not used, because a dealer with branches in two cities
      is one seller and splitting it would leak.
    """
    return seller_group_key(frame).map(
        lambda key: hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:16]
    )


# --------------------------------------------------------------------------
# The row rules, in order
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RowRule:
    """One row rule: the name the funnel reports it under, and what it keeps.

    `keeps` returns a boolean mask over the frame it is handed, never a frame,
    so the runner can count what each rule removed without every rule having
    to report it itself.
    """

    name: str
    keeps: Callable[[pd.DataFrame, dict[str, Any]], pd.Series]


def _is_a_used_passenger_car(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """EDN-04's scope: `offer_type = U`, not pre-registered, `vehicle_type = Car`.

    `is_used` is not consulted. It disagrees with `offer_type` in 18,446 of the
    `offer_type = U` rows, and a `False` there means "not asserted" rather than
    "new" (EDN-23), so it is not a usable negative. `is_preregistered` has the
    same weakness and is used anyway, because it is the only signal there is;
    the residue of unmarked pre-registered listings is a documented limitation
    rather than something this stage can filter (EDN-24).
    """
    rules = params["preprocess"]
    return (
        (frame["offer_type"] == rules["offer_type"])
        & ~frame["is_preregistered"]
        & (frame["vehicle_type"] == rules["vehicle_type"])
    )


def _is_registered_by_the_reference_date(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """EDN-22: age is the reference date minus the registration date, so it may not be negative.

    A missing date is kept. It makes age missing, which the feature stage and
    the model handle, while a date after the snapshot would make age negative,
    which nothing downstream is built for.

    The bound is inclusive: age zero is a real car. Nothing in the snapshot sits
    on it, because every registration date there is the first of a month and the
    reference date is the 8th, but the reference date is a parameter and is the
    request date in serving, where a strict bound would drop every listing
    registered that day.
    """
    registered = frame["registration_date"]
    return registered.isna() | (registered <= pd.Timestamp(params["reference_date"]))


def _is_priced_in_the_training_range(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """The params price range, inclusive at both ends.

    Below the floor the listings are placeholders and data-entry errors - the
    raw file starts at 1 EUR - and above the ceiling they are collector cars
    the model has no support for.
    """
    rules = params["preprocess"]
    return frame["price"].between(rules["price_min_eur"], rules["price_max_eur"], inclusive="both")


def _is_the_first_of_its_kind(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    """The 7-column duplicate key, keeping the copy with the lowest `id`.

    `vin` is only 34 % filled, so a VIN key cannot even see a pair unless both
    of its copies carry one.

    Which copy survives is decided by `id` and not by the order the rows
    arrived in, so that the artefact is a function of the input's content alone
    (NFR-06). Without the sort it is a function of the row order too: the copies
    differ outside the key, in equipment, colour, previous owners and the seller
    they group by, so a re-published, chunk-read or upstream-sorted snapshot
    silently moves about 1.5 % of the 105,405 surviving listings while the row
    count, and every count this stage reports, stays the same. `id` is still in
    the frame when the rules run and `Schema.conform` drops it at the write, so
    it costs the artefact nothing.
    """
    # Stable, so equal ids keep their relative order instead of an arbitrary one.
    ordered = frame.sort_values("id", kind="stable")
    first = ~ordered.duplicated(subset=params["preprocess"]["dedup_key"], keep="first")
    return first.reindex(frame.index)


#: The row rules of issue #34, in the order it fixes. See the module docstring:
#: each rule sees what the rule above it left, so reordering this tuple changes
#: the counts the report and the dataset card publish even where it leaves the
#: artefact identical, and deduplication in particular has to stay last. The
#: order is asserted in `tests/test_preprocess.py`, because a comment cannot
#: hold a contract.
ROW_RULES: tuple[RowRule, ...] = (
    RowRule("used passenger cars", _is_a_used_passenger_car),
    RowRule("registered by the reference date", _is_registered_by_the_reference_date),
    RowRule("price in the training range", _is_priced_in_the_training_range),
    RowRule("deduplicated", _is_the_first_of_its_kind),
)


# --------------------------------------------------------------------------
# The funnel
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FunnelStep:
    """One line of the funnel: what a rule left, and what it cost."""

    rule: str
    rows: int
    #: None for the input line, which no rule produced.
    removed: int | None


@dataclass(frozen=True)
class Funnel:
    """The row count after each rule, in the order the rules ran.

    The report cites these numbers, and the tests assert them, so they are a
    return value rather than only a log line: a count that exists only in a log
    is a count nothing can check.
    """

    steps: tuple[FunnelStep, ...]

    def rows(self, rule: str) -> int:
        """How many rows `rule` left, by the name it reports under."""
        for step in self.steps:
            if step.rule == rule:
                return step.rows
        known = ", ".join(step.rule for step in self.steps)
        raise KeyError(f"no funnel step {rule!r}; the steps are {known}")

    def removed(self, rule: str) -> int:
        """How many rows `rule` removed."""
        for step in self.steps:
            if step.rule == rule and step.removed is not None:
                return step.removed
        raise KeyError(f"no rule {rule!r} removed rows; {INPUT_STEP!r} is not a rule")

    def render(self) -> str:
        """The funnel as an aligned table, ready to be quoted in the report."""
        width = max(len(step.rule) for step in self.steps)
        lines = []
        for step in self.steps:
            removed = "" if step.removed is None else f"-{step.removed:,}"
            # Right-aligned, then stripped: the input line has nothing in the
            # last column, and trailing whitespace is what the pre-commit hooks
            # delete from anything quoting this.
            lines.append(f"  {step.rule:<{width}}  {step.rows:>9,}  {removed:>9}".rstrip())
        return "\n".join(lines)


def apply_row_rules(frame: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, Funnel]:
    """Apply `ROW_RULES` in order, and report what each one left.

    The registration date is parsed here rather than by the caller, because the
    rule that reads it is a date comparison and the stage that writes the frame
    would otherwise have to remember to parse it for the rule's benefit.
    """
    frame = frame.assign(registration_date=pd.to_datetime(frame["registration_date"]))

    steps = [FunnelStep(INPUT_STEP, len(frame), None)]
    for rule in ROW_RULES:
        before = len(frame)
        frame = frame[rule.keeps(frame, params)]
        steps.append(FunnelStep(rule.name, len(frame), before - len(frame)))
    return frame, Funnel(tuple(steps))


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------


def assert_the_contract_excludes(columns: Iterable[str], schema: Schema) -> None:
    """Fail the stage if `schema` names a column that must not survive it.

    `Schema.conform` keeps the excluded columns out by *selecting* the
    contract's columns, which makes the exclusion table of problem-spec section
    4 structural instead of a drop list someone has to remember. That is
    airtight only while the contract names none of them, and the contract lives
    in another module, so the assumption is checked here rather than trusted: a
    column added to `schema.py` in good faith would otherwise turn NFR-08 from
    a guarantee into a coincidence.
    """
    named = sorted(set(columns) & set(schema.names))
    if named:
        raise ValueError(
            f"the {schema.name!r} contract names column(s) that must not survive "
            f"preprocessing: {', '.join(named)}"
        )


def derive_group_key_and_drop_pii(frame: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    """Steps 1 and 2: hash the seller into the group key, then drop the PII columns.

    In that order, because the key is derived from one of the columns the drop
    removes; hashing after the drop could only produce a constant.

    What reaches the artefact is decided by the interim contract, which names
    none of the excluded columns, so this drop is not what keeps PII out of the
    output. It is here so that nothing between the read and the write - a log
    line, a traceback, an intermediate artefact someone adds later - can carry
    PII (NFR-08), and so that `preprocess.pii_columns` is a parameter something
    actually reads.

    A step of its own, and not inlined into `main`, so that
    `reports/analysis/preprocess_funnel.py` measures this stage on the real file
    rather than its own copy of it.
    """
    hashed = frame.assign(seller_group_id=hash_seller_group(frame))
    return hashed.drop(columns=list(params["preprocess"]["pii_columns"]))


def add_the_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Step 7: the target, after the price range and never before.

    The raw file starts at 1 EUR, and `log(0)` is `-inf`, which no contract can
    catch - it is a float like any other.
    """
    return frame.assign(log_price=np.log(frame["price"]))


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "listings.parquet",
    output_path: Path = INTERIM_DATA_DIR / "listings.parquet",
    params_path: Path = PARAMS_FILE,
) -> Funnel:
    params = load_params(params_path)
    # Before the read, not after: a stage whose output contract could leak PII
    # has no business touching the data at all.
    assert_the_contract_excludes(params["preprocess"]["pii_columns"], INTERIM_SCHEMA)

    frame = read_frame(input_path, RAW_SCHEMA)
    frame = derive_group_key_and_drop_pii(frame, params)

    # Steps 3 to 6: the row rules, in the order the module docstring explains.
    frame, funnel = apply_row_rules(frame, params)
    logger.info(
        f"Row funnel, each rule applied to what the rule above it left "
        f"(reference date {params['reference_date']}):\n{funnel.render()}"
    )

    write_frame(add_the_target(frame), output_path, INTERIM_SCHEMA)
    return funnel


if __name__ == "__main__":
    app()
