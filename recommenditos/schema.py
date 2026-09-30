"""The data contract every pipeline stage reads and writes against.

This module is the boundary between the stages. A stage does not code against
the stage before it, it codes against the schema below, which is why the stages
can be built in parallel.

It describes *structure* only: which columns exist, their dtype, whether they
may be null and - for a categorical - the levels its codes are positions in.
Value rules - price ranges, the supported make list, fill rates - are data
quality, and they live in the Great Expectations suites (#25). The two answer
different questions: a structural break is a bug in our code, a value break is a
change in the data. The level list sits on the structural side because a code
*is* a level's position: two frames over the same levels in a different order
hold different numbers under the same `str(dtype)`, so a contract that omitted
them would pass a frame whose every code means something else.

Two entry points matter:

- `Schema.conform(frame)` selects the contract's columns in its order, casts
  them to the declared dtypes and validates the result. Every stage calls it
  before writing, and it is also the one supported way for a consumer to cast a
  frame of its own to a loaded contract. Because it selects, a column the
  contract does not name cannot reach an artefact - which is how the PII columns
  of NFR-08 are kept out structurally rather than by remembering to drop them.
- `Schema.validate(frame)` checks a frame and raises `SchemaError` listing
  every problem at once. Every stage calls it after reading.

Both exist because Parquet preserves whatever dtype it is handed rather than
normalising it, so without an explicit cast at each boundary the frames drift
apart stage by stage.
"""

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import pandas as pd

# How many offending row positions a null-value problem names before it stops.
_MAX_REPORTED_POSITIONS = 5

# How many level names a level problem prints before it stops. The full list can
# be hundreds of trims long, and the first few plus the counts already say what
# went wrong.
_MAX_REPORTED_LEVELS = 4

#: The one dtype whose meaning depends on data, so the one that needs `levels`.
_CATEGORY = "category"


class SchemaError(ValueError):
    """A DataFrame does not match the schema it was checked against."""


@dataclass(frozen=True)
class Column:
    """One column of a contract.

    `dtype` is the pandas dtype as `str(series.dtype)` renders it, so it is
    compared, and cast to, without any translation table. On pandas 3 a text
    column is `str`, never `object`: `object` passes in the stage that writes
    it and fails in the stage that reads it back from Parquet.

    `levels` belongs to a `category` and to nothing else, and a `category`
    cannot be declared without it. Everything else about a categorical column is
    the same whatever its levels are - `str(dtype)` is `'category'` either way -
    so a contract that left them out would accept a frame whose codes number the
    same cars differently, and a consumer holding that contract would have to
    infer the levels from the rows in front of it.
    """

    name: str
    dtype: str
    nullable: bool
    description: str
    levels: "tuple[str, ...] | None" = None

    def __post_init__(self) -> None:
        # JSON has no tuple, so a loaded row arrives with a list; coercing here
        # keeps a loaded contract equal to the built one instead of merely
        # equivalent, which is what lets a test compare the two directly.
        if self.levels is not None and not isinstance(self.levels, tuple):
            object.__setattr__(self, "levels", tuple(self.levels))
        if self.dtype == _CATEGORY and self.levels is None:
            raise SchemaError(
                f"column {self.name!r} is {_CATEGORY!r} but declares no levels; a code is "
                f"a level's position, so a contract without them cannot say what a code "
                f"means - build it with Schema.as_category"
            )
        if self.dtype != _CATEGORY and self.levels is not None:
            raise SchemaError(
                f"column {self.name!r} declares levels but its dtype is {self.dtype!r}; "
                f"only a {_CATEGORY!r} column has levels"
            )


#: The columns a feature matrix carries as the label rather than as an input.
#:
#: Defined here rather than beside the contracts below because `Schema` itself
#: has to be able to tell a feature from a target: `train` publishes the two
#: lists separately so that no consumer has to slice the target off by hand, and
#: a serving frame is checked against the features alone.
_TARGET_COLUMNS: "tuple[Column, ...]" = (
    Column("price", "float64", False, "Asking price in EUR, inside the params price range."),
    Column("log_price", "float64", False, "log(price). The training target (problem-spec 3)."),
)

#: The names of the above, which is what a consumer compares against.
TARGET_NAMES: tuple[str, ...] = tuple(column.name for column in _TARGET_COLUMNS)


@dataclass(frozen=True)
class Schema:
    """An ordered set of columns, with the checks that enforce it."""

    name: str
    columns: tuple[Column, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def dtypes(self) -> dict[str, str]:
        return {column.name: column.dtype for column in self.columns}

    @property
    def levels(self) -> dict[str, tuple[str, ...]]:
        """Every categorical column's levels, in the order their codes use."""
        return {column.name: column.levels for column in self.columns if column.levels is not None}

    @property
    def target_names(self) -> tuple[str, ...]:
        """The label columns this contract carries, in its own order."""
        return tuple(name for name in self.names if name in TARGET_NAMES)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Every column of this contract that is not a label.

        What a model may be fitted on. `train` writes this list into the model
        artefact and `evaluate` and the API read it back, so none of them has to
        know that the last two columns of a matrix happen to be the target.
        """
        return tuple(name for name in self.names if name not in TARGET_NAMES)

    def features(self, *, name: str = "") -> "Schema":
        """This contract without its label columns.

        What a serving frame is checked against: a request describes the car and
        not its price, so conforming it to the full contract would reject it for
        a missing column no caller could supply.
        """
        return self.select(self.feature_names, name=name or f"{self.name}-features")

    def column(self, name: str) -> Column:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"{self.name!r} has no column {name!r}")

    def renamed(self, name: str) -> "Schema":
        """The same columns under a different contract name."""
        return replace(self, name=name)

    def to_dicts(self) -> list[dict]:
        """This contract as plain JSON-serialisable rows.

        A contract whose columns depend on the data cannot be a constant in this
        module, so the stage that builds it writes it out and the stages that
        read its artefacts load it back. `from_dicts` is the other half.
        """
        rows = []
        for column in self.columns:
            row = asdict(column)
            if row["levels"] is None:
                # Writing `"levels": null` on the 30-odd non-categorical columns
                # would be noise in an artefact people read; `Column`'s default
                # puts it back on the way in.
                del row["levels"]
            else:
                row["levels"] = list(row["levels"])
            rows.append(row)
        return rows

    @classmethod
    def from_dicts(cls, rows: "list[dict]", *, name: str) -> "Schema":
        """A contract from what `to_dicts` wrote.

        `Column(**row)` rather than a field-by-field read: a row that has lost a
        field, or gained one, fails here naming it instead of producing a
        contract that silently checks less than it should. An empty list is
        refused for the same reason: a contract with no columns does not check
        less, it reports every column of every frame as unexpected.
        """
        if not rows:
            raise SchemaError(f"{name!r} would have no columns, so it could check nothing")
        return cls(name=name, columns=tuple(Column(**row) for row in rows))

    def select(self, names: "list[str] | tuple[str, ...]", *, name: str) -> "Schema":
        """A sub-contract over `names`, in the order given."""
        return Schema(name=name, columns=tuple(self.column(each) for each in names))

    def extend(self, columns: "tuple[Column, ...]", *, name: str = "") -> "Schema":
        """This contract plus `columns`, for the columns a stage derives itself."""
        added = [column.name for column in columns]
        known = set(self.names)
        # Two equipment items that normalise to the same column name would
        # otherwise produce a schema with a repeated column, and the failure
        # would surface much later as a TypeError out of `conform`.
        clashes = sorted({name for name in added if name in known or added.count(name) > 1})
        if clashes:
            raise SchemaError(f"{self.name!r} would repeat column(s) {', '.join(clashes)}")
        return Schema(name=name or self.name, columns=self.columns + columns)

    def drop(self, names: "list[str] | tuple[str, ...]", *, name: str = "") -> "Schema":
        """This contract without `names`.

        For a stage that replaces a column with something derived from it: the
        equipment lists become multi-hot columns, so `build_features` drops the
        four repr strings and extends with what it built. Doing that here keeps
        the change inside that stage instead of in this module.
        """
        unknown = sorted(set(names) - set(self.names))
        if unknown:
            raise SchemaError(f"{self.name!r} has no column(s) {', '.join(unknown)}")
        kept = tuple(column for column in self.columns if column.name not in set(names))
        return Schema(name=name or self.name, columns=kept)

    def with_dtype(self, column: str, dtype: str, *, name: str = "") -> "Schema":
        """This contract with one column's dtype changed.

        A stage that parses a column keeps its name but changes its type:
        `weight_kg` arrives as the text `'1,945 kg'` and leaves `build_features`
        as a number. That is the stage's business, so it is expressed here rather
        than by editing the constants below.

        `category` cannot be reached through this method. A categorical needs its
        levels as well as its dtype, and `as_category` is the only way to declare
        one, so no code path can produce a contract that says `'category'` and
        leaves the levels for a consumer to guess.
        """
        if dtype == _CATEGORY:
            raise SchemaError(
                f"use as_category to declare {column!r} {_CATEGORY!r}, because a "
                f"categorical needs its levels as well as its dtype"
            )
        return self._replacing(column, name=name, dtype=dtype, levels=None)

    def as_category(self, column: str, levels: "tuple[str, ...]", *, name: str = "") -> "Schema":
        """This contract with one column declared categorical over exactly `levels`.

        The levels travel in the contract because the codes mean nothing without
        them: they reach the stage that reads the matrix, and the API, through
        the artefact rather than being re-derived from whatever rows that
        consumer happens to hold (EDN-02).
        """
        return self._replacing(column, name=name, dtype=_CATEGORY, levels=tuple(levels))

    def _replacing(self, column: str, *, name: str = "", **fields) -> "Schema":
        if column not in self.names:
            raise SchemaError(f"{self.name!r} has no column {column!r}")
        changed = tuple(
            replace(each, **fields) if each.name == column else each for each in self.columns
        )
        return Schema(name=name or self.name, columns=changed)

    def validate(self, frame: pd.DataFrame) -> None:
        """Raise `SchemaError` naming every way `frame` breaks this contract."""
        problems = [
            *self._missing_problems(frame),
            *self._unexpected_problems(frame),
            *self._dtype_problems(frame),
            *self._null_problems(frame),
        ]
        if problems:
            listed = "\n".join(f"  - {problem}" for problem in problems)
            raise SchemaError(
                f"frame does not match the {self.name!r} contract "
                f"({len(problems)} problem(s)):\n{listed}"
            )

    def conform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Select, order and cast `frame` to this contract, then validate it.

        Also the one supported way for a consumer to cast a frame of its own -
        a rebuilt matrix, a masked copy, a one-row request - to a contract it
        loaded. A categorical is cast to the *declared* levels, so a value the
        contract does not know becomes missing rather than a level of its own:
        `.cat.codes` on the result therefore means the same thing here as it did
        in the matrix the model was fitted on.
        """
        missing = self._missing_problems(frame)
        if missing:
            listed = "\n".join(f"  - {problem}" for problem in missing)
            raise SchemaError(f"cannot conform to {self.name!r}:\n{listed}")

        conformed = frame.loc[:, list(self.names)].copy()

        # Casting before the null check would hide nulls rather than report
        # them: `astype("bool")` turns a missing value into True.
        nulls = self._null_problems(conformed)
        if nulls:
            listed = "\n".join(f"  - {problem}" for problem in nulls)
            raise SchemaError(f"cannot conform to {self.name!r}:\n{listed}")

        for column in self.columns:
            conformed[column.name] = _cast(conformed[column.name], column)
        self.validate(conformed)
        return conformed

    def _missing_problems(self, frame: pd.DataFrame) -> list[str]:
        present = set(frame.columns)
        return [f"missing column {name!r}" for name in self.names if name not in present]

    def _unexpected_problems(self, frame: pd.DataFrame) -> list[str]:
        known = set(self.names)
        return [
            f"unexpected column {name!r}, which the contract does not allow"
            for name in frame.columns
            if name not in known
        ]

    def _dtype_problems(self, frame: pd.DataFrame) -> list[str]:
        problems = []
        for column in self.columns:
            if column.name not in frame.columns:
                continue
            actual = frame[column.name].dtype
            if str(actual) != column.dtype:
                problems.append(
                    f"column {column.name!r}: expected dtype {column.dtype}, got {actual}"
                )
            elif column.levels is not None:
                problems.extend(_level_problems(column, actual))
        return problems

    def _null_problems(self, frame: pd.DataFrame) -> list[str]:
        problems = []
        for column in self.columns:
            if column.nullable or column.name not in frame.columns:
                continue
            missing = frame[column.name].isna()
            count = int(missing.sum())
            if count:
                where = [int(position) for position in missing.to_numpy().nonzero()[0]]
                shown = where[:_MAX_REPORTED_POSITIONS]
                more = "" if len(where) <= _MAX_REPORTED_POSITIONS else ", ..."
                problems.append(
                    f"column {column.name!r}: {count} null value(s) in a column that "
                    f"may not be null, at row position(s) {shown}{more}"
                )
        return problems


def _level_problems(column: Column, actual: pd.CategoricalDtype) -> list[str]:
    """How `actual`'s levels differ from the declared ones, if they do."""
    observed = tuple(str(level) for level in actual.categories)
    if observed == column.levels:
        return []
    declared = set(column.levels)
    if declared == set(observed):
        detail = "the same levels in a different order"
    else:
        detail = (
            f"{_listed(sorted(declared - set(observed)))} absent and "
            f"{_listed(sorted(set(observed) - declared))} added"
        )
    return [
        (
            f"column {column.name!r}: expected {len(column.levels)} level(s) "
            f"{_listed(column.levels)}, got {len(observed)} {_listed(observed)} - {detail}. "
            f"A code is a level's position, so this changes what every code means."
        )
    ]


def _listed(levels: "tuple[str, ...] | list[str]") -> str:
    shown = ", ".join(repr(level) for level in levels[:_MAX_REPORTED_LEVELS])
    return f"[{shown}{', ...' if len(levels) > _MAX_REPORTED_LEVELS else ''}]"


def _cast(series: pd.Series, column: Column) -> pd.Series:
    if column.levels is not None:
        return _cast_levels(series, column.levels)
    if str(series.dtype) == column.dtype:
        return series
    if column.dtype.startswith("datetime64"):
        # Parse first: pandas 3 parses to microsecond resolution, so a bare
        # astype from text would not reach the declared unit.
        return pd.to_datetime(series).astype(column.dtype)
    return series.astype(column.dtype)


def _cast_levels(series: pd.Series, levels: "tuple[str, ...]") -> pd.Series:
    """`series` as a categorical over exactly `levels`, whichever of them occur.

    Never over the levels the data happens to hold. `astype('category')` would
    infer them, so a 50-row subset, a rebuilt frame or a one-row request would
    each come back numbered differently from the matrix the model was fitted on,
    and `.cat.codes` would silently mean something else. A value outside the
    declared levels becomes missing instead: it carries no fitted signal, and
    giving it a code of its own would renumber every other level.
    """
    if isinstance(series.dtype, pd.CategoricalDtype) and (
        tuple(str(level) for level in series.dtype.categories) == levels
    ):
        return series
    # Via `object`, because `where` on a categorical is restricted to its own
    # levels and would raise on a value this contract is about to drop.
    plain = series.astype("object") if isinstance(series.dtype, pd.CategoricalDtype) else series
    return plain.where(plain.isin(levels)).astype(pd.CategoricalDtype(levels))


# --------------------------------------------------------------------------
# Raw: the AutoScout24 snapshot exactly as it is published.
#
# Dtypes and nullability are measured on the Zenodo file with MD5
# b23a122cc51baf7de39f449193ff0d28, which is the file data/raw/*.dvc points at.
# The 22 columns marked not-null are the fully populated ones the dataset card
# reports. Nothing here is a modelling choice; it is what the upstream file is,
# and a change to it should fail the `download` stage loudly.
# --------------------------------------------------------------------------
RAW_SCHEMA = Schema(
    name="raw",
    columns=(
        Column("id", "str", False, "Listing identifier. Excluded: identifier."),
        Column("description", "str", True, "Free text, multilingual. Excluded: leaks the price."),
        Column("ratings_average", "str", True, "Seller rating, comma decimal. Excluded: seller."),
        Column("ratings_count", "float64", True, "Seller rating count. Excluded: seller."),
        Column("ratings_recommend_percentage", "float64", True, "Seller score. Excluded: seller."),
        Column("price_currency", "str", False, "EUR for every row. Constant after scoping."),
        Column("price", "float64", False, "Asking price in EUR. The target."),
        Column("price_tax_deductible", "bool", False, "Listing option. Excluded: tied to price."),
        Column("price_negotiable", "bool", False, "Listing option. Excluded: tied to price."),
        Column("price_net", "float64", True, "Excluded: derived from the target."),
        Column("price_vat_rate", "float64", True, "Excluded: derived from the target."),
        Column("vin", "str", True, "Vehicle identification number. PII."),
        Column("make", "str", False, "Manufacturer. Required by the scope check (FR-04)."),
        Column("model", "str", True, "Model name."),
        Column("model_version", "str", True, "Free-text trim; normalised in the extended set."),
        Column("german_hsn_tsn", "str", True, "German type key. Excluded: identifier."),
        Column("mileage_km_raw", "float64", True, "Odometer reading in km."),
        Column("mileage_km", "str", True, "Excluded: text duplicate of mileage_km_raw."),
        Column("registration_date", "str", True, "First registration, always the 1st of a month."),
        Column("production_year", "float64", True, "Excluded: 18.8 % filled."),
        Column("vehicle_type", "str", False, "Car or Transporter. Constant after scoping."),
        Column("body_type", "str", False, "Body style."),
        Column("nr_seats", "float64", True, "Number of seats."),
        Column("nr_doors", "float64", True, "Number of doors."),
        Column("body_color", "str", True, "Normalised exterior colour."),
        Column("paint_type", "str", True, "Metallic or Others."),
        Column("body_color_original", "str", True, "Excluded: free-text variant of body_color."),
        Column("upholstery", "str", True, "Upholstery material."),
        Column("upholstery_color", "str", True, "Normalised interior colour."),
        Column("power_kw", "float64", True, "Engine power in kW."),
        Column("power_hp", "float64", True, "Excluded: power_kw in another unit."),
        Column("transmission", "str", True, "Manual, Automatic or Semi-automatic."),
        Column("gears", "float64", True, "Number of gears."),
        Column("drive_train", "str", True, "Front Wheel Drive, Rear Wheel Drive or 4WD."),
        Column("cylinders", "float64", True, "Number of cylinders."),
        Column("cylinders_volume_cc", "float64", True, "Displacement in cc."),
        Column("weight_kg", "str", True, "Kerb weight as text, e.g. '1,945 kg'. Parsed later."),
        Column("has_particle_filter", "bool", False, "Particle filter fitted."),
        Column("fuel_category", "str", True, "Coarse fuel type."),
        Column("primary_fuel", "str", True, "Excluded: finer-grained fuel_category."),
        Column(
            "electric_range_km", "float64", True, "Electric range; applies to some drivetrains."
        ),
        Column("electric_range_city_km", "float64", True, "Excluded: 0.5 % filled."),
        Column("fuel_cons_comb_l100_km", "float64", True, "Excluded: sparse."),
        Column("fuel_cons_city_l100_km", "float64", True, "Excluded: empty in every row."),
        Column("fuel_cons_highway_l100_km", "float64", True, "Excluded: empty in every row."),
        Column("co2_emission_grper_km", "float64", True, "Excluded: sparse."),
        Column("fuel_cons_comb_l100_wltp_km", "float64", True, "Excluded: sparse."),
        Column("fuel_cons_electric_comb_l100_wltp_km", "float64", True, "Excluded: sparse."),
        Column("co2_emission_grper_wltp_km", "float64", True, "Excluded: sparse."),
        Column(
            "equipment_comfort", "str", False, "Python-repr list; '[]' when empty, never null."
        ),
        Column("equipment_entertainment", "str", False, "Python-repr list, never null."),
        Column("equipment_extra", "str", False, "Python-repr list, never null."),
        Column("equipment_safety", "str", False, "Python-repr list, never null."),
        Column("is_used", "bool", False, "Excluded: contradicts offer_type (EDN-23)."),
        Column("is_new", "bool", False, "Excluded: constant after scoping."),
        Column(
            "is_preregistered", "bool", False, "Defines the scope filter (EDN-24), not a feature."
        ),
        Column("had_accident", "bool", False, "Excluded: True in 3 rows of 118,382."),
        Column("has_full_service_history", "bool", False, "One-sided assertion (EDN-23)."),
        Column("non_smoking", "bool", False, "One-sided assertion (EDN-23)."),
        Column("nr_prev_owners", "float64", True, "Previous owners; 54.7 % filled."),
        Column("is_rental", "bool", False, "One-sided assertion (EDN-23)."),
        Column("envir_standard", "str", True, "Euro emission standard."),
        Column("original_market", "str", True, "Country the car was first sold in."),
        Column("offer_type", "str", False, "U, N or A. Defines the scope; constant afterwards."),
        Column("country_code", "str", True, "One of the 8 countries; ES is held out (EDN-03)."),
        Column("zip", "str", True, "Postal code. PII."),
        Column("city", "str", True, "City. PII."),
        Column("street", "str", True, "Street address. PII."),
        Column("latitude", "float64", True, "Exact location. PII."),
        Column("longitude", "float64", True, "Exact location. PII."),
        Column("seller_is_dealer", "bool", False, "Excluded: duplicates seller_type exactly."),
        Column("seller_type", "str", True, "Dealer or PrivateSeller."),
        Column("seller_company_name", "str", True, "PII; survives only as the hashed group key."),
        Column("has_warranty", "float64", True, "Excluded: empty in every row."),
        Column("warranty", "float64", True, "Excluded: empty in every row."),
    ),
)

# --------------------------------------------------------------------------
# Interim: what `preprocess` writes (#34).
#
# Row-filtered, PII-free, deduplicated and carrying the target, but not yet
# feature-engineered: `registration_date` is still a date rather than an age,
# the equipment lists are still repr strings and `weight_kg` is still text.
# Every make and every country is still present; the supported-make filter and
# the ES holdout belong to `split` (EDN-05 counts support after the holdout).
#
# The columns here are the raw ones that survive problem-spec section 4, so the
# exclusion table is enforced by this schema rather than by a drop list a stage
# has to remember. Nullability is structural only: `body_type` is nullable here
# even though it is filled in 100 % of the training listings, because that is a
# fill rate the Great Expectations suite asserts, not a guarantee of the frame.
# --------------------------------------------------------------------------
_GROUP_COLUMNS: tuple[Column, ...] = (
    Column(
        "seller_group_id",
        "str",
        False,
        "Hashed seller_company_name, location for private sellers. The split groups by it "
        "so no seller appears in two sets (EDN-14). Never reversible to the name.",
    ),
)

_KEPT_RAW_COLUMNS: tuple[str, ...] = (
    # Identity and market (basic set).
    "make",
    "model",
    "body_type",
    "registration_date",
    "mileage_km_raw",
    "nr_prev_owners",
    "power_kw",
    "fuel_category",
    "transmission",
    "drive_train",
    "gears",
    "cylinders_volume_cc",
    "nr_seats",
    "nr_doors",
    "country_code",
    "seller_type",
    # Extended set.
    "equipment_comfort",
    "equipment_entertainment",
    "equipment_extra",
    "equipment_safety",
    "has_full_service_history",
    "non_smoking",
    "is_rental",
    "body_color",
    "paint_type",
    "upholstery",
    "upholstery_color",
    "model_version",
    "weight_kg",
    "cylinders",
    "electric_range_km",
    "envir_standard",
    "original_market",
    # Named in neither feature set nor the exclusion table of problem-spec
    # section 4. Kept so that promoting it later is a params change rather than
    # a pipeline change; no feature set references it today.
    "has_particle_filter",
)


def _as_interim(column: Column) -> Column:
    """One raw column as the interim contract sees it.

    Two differences from the raw frame, both deliberate.

    `preprocess` parses the date, so the interim frame carries a real
    timestamp and EDN-22's "registered after the snapshot" rule is a date
    comparison rather than a string one.

    Nullability becomes *structural* rather than measured. In the raw contract
    a column is non-null because the published file happens to fill it; here it
    is non-null only when it cannot be otherwise - a `bool` column, because the
    dtype has no way to represent a missing value. `make` and `body_type` are
    filled in every row of the snapshot, but that is a fill rate, and a fill
    rate is an expectation for the Great Expectations suite to assert with a
    tolerance, not a promise this contract should make on the data's behalf. If
    it made it, one missing `body_type` in a future scrape would fail five
    stages with a message that reads like a bug in our code.
    """
    if column.name == "registration_date":
        return replace(column, dtype="datetime64[ns]", nullable=True)
    return replace(column, nullable=column.dtype != "bool")


INTERIM_SCHEMA = (
    Schema(
        name="interim",
        columns=tuple(_as_interim(RAW_SCHEMA.column(name)) for name in _KEPT_RAW_COLUMNS),
    )
    .extend(_TARGET_COLUMNS)
    .extend(_GROUP_COLUMNS, name="interim")
)

# --------------------------------------------------------------------------
# Processed: what `split` writes, once per set (#35).
#
# The split moves rows between files; it does not change the columns, so the
# contract is the interim one under a name the stage can assert against. If a
# split ever needs a column of its own, add it here and nowhere else.
# --------------------------------------------------------------------------
PROCESSED_SCHEMA = INTERIM_SCHEMA.renamed("processed")

# --------------------------------------------------------------------------
# Features: what `build_features` writes, once per set and split (#36).
#
# The columns depend on `features.sets` in params.yaml, so this is a function
# rather than a constant. `age_years` is the one column the stage derives from
# a raw column rather than passing through; the equipment multi-hot columns are
# data-dependent, so the stage adds them with `Schema.extend`.
#
# `feature_schema` is therefore only the catalogue's view of a feature set: the
# columns params.yaml names, before the stage reshapes them. What the stage
# actually wrote is in the artefact below, and that is what a stage reading a
# matrix must validate against.
# --------------------------------------------------------------------------
DERIVED_COLUMNS: tuple[Column, ...] = (
    Column(
        "age_years",
        "float64",
        True,
        "reference_date minus registration_date, in years. Null where the date is. "
        "Computed by one function the API reuses, so training and serving cannot drift.",
    ),
)

_FEATURE_CATALOGUE = INTERIM_SCHEMA.extend(DERIVED_COLUMNS, name="feature-catalogue")


def feature_schema(feature_columns: "list[str] | tuple[str, ...]", *, name: str) -> Schema:
    """The contract for one feature matrix: the named columns, plus the target.

    `feature_columns` comes straight from `features.sets.<name>` in
    params.yaml, so a typo there fails here with the offending name instead of
    producing a matrix that is quietly missing a column.
    """
    unknown = sorted(set(feature_columns) - set(_FEATURE_CATALOGUE.names))
    if unknown:
        raise SchemaError(
            f"feature set {name!r} names column(s) no stage produces: {', '.join(unknown)}"
        )
    selected = _FEATURE_CATALOGUE.select(feature_columns, name=name)
    return selected.extend(_TARGET_COLUMNS, name=name)


#: The artefact `build_features` writes beside each feature set's matrices.
#:
#: It carries the contract those matrices were written against, plus the
#: vocabulary the training rows decided (which equipment items got a column,
#: which levels each categorical has). Both are data-dependent, so neither can
#: live in this module - but a stage that reads a matrix must still code against
#: the contract rather than against `build_features`, and this file is how the
#: contract reaches it.
FEATURE_SPACE_FILE = "feature_space.json"

#: The sections the artefact must carry. Named here so a file missing one fails
#: on the read rather than at the first `KeyError` some consumer happens to hit.
_FEATURE_SPACE_SECTIONS: tuple[str, ...] = ("schema", "vocabulary")


class FeatureSpaceError(ValueError):
    """The `features` stage's artefact is absent, unreadable or incomplete."""


def read_feature_space(directory: Path) -> dict:
    """The `features` stage's artefact for `directory`, or a failure that names it.

    Every way this file can be wrong - absent, truncated, empty, not an object,
    missing a section - reaches the caller as one `FeatureSpaceError` naming the
    path and the stage that writes it. Without that a stale artefact surfaces as
    a bare `JSONDecodeError` or `KeyError: 'schema'` inside whichever stage
    happened to read it next, which says nothing about the file that has to be
    rebuilt.
    """
    path = directory / FEATURE_SPACE_FILE
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise FeatureSpaceError(
            _feature_space_advice(path, f"cannot be read ({error})")
        ) from error
    try:
        space = json.loads(text)
    except json.JSONDecodeError as error:
        raise FeatureSpaceError(
            _feature_space_advice(path, f"is not valid JSON ({error})")
        ) from error
    if not isinstance(space, dict):
        raise FeatureSpaceError(
            _feature_space_advice(path, f"holds a {type(space).__name__}, not an object")
        )
    absent = [section for section in _FEATURE_SPACE_SECTIONS if section not in space]
    if absent:
        raise FeatureSpaceError(
            _feature_space_advice(path, f"has no {' or '.join(absent)} section")
        )
    return space


def _feature_space_advice(path: Path, problem: str) -> str:
    return (
        f"{path} {problem}. It is the artefact of the `features` stage, which carries the "
        f"contract and the vocabulary of the matrices beside it, so rebuild it with "
        f"`dvc repro features` rather than editing it by hand."
    )


def load_feature_schema(directory: Path, *, name: str) -> Schema:
    """The contract of the feature matrices in `directory`, as their stage wrote it.

    What a stage that only reads matrices needs. A stage or a caller that has to
    *build* a frame wants `build_features.FeatureSpace`, which carries this
    contract and the vocabulary together.
    """
    path = directory / FEATURE_SPACE_FILE
    space = read_feature_space(directory)
    try:
        return Schema.from_dicts(space["schema"], name=name)
    except (SchemaError, TypeError) as error:
        raise FeatureSpaceError(
            _feature_space_advice(path, f"carries a contract this code cannot read ({error})")
        ) from error


#: Every schema a stage can validate against, by the name it is known by.
SCHEMAS: dict[str, Schema] = {
    schema.name: schema for schema in (RAW_SCHEMA, INTERIM_SCHEMA, PROCESSED_SCHEMA)
}
