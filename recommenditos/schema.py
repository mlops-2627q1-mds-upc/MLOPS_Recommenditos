"""The data contract every pipeline stage reads and writes against.

This module is the boundary between the stages. A stage does not code against
the stage before it, it codes against the schema below, which is why the stages
can be built in parallel.

It describes *structure* only: which columns exist, their dtype and whether they
may be null. Value rules - price ranges, the supported make list, fill rates -
are data quality, and they live in the Great Expectations suites (#25). The two
answer different questions: a structural break is a bug in our code, a value
break is a change in the data.

Two entry points matter:

- `Schema.conform(frame)` selects the contract's columns in its order, casts
  them to the declared dtypes and validates the result. Every stage calls it
  before writing. Because it selects, a column the contract does not name
  cannot reach an artefact - which is how the PII columns of NFR-08 are kept
  out structurally rather than by remembering to drop them.
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


class SchemaError(ValueError):
    """A DataFrame does not match the schema it was checked against."""


@dataclass(frozen=True)
class Column:
    """One column of a contract.

    `dtype` is the pandas dtype as `str(series.dtype)` renders it, so it is
    compared, and cast to, without any translation table. On pandas 3 a text
    column is `str`, never `object`: `object` passes in the stage that writes
    it and fails in the stage that reads it back from Parquet.
    """

    name: str
    dtype: str
    nullable: bool
    description: str


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
        return [asdict(column) for column in self.columns]

    @classmethod
    def from_dicts(cls, rows: "list[dict]", *, name: str) -> "Schema":
        """A contract from what `to_dicts` wrote.

        `Column(**row)` rather than a field-by-field read: a row that has lost a
        field, or gained one, fails here naming it instead of producing a
        contract that silently checks less than it should.
        """
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
        as a number, and `train` may want the categoricals as `category` for
        LightGBM's native handling (EDN-02). Both are that stage's business, so
        they are expressed here rather than by editing the constants below.
        """
        changed = tuple(
            replace(each, dtype=dtype) if each.name == column else each for each in self.columns
        )
        if changed == self.columns:
            raise SchemaError(f"{self.name!r} has no column {column!r}")
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
        """Select, order and cast `frame` to this contract, then validate it."""
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
            conformed[column.name] = _cast(conformed[column.name], column.dtype)
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
            actual = str(frame[column.name].dtype)
            if actual != column.dtype:
                problems.append(
                    f"column {column.name!r}: expected dtype {column.dtype}, got {actual}"
                )
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


def _cast(series: pd.Series, dtype: str) -> pd.Series:
    if str(series.dtype) == dtype:
        return series
    if dtype.startswith("datetime64"):
        # Parse first: pandas 3 parses to microsecond resolution, so a bare
        # astype from text would not reach the declared unit.
        return pd.to_datetime(series).astype(dtype)
    return series.astype(dtype)


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
_TARGET_COLUMNS: tuple[Column, ...] = (
    Column("price", "float64", False, "Asking price in EUR, inside the params price range."),
    Column("log_price", "float64", False, "log(price). The training target (problem-spec 3)."),
)

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


def load_feature_schema(directory: Path, *, name: str) -> Schema:
    """The contract of the feature matrices in `directory`, as their stage wrote it."""
    space = json.loads((directory / FEATURE_SPACE_FILE).read_text(encoding="utf-8"))
    return Schema.from_dicts(space["schema"], name=name)


#: Every schema a stage can validate against, by the name it is known by.
SCHEMAS: dict[str, Schema] = {
    schema.name: schema for schema in (RAW_SCHEMA, INTERIM_SCHEMA, PROCESSED_SCHEMA)
}
