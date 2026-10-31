"""What every request and response body shares: field types, warnings, errors.

A field is declared once, as an `Annotated` type that carries its JSON type, its
bounds, its description and an example. A request body then only says which
fields it has and which of them are required: `/predict` requires the ten fields
of FR-01, and an endpoint that requires less, such as UC2's `/price-range`
(FR-02), reuses the same types with `| None = None`. Every listing request body
derives from `ListingFields`, which is how the error handler knows to run the
model's own checks on it (`recommenditos/api/serving.py`).

What the static schema cannot know is which values the loaded model accepts:
the supported makes and the levels of the categoricals come from the bundle at
startup (FR-03, FR-04, FR-12), so a field like `make` is a plain string here and
the served model checks it. `create_app` writes those values into the OpenAPI
document as enums, so `/docs` shows them anyway.

Every string and every equipment list has a maximum length, so that a body of
ASCII text without escapes within every bound stays under the 10 KB limit of
NFR-09 and the OpenAPI document states what that limit enforces. A bound counts
characters and the limit bytes, so multi-byte or escaped text within the bounds
can still make a body larger, which is a 413; no real listing comes near it (the
largest is about 3.5 KB). `tests/api/test_request_bounds.py` holds the largest
such ASCII body under the limit.

`Problem` and `FieldError` are the error shape every endpoint shares (FR-16,
EDN-84); `recommenditos/api/errors.py` builds and documents them.
"""

from datetime import UTC, datetime
import re
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

# --------------------------------------------------------------------------
# Field types
# --------------------------------------------------------------------------

#: A month from 1900-01 to 2099-12, so the OpenAPI document states the lower bound
#: of FR-03 itself; the upper bound moves with the calendar and is checked below.
_YEAR_MONTH = re.compile(r"^((?:19|20)\d{2})-(0[1-9]|1[0-2])$")

#: Any year: a month outside the range then gets the range in its error rather
#: than the pattern.
_ANY_YEAR_MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

#: The earliest registration month FR-03 accepts.
EARLIEST_REGISTRATION = "1900-01"

# The length bounds come from the 118,382 raw listings (`data/raw/listings.parquet`),
# whose every value they accept. A string outside the equipment lists gets at
# least twice its longest value: 23 characters for a category (the longest is
# `body_type`; no colour is longer than 6), 22 for `model`, 121 for
# `model_version`. The equipment lists get at least 1.2 times
# theirs, rounded up to a multiple of 5, because their items dominate a body and
# twice would not fit in 10 KB: 37 characters for an item, and 42, 16, 37 and 36
# items for comfort, entertainment, extra and safety.

#: Characters of a category, a colour, a make or a model name.
MAX_NAME_LENGTH = 50
#: Characters of `model_version`, the listing's free-text version line.
MAX_MODEL_VERSION_LENGTH = 250
#: Characters of one equipment item.
MAX_EQUIPMENT_ITEM_LENGTH = 45
#: Items per equipment list.
MAX_EQUIPMENT_ITEMS = {
    "equipment_comfort": 55,
    "equipment_entertainment": 20,
    "equipment_extra": 45,
    "equipment_safety": 45,
}


def _within_registration_range(value: Any) -> Any:
    """FR-03: a registration month lies between 1900-01 and the request month.

    It runs before the pattern, so `"1899-12"` and next month are both told the
    range; a value that is no month at all is left to the pattern and the type.
    """
    if not isinstance(value, str) or not _ANY_YEAR_MONTH.match(value):
        return value
    now = datetime.now(UTC)
    this_month = f"{now.year:04d}-{now.month:02d}"
    if not EARLIEST_REGISTRATION <= value <= this_month:
        raise ValueError(
            f"must lie between {EARLIEST_REGISTRATION} and the current month, {this_month}"
        )
    return value


Make = Annotated[
    str,
    Field(
        max_length=MAX_NAME_LENGTH,
        description="Manufacturer. Must be one of the makes the served model supports (FR-04); "
        "a 422 lists them.",
        examples=["Volkswagen"],
    ),
]
ModelName = Annotated[
    str,
    Field(
        min_length=1,
        max_length=MAX_NAME_LENGTH,
        description="Model name. A model the training data does not hold is accepted, passed "
        "to the model as unknown and reported in `warnings` (FR-05).",
        examples=["Golf"],
    ),
]
RegistrationDate = Annotated[
    str,
    Field(
        pattern=_YEAR_MONTH.pattern,
        description="First registration as year and month, `YYYY-MM`, between 1900-01 and the "
        "current month. The car's age is computed from it at the request date.",
        examples=["2019-06"],
    ),
    BeforeValidator(_within_registration_range),
]
MileageKm = Annotated[
    float,
    Field(
        ge=0,
        le=1_000_000,
        allow_inf_nan=False,
        description="Mileage in km, 0 to 1,000,000.",
        examples=[45000],
    ),
]
PowerKw = Annotated[
    float,
    Field(
        ge=1,
        le=1_200,
        allow_inf_nan=False,
        description="Engine power in kW, 1 to 1,200.",
        examples=[110],
    ),
]
FuelCategory = Annotated[
    str,
    Field(
        max_length=MAX_NAME_LENGTH,
        description="Fuel category, one of the served model's.",
        examples=["Gasoline"],
    ),
]
Transmission = Annotated[
    str,
    Field(
        max_length=MAX_NAME_LENGTH,
        description="Transmission, one of the served model's.",
        examples=["Manual"],
    ),
]
CountryCode = Annotated[
    str,
    Field(
        max_length=MAX_NAME_LENGTH,
        description="ISO country code of the listing, one of the 8 countries of the data. "
        "`ES` is accepted although the model was not trained on it, and reported in `warnings` "
        "(FR-05).",
        examples=["DE"],
    ),
]
BodyType = Annotated[
    str,
    Field(
        max_length=MAX_NAME_LENGTH,
        description="Body type, one of the served model's.",
        examples=["Compact"],
    ),
]
SellerType = Annotated[
    str,
    Field(
        max_length=MAX_NAME_LENGTH,
        description="Seller type, one of the served model's.",
        examples=["Dealer"],
    ),
]
DriveTrain = Annotated[
    str,
    Field(
        max_length=MAX_NAME_LENGTH,
        description="Drive train, one of the served model's.",
        examples=["Front Wheel Drive"],
    ),
]


def _integral_float_as_int(value: Any) -> Any:
    """`4.0` as `4`: JSON Schema counts it as an integer, and strict mode would not."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


#: Follows the `Field` of an integer type, which keeps the bounds in the OpenAPI
#: document; before it, pydantic writes them as unknown keywords.
_WHOLE_NUMBER = BeforeValidator(_integral_float_as_int)

# The optional numbers are bounded by what a passenger car can have, not by the
# training data, which holds data-entry errors such as 99 previous owners or a
# 16,000 kg car. A value outside reaches no model: it is a 422 (FR-03).
PreviousOwners = Annotated[
    int, Field(ge=0, le=20, description="Previous owners, 0 to 20."), _WHOLE_NUMBER
]
Gears = Annotated[
    int, Field(ge=0, le=12, description="Gears, 0 to 12 (0 for a single-speed EV)."), _WHOLE_NUMBER
]
Seats = Annotated[int, Field(ge=1, le=9, description="Seats, 1 to 9."), _WHOLE_NUMBER]
Doors = Annotated[int, Field(ge=1, le=7, description="Doors, 1 to 7."), _WHOLE_NUMBER]
Cylinders = Annotated[
    int, Field(ge=0, le=16, description="Cylinders, 0 to 16 (0 for an EV)."), _WHOLE_NUMBER
]
DisplacementCc = Annotated[
    float,
    Field(
        ge=0, le=10_000, allow_inf_nan=False, description="Engine displacement in cc, 0 to 10,000."
    ),
]
WeightKg = Annotated[
    int, Field(ge=1, le=10_000, description="Kerb weight in kg, 1 to 10,000."), _WHOLE_NUMBER
]
ElectricRangeKm = Annotated[
    float,
    Field(ge=0, le=2_000, allow_inf_nan=False, description="Electric range in km, 0 to 2,000."),
]
EquipmentItem = Annotated[str, Field(max_length=MAX_EQUIPMENT_ITEM_LENGTH)]
_EQUIPMENT_DESCRIPTION = (
    "Equipment items as the listing names them. An omitted list is an empty list, never null "
    "(FR-01)."
)
ComfortEquipment = Annotated[
    list[EquipmentItem],
    Field(
        max_length=MAX_EQUIPMENT_ITEMS["equipment_comfort"],
        examples=[["Air conditioning", "Navigation system"]],
        description=_EQUIPMENT_DESCRIPTION,
    ),
]
EntertainmentEquipment = Annotated[
    list[EquipmentItem],
    Field(
        max_length=MAX_EQUIPMENT_ITEMS["equipment_entertainment"],
        examples=[["Bluetooth", "Radio"]],
        description=_EQUIPMENT_DESCRIPTION,
    ),
]
ExtraEquipment = Annotated[
    list[EquipmentItem],
    Field(
        max_length=MAX_EQUIPMENT_ITEMS["equipment_extra"],
        examples=[["Alloy wheels", "Sport seats"]],
        description=_EQUIPMENT_DESCRIPTION,
    ),
]
SafetyEquipment = Annotated[
    list[EquipmentItem],
    Field(
        max_length=MAX_EQUIPMENT_ITEMS["equipment_safety"],
        examples=[["ABS", "Electronic stability control"]],
        description=_EQUIPMENT_DESCRIPTION,
    ),
]
Flag = Annotated[
    bool,
    Field(description="A ticked box of the listing. Omitted means not ticked (EDN-23)."),
]
Text = Annotated[str, Field(min_length=1, max_length=MAX_NAME_LENGTH)]
ModelVersion = Annotated[
    str,
    Field(
        max_length=MAX_MODEL_VERSION_LENGTH, description="The listing's free-text version line."
    ),
]


class ListingFields(BaseModel):
    """The base of every request body that describes one listing.

    It refuses fields it does not know, so that a misspelt optional field is a
    422 instead of a silently ignored input. It is strict, so a value has the
    JSON type the OpenAPI document gives it: `0` is not a boolean and `"45000"`
    is not a number (#144). The error handler runs the served
    model's checks on any body of this type, so a request with a type error and
    an unsupported make gets both in one response (FR-03).
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    #: The request fields whose values must be levels of the served model (FR-03,
    #: FR-04). `make` is checked against the supported makes, `country_code`
    #: against the 8 countries, the others against the bundle's levels.
    checked_fields: ClassVar[tuple[str, ...]] = (
        "make",
        "fuel_category",
        "transmission",
        "body_type",
        "drive_train",
        "seller_type",
        "country_code",
    )


# --------------------------------------------------------------------------
# Shared parts of responses
# --------------------------------------------------------------------------


class FieldWarning(BaseModel):
    """A value the request carried that the served model has not seen (FR-05)."""

    pointer: str = Field(description="JSON Pointer to the field in the request body.")
    detail: str = Field(description="What the model did with the value.")


class ModelInfo(BaseModel):
    """Which model answered."""

    variant: str = Field(description="The model variant, e.g. `lgbm-basic`.")
    version: str = Field(
        description="The DVC hash of the bundle directory, the same value `dvc.lock` records for "
        "`models/<variant>` (FR-12)."
    )
    trained_at: str | None = Field(description="When the bundle was trained, ISO 8601.")


MODEL_EXAMPLE: dict[str, Any] = {
    "variant": "lgbm-basic",
    "version": "2b53bd3487f97475ebf0e6bd268ba0f0",
    "trained_at": "2026-10-13T12:05:02+00:00",
}


# --------------------------------------------------------------------------
# The error shape (RFC 9457)
# --------------------------------------------------------------------------


class FieldError(BaseModel):
    """One invalid field of the request body."""

    pointer: str = Field(
        description="JSON Pointer (RFC 6901) to the field in the request body, as a URI "
        "fragment: `#/power_kw`. `#` is the body as a whole."
    )
    detail: str = Field(description="What is wrong with the value.")
    allowed: list[str] | None = Field(
        default=None,
        description="The values the served model accepts, where the field takes one of a set.",
    )


class Problem(BaseModel):
    """An error, as RFC 9457 Problem Details."""

    type: str = Field(description="URI that identifies the kind of problem.")
    title: str = Field(description="Short summary of the kind of problem.")
    status: int = Field(description="The HTTP status code.")
    detail: str = Field(description="What went wrong with this request.")
    instance: str = Field(description="This occurrence: `urn:uuid:` and the request ID.")
    errors: list[FieldError] | None = Field(
        default=None, description="One entry per invalid field, for a 422."
    )
