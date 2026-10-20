"""The request and response bodies of the API, and the field types they share.

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
"""

from datetime import UTC, datetime
import re
from typing import Annotated, Any, ClassVar, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------
# Field types
# --------------------------------------------------------------------------

_YEAR_MONTH = re.compile(r"^(\d{4})-(0[1-9]|1[0-2])$")

#: The earliest registration month FR-03 accepts.
EARLIEST_REGISTRATION = "1900-01"


def _not_after_this_month(value: str) -> str:
    """FR-03: a registration month lies between 1900-01 and the request month."""
    match = _YEAR_MONTH.match(value)
    if match is None:  # the pattern has already refused it; kept for the type checker
        raise ValueError("must be a month in the form YYYY-MM")
    now = datetime.now(UTC)
    if value < EARLIEST_REGISTRATION or value > f"{now.year:04d}-{now.month:02d}":
        raise ValueError(
            f"must lie between {EARLIEST_REGISTRATION} and the current month, "
            f"{now.year:04d}-{now.month:02d}"
        )
    return value


Make = Annotated[
    str,
    Field(
        description="Manufacturer. Must be one of the makes the served model supports (FR-04); "
        "a 422 lists them.",
        examples=["Volkswagen"],
    ),
]
ModelName = Annotated[
    str,
    Field(
        min_length=1,
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
    AfterValidator(_not_after_this_month),
]
MileageKm = Annotated[
    float,
    Field(ge=0, le=1_000_000, description="Mileage in km, 0 to 1,000,000.", examples=[45000]),
]
PowerKw = Annotated[
    float,
    Field(ge=1, le=1_200, description="Engine power in kW, 1 to 1,200.", examples=[110]),
]
FuelCategory = Annotated[
    str, Field(description="Fuel category, one of the served model's.", examples=["Gasoline"])
]
Transmission = Annotated[
    str, Field(description="Transmission, one of the served model's.", examples=["Manual"])
]
CountryCode = Annotated[
    str,
    Field(
        description="ISO country code of the listing, one of the 8 countries of the data. "
        "`ES` is accepted although the model was not trained on it, and reported in `warnings` "
        "(FR-05).",
        examples=["DE"],
    ),
]
BodyType = Annotated[
    str, Field(description="Body type, one of the served model's.", examples=["Compact"])
]
SellerType = Annotated[
    str, Field(description="Seller type, one of the served model's.", examples=["Dealer"])
]
DriveTrain = Annotated[
    str,
    Field(description="Drive train, one of the served model's.", examples=["Front Wheel Drive"]),
]
Count = Annotated[int, Field(description="A count, such as doors or previous owners.")]
EquipmentList = Annotated[
    list[str],
    Field(
        description="Equipment items as the listing names them. An omitted list is an empty "
        "list, never null (FR-01).",
        examples=[["Air conditioning", "Navigation system"]],
    ),
]
Flag = Annotated[
    bool,
    Field(description="A ticked box of the listing. Omitted means not ticked (EDN-23)."),
]
Text = Annotated[str, Field(min_length=1)]


class ListingFields(BaseModel):
    """The base of every request body that describes one listing.

    It refuses fields it does not know, so that a misspelt optional field is a
    422 instead of a silently ignored input. The error handler runs the served
    model's checks on any body of this type, so a request with a type error and
    an unsupported make gets both in one response (FR-03).
    """

    model_config = ConfigDict(extra="forbid")

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


class ValuationRequest(ListingFields):
    """One listing to value (UC1): the ten fields FR-01 requires, the rest optional.

    The optional fields are the inputs of the basic and the extended feature set.
    The served model reads the ones its feature set holds and ignores the rest,
    so a model trained on another set needs no change here (FR-12).
    """

    make: Make
    model: ModelName
    registration_date: RegistrationDate
    mileage_km_raw: MileageKm
    power_kw: PowerKw
    fuel_category: FuelCategory
    transmission: Transmission
    country_code: CountryCode
    body_type: BodyType
    seller_type: SellerType

    nr_prev_owners: Count | None = None
    drive_train: DriveTrain | None = None
    gears: Count | None = None
    cylinders_volume_cc: Annotated[
        float | None, Field(description="Engine displacement in cc.")
    ] = None
    nr_seats: Count | None = None
    nr_doors: Count | None = None
    cylinders: Count | None = None
    weight_kg: Annotated[int | None, Field(description="Kerb weight in kg.")] = None
    electric_range_km: Annotated[float | None, Field(description="Electric range in km.")] = None
    equipment_comfort: EquipmentList = []
    equipment_entertainment: EquipmentList = []
    equipment_extra: EquipmentList = []
    equipment_safety: EquipmentList = []
    has_full_service_history: Flag = False
    non_smoking: Flag = False
    is_rental: Flag = False
    body_color: Text | None = None
    paint_type: Text | None = None
    upholstery: Text | None = None
    upholstery_color: Text | None = None
    model_version: Annotated[
        str | None, Field(description="The listing's free-text version line.")
    ] = None
    envir_standard: Text | None = None
    original_market: Text | None = None

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "make": "Volkswagen",
                    "model": "Golf",
                    "registration_date": "2019-06",
                    "mileage_km_raw": 45000,
                    "power_kw": 110,
                    "fuel_category": "Gasoline",
                    "transmission": "Manual",
                    "country_code": "DE",
                    "body_type": "Compact",
                    "seller_type": "Dealer",
                    "nr_prev_owners": 1,
                }
            ]
        },
    )


# --------------------------------------------------------------------------
# Responses
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


_MODEL_EXAMPLE: dict[str, Any] = {
    "variant": "lgbm-basic",
    "version": "2b53bd3487f97475ebf0e6bd268ba0f0",
    "trained_at": "2026-10-13T12:05:02+00:00",
}


class Valuation(BaseModel):
    """The estimated asking price of one listing (UC1, FR-06)."""

    price_eur: float = Field(description="Point estimate of the asking price in EUR.")
    model: ModelInfo
    request_id: str = Field(
        description="Identifies this request; also sent as the `X-Request-ID` header."
    )
    warnings: list[FieldWarning] = Field(
        description="Values the model has not seen and treated as unknown (FR-05). Empty when "
        "there are none."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "price_eur": 21890.0,
                    "model": _MODEL_EXAMPLE,
                    "request_id": "0b9f3c1e6c0a4b5e9d1f2a3b4c5d6e7f",
                    "warnings": [],
                }
            ]
        }
    )


class Health(BaseModel):
    """Whether the service is up, and which model it serves (FR-11)."""

    status: Literal["ok"] = Field(description="`ok` whenever the service answers.")
    model: ModelInfo

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "ok", "model": _MODEL_EXAMPLE}]}
    )
