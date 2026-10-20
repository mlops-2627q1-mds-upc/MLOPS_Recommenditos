"""The bodies of `POST /v1/predict` (UC1)."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from recommenditos.api.schemas.common import (
    MODEL_EXAMPLE,
    BodyType,
    CountryCode,
    Cylinders,
    DisplacementCc,
    Doors,
    DriveTrain,
    ElectricRangeKm,
    EquipmentList,
    FieldWarning,
    Flag,
    FuelCategory,
    Gears,
    ListingFields,
    Make,
    MileageKm,
    ModelInfo,
    ModelName,
    PowerKw,
    PreviousOwners,
    RegistrationDate,
    Seats,
    SellerType,
    Text,
    Transmission,
    WeightKg,
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

    nr_prev_owners: PreviousOwners | None = None
    drive_train: DriveTrain | None = None
    gears: Gears | None = None
    cylinders_volume_cc: DisplacementCc | None = None
    nr_seats: Seats | None = None
    nr_doors: Doors | None = None
    cylinders: Cylinders | None = None
    weight_kg: WeightKg | None = None
    electric_range_km: ElectricRangeKm | None = None
    equipment_comfort: EquipmentList = Field(default_factory=list)
    equipment_entertainment: EquipmentList = Field(default_factory=list)
    equipment_extra: EquipmentList = Field(default_factory=list)
    equipment_safety: EquipmentList = Field(default_factory=list)
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


class FeatureEffect(BaseModel):
    """How much one request field moves the estimate (FR-08)."""

    field: str = Field(
        description="The request field, as the request body names it. Every model feature is "
        "attributed to the field it comes from: `registration_date` covers the car's age, and an "
        "equipment list covers all of its items together."
    )
    effect_pct: float = Field(
        description="The field's multiplicative effect on the estimate in percent: `12.5` "
        "raises it by 12.5 %, `-30` lowers it by 30 %. It is `exp(φ) - 1`, where φ is the "
        "field's SHAP value on the model's log price scale."
    )


class Explanation(BaseModel):
    """Why the estimate is what it is: where it starts and what moves it (FR-08).

    The effects multiply: `base_price_eur` times `1 + effect_pct / 100` over the
    entries of `top_features` and `other_features_effect_pct` gives the estimate.
    """

    base_price_eur: float = Field(
        description="Where every estimate starts before any field moves it, in EUR: the "
        "model's average log price over its training listings, as a price."
    )
    top_features: list[FeatureEffect] = Field(
        description="The five fields with the largest effect, largest first, whether it raises "
        "or lowers the estimate. Equal effects keep the order of the request body; a model that "
        "reads fewer than five fields lists all of them."
    )
    other_features_effect_pct: float = Field(
        description="The combined effect of every field the model reads that `top_features` "
        "does not list, in percent; 0 when there is none."
    )


class Valuation(BaseModel):
    """The estimated asking price of one listing (UC1, FR-06)."""

    price_eur: float = Field(description="Point estimate of the asking price in EUR.")
    model: ModelInfo
    request_id: str = Field(
        description="Identifies this request; also sent as the `X-Request-ID` header."
    )
    explanation: Explanation
    warnings: list[FieldWarning] = Field(
        description="Values the model has not seen and treated as unknown (FR-05), and an "
        "estimate bounded to the training price range. Empty when there are none."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "price_eur": 14639.2,
                    "model": MODEL_EXAMPLE,
                    "request_id": "0b9f3c1e-6c0a-4b5e-9d1f-2a3b4c5d6e7f",
                    "explanation": {
                        "base_price_eur": 38389.95741359577,
                        "top_features": [
                            {"field": "power_kw", "effect_pct": -35.26084112077258},
                            {"field": "model", "effect_pct": -23.360818674067794},
                            {"field": "make", "effect_pct": -17.95976142027397},
                            {"field": "mileage_km_raw", "effect_pct": 14.580681443498216},
                            {"field": "registration_date", "effect_pct": -6.263490267105494},
                        ],
                        "other_features_effect_pct": -12.776273101340156,
                    },
                    "warnings": [],
                }
            ]
        }
    )
