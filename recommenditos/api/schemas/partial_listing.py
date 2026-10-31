"""A listing described as far as the client knows it, down to the make alone (FR-02).

The request body of UC2's `/v1/price-range`, and of `/v1/comparables` (#138),
which will take the same input. It has every field of `ValuationRequest`, with the
same types and therefore the same checks for every field that is given (FR-03,
FR-04); only `make` is required. A field left out reaches the model as absent,
which is how the interval models were trained and calibrated (EDN-85).
"""

from pydantic import ConfigDict, Field

from recommenditos.api.schemas.common import (
    BodyType,
    ComfortEquipment,
    CountryCode,
    Cylinders,
    DisplacementCc,
    Doors,
    DriveTrain,
    ElectricRangeKm,
    EntertainmentEquipment,
    ExtraEquipment,
    Flag,
    FuelCategory,
    Gears,
    ListingFields,
    Make,
    MileageKm,
    ModelName,
    ModelVersion,
    PowerKw,
    PreviousOwners,
    RegistrationDate,
    SafetyEquipment,
    Seats,
    SellerType,
    Text,
    Transmission,
    WeightKg,
)


class PartialListing(ListingFields):
    """One car, described by any subset of the listing fields that names its make (FR-02).

    The fields and their order are `ValuationRequest`'s; a test holds the two
    together, so a field added to one is added to the other.
    """

    make: Make
    model: ModelName | None = None
    registration_date: RegistrationDate | None = None
    mileage_km_raw: MileageKm | None = None
    power_kw: PowerKw | None = None
    fuel_category: FuelCategory | None = None
    transmission: Transmission | None = None
    country_code: CountryCode | None = None
    body_type: BodyType | None = None
    seller_type: SellerType | None = None

    nr_prev_owners: PreviousOwners | None = None
    drive_train: DriveTrain | None = None
    gears: Gears | None = None
    cylinders_volume_cc: DisplacementCc | None = None
    nr_seats: Seats | None = None
    nr_doors: Doors | None = None
    cylinders: Cylinders | None = None
    weight_kg: WeightKg | None = None
    electric_range_km: ElectricRangeKm | None = None
    equipment_comfort: ComfortEquipment = Field(default_factory=list)
    equipment_entertainment: EntertainmentEquipment = Field(default_factory=list)
    equipment_extra: ExtraEquipment = Field(default_factory=list)
    equipment_safety: SafetyEquipment = Field(default_factory=list)
    has_full_service_history: Flag = False
    non_smoking: Flag = False
    is_rental: Flag = False
    body_color: Text | None = None
    paint_type: Text | None = None
    upholstery: Text | None = None
    upholstery_color: Text | None = None
    model_version: ModelVersion | None = None
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
                },
                {"make": "Volkswagen"},
            ]
        },
    )
