"""The bodies of `POST /v1/comparables` (UC2, FR-09), and the index's description.

The request is `PartialListing`, the body of `/v1/price-range`, plus `k`, so a
client sends the same car to both endpoints.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from recommenditos.api.comparables import DEFAULT_K, MAX_K, ComparablesIndex
from recommenditos.api.schemas.common import MODEL_EXAMPLE, FieldWarning, ModelInfo
from recommenditos.api.schemas.partial_listing import PartialListing

#: The version and size of the index in `ComparablesIndexInfo`'s example.
INDEX_EXAMPLE: dict[str, Any] = {"version": "9993b933797d905b2b6271b03bac50ee", "listings": 70172}


class ComparablesRequest(PartialListing):
    """A car described as for `/v1/price-range`, and how many listings to return."""

    k: int = Field(
        default=DEFAULT_K,
        ge=1,
        le=MAX_K,
        description=f"How many listings to return at most, 1 to {MAX_K}; {DEFAULT_K} if left out.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "make": "Volkswagen",
                    "model": "Golf",
                    "registration_date": "2019-06",
                    "mileage_km_raw": 45000,
                    "k": 3,
                },
                {"make": "Volkswagen", "model": "Golf"},
            ]
        },
    )


class ComparablesIndexInfo(BaseModel):
    """Which comparables index the service searches (FR-09, FR-12)."""

    version: str = Field(
        description="The md5 of the index file, which `dvc.lock` records for "
        "`models/comparables.parquet`."
    )
    listings: int = Field(description="How many listings the index holds.")

    model_config = ConfigDict(json_schema_extra={"examples": [INDEX_EXAMPLE]})

    @classmethod
    def of(cls, index: ComparablesIndex) -> "ComparablesIndexInfo":
        return cls(version=index.version, listings=len(index))


class Comparable(BaseModel):
    """One real listing of the training or validation data, without PII (NFR-08)."""

    make: str
    model: str | None
    model_version: str | None = Field(description="The listing's free-text version line.")
    registration_date: str | None = Field(description="First registration, `YYYY-MM`.")
    mileage_km_raw: float | None = Field(description="Mileage in km.")
    power_kw: float | None = Field(description="Power in kW.")
    fuel_category: str | None
    transmission: str | None
    body_type: str | None
    seller_type: str | None
    country_code: str | None
    price_eur: float = Field(description="The asking price of the listing in EUR.")


class Comparables(BaseModel):
    """Listings comparable to a car, closest first (UC2, FR-09)."""

    listings: list[Comparable] = Field(
        description="At most `k` listings with the request's `make` and `model`, an age at "
        "most 2 years and a mileage at most 25 % from the request's, sorted by mileage "
        "distance, then age distance. A field the request leaves out sets no filter. Empty "
        "when no listing lies in every window: no filter is relaxed."
    )
    index: ComparablesIndexInfo
    model: ModelInfo
    request_id: str = Field(
        description="Identifies this request; also sent as the `X-Request-ID` header."
    )
    warnings: list[FieldWarning] = Field(
        description="A `model` of which the index holds no listing for the make, so that the "
        "list is empty whatever the other fields say. Empty when there is none."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "listings": [
                        {
                            "make": "Volkswagen",
                            "model": "Golf",
                            "model_version": "1.5 TSI ACT Comfortline",
                            "registration_date": "2019-03",
                            "mileage_km_raw": 46100,
                            "power_kw": 110,
                            "fuel_category": "Gasoline",
                            "transmission": "Manual",
                            "body_type": "Compact",
                            "seller_type": "Dealer",
                            "country_code": "DE",
                            "price_eur": 17490.0,
                        }
                    ],
                    "index": INDEX_EXAMPLE,
                    "model": MODEL_EXAMPLE,
                    "request_id": "0b9f3c1e-6c0a-4b5e-9d1f-2a3b4c5d6e7f",
                    "warnings": [],
                }
            ]
        }
    )
