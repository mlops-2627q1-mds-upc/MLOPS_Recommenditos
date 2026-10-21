"""The response body of `POST /v1/price-range` (UC2); its request is `PartialListing`."""

from pydantic import BaseModel, ConfigDict, Field

from recommenditos.api.schemas.common import MODEL_EXAMPLE, FieldWarning, ModelInfo


class PriceRange(BaseModel):
    """The typical price of a car and the range its price lies in (UC2, FR-07)."""

    price_eur: float = Field(
        description="The typical price in EUR: the point model's, as `/v1/predict` would give "
        "it, when the request carries every field `/v1/predict` requires, otherwise the "
        "median of the price range (EDN-85)."
    )
    lower_eur: float = Field(description="Lower bound of the range in EUR, at most `price_eur`.")
    upper_eur: float = Field(description="Upper bound of the range in EUR, at least `price_eur`.")
    confidence_level: float = Field(
        description="The nominal share of prices the range contains, 0.9 for a 90 % range. It "
        "is calibrated over requests with fields left out as well as full ones (SC-05)."
    )
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
                    "price_eur": 19870.0,
                    "lower_eur": 14420.5,
                    "upper_eur": 28305.25,
                    "confidence_level": 0.9,
                    "model": MODEL_EXAMPLE,
                    "request_id": "0b9f3c1e-6c0a-4b5e-9d1f-2a3b4c5d6e7f",
                    "warnings": [],
                }
            ]
        }
    )
