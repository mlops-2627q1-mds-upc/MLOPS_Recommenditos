"""The bodies of `POST /v1/feedback` (FR-10)."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

_REQUEST_ID = "0b9f3c1e-6c0a-4b5e-9d1f-2a3b4c5d6e7f"


class FeedbackRequest(BaseModel):
    """The price a listing was observed at, for an answer of the API."""

    request_id: UUID = Field(
        description="The `request_id` of the answer the price belongs to, as the API returned it."
    )
    observed_price_eur: float = Field(
        gt=0, allow_inf_nan=False, description="The observed asking price in EUR."
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"request_id": _REQUEST_ID, "observed_price_eur": 20500.0}]
        },
    )


class FeedbackReceipt(BaseModel):
    """The price the service holds for an answer."""

    request_id: UUID = Field(description="The answer the price belongs to.")
    observed_price_eur: float = Field(description="The stored price in EUR.")
    received_at: datetime = Field(description="When the service stored it, in UTC.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "request_id": _REQUEST_ID,
                    "observed_price_eur": 20500.0,
                    "received_at": "2026-10-20T09:15:02.481000Z",
                }
            ]
        }
    )
