"""The body of `GET /health`."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from recommenditos.api.schemas.common import MODEL_EXAMPLE, ModelInfo
from recommenditos.api.schemas.comparables import INDEX_EXAMPLE, ComparablesIndexInfo


class Health(BaseModel):
    """Whether the service is up, and which model and comparables index it serves (FR-11)."""

    status: Literal["ok"] = Field(description="`ok` whenever the service answers.")
    model: ModelInfo
    price_range: bool = Field(
        description="Whether the served bundle carries the interval models `/v1/price-range` "
        "needs (EDN-85). Without them that endpoint answers 503 while the rest of the service "
        "works."
    )
    comparables: ComparablesIndexInfo | None = Field(
        description="The comparables index `/v1/comparables` searches (FR-09), or null when the "
        "service was started without one."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "model": MODEL_EXAMPLE,
                    "price_range": True,
                    "comparables": INDEX_EXAMPLE,
                }
            ]
        }
    )
