"""The body of `GET /health`."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from recommenditos.api.schemas.common import MODEL_EXAMPLE, ModelInfo


class Health(BaseModel):
    """Whether the service is up, and which model it serves (FR-11)."""

    status: Literal["ok"] = Field(description="`ok` whenever the service answers.")
    model: ModelInfo
    price_range: bool = Field(
        description="Whether the served bundle carries the interval models `/v1/price-range` "
        "needs (EDN-85). Without them that endpoint answers 503 while the rest of the service "
        "works."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"status": "ok", "model": MODEL_EXAMPLE, "price_range": True}]
        }
    )
