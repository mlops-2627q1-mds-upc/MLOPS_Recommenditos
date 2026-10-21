"""The body of `GET /health`."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from recommenditos.api.schemas.common import MODEL_EXAMPLE, ModelInfo


class Health(BaseModel):
    """Whether the service is up, and which model it serves (FR-11)."""

    status: Literal["ok"] = Field(description="`ok` whenever the service answers.")
    model: ModelInfo

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "ok", "model": MODEL_EXAMPLE}]}
    )
