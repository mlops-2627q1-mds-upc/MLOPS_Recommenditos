"""The body of `GET /v1/makes`."""

from pydantic import BaseModel, ConfigDict, Field

from recommenditos.api.schemas.common import MODEL_EXAMPLE, ModelInfo


class Make(BaseModel):
    """One supported make and the model names the served model knows for it."""

    make: str
    models: list[str] = Field(
        description="Model names in the served model's training data that the comparables "
        "index holds listings of for this make, sorted. Another name is still accepted, and "
        "answered with a warning (FR-05)."
    )


class Makes(BaseModel):
    """The supported makes with their known model names, for a client's suggestions."""

    makes: list[Make]
    model: ModelInfo

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "makes": [{"make": "BMW", "models": ["116", "118", "320", "X1"]}],
                    "model": MODEL_EXAMPLE,
                }
            ]
        }
    )
