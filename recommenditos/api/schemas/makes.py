"""The body of `GET /v1/makes`."""

from pydantic import BaseModel, ConfigDict, Field

from recommenditos.api.comparables import ComparablesIndex
from recommenditos.api.schemas.common import MODEL_EXAMPLE, ModelInfo
from recommenditos.api.serving import ServedModel


class Make(BaseModel):
    """One supported make and the model names the served model knows for it."""

    make: str
    models: list[str] = Field(
        description="Model names in the served model's training data that the comparables "
        "index holds listings of for this make, sorted regardless of case. Another name is "
        "still accepted, and answered with a warning (FR-05)."
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

    @classmethod
    def of(cls, served: ServedModel, index: ComparablesIndex) -> "Makes":
        """Each make `served` accepts, with the trained names `index` pairs with it.

        The model's own vocabulary has no make, which is why the index pairs them.
        """
        known = set(served.model.input_schema.levels.get("model", ()))
        return cls(
            makes=[
                Make(
                    make=make, models=[m for m in index.models_by_make.get(make, ()) if m in known]
                )
                for make in served.allowed["make"]
            ],
            model=served.info,
        )
