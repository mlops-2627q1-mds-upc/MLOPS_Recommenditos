"""The body of `GET /`."""

from pydantic import BaseModel, ConfigDict, Field

from recommenditos.api.schemas.common import MODEL_EXAMPLE, ModelInfo


class ServiceIndex(BaseModel):
    """What the service is, which model it serves, and where its documentation is."""

    service: str = Field(description="The name of the service.")
    api_version: str = Field(description="The version of the API document.")
    model: ModelInfo
    docs: str = Field(description="Swagger UI, the interactive API reference.")
    redoc: str = Field(description="ReDoc, the same reference as a document.")
    openapi: str = Field(description="The OpenAPI document itself.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "service": "Recommenditos used-car price API",
                    "api_version": "1.0.0",
                    "model": MODEL_EXAMPLE,
                    "docs": "/docs",
                    "redoc": "/redoc",
                    "openapi": "/openapi.json",
                }
            ]
        }
    )
