"""`GET /v1/makes`: the supported makes and the model names the served model knows."""

from http import HTTPStatus

from fastapi import APIRouter

from recommenditos.api.dependencies import Index, Served
from recommenditos.api.errors import ProblemError, problem_responses
from recommenditos.api.schemas.makes import Make, Makes

router = APIRouter(tags=["reference"])


@router.get(
    "/makes",
    summary="The supported makes and the model names the served model knows for each",
    responses=problem_responses(HTTPStatus.SERVICE_UNAVAILABLE),
)
def makes(served: Served, index: Index) -> Makes:
    """Every make `/v1/predict` accepts, with the model names it knows for that make.

    A model name is listed under a make when the comparables index holds a
    listing of both and the served model saw the name in training, so a client
    can suggest names that reach the model rather than an unknown level (FR-05).
    The model's own vocabulary has no make, which is why the index pairs them.
    A service built without the index answers 503.
    """
    if index is None:
        raise ProblemError(
            HTTPStatus.SERVICE_UNAVAILABLE, "The service was started without a comparables index."
        )
    known = set(served.model.input_schema.levels.get("model", ()))
    return Makes(
        makes=[
            Make(make=make, models=[m for m in index.models_by_make.get(make, ()) if m in known])
            for make in served.allowed["make"]
        ],
        model=served.info,
    )
