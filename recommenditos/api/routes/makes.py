"""`GET /v1/makes`: the supported makes and the model names the served model knows."""

from http import HTTPStatus

from fastapi import APIRouter, Request

from recommenditos.api.errors import ProblemError, problem_responses
from recommenditos.api.schemas.makes import Makes

router = APIRouter(tags=["reference"])


@router.get(
    "/makes",
    summary="The supported makes and the model names the served model knows for each",
    responses=problem_responses(HTTPStatus.SERVICE_UNAVAILABLE),
)
async def makes(request: Request) -> Makes:
    """Every make `/v1/predict` accepts, with the model names it knows for that make.

    A model name is listed under a make when the comparables index holds a
    listing of both and the served model saw the name in training, so a client
    can suggest names that reach the model rather than an unknown level (FR-05).
    The answer is built once at startup and never changes while the service
    runs, so the route is `async` and waits for no thread. A service built
    without the index answers 503.
    """
    answer = request.app.state.makes
    if answer is None:
        raise ProblemError(
            HTTPStatus.SERVICE_UNAVAILABLE, "The service was started without a comparables index."
        )
    if not isinstance(answer, Makes):
        raise TypeError(f"app.state.makes is a {type(answer).__name__}, not a Makes")
    return answer
