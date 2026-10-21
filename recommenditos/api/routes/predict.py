"""`POST /v1/predict`: the asking price of one listing (UC1, FR-06) and why (FR-08)."""

from http import HTTPStatus
import math

from fastapi import APIRouter

from recommenditos.api.dependencies import Log, Metrics, RequestId, Served
from recommenditos.api.errors import ProblemError, problem_responses, validation_problem
from recommenditos.api.schemas.predict import Valuation, ValuationRequest

router = APIRouter(tags=["valuation"])


@router.post(
    "/predict",
    summary="Estimate the asking price of a used car",
    responses=problem_responses(
        HTTPStatus.UNPROCESSABLE_ENTITY,
        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        HTTPStatus.SERVICE_UNAVAILABLE,
    ),
)
def predict(
    listing: ValuationRequest,
    served: Served,
    metrics: Metrics,
    request_id: RequestId,
    log: Log,
) -> Valuation:
    """Values one listing with the served model.

    The ten required fields are those of FR-01; every other field is optional,
    and an omitted one reaches the model as absent. A make the model does not
    support is refused with a 422 that lists the supported ones (FR-04). A
    model name or a country the training data does not hold is accepted and
    reported in `warnings` (FR-05).

    The `explanation` says where the estimate starts and which five fields move
    it most, each as a percentage, so a client can check the estimate (FR-08).
    Every answer is one record of the prediction log (FR-13); when it cannot
    be written, the answer is a 503. A price that is not a finite number is a
    500 and is neither answered nor logged.
    """
    faults = served.faults(listing)
    if faults:
        raise validation_problem(faults)
    estimate = served.value(listing)
    if not math.isfinite(estimate.price_eur):
        raise ProblemError(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "The model produced no valid price for this listing. Quote the instance when you "
            "report it.",
        )
    valuation = Valuation(
        price_eur=round(estimate.price_eur, 2),
        model=served.info,
        request_id=request_id,
        explanation=estimate.explanation,
        warnings=estimate.warnings,
    )
    log.record(listing, valuation)
    metrics.observe_valuation(valuation.price_eur, valuation.warnings)
    return valuation
