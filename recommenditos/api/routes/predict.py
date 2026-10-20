"""`POST /v1/predict`: the asking price of one listing (UC1, FR-06) and why (FR-08)."""

from http import HTTPStatus

from fastapi import APIRouter

from recommenditos.api.dependencies import Metrics, RequestId, Served
from recommenditos.api.errors import problem_responses, validation_problem
from recommenditos.api.schemas.predict import Valuation, ValuationRequest

router = APIRouter(tags=["valuation"])


@router.post(
    "/predict",
    summary="Estimate the asking price of a used car",
    responses=problem_responses(
        HTTPStatus.UNPROCESSABLE_ENTITY, HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    ),
)
def predict(
    listing: ValuationRequest, served: Served, metrics: Metrics, request_id: RequestId
) -> Valuation:
    """Values one listing with the served model.

    The ten required fields are those of FR-01; every other field is optional,
    and an omitted one reaches the model as absent. A make the model does not
    support is refused with a 422 that lists the supported ones (FR-04). A
    model name or a country the training data does not hold is accepted and
    reported in `warnings` (FR-05).

    The `explanation` says where the estimate starts and which five fields move
    it most, each as a percentage, so a client can check the estimate (FR-08).
    """
    faults = served.faults(listing)
    if faults:
        raise validation_problem(faults)
    estimate = served.value(listing)
    valuation = Valuation(
        price_eur=round(estimate.price_eur, 2),
        model=served.info,
        request_id=request_id,
        explanation=estimate.explanation,
        warnings=estimate.warnings,
    )
    metrics.observe_valuation(valuation.price_eur, valuation.warnings)
    return valuation
