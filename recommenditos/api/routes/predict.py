"""`POST /v1/predict`: the asking price of one listing (UC1, FR-06)."""

from http import HTTPStatus

from fastapi import APIRouter

from recommenditos.api.dependencies import RequestId, Served
from recommenditos.api.errors import pointer, problem_responses, validation_problem
from recommenditos.api.schemas.common import FieldWarning
from recommenditos.api.schemas.predict import Valuation, ValuationRequest

router = APIRouter(tags=["valuation"])


@router.post(
    "/predict",
    summary="Estimate the asking price of a used car",
    responses=problem_responses(
        HTTPStatus.UNPROCESSABLE_ENTITY, HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    ),
)
def predict(listing: ValuationRequest, served: Served, request_id: RequestId) -> Valuation:
    """Values one listing with the served model.

    The ten required fields are those of FR-01; every other field is optional,
    and an omitted one reaches the model as absent. A make the model does not
    support is refused with a 422 that lists the supported ones (FR-04). A
    model name or a country the training data does not hold is accepted and
    reported in `warnings` (FR-05).
    """
    faults = served.faults(listing)
    if faults:
        raise validation_problem(faults)
    price, unseen = served.value(listing)
    return Valuation(
        price_eur=round(price, 2),
        model=served.info,
        request_id=request_id,
        warnings=[FieldWarning(pointer=pointer([name]), detail=detail) for name, detail in unseen],
    )
