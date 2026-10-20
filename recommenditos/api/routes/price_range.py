"""`POST /v1/price-range`: the price range of a car described in part (UC2, FR-02, FR-07)."""

from http import HTTPStatus

from fastapi import APIRouter

from recommenditos.api.dependencies import RequestId, Served
from recommenditos.api.errors import (
    ProblemError,
    pointer,
    problem_responses,
    validation_problem,
)
from recommenditos.api.schemas.common import FieldWarning
from recommenditos.api.schemas.partial_listing import PartialListing
from recommenditos.api.schemas.price_range import PriceRange

router = APIRouter(tags=["valuation"])


@router.post(
    "/price-range",
    summary="Estimate the price range of a used car described in part",
    responses=problem_responses(
        HTTPStatus.UNPROCESSABLE_ENTITY,
        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        HTTPStatus.SERVICE_UNAVAILABLE,
    ),
)
def price_range(car: PartialListing, served: Served, request_id: RequestId) -> PriceRange:
    """The nominal 90 % price range of a car, and its typical price (FR-07).

    Only `make` is required (FR-02): any other field may be left out, and the
    range widens the less the request says. Every field that is given is
    checked as `/v1/predict` checks it (FR-03, FR-04), and an unseen `model` or
    country is answered and reported in `warnings` (FR-05). The typical price is
    the one `/v1/predict` gives when the request carries all of its required
    fields, and the range's median otherwise; the bounds always contain it. A
    bundle without interval models, such as `lgbm-extended`, answers 503.
    """
    faults = served.faults(car)
    if faults:
        raise validation_problem(faults)
    if not served.has_price_range:
        raise ProblemError(
            HTTPStatus.SERVICE_UNAVAILABLE,
            f"The served model {served.info.variant!r} carries no price-range models.",
        )
    estimate, unseen = served.price_range(car)
    return PriceRange(
        price_eur=round(estimate.price_eur, 2),
        lower_eur=round(estimate.lower_eur, 2),
        upper_eur=round(estimate.upper_eur, 2),
        confidence_level=estimate.confidence_level,
        model=served.info,
        request_id=request_id,
        warnings=[FieldWarning(pointer=pointer([name]), detail=detail) for name, detail in unseen],
    )
