"""`POST /v1/comparables`: real listings like a car described in part (UC2, FR-09)."""

from http import HTTPStatus

from fastapi import APIRouter

from recommenditos.api.comparables import query_of
from recommenditos.api.dependencies import Index, Log, Metrics, RequestId, Served
from recommenditos.api.errors import ProblemError, pointer, problem_responses, validation_problem
from recommenditos.api.schemas.common import FieldWarning
from recommenditos.api.schemas.comparables import (
    Comparable,
    Comparables,
    ComparablesIndexInfo,
    ComparablesRequest,
)
from recommenditos.api.serving import sent_fields

router = APIRouter(tags=["valuation"])


@router.post(
    "/comparables",
    summary="Find real listings comparable to a used car described in part",
    responses=problem_responses(
        HTTPStatus.UNPROCESSABLE_ENTITY,
        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        HTTPStatus.SERVICE_UNAVAILABLE,
    ),
)
def comparables(
    car: ComparablesRequest,
    served: Served,
    index: Index,
    metrics: Metrics,
    request_id: RequestId,
    log: Log,
) -> Comparables:
    """Up to `k` listings of the training and validation data like the car (FR-09).

    The body is `/v1/price-range`'s plus `k` (5 by default, at most 20), and it
    is checked the same way (FR-03, FR-04). A listing is comparable when it has
    the car's `make` and `model`, its age differs by at most 2 years and its
    mileage by at most 25 %; the closest mileage comes first, then the closest
    age. A field the request leaves out sets no filter, and no filter is ever
    relaxed, so a car outside every window gets an empty list. A service built
    without the index answers 503. Every answer is one record of the prediction
    log (FR-13); when it cannot be written, the answer is a 503.
    """
    faults = served.faults(car)
    if faults:
        raise validation_problem(faults)
    if index is None:
        raise ProblemError(
            HTTPStatus.SERVICE_UNAVAILABLE, "The service was started without a comparables index."
        )
    sent = sent_fields(car)
    query = query_of({name: getattr(car, name) for name in sent | {"make"}})
    found = index.search(query, k=car.k)
    warnings = []
    if query.model is not None and not index.has_model(query.make, query.model):
        warnings.append(
            FieldWarning(
                pointer=pointer(["model"]),
                detail=f"The index holds no {query.make} {query.model!r}, so no listing can "
                "be comparable.",
            )
        )
    answer = Comparables(
        listings=[
            Comparable(**_none_for_missing(row), price_eur=row["price"])
            for row in found.to_dict("records")
        ],
        index=ComparablesIndexInfo.of(index),
        model=served.info,
        request_id=request_id,
        warnings=warnings,
    )
    log.record(car, answer)
    metrics.observe_comparables(len(answer.listings), answer.warnings)
    return answer


def _none_for_missing(row: dict) -> dict:
    """A listing's fields, a missing value as None rather than NaN."""
    fields = set(Comparable.model_fields) - {"price_eur"}
    return {name: None if row[name] != row[name] else row[name] for name in fields}
