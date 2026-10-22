"""`POST /v1/feedback`: an observed price for an answer, the delayed label of FR-10.

Internal: the reverse proxy refuses it from outside, and it requires the API
key as a second layer (EDN-13). Only the `ES` replay job calls it.
"""

from datetime import UTC, datetime
from http import HTTPStatus

from fastapi import APIRouter, Response

from recommenditos.api.dependencies import RequireApiKey, Store
from recommenditos.api.errors import ProblemError, problem_responses
from recommenditos.api.prediction_log import Feedback, FeedbackOutcome
from recommenditos.api.schemas.feedback import FeedbackReceipt, FeedbackRequest

router = APIRouter(tags=["feedback"])


@router.post(
    "/feedback",
    summary="Store the observed price of an answered listing",
    status_code=HTTPStatus.CREATED,
    dependencies=[RequireApiKey],
    responses={
        HTTPStatus.OK: {
            "model": FeedbackReceipt,
            "description": "The same price was stored for this request before; nothing changed.",
        },
        **problem_responses(
            HTTPStatus.UNAUTHORIZED,
            HTTPStatus.NOT_FOUND,
            HTTPStatus.CONFLICT,
            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            HTTPStatus.UNPROCESSABLE_ENTITY,
            HTTPStatus.SERVICE_UNAVAILABLE,
        ),
    },
)
def feedback(body: FeedbackRequest, store: Store, response: Response) -> FeedbackReceipt:
    """Stores the price a listing was observed at against the answer's `request_id`.

    The drift job joins it to the prediction log to report the real error and
    interval coverage (FR-14), so a price is stored only for an answer the log
    holds, and only one per answer:

    - a new price is stored, `201`;
    - the same price again is answered `200` and changes nothing, so a retry is safe;
    - another price for the same answer is refused with `409`, and so is any price for
      an answer that gave none, such as one of `/v1/comparables`;
    - a `request_id` the log does not hold, or no longer holds after rotation, is `404`.
    """
    result = store.record_feedback(
        Feedback(str(body.request_id), body.observed_price_eur, datetime.now(UTC))
    )
    if result.outcome is FeedbackOutcome.UNKNOWN_REQUEST:
        raise ProblemError(
            HTTPStatus.NOT_FOUND,
            f"No answer with request ID {body.request_id} is in the prediction log.",
        )
    if result.outcome is FeedbackOutcome.UNPRICED_ANSWER:
        raise ProblemError(
            HTTPStatus.CONFLICT,
            f"The answer with request ID {body.request_id} gave no price, so there is no "
            "estimate an observed price could be compared with.",
        )
    if result.stored is None:
        raise RuntimeError(f"a {result.outcome.name} feedback outcome carries no stored price")
    if result.outcome is FeedbackOutcome.CONFLICT:
        raise ProblemError(
            HTTPStatus.CONFLICT,
            f"Another price ({result.stored.observed_price_eur} EUR) is stored for request ID "
            f"{body.request_id}; a stored price is not replaced.",
        )
    if result.outcome is FeedbackOutcome.REPEATED:
        response.status_code = HTTPStatus.OK
    return FeedbackReceipt(
        request_id=body.request_id,
        observed_price_eur=result.stored.observed_price_eur,
        received_at=result.stored.received_at,
    )
