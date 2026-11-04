"""A processed listing as the request body a client would send for it.

The one mapping from a row of `data/processed/` to a `/v1/predict` body, used by
the load test (`tools/loadtest/bodies.py`, #143) and by the `monitoring_samples`
stage, whose replay files M6 sends to the API (#227). Sharing it keeps the two
from disagreeing about what a client sends for the same listing.

A field the listing lacks is left out of the body, never sent as `null`, because
that is what a client without the value does, and the prediction log records only
what was sent (#191). The registration date is sent as `YYYY-MM`, the kerb weight
as an integer, the equipment lists as lists, and counts as integers.
"""

import ast
import re
from typing import Any

import pandas as pd
from pydantic import ValidationError

from recommenditos.api.schemas.predict import ValuationRequest

_EQUIPMENT = (
    "equipment_comfort",
    "equipment_entertainment",
    "equipment_extra",
    "equipment_safety",
)
_INTEGER = {"nr_prev_owners", "gears", "nr_seats", "nr_doors", "cylinders", "cylinders_volume_cc"}


def _missing(value: Any) -> bool:
    """None, NaN, `pd.NaT` or `pd.NA`; a list (the equipment fields) is never missing."""
    return bool(pd.api.types.is_scalar(value) and pd.isna(value))


def _weight_kg(text: str) -> int | None:
    """`'2,355 kg'` as 2355, the way a client sends the kerb weight."""
    digits = re.sub(r"[^0-9]", "", text)
    return int(digits) if digits else None


def to_body(listing: dict[str, Any]) -> dict[str, Any] | None:
    """The `/v1/predict` body of one processed listing, or None if it is no valid request.

    The body holds the fields in the order `ValuationRequest` declares them, so
    the same listing always gives the same JSON.
    """
    body: dict[str, Any] = {}
    for field in ValuationRequest.model_fields:
        value: Any = listing.get(field)
        if _missing(value):
            continue
        if field == "registration_date":
            value = pd.Timestamp(value).strftime("%Y-%m")
        elif field in _EQUIPMENT:
            value = ast.literal_eval(value) if isinstance(value, str) else list(value)
        elif field == "weight_kg":
            value = _weight_kg(str(value))
        elif field in _INTEGER:
            value = int(value)
        elif hasattr(value, "item"):
            value = value.item()
        if value is not None:
            body[field] = value
    try:
        ValuationRequest.model_validate(body)
    except ValidationError:
        return None
    return body
