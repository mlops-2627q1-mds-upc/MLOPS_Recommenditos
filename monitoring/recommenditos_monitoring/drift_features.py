"""The drift features of NFR-11 and the age rule the replay and the drift job share.

The `monitoring_samples` stage writes the drift reference with `drift_features`,
and the drift job (#228) computes the same features from each logged request
with it, so the two sides cannot disagree about a feature's representation.
The drift job runs in its own uv project (`monitoring/`, pandas 2 and numpy 1.26
for Alibi Detect), which cannot install this package, so it carries a copy of
this file that `tests/test_monitoring_copy.py` keeps byte-identical to this
one. This module therefore imports nothing but numpy, pandas and the standard
library (`tests/test_monitoring_samples.py` checks that), and it computes the
age itself rather than through `build_features.age_years`; a test checks the
two agree.

The age rule. The model was trained on ages at `reference_date` (2025-11-08),
while the API computes a request's age at the day it answers it. The replay
(#227) sends listings of the 2025 snapshot a year later, so it shifts each
`registration_date` forward by `months_since_reference(send time)`, the whole
months from `reference_date` to the send time, with `shifted_registration`. The
drift job computes the age of a logged request at `as_of_for(answered_at)`:
`reference_date` advanced by the whole months to the answer time. The two shifts
cancel, and the age of a replayed listing is its age at `reference_date` to
within four days (three for a send in the year after `reference_date`), the
most by which the shifted months and the months after `reference_date` differ
in length for any registration month since 1990 and any send in the five years
after it, so a control window compares like with like. Without the second
shift the age would be off by up to one month (+0.066 to +0.079 years on the
4th of a month), which flags `age_years` in 1 or 2 of 20 control windows and
breaks NFR-11's control clause.

A month is counted at 00:00 UTC on `reference_date`'s day of the month (the
8th), not at the end of a calendar month. A request sent before that instant
and answered after it is computed one month older, by 1/12 of a year, than it
was sent. That is accepted: it touches only the requests in flight at that
instant, a handful at the replay's rates against a window of 1,000. Real
traffic, whose registration dates are not shifted, would get an age up to one
month younger than the API computes; all M6 traffic is the replay (EDN-13: no
real users), so the rule is stated rather than corrected for.

The features are EDN-14's: eight numeric ones compared on their non-missing
values (KS), seven categorical ones and a missing indicator per numeric feature
(chi-square). `age_years`, `mileage_km_raw` and `power_kw` come from fields
FR-01 requires, so their indicators are always `present` in an accepted
request: a constant feature, which the drift job has to give a test that
accepts one category or leave out and say so.
"""

from collections.abc import Iterable, Mapping
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

#: The numeric features EDN-14 tested, compared on their non-missing values (KS).
NUMERIC: tuple[str, ...] = (
    "age_years",
    "mileage_km_raw",
    "power_kw",
    "nr_prev_owners",
    "gears",
    "cylinders_volume_cc",
    "nr_seats",
    "nr_doors",
)
#: The categorical features EDN-14 tested (chi-square). `model` is not one: its
#: 349 levels leave well under one expected count per cell in a window of 1,000.
CATEGORICAL: tuple[str, ...] = (
    "make",
    "body_type",
    "fuel_category",
    "transmission",
    "drive_train",
    "seller_type",
    "country_code",
)
#: Whether each numeric feature was sent, tested as a categorical, because a KS
#: test on the non-missing values cannot see a change in how often one is missing.
MISSING_INDICATORS: tuple[str, ...] = tuple(f"{name}__missing" for name in NUMERIC)
#: The 23 features of NFR-11, in EDN-14's order.
DRIFT_FEATURES: tuple[str, ...] = NUMERIC + CATEGORICAL + MISSING_INDICATORS

#: The value of a categorical feature the request did not send.
MISSING = "__MISSING__"
#: The two values of a missing indicator.
ABSENT, PRESENT = "missing", "present"

#: Days per year of the age, as `build_features` computes it for training.
DAYS_PER_YEAR = 365.25

#: A point in time: an ISO string, a date, a datetime or a pandas timestamp.
type Instant = str | date | datetime | pd.Timestamp | np.datetime64


def _utc(when: Instant) -> pd.Timestamp:
    """`when` as a naive UTC timestamp; a naive value is taken to be UTC already."""
    stamp = pd.Timestamp(when)
    return stamp.tz_convert("UTC").tz_localize(None) if stamp.tzinfo is not None else stamp


def months_since_reference(when: Instant, reference_date: Instant) -> int:
    """The whole months from `reference_date` to `when`.

    A month is complete at 00:00 UTC on `reference_date`'s day of the month, so
    for 2025-11-08 it is 11 on 2026-11-07 and 12 from 2026-11-08 on.
    """
    start, end = _utc(reference_date).normalize(), _utc(when)
    months = (end.year - start.year) * 12 + (end.month - start.month)
    return months - 1 if end < start + pd.DateOffset(months=months) else months


def as_of_for(answered_at: Instant, reference_date: Instant) -> pd.Timestamp:
    """The date a logged request's age is computed at: `reference_date` plus the whole months."""
    start = _utc(reference_date).normalize()
    return start + pd.DateOffset(months=months_since_reference(answered_at, reference_date))


def drift_features(
    bodies: Iterable[Mapping[str, Any]], as_of: Instant | Iterable[Instant]
) -> pd.DataFrame:
    """The drift features of request bodies, as the reference holds them.

    `as_of` is the date the age is computed at: one for every body, such as
    `reference_date` for the reference, or one per body in any iterable (a list,
    a numpy array or a pandas Series), `as_of_for` of each request's answer time
    for the log. No bodies give an empty frame with the 23 columns. A field the
    body does not carry is null for a numeric feature and `MISSING` for a
    categorical one.
    """
    frame = pd.DataFrame.from_records(list(bodies))
    rows = len(frame)
    out = pd.DataFrame(index=range(rows))

    def column(name: str) -> pd.Series:
        if name in frame:
            return frame[name].reset_index(drop=True)
        return pd.Series([None] * rows, dtype="object")

    if isinstance(as_of, str | date | np.datetime64):
        dates = [_utc(as_of).normalize()] * rows
    else:
        dates = [_utc(when).normalize() for when in as_of]
        if len(dates) != rows:
            raise ValueError(f"{len(dates)} `as_of` dates for {rows} bodies")
    # An explicit dtype, because an empty list would otherwise give an object
    # Series that cannot be subtracted from the registration dates.
    at = pd.Series(dates, dtype="datetime64[ns]")
    registered = pd.to_datetime(column("registration_date"), format="%Y-%m").astype(
        "datetime64[ns]"
    )
    out["age_years"] = ((at - registered).dt.days / DAYS_PER_YEAR).astype("float64")
    for name in NUMERIC[1:]:
        out[name] = pd.to_numeric(column(name)).astype("float64")
    for name in CATEGORICAL:
        out[name] = column(name).astype("str").where(column(name).notna(), MISSING)
    for name in NUMERIC:
        out[f"{name}__missing"] = np.where(out[name].isna(), ABSENT, PRESENT)
    return out.astype({name: "str" for name in CATEGORICAL + MISSING_INDICATORS})


def shifted_registration(registration_date: str, months: int) -> str:
    """A `YYYY-MM` registration date moved forward by `months`, as the replay sends it."""
    shifted = pd.Timestamp(f"{registration_date}-01") + pd.DateOffset(months=months)
    return shifted.strftime("%Y-%m")
