"""The comparables index the API serves (FR-09): loaded once, searched per request.

`ComparablesIndex.load` reads the file the `comparables` stage wrote
(`recommenditos/data/build_comparables.py`), checks it against that stage's
contract, and reports its version as the md5 `dvc.lock` records for it, so
`/health` and every answer say which index was searched (FR-12).

A listing is comparable to a request when it has the request's `make` and
`model`, its age differs from the requested car's by at most `MAX_AGE_DIFF_YEARS`
and its mileage by at most `MAX_MILEAGE_DIFF_SHARE` of the requested mileage.
A field the request leaves out sets no filter and adds no distance: a request
without `model` matches every model of the make, one without a mileage leaves
the order to the age. No filter is ever relaxed, so a request outside every
window gets an empty list. The matches are sorted by mileage distance, then by
age distance, then by their position in the index, which the stage writes in a
fixed order, so the same request always gets the same listings.

The listing's age is its age on the snapshot date, the age its price was asked
at; the requested car's is its age today, computed by the same function the
model's feature uses (`build_features.age_years`).

The search runs on NumPy arrays held beside the frame: `make` and `model` as
integer codes, age and mileage as floats. That keeps one request well below a
millisecond on the real index of about 70,000 listings.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recommenditos.data.build_comparables import COMPARABLES_SCHEMA
from recommenditos.data.build_features import age_years

#: FR-09's windows: a comparable listing's age differs by at most this many years
#: from the requested car's ...
MAX_AGE_DIFF_YEARS = 2.0
#: ... and its mileage by at most this share of the requested mileage.
MAX_MILEAGE_DIFF_SHARE = 0.25

#: How many listings an answer holds by default, and at most (FR-09).
DEFAULT_K = 5
MAX_K = 20


def file_md5(path: Path) -> str:
    """The md5 of the file's bytes, which is what `dvc.lock` records for a file output."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Query:
    """What a request asks of the index: `make` always, the rest only when it was sent."""

    make: str
    model: str | None = None
    registration_date: str | None = None
    mileage_km: float | None = None


@dataclass(frozen=True)
class ComparablesIndex:
    """The loaded index and the arrays it is searched on."""

    listings: pd.DataFrame
    version: str
    _make_codes: dict[str, int]
    _model_codes: dict[str, int]
    _make: np.ndarray
    _model: np.ndarray
    _age: np.ndarray
    _mileage: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "ComparablesIndex":
        listings = pd.read_parquet(path)
        COMPARABLES_SCHEMA.validate(listings)
        make = pd.Categorical(listings["make"])
        model = pd.Categorical(listings["model"])
        return cls(
            listings=listings,
            version=file_md5(path),
            _make_codes={level: code for code, level in enumerate(make.categories)},
            _model_codes={level: code for code, level in enumerate(model.categories)},
            _make=make.codes,
            _model=model.codes,
            _age=listings["age_years"].to_numpy(dtype="float64"),
            _mileage=listings["mileage_km_raw"].to_numpy(dtype="float64"),
        )

    def __len__(self) -> int:
        return len(self.listings)

    def has_model(self, make: str, model: str) -> bool:
        """Whether any listing of the index has this make and model."""
        make_code, model_code = self._make_codes.get(make), self._model_codes.get(model)
        if make_code is None or model_code is None:
            return False
        return bool(np.any((self._make == make_code) & (self._model == model_code)))

    def search(self, query: Query, k: int = DEFAULT_K, today: date | None = None) -> pd.DataFrame:
        """The at most `k` listings comparable to `query`, closest first."""
        make_code = self._make_codes.get(query.make)
        if make_code is None:
            return self.listings.iloc[:0]
        mask = self._make == make_code
        if query.model is not None:
            model_code = self._model_codes.get(query.model)
            if model_code is None:
                return self.listings.iloc[:0]
            mask &= self._model == model_code
        # np.lexsort sorts by the last key first: the position breaks every tie.
        keys: list[np.ndarray] = []
        if query.registration_date is not None:
            age_distance = np.abs(self._age - _age_today(query.registration_date, today))
            # A listing without a registration date has a NaN distance, which
            # no comparison passes: it cannot be shown to lie in the window.
            mask &= age_distance <= MAX_AGE_DIFF_YEARS
            keys.append(age_distance)
        if query.mileage_km is not None:
            mileage_distance = np.abs(self._mileage - query.mileage_km)
            mask &= mileage_distance <= MAX_MILEAGE_DIFF_SHARE * query.mileage_km
            keys.append(mileage_distance)
        positions = np.flatnonzero(mask)
        order = np.lexsort([positions, *(key[positions] for key in keys)])
        return self.listings.iloc[positions[order[:k]]]


def query_of(fields: dict[str, Any]) -> Query:
    """The `Query` of a request's fields, given as the fields the client sent."""
    mileage = fields.get("mileage_km_raw")
    return Query(
        make=fields["make"],
        model=fields.get("model"),
        registration_date=fields.get("registration_date"),
        mileage_km=None if mileage is None else float(mileage),
    )


def _age_today(registration_date: str, today: date | None) -> float:
    """The age of a car first registered in `registration_date` (YYYY-MM), today."""
    when = (today or datetime.now(UTC).date()).isoformat()
    return float(age_years(pd.Series([f"{registration_date}-01"]), when).iloc[0])
