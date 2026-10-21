"""The model the API serves: loaded once, checked once, asked per request.

`ServedModel.load` reads a bundle through `load_model` and takes everything a
request is checked against from it, never from the API schema: the supported
makes from its metadata (FR-04) and the levels of the categoricals from its
feature space (FR-03, FR-12). A retrained model with another make or another
fuel category therefore needs no change to the API.

A request reaches the model the way a training row did: as a `processed`-shaped
row that the bundle's own feature space turns into a matrix, with every optional
field the client left out set by `Model.mask_absent`, the one implementation of
"the caller did not send this" that SC-06 and the blanked copies use too.
`/v1/predict` and `/v1/price-range` build that row by the same code, so a field
means the same to both; they differ only in which fields a request must carry.

Every valuation is explained by the booster's own SHAP export (FR-08, EDN-11),
grouped by request field, so a bundle that has none cannot be served.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

import numpy as np
import pandas as pd

from recommenditos.api.errors import pointer
from recommenditos.api.schemas.common import FieldError, FieldWarning, ListingFields, ModelInfo
from recommenditos.api.schemas.partial_listing import PartialListing
from recommenditos.api.schemas.predict import Explanation, FeatureEffect, ValuationRequest
from recommenditos.data.build_features import build_matrix
from recommenditos.modeling.model import (
    BASE_LOG_PRICE,
    INTERVAL_COLUMNS,
    LightGBMModel,
    Model,
    ModelError,
    RequestContract,
    input_field_columns,
    load_model,
    seam_names,
)

#: The 8 countries of the data (problem specification section 2). A request from
#: any other country is out of scope (FR-03); one of these the model was not
#: trained on, `ES` today, is answered with a warning (FR-05).
COUNTRIES: tuple[str, ...] = ("AT", "BE", "DE", "ES", "FR", "IT", "LU", "NL")

#: The request contract the API enforces, in the form the serving seam reads it.
#: The required fields are `ValuationRequest`'s own, those of a price range
#: `PartialListing`'s; the rename is the one input field the matrix carries under
#: another name. `tests/api/test_predict.py`
#: checks both against `params.yaml`'s `request` block, which the pipeline reads,
#: so the image needs no `params.yaml` and the two cannot drift apart unnoticed.
REQUEST_CONTRACT = RequestContract(
    required_fields=tuple(
        name for name, field in ValuationRequest.model_fields.items() if field.is_required()
    ),
    field_to_feature={"registration_date": "age_years"},
    price_range_required_fields=tuple(
        name for name, field in PartialListing.model_fields.items() if field.is_required()
    ),
)


#: How many fields FR-08's explanation lists by name.
TOP_FEATURES = 5


class ServingError(ModelError):
    """The bundle cannot be served by this API."""


@dataclass(frozen=True)
class Estimate:
    """One valuation: the price, what it rests on, and what the client should know."""

    price_eur: float
    explanation: Explanation
    warnings: list[FieldWarning]


@dataclass(frozen=True)
class PriceRangeEstimate:
    """A price range in EUR (FR-07): the typical price and the bounds that contain it."""

    price_eur: float
    lower_eur: float
    upper_eur: float
    confidence_level: float


def price_range(model: Model, matrix: pd.DataFrame, *, complete: bool) -> pd.DataFrame:
    """The price range of every row of `matrix`: `price_eur`, `lower_eur`, `upper_eur`.

    The typical price is FR-07's: the point model's when the rows carry every
    field FR-01 requires (`complete`), otherwise the 50 % quantile model's
    (`median_eur`), because the point model has never seen a required field
    missing (EDN-85). The bounds are the conformalized ones, widened where the
    typical price lies outside them, so `lower_eur <= price_eur <= upper_eur` in
    every row; widening only raises coverage. One function for the endpoint and
    for the measurement of its widths, so the two cannot disagree.
    """
    interval = model.predict_interval_eur(matrix)
    typical = (
        model.predict_eur(matrix).to_numpy()
        if complete
        else interval[INTERVAL_COLUMNS["median"]].to_numpy()
    )
    return pd.DataFrame(
        {
            "price_eur": typical,
            "lower_eur": np.minimum(interval[INTERVAL_COLUMNS["lower"]].to_numpy(), typical),
            "upper_eur": np.maximum(interval[INTERVAL_COLUMNS["upper"]].to_numpy(), typical),
        },
        index=matrix.index,
    )


def bundle_version(directory: Path) -> str:
    """The hash DVC records for `directory` in `dvc.lock`, computed from its files.

    DVC hashes a directory as the MD5 of the sorted list of its files' MD5s and
    relative paths, in JSON. Reporting the same value lets FR-12's check compare
    `/health` with `dvc.lock` directly.
    """
    entries = sorted(
        (
            {
                "md5": hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest(),
                "relpath": path.relative_to(directory).as_posix(),
            }
            for path in directory.rglob("*")
            if path.is_file()
        ),
        key=lambda entry: entry["relpath"],
    )
    return hashlib.md5(
        json.dumps(entries, sort_keys=True).encode(), usedforsecurity=False
    ).hexdigest()


@dataclass(frozen=True)
class ServedModel:
    """A loaded bundle, the values it accepts, and the fields it reads."""

    model: LightGBMModel
    info: ModelInfo
    #: Per checked request field, the values the model accepts.
    allowed: Mapping[str, tuple[str, ...]]
    #: Per input field the model reads, the matrix columns it became.
    fields: Mapping[str, tuple[str, ...]]
    #: Held while the model scores a request, so one request at a time is scored
    #: per process. Concurrent requests in uvicorn's thread pool otherwise take
    #: turns at the GIL every few milliseconds, and each one's latency grows to
    #: that of all of them (issue #179); waiting on a lock costs no CPU.
    _scoring: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    @classmethod
    def load(cls, directory: Path) -> "ServedModel":
        model = load_model(directory)
        if not isinstance(model, LightGBMModel):
            raise ServingError(
                f"{model.variant!r} is a {model.estimator!r} model, which has no SHAP export, so "
                "no valuation could be explained (FR-08). The API serves LightGBM bundles."
            )
        fields = input_field_columns(model, REQUEST_CONTRACT)
        undefined = sorted(set(fields) - set(ValuationRequest.model_fields))
        if undefined:
            raise ServingError(
                f"{model.variant!r} reads the input field(s) {', '.join(undefined)}, which the "
                "request body does not define, so no request could ever supply them. Add them "
                "to `ValuationRequest` before serving this bundle."
            )
        # In the order of the request body, which is how FR-08's explanation
        # breaks a tie between two equal effects.
        fields = {name: fields[name] for name in ValuationRequest.model_fields if name in fields}
        levels = model.input_schema.levels
        allowed = {
            name: tuple(levels[name])
            for name in ListingFields.checked_fields
            if name in levels and name not in {"make", "country_code"}
        }
        allowed["make"] = tuple(sorted(model.metadata["training"]["supported_makes"]))
        allowed["country_code"] = COUNTRIES
        # In the order of the request body, so that faults are listed as a client
        # reads its own request.
        allowed = {
            name: allowed[name] for name in ValuationRequest.model_fields if name in allowed
        }
        trained_on = set(levels.get("country_code", ()))
        if not trained_on <= set(COUNTRIES):
            raise ServingError(
                f"{model.variant!r} was trained on the countries {sorted(trained_on)}, which are "
                f"not all among the {len(COUNTRIES)} the API accepts ({', '.join(COUNTRIES)}). "
                "A request from such a country would be refused although the model knows it."
            )
        info = ModelInfo(
            variant=model.variant,
            version=bundle_version(directory),
            trained_at=model.metadata.get("trained_at"),
        )
        return cls(model=model, info=info, allowed=allowed, fields=fields)

    # -- the checks the schema cannot make ----------------------------------

    def faults(self, body: Any) -> list[FieldError]:
        """The checked fields of `body` whose value the model does not accept.

        `body` is the raw JSON body or a validated request, so the same check
        runs on a request pydantic refused (to report every fault at once) and
        on one it accepted. A value of the wrong type is left to pydantic.
        """
        values = body.model_dump() if isinstance(body, ListingFields) else body
        if not isinstance(values, Mapping):
            return []
        faults = []
        for name, allowed in self.allowed.items():
            value = values.get(name)
            if isinstance(value, str) and value not in allowed:
                faults.append(
                    FieldError(
                        pointer=pointer([name]),
                        detail=self._fault_detail(name, value),
                        allowed=list(allowed),
                    )
                )
        return faults

    @staticmethod
    def _fault_detail(name: str, value: str) -> str:
        if name == "make":
            return f"{value!r} is not a make the served model supports."
        if name == "country_code":
            return f"{value!r} is not one of the {len(COUNTRIES)} countries of the data."
        return f"{value!r} is not a value of {name} in the served model's training data."

    # -- valuation ----------------------------------------------------------

    def value(self, request: ValuationRequest) -> Estimate:
        """The price of `request` in EUR, its explanation, and the warnings for it."""
        with self._scoring:
            return self._value(request)

    def _value(self, request: ValuationRequest) -> Estimate:
        matrix = self.matrix(request)
        price = float(self.model.predict_eur(matrix).iloc[0])
        contributions = self.model.log_price_contributions(matrix).iloc[0]
        warnings = [
            FieldWarning(pointer=pointer([name]), detail=detail)
            for name, detail in self._unseen(request, matrix)
        ]
        unbounded = float(np.exp(contributions.sum()))
        if not np.isclose(price, unbounded, rtol=1e-9, atol=0.0):
            warnings.append(
                FieldWarning(
                    pointer=pointer([]),
                    detail=f"The model estimated {unbounded:,.2f} EUR, outside the price range "
                    f"of its training data, so the price is that range's bound of {price:,.2f} "
                    "EUR. The explanation describes the model's estimate before the bound.",
                )
            )
        return Estimate(
            price_eur=price,
            explanation=explain(contributions, self.fields),
            warnings=warnings,
        )

    @property
    def has_price_range(self) -> bool:
        """Whether the bundle carries the interval models a price range needs (EDN-85)."""
        return self.model.has_intervals

    def price_range(
        self, request: PartialListing
    ) -> tuple[PriceRangeEstimate, list[tuple[str, str]]]:
        """The price range of `request` (FR-07), and the fields it reported as unseen."""
        with self._scoring:
            return self._price_range(request)

    def _price_range(
        self, request: PartialListing
    ) -> tuple[PriceRangeEstimate, list[tuple[str, str]]]:
        matrix = self.matrix(request)
        sent = sent_fields(request)
        complete = all(name in sent for name in REQUEST_CONTRACT.required_fields)
        row = price_range(self.model, matrix, complete=complete).iloc[0]
        estimate = PriceRangeEstimate(
            price_eur=float(row["price_eur"]),
            lower_eur=float(row["lower_eur"]),
            upper_eur=float(row["upper_eur"]),
            confidence_level=float(self.model.metadata["intervals"]["confidence_level"]),
        )
        return estimate, self._unseen(request, matrix)

    def matrix(self, request: ListingFields) -> pd.DataFrame:
        """`request` as the one-row matrix the model scores.

        Built against the model's input contract rather than `space.matrix`,
        whose contract includes the price columns a request does not carry, at
        the request date (problem specification section 4). Every field the
        model reads and the client did not send is then omitted through
        `mask_absent`: an optional one of `/predict`, or any but `make` of a
        price range.
        """
        matrix = build_matrix(
            _processed_row(request),
            self.model.input_schema,
            self.model.space.vocabulary,
            reference_date=datetime.now(UTC).date().isoformat(),
        )
        sent = sent_fields(request)
        omitted = [name for name in self.fields if name not in sent]
        seam = seam_names(self.fields, REQUEST_CONTRACT)
        return self.model.mask_absent(matrix, [seam[name] for name in omitted])

    def _unseen(self, request: ListingFields, matrix: pd.DataFrame) -> list[tuple[str, str]]:
        """The categorical fields that were sent and reached the model as unknown (FR-05)."""
        sent = sent_fields(request)
        unseen = []
        for name, columns in self.fields.items():
            column = columns[0]
            categorical = column in self.model.categorical_columns
            if name in sent and categorical and pd.isna(matrix[column].iloc[0]):
                detail = (
                    f"{getattr(request, name)!r} is not in the served model's training data; "
                    "the model treats it as unknown, so the estimate rests on the other fields."
                )
                unseen.append((name, detail))
        return unseen


def explain(
    contributions: pd.Series, fields: Mapping[str, tuple[str, ...]], top: int = TOP_FEATURES
) -> Explanation:
    """FR-08's explanation of one row from its SHAP values in log price.

    A field's contribution is the sum over the matrix columns it became, which
    SHAP's additivity allows, so an equipment list counts once with all its
    items and `registration_date` stands for the age. Fields are ranked by the
    size of their contribution, ties in the order of `fields`, and each one is
    reported as its multiplicative effect on the price, `exp(φ) - 1`, because
    the model predicts log price and contributions there add up to a product.
    Nothing is rounded, so that the product reproduces the unrounded estimate
    to float precision and `price_eur`, its rounding, to the cent.
    """
    # Summed on the NumPy array rather than through pandas, whose label lookup
    # per field cost about a fifth of a valuation's CPU time (issue #179); the
    # same values in the same order give the same sums.
    values = contributions.to_numpy(dtype="float64")
    position = {column: index for index, column in enumerate(contributions.index)}
    by_field = {
        name: float(values[[position[column] for column in columns]].sum())
        for name, columns in fields.items()
    }
    ranked = sorted(by_field, key=lambda name: abs(by_field[name]), reverse=True)
    others = ranked[top:]
    return Explanation(
        base_price_eur=float(np.exp(contributions[BASE_LOG_PRICE])),
        top_features=[
            FeatureEffect(field=name, effect_pct=_effect_pct(by_field[name]))
            for name in ranked[:top]
        ],
        other_features_effect_pct=_effect_pct(sum(by_field[name] for name in others)),
    )


def _effect_pct(log_contribution: float) -> float:
    """A contribution to the log price as a percentage change of the price."""
    return float(np.expm1(log_contribution) * 100)


def sent_fields(request: ListingFields) -> set[str]:
    """The fields the client sent with a value; a field sent as null was not sent."""
    return {name for name in request.model_fields_set if getattr(request, name) is not None}


def _processed_row(request: ListingFields) -> pd.DataFrame:
    """`request` as one row of the `processed` frame, the shape the feature space reads.

    The equipment lists become the Python-repr strings the raw data stores, the
    kerb weight the text form the `features` stage parses, and the registration
    month the first of that month, or no date where a price range left it out.
    """
    row: dict[str, Any] = request.model_dump()
    if row["registration_date"] is None:
        row["registration_date"] = pd.NaT
    else:
        year, month = (int(part) for part in row["registration_date"].split("-"))
        row["registration_date"] = pd.Timestamp(year=year, month=month, day=1)
    for name in (
        "equipment_comfort",
        "equipment_entertainment",
        "equipment_extra",
        "equipment_safety",
    ):
        row[name] = repr(list(row[name]))
    if row["weight_kg"] is not None:
        row["weight_kg"] = f"{row['weight_kg']:,} kg"
    frame = pd.DataFrame([row])
    for name, value in row.items():
        if value is None:
            frame[name] = np.nan if name in _NUMERIC else pd.Series([None], dtype="str")
    return frame


_NUMERIC = frozenset(
    {
        "mileage_km_raw",
        "power_kw",
        "nr_prev_owners",
        "gears",
        "cylinders_volume_cc",
        "nr_seats",
        "nr_doors",
        "cylinders",
        "electric_range_km",
    }
)
