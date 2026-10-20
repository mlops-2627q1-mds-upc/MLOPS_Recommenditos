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
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recommenditos.api.errors import FieldError, pointer
from recommenditos.api.schemas import ListingFields, ModelInfo, ValuationRequest
from recommenditos.data.build_features import build_matrix
from recommenditos.modeling.model import (
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
#: The required fields are `ValuationRequest`'s own; the rename is the one input
#: field the matrix carries under another name. `tests/api/test_predict.py`
#: checks both against `params.yaml`'s `request` block, which the pipeline reads,
#: so the image needs no `params.yaml` and the two cannot drift apart unnoticed.
REQUEST_CONTRACT = RequestContract(
    required_fields=tuple(
        name for name, field in ValuationRequest.model_fields.items() if field.is_required()
    ),
    field_to_feature={"registration_date": "age_years"},
)


class ServingError(ModelError):
    """The bundle cannot be served by this API."""


def bundle_version(directory: Path) -> str:
    """The hash DVC records for `directory` in `dvc.lock`, computed from its files.

    DVC hashes a directory as the MD5 of the sorted list of its files' MD5s and
    relative paths, in JSON. Reporting the same value lets FR-12's check compare
    `/health` with `dvc.lock` directly.
    """
    entries = sorted(
        (
            {
                "md5": hashlib.md5(path.read_bytes()).hexdigest(),
                "relpath": path.relative_to(directory).as_posix(),
            }
            for path in directory.rglob("*")
            if path.is_file()
        ),
        key=lambda entry: entry["relpath"],
    )
    return hashlib.md5(json.dumps(entries, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ServedModel:
    """A loaded bundle, the values it accepts, and the fields it reads."""

    model: Model
    info: ModelInfo
    #: Per checked request field, the values the model accepts.
    allowed: Mapping[str, tuple[str, ...]]
    #: Per input field the model reads, the matrix columns it became.
    fields: Mapping[str, tuple[str, ...]]

    @classmethod
    def load(cls, directory: Path) -> "ServedModel":
        model = load_model(directory)
        fields = input_field_columns(model, REQUEST_CONTRACT)
        levels = model.input_schema.levels
        allowed = {
            name: tuple(levels[name])
            for name in ListingFields.checked_fields
            if name in levels and name not in {"make", "country_code"}
        }
        allowed["make"] = tuple(sorted(model.metadata["training"]["supported_makes"]))
        allowed["country_code"] = COUNTRIES
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

    def value(self, request: ValuationRequest) -> tuple[float, list[tuple[str, str]]]:
        """The price of `request` in EUR, and the fields it reported as unseen."""
        matrix = self.matrix(request)
        price = float(self.model.predict_eur(matrix).iloc[0])
        return price, self._unseen(request, matrix)

    def matrix(self, request: ValuationRequest) -> pd.DataFrame:
        """`request` as the one-row matrix the model scores.

        Built against the model's input contract rather than `space.matrix`,
        whose contract includes the price columns a request does not carry, at
        the request date (problem specification section 4). Every optional field
        the model reads and the client did not send is then omitted through
        `mask_absent`.
        """
        matrix = build_matrix(
            _processed_row(request),
            self.model.input_schema,
            self.model.space.vocabulary,
            reference_date=datetime.now(UTC).date().isoformat(),
        )
        sent = _sent(request)
        omitted = [
            name
            for name in self.fields
            if name not in REQUEST_CONTRACT.required_fields and name not in sent
        ]
        seam = seam_names(self.fields, REQUEST_CONTRACT)
        return self.model.mask_absent(matrix, [seam[name] for name in omitted])

    def _unseen(self, request: ValuationRequest, matrix: pd.DataFrame) -> list[tuple[str, str]]:
        """The categorical fields that were sent and reached the model as unknown (FR-05)."""
        sent = _sent(request)
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


def _sent(request: ValuationRequest) -> set[str]:
    """The fields the client sent with a value; a field sent as null was not sent."""
    return {name for name in request.model_fields_set if getattr(request, name) is not None}


def _processed_row(request: ValuationRequest) -> pd.DataFrame:
    """`request` as one row of the `processed` frame, the shape the feature space reads.

    The equipment lists become the Python-repr strings the raw data stores, the
    kerb weight the text form the `features` stage parses, and the registration
    month the first of that month.
    """
    row: dict[str, Any] = request.model_dump()
    year, month = (int(part) for part in request.registration_date.split("-"))
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
