"""The four estimators of the ladder, and the one seam that persists and serves them.

This module is the boundary between "a model" and everything that uses one. The
`train` stage fits through `fit_variant`; `evaluate` and the API load through
`load_model` and ask for `predict_eur`. Nothing outside this module opens a file
in `models/<variant>/`, and nothing outside it knows that three of the four
estimators are fitted on `log_price`.

It is a module of its own rather than part of `train.py` because `evaluate`'s
`dvc.yaml` `deps` name the code that stage runs: with the seam inside `train.py`,
every hyperparameter-only edit would rerun the whole evaluation.

Two properties are the reason the seam is shaped this way.

**The unit is in the method name.** `predict_eur` returns euros, always, so the
inverse of the log transform lives inside the model and `evaluate` and the API
cannot come to disagree about it. `predict_log_price` is defined as the log of
that, which is exact rather than approximate: every estimator here is a
median-flavoured estimator, so the log of the prediction is the prediction of
the log.

**The feature contract comes from the artefact, never from params.yaml.** The
matrices carry data-dependent columns - the equipment multi-hot columns - and
data-dependent category levels, so a contract rebuilt from `features.sets` would
not describe them. `train` copies the `features` stage's own `feature_space.json`
into the model bundle, and `load_model` reads it back, which is how a served
request is encoded exactly as a training row was (EDN-02).

**UC2's price range is served from the same bundle (EDN-85).** A variant whose
`params.yaml` block names `intervals` also carries three quantile boosters, at
5 %, 95 % and 50 %, and the two conformity offsets MAPIE's conformalized
quantile regression computed for them on the `calibration` split.
`predict_interval_eur` adds them up: the lower booster minus the lower offset
and the upper booster plus the upper offset, in log price. MAPIE itself is a
`pipeline` package and is imported only by `intervals.py`, which the `train`
stage runs; this module, which the API imports, needs LightGBM alone for it.
`has_intervals` is what `evaluate` checks before it measures SC-05, so a bundle
without the boosters reports the criterion as not measured rather than failing.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import version as distribution_version
import json
from pathlib import Path
import platform
from types import MappingProxyType

import joblib
import lightgbm
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from recommenditos.data.build_features import FeatureSpace, equipment_feature_name
from recommenditos.pipeline import write_json
from recommenditos.schema import FEATURE_SPACE_FILE, Schema

#: The name every prediction Series carries, so a frame that has been joined
#: with predictions says which column is which without a convention.
PREDICTION_NAME = "predicted_price_eur"

#: The column of `LightGBMModel.log_price_contributions` that holds the base
#: value, the log price every row's contributions start from.
BASE_LOG_PRICE = "base_log_price"

#: The metadata record, and the only file `load_model` opens before it knows
#: which estimator it is loading.
MODEL_FILE = "model.json"

_BOOSTER_FILE = "booster.txt"
_PIPELINE_FILE = "pipeline.joblib"
_LOOKUP_FILE = "lookup.parquet"

#: The three quantile boosters of UC2's price range (EDN-85), by the role MAPIE
#: gives them: the lower and upper quantile it conformalizes, and the median it
#: requires beside them. The order is MAPIE's order for a prefit estimator list.
INTERVAL_QUANTILES: tuple[str, ...] = ("lower", "upper", "median")
_INTERVAL_FILES: dict[str, str] = {name: f"interval_{name}.txt" for name in INTERVAL_QUANTILES}

#: The columns `predict_interval_eur` returns, in euros.
INTERVAL_COLUMNS: dict[str, str] = {
    "lower": "lower_eur",
    "upper": "upper_eur",
    "median": "median_eur",
}

#: The level a Ridge one-hot encoder gets for an absent categorical value. A
#: literal level rather than a dropped row, so absence is something the model
#: can carry a coefficient for.
MISSING_LEVEL = "__missing__"

#: The levels of B0's lookup table, coarsest last. The order is the fallback
#: order, and `_LOOKUP_LEVELS.index` is what sorts the table.
_LOOKUP_LEVELS: tuple[str, ...] = ("bucket", "make", "global")

#: The columns B0 keys its finest level on. `age_years` becomes a bucket.
_B0_KEY_COLUMNS: tuple[str, ...] = ("make", "model", "age_years")

#: What `_align` can turn into a number. Anything else in a feature matrix is a
#: column no estimator here knows how to encode, and guessing would be worse
#: than failing: a datetime cast to float64 is a plausible-looking feature.
_NUMERIC_DTYPES: frozenset[str] = frozenset({"float64", "bool", "boolean"})

#: The name LightGBM's early stopping reports its scores under.
_EVAL_NAME = "validation"


class ModelError(ValueError):
    """A model bundle, or a frame handed to one, cannot be used as asked."""


def _dtype_name(dtype) -> str:
    """A dtype as an error message has to name it: two categoricals differ by their levels."""
    if isinstance(dtype, pd.CategoricalDtype):
        return f"a categorical over {len(dtype.categories)} level(s)"
    return str(dtype)


def _versions(*libraries: str) -> dict[str, str]:
    """The versions of the libraries whose output ends up in the bundle.

    Recorded per estimator rather than for everything installed: the point is
    that `load_model` can warn when the library that *wrote* a payload is not
    the one reading it, and a version list full of libraries the payload does
    not depend on would make that warning fire for irrelevant upgrades.

    The names are distribution names, read from the installed metadata, so
    `scikit-learn` is spelled the way `pyproject.toml` and a changelog spell it
    rather than as the module `sklearn`.
    """
    return {
        "python": platform.python_version(),
        **{name: distribution_version(name) for name in libraries},
    }


# --------------------------------------------------------------------------
# What a fit sees
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestContract:
    """What FR-01 says a request carries, in the two facts a model's fields are read by.

    `required_fields` are the input fields `/predict` refuses a request without
    (EDN-15), and `field_to_feature` names the matrix column of an input field
    that reaches the model under another name: a request carries
    `registration_date` and the matrix the derived `age_years`. Every other field
    of a feature set is optional, which is what SC-06 masks and what a fit blanks
    in the copies it adds (EDN-82).

    The seam itself knows only the matrix and never reads this from
    `params.yaml`: a stage that has a view on requests builds the contract and
    hands it over, to `evaluate`'s masking sweep or to a fit through
    `TrainingData`. `from_params` is the one place that says where `params.yaml`
    keeps the two facts, so that the stages cannot come to read them from two.
    """

    required_fields: tuple[str, ...]
    field_to_feature: Mapping[str, str]
    #: The input fields `/price-range` refuses a request without (FR-02): `make`
    #: alone. Every other field of the feature set may be left out of a UC2
    #: request, so it is what the interval models' copies omit and calibrate on.
    price_range_required_fields: tuple[str, ...] = ()

    @classmethod
    def from_params(cls, params: Mapping) -> "RequestContract":
        block = params["request"]
        return cls(
            required_fields=tuple(block["required_input_fields"]),
            field_to_feature=dict(block["input_field_to_feature"]),
            price_range_required_fields=tuple(block.get("price_range_required_input_fields", ())),
        )


@dataclass(frozen=True)
class TrainingData:
    """The frames a fit sees, the contract they were written against, and the scope.

    One object rather than five arguments, so that `fit_variant` stays the single
    call issue #38's energy measurement has to wrap.

    `supported_makes` is here rather than beside the parameters because it is a
    property of these rows: `features` restricts both frames to the makes the API
    serves and `train` checks that it did (EDN-05), and the list has to
    reach the model bundle so that the API answers FR-04's scope check from the
    model it is serving.

    `request` is what tells a fit which of its fields a request may leave out.
    Only a variant that adds blanked copies of its training rows reads it
    (EDN-82), so it may be absent for any other, and `fit_variant` refuses a
    variant that needs it and was not given it.
    """

    space: FeatureSpace
    train: pd.DataFrame
    validation: pd.DataFrame
    supported_makes: tuple[str, ...]
    request: RequestContract | None = None
    #: The `calibration` split, for a variant with a price range alone (EDN-85):
    #: `intervals.py` computes the conformity offsets on it, and nothing else in
    #: a fit reads it. `None` for every other variant.
    calibration: pd.DataFrame | None = None


# --------------------------------------------------------------------------
# The seam
# --------------------------------------------------------------------------


class Model:
    """A fitted variant: what it consumes, what it predicts, and how it was made.

    `predict_eur` is final. Subclasses implement `_predict_log_price`, which
    receives an already-aligned frame and returns log euros, so neither the
    dtype alignment nor the training-range bound can be bypassed by an
    estimator.

    Why the hook works in log space although the public unit is euros: the two
    log-space estimator families are fitted with an unbounded link, and `exp()`
    of a linear extrapolation overflows to `inf` at about 710 in log space and
    underflows to exactly `0.0` at about -746. Bounding *before* the exponential
    is what makes "finite and strictly positive" a property of the arithmetic
    rather than a hope, and it costs the median-preserving estimators nothing,
    because `exp` is monotone.

    The bound is not only a guard against a float limit: measured on the real
    snapshot, B1's unbounded prediction leaves the training price range once in
    19,665 test rows, at 10,635,538 EUR. It is a guard rail rather than a
    calibration, though, and the comment is not a claim that the bounded number
    is right - it is the price of the most expensive car in the training rows.
    """

    #: The libraries whose version a subclass records, beyond python itself.
    #: `numpy` and `pandas` are in every subclass's list because the alignment
    #: below is theirs, not the payload's.
    libraries: tuple[str, ...] = ("numpy", "pandas")

    #: The `estimator` key in params.yaml that selects this class.
    estimator_name: str = ""

    #: The keys this estimator reads from its variant's `params` block. A key it
    #: does not know is refused rather than ignored: a parameter nothing reads
    #: makes `dvc repro` report a retrain for a change that cannot move a number.
    hyperparameters: tuple[str, ...] = ()

    #: Keys the `params` block may leave out, each with the value it then takes.
    #: Only for a knob whose default is the library's own, so that naming it in
    #: params.yaml at that value changes nothing about the fit, and leaving it out
    #: keeps `params.yaml` and `dvc.lock` as they were before the knob became one.
    #: Read-only, because the class attribute is shared by every subclass.
    optional_hyperparameters: Mapping[str, object] = MappingProxyType({})

    #: Whether a variant of this estimator may name `blanked_copies` (EDN-82).
    #: Off unless an estimator says otherwise, because the copies are rows to an
    #: estimator and not every one may count them as listings: the Ridge folds a
    #: level by how many training rows carry it and B0's table reports the rows
    #: behind a median, so both would count a listing once per copy.
    supports_blanked_copies: bool = False

    def __init__(self, metadata: dict, space: FeatureSpace) -> None:
        self.metadata = metadata
        self.space = space
        self.input_schema: Schema = space.schema.features(name=f"{metadata['feature_set']}-input")
        self._check_the_bundle_agrees_with_itself()
        training = metadata["training"]
        low, high = float(training["price_min_eur"]), float(training["price_max_eur"])
        # `predict_eur` promises a strictly positive price, and it keeps that
        # promise by bounding to this range, so a range that does not bound one
        # has to be refused here rather than produce a zero. `preprocess` makes it
        # impossible from the pipeline (`price_min_eur: 500`); a hand-edited
        # bundle is what this guard is for.
        if not 0 < low <= high:
            raise ModelError(
                f"the training price range of {self.metadata['variant']!r} is "
                f"[{low}, {high}], which cannot bound a price: the lower edge has to be "
                f"above 0 and at or below the upper one."
            )
        # In log space, because that is where the bound is applied. Kept as euros
        # in the metadata, so a reader - and #39's report of how many predictions
        # sit on a bound - sees the number in the unit it means.
        self._log_bounds = (float(np.log(low)), float(np.log(high)))

    def _check_the_bundle_agrees_with_itself(self) -> None:
        """Refuse a bundle whose record and feature space describe different models.

        The two files are written together and restored together, so this can
        only happen to a hand-edited or half-restored bundle. It is worth a check
        because the symptom otherwise is a prediction from a model encoding its
        input by one level order and scoring it by another, with no error.
        """
        declared = tuple(self.metadata["features"])
        expected = self.input_schema.names
        if declared != expected:
            misplaced = [
                f"position {index} is {was!r} in {MODEL_FILE} and {now!r} in {FEATURE_SPACE_FILE}"
                for index, (was, now) in enumerate(zip(declared, expected, strict=False))
                if was != now
            ]
            raise ModelError(
                f"the two halves of the {self.metadata['variant']!r} bundle disagree about what "
                f"it consumes: {MODEL_FILE} lists {len(declared)} feature(s), "
                f"{FEATURE_SPACE_FILE} describes {len(expected)}, and {len(misplaced)} are not in "
                f"the same place"
                + (f" ({misplaced[0]})" if misplaced else "")
                + ". The order is as load-bearing as the membership, because a column is a "
                "position and so is a code: a bundle ordered one way in one file and another way "
                "in the other encodes a request by one order and scores it by another, with no "
                "error anywhere. Retrain the variant rather than editing either file."
            )
        unknown = {
            column.name: column.dtype
            for column in self.input_schema.columns
            if column.levels is None and column.dtype not in _NUMERIC_DTYPES
        }
        if unknown:
            listed = ", ".join(f"{name} ({dtype})" for name, dtype in sorted(unknown.items()))
            raise ModelError(
                f"the {self.feature_set!r} matrix carries column(s) no estimator here can "
                f"encode: {listed}. A categorical becomes levels and a number becomes "
                f"float64; anything else needs a decision, not a cast."
            )

    # -- what this model is -------------------------------------------------

    @property
    def variant(self) -> str:
        return str(self.metadata["variant"])

    @property
    def estimator(self) -> str:
        return str(self.metadata["estimator"])

    @property
    def feature_set(self) -> str:
        return str(self.metadata["feature_set"])

    @property
    def features(self) -> tuple[str, ...]:
        """The matrix columns this model was fitted against, in its own order.

        The authority on what a caller has to supply. Not the same as the columns
        the estimator's arithmetic reads: B0 keys on three of them and ignores
        the rest, which is a property of that estimator rather than of the
        interface, and which is why masking a column B0 ignores moves nothing.
        """
        return self.input_schema.names

    @property
    def categorical_columns(self) -> tuple[str, ...]:
        """The aligned columns that reach an estimator as levels, not as numbers."""
        return tuple(name for name in self.features if name in self.input_schema.levels)

    @property
    def numeric_columns(self) -> tuple[str, ...]:
        return tuple(name for name in self.features if name not in self.input_schema.levels)

    def columns_of(self, field: str) -> tuple[str, ...]:
        """The matrix columns one **feature** field became, in the model's own order.

        One column for almost every field, and one per kept item for an equipment
        list: `equipment_comfort` is not a matrix column at all, it is the
        multi-hot columns the vocabulary derived from it. SC-06 masks fields and
        the API receives fields, so the expansion lives here rather than being
        derived twice - two derivations of it are two answers to "what did masking
        `equipment_comfort` mask".

        **The boundary, which is deliberate.** `field` is a name in the feature
        space, not a name in a request. `columns_of("registration_date")` raises,
        because the matrix carries the derived `age_years` and this method knows
        only the matrix. Which request field a feature column belongs to is the
        request contract's to say, and no method of this class reads it: a stage
        that has a view on requests builds a `RequestContract` from `params.yaml`
        and hands it to `input_field_columns` below, which applies the rename and
        then calls this.

        A caller that starts from a *request* field therefore goes through
        `input_field_columns` rather than composing the two itself. `evaluate`'s
        masking sweep and the blanked copies of a fit both do. The API meets
        exactly the same thing - a request carries `registration_date` where the
        matrix carries `age_years` - so it does too, rather than growing a second
        expansion of its own, which is the whole point of there being one
        `columns_of`.
        """
        items = self.space.vocabulary.equipment.get(field)
        candidates = (
            {field} if items is None else {equipment_feature_name(field, item) for item in items}
        )
        columns = tuple(name for name in self.features if name in candidates)
        if not columns:
            raise ModelError(
                f"{self.variant!r} consumes no column of field {field!r}, so there is nothing "
                f"to mask or to supply. Its {len(self.features)} feature(s) come from the "
                f"{self.feature_set!r} set."
            )
        return columns

    def mask_absent(
        self, frame: pd.DataFrame, fields: Iterable[str], *, where: np.ndarray | None = None
    ) -> pd.DataFrame:
        """`frame` with every column of `fields` set to what an omitted field is.

        The single implementation of "the caller did not send this", for SC-06's
        masking sweep, the API's partial-request path, the blanked copies a fit
        adds to its training rows (EDN-82) and the tests alike. It is
        shared because no one value is right for all four dtypes:

        - a `bool` assertion flag becomes `False`. A tick is evidence and an empty
          box is the absence of it, which is all the source can mean (EDN-23), and
          the contract makes those three columns non-nullable to say so.
        - a nullable `boolean` equipment column becomes `False` as well, because
          an omitted equipment list is an **empty** list. The raw field is always
          a list and `"[]"` when empty: measured over the 105,405 scoped listings
          there is not one null in any of the four, while the literal `"[]"` is
          5.2 % to 8.0 % of them per field. So empty is a state the model was
          fitted on and absent is one it never saw (EDN-61).
        - a categorical becomes null over the contract's own levels, so `_align`
          produces the missing code rather than a level of its own.
        - a number becomes NaN.

        The one-liner this replaces is `frame[column] = np.nan`, which is wrong
        twice over and loudly only once. For the three flags it is refused, because
        a float64 NaN in a non-nullable `bool` column fails the contract's null
        check, so a caller notices. For an equipment column it is not: the column
        upcasts to float64, survives `conform`, and `astype("boolean")` turns it
        back into `pd.NA` - so the field arrives as absent rather than empty, which
        is the one thing this method exists to decide, and nothing says a word.

        `where` narrows it to single cells: something array-like of truth values
        with one row per row of `frame`, by position, and one column per field,
        in the order of `fields`. A field is omitted from a row only where it is
        true, and a field named twice is omitted where either says so. Left out,
        every field is omitted from every row. So a request that carries some
        optional fields and not others is built by the same method that builds
        one carrying none, and a cell is either the frame's own value or the
        omitted one.

        `frame` is not mutated.
        """
        fields = tuple(fields)
        masked = frame.copy()
        if where is None:
            for field in fields:
                for name in self.columns_of(field):
                    column = self.input_schema.column(name)
                    if column.dtype == "bool":
                        masked[name] = False
                    elif column.dtype == "boolean":
                        masked[name] = pd.array([False] * len(masked), dtype="boolean")
                    elif column.levels is not None:
                        masked[name] = pd.Categorical(
                            [None] * len(masked), categories=column.levels
                        )
                    else:
                        masked[name] = np.nan
            return masked

        cells = np.asarray(where, dtype=bool)
        if cells.shape != (len(frame), len(fields)):
            raise ModelError(
                f"`where` has the shape {cells.shape} for {len(frame)} row(s) and "
                f"{len(fields)} field(s). It needs one row per row of the frame and one "
                f"column per field, in the order of `fields`."
            )
        # What every cell would be if omitted, from the branch above, so the two
        # ways of calling this cannot come to mean two things by "omitted".
        omitted = self.mask_absent(frame, fields)
        for position, field in enumerate(fields):
            # By position, like `cells`, and carried on the frame's own index so
            # that pandas does not align it by label.
            keep = pd.Series(~cells[:, position], index=frame.index)
            for name in self.columns_of(field):
                if omitted[name].dtype != frame[name].dtype:
                    raise ModelError(
                        f"column {name!r} of the frame is {_dtype_name(frame[name].dtype)} and "
                        f"the contract's omitted value for it is "
                        f"{_dtype_name(omitted[name].dtype)}, so a column omitted from some "
                        f"rows only would be neither. Conform the frame to {self.variant!r}'s "
                        f"input schema before masking single rows."
                    )
                masked[name] = masked[name].where(keep, omitted[name])
        return masked

    # -- prediction ---------------------------------------------------------

    def predict_eur(self, frame: pd.DataFrame) -> pd.Series:
        """`frame`'s rows as a price in euros, one per row, in `frame`'s order.

        Float64, named `PREDICTION_NAME`, finite, strictly positive and inside
        the training price range, with the same index as `frame`. An empty frame
        gives an empty Series, because the length promise is the promise. Extra
        columns are ignored; a missing one raises and names itself. `frame` is not
        mutated: the alignment works on a copy.

        A missing value is passed through unimputed and the estimator decides
        what to do with it (EDN-15). What "missing" means for a column whose
        domain already represents absence is not the caller's to guess:
        `mask_absent` is the one implementation of it, and an omitted field that
        does not go through it is the defect EDN-61 records. An omitted assertion
        flag is `False`, and the contract makes those three columns non-nullable,
        so passing `None` for one of them fails here rather than quietly becoming
        a third state (EDN-23). An omitted equipment list is `False` throughout
        too, because it is an *empty* list and that is a state the training rows
        contain (EDN-61).
        """
        aligned = self._align(frame)
        if aligned.empty:
            # Short-circuited rather than passed on: "the same length and the same
            # index as `frame`" is the whole interface, and n = 0 satisfies it
            # trivially while three of the four estimators refuse it - scikit-learn
            # with "Found array with 0 sample(s)" and LightGBM with "Input data must
            # be 2 dimensional and non empty". SC-06's masking sweep can produce an
            # empty segment and a batch endpoint can be handed an empty list, and
            # neither is an error. After `_align`, so a frame that is empty *and*
            # missing a column still names the column.
            return pd.Series(np.empty(0, dtype="float64"), index=frame.index, name=PREDICTION_NAME)
        log_price = np.asarray(self._predict_log_price(aligned), dtype="float64")
        if log_price.shape != (len(frame),):
            raise ModelError(
                f"{self.estimator!r} returned {log_price.shape} predictions for "
                f"{len(frame)} row(s)"
            )
        bounded = np.clip(log_price, *self._log_bounds)
        return pd.Series(np.exp(bounded), index=frame.index, name=PREDICTION_NAME)

    def predict_log_price(self, frame: pd.DataFrame) -> pd.Series:
        """The log of `predict_eur`, which is what the training target was.

        Exact rather than reconstructed from the estimator's own log-space
        output: taking the log of the served number is the only definition under
        which the two methods cannot disagree about the bound.
        """
        return np.log(self.predict_eur(frame)).rename("predicted_log_price")

    def _predict_log_price(self, aligned: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    @property
    def has_intervals(self) -> bool:
        """Whether this bundle carries calibrated UC2 intervals (EDN-85).

        What `evaluate` checks before it measures SC-05: a bundle without them
        reports the criterion as not measured, which blocks NFR-01's gate.
        """
        return False

    def predict_interval_eur(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The nominal 90 % price range of each row, in euros (FR-07).

        One row per row of `frame`, with its index, and three float64 columns:
        `lower_eur` and `upper_eur`, the conformalized bounds, and `median_eur`,
        the 50 % quantile model MAPIE requires beside them. Only a bundle that
        `has_intervals` implements it.
        """
        raise ModelError(
            f"{self.variant!r} carries no price-range models. A LightGBM variant gains them "
            f"when its params.yaml block names `intervals` (EDN-85)."
        )

    def _align(self, frame: pd.DataFrame) -> pd.DataFrame:
        """`frame` as the matrix this model was fitted on, one row for one row.

        `Schema.conform` is the whole cast: it selects the contract's columns in
        the contract's order, drops everything else, and rebuilds each
        categorical over the levels recorded at fit time, so a value the training
        rows never saw becomes a missing code rather than a level of its own
        (EDN-18) and `.cat.codes` means here what it meant then. Casting with
        `astype("category")` instead would be a silent no-op between two
        categoricals over the same set of levels, keeping whatever order the
        frame happened to arrive in.

        What is left afterwards is the second half only: the contract's `bool`
        and nullable `boolean` columns become float64, so that every column of the
        returned frame is float64 or category. That is the dtype contract
        `_predict_log_price` is written against, and the reason it is stated here
        is that it is the only thing a new estimator may assume about its input.

        Worth being exact, because the docstring previously called this
        load-bearing and it is not: removing the cast changes **no prediction
        anywhere** today. Measured on the real snapshot, all four variants predict
        bit-identically without it, a Ridge fitted on `extended` fits and predicts
        bit-identically without it, and so does `lgbm-extended` on a frame with
        `pd.NA` actually present in an equipment column. LightGBM accepts either
        dtype; scikit-learn's `MissingIndicator` refuses `bool` only when handed
        a `bool`-only frame, and inside the `ColumnTransformer` it gets a mixed
        one that `check_array` has already made float64.

        So this is insurance of the same kind as `deterministic=True` and
        `solver="lsqr"`, kept because a property a future estimator will
        depend on should not rest on two libraries independently choosing to be
        tolerant - not because anything here would fail without it.
        """
        aligned = self.input_schema.conform(frame)
        for name in aligned.columns:
            if str(aligned[name].dtype) in {"bool", "boolean"}:
                aligned[name] = pd.Series(
                    aligned[name].to_numpy(dtype="float64", na_value=np.nan),
                    index=aligned.index,
                    name=name,
                )
        return aligned

    # -- persistence --------------------------------------------------------

    def save(self, directory: Path) -> None:
        """Write the whole bundle: the record, the feature space and the payload."""
        directory.mkdir(parents=True, exist_ok=True)
        write_json(self.metadata, directory / MODEL_FILE)
        # A copy of the `features` stage's own artefact rather than a second,
        # model-shaped rendering of the same facts. It is what `load_model`,
        # `read_feature_space` and the API already know how to read, and one file
        # cannot disagree with itself about which levels a code means.
        space = {
            "schema": self.space.schema.to_dicts(),
            "vocabulary": self.space.vocabulary.to_dict(),
        }
        write_json(space, directory / FEATURE_SPACE_FILE)
        self._save_payload(directory)
        logger.success(f"Wrote the {self.variant!r} bundle to {directory}.")

    def _save_payload(self, directory: Path) -> None:
        raise NotImplementedError

    # -- fitting ------------------------------------------------------------

    @classmethod
    def fit(cls, metadata: dict, data: TrainingData, *, seed: int, num_threads: int) -> "Model":
        raise NotImplementedError

    @classmethod
    def _load_payload(cls, directory: Path, metadata: dict, space: FeatureSpace) -> "Model":
        raise NotImplementedError

    @classmethod
    def _tuning(cls, metadata: dict) -> dict:
        """This variant's `params` block, checked against what the estimator reads.

        Both directions, because both are silent otherwise: a missing key would
        surface as a `KeyError` from inside a fit, and an extra one as a
        parameter somebody set and nothing used. An optional key the block leaves
        out takes its default here, so the fit always receives every key it reads.
        """
        given = dict(metadata["params"])
        expected = set(cls.hyperparameters)
        unknown = sorted(set(given) - expected - set(cls.optional_hyperparameters))
        missing = sorted(expected - set(given))
        if unknown or missing:
            problems = []
            if missing:
                problems.append(f"does not set {', '.join(missing)}")
            if unknown:
                problems.append(f"sets {', '.join(unknown)}, which this estimator does not read")
            optional = (
                f", and optionally {', '.join(cls.optional_hyperparameters)}"
                if cls.optional_hyperparameters
                else ""
            )
            raise ModelError(
                f"the params block of variant {metadata['variant']!r} "
                f"{' and '.join(problems)}. {cls.estimator_name!r} reads exactly: "
                f"{', '.join(cls.hyperparameters)}{optional}."
            )
        return {**cls.optional_hyperparameters, **given}


# --------------------------------------------------------------------------
# B0: the median baseline
# --------------------------------------------------------------------------


class MedianBaselineModel(Model):
    """Ladder step 1: the median price of comparable cars, with a fallback chain.

    Median `price` per `(make, model, age bucket)`, falling back to the median
    per `make` and then to the global median. The whole model is one table a
    person can read in euros, which is what makes it the estimator whose
    arithmetic a test can check by hand.

    Fitted on `price` rather than on `log_price` because the median commutes with
    a monotone transform, so `exp(median(log price))` and `median(price)` are the
    same number and the readable one is worth storing. The prediction path takes
    the log of the stored median and `predict_eur` exponentiates it again, so what
    it returns equals the stored median to within floating-point rounding rather
    than bit-exactly - which is why the hand-computed test compares with a
    tolerance.

    The fallback chain is also what makes the baseline degrade gracefully under a
    partial request: a row whose key is absent - an unknown model, or no
    registration date, so no age - has no bucket entry and falls to the make
    median. That is `groupby` dropping null keys, not a special case in the
    prediction path.
    """

    estimator_name = "median_baseline"
    libraries = ("numpy", "pandas")
    hyperparameters = ("age_bucket_years",)

    def __init__(self, metadata: dict, space: FeatureSpace, lookup: pd.DataFrame) -> None:
        super().__init__(metadata, space)
        missing = [name for name in _B0_KEY_COLUMNS if name not in self.features]
        if missing:
            raise ModelError(
                f"{self.estimator_name!r} keys on {', '.join(_B0_KEY_COLUMNS)}, and the "
                f"{self.feature_set!r} feature set does not carry {', '.join(missing)}"
            )
        self.lookup = lookup
        # Through `_tuning`, so a loaded bundle's params block is checked on the
        # way in and not only at fit time: this is the one estimator that reads a
        # parameter at predict time, so a bundle carrying the wrong one would
        # silently key a different age band than it was fitted with.
        self._width = float(self._tuning(metadata)["age_bucket_years"])
        # Object keys on both sides of the merge and of the map. Parquet hands a
        # text column back as pandas 3's `str` dtype, and a `str` key does not
        # match an `object` key, so normalising here is what keeps a loaded model
        # predicting what the fitted one did instead of falling to the global
        # median for every row.
        buckets = lookup[lookup["level"] == "bucket"]
        self._buckets = buckets[["make", "model", "age_bucket", "price_eur"]].astype(
            {"make": "object", "model": "object"}
        )
        makes = lookup[lookup["level"] == "make"]
        self._makes = pd.Series(
            makes["price_eur"].to_numpy(dtype="float64"),
            index=pd.Index(makes["make"].astype("object")),
        )
        self._global = float(lookup.loc[lookup["level"] == "global", "price_eur"].iloc[0])

    @classmethod
    def fit(
        cls, metadata: dict, data: TrainingData, *, seed: int, num_threads: int
    ) -> "MedianBaselineModel":
        """Group the training prices. Neither the seed nor the threads are read.

        Signed the same way as every other estimator's `fit` so that
        `fit_variant` does not branch on the class, and taking neither argument
        seriously because a median consumes no randomness and no parallelism.
        """
        width = cls._tuning(metadata)["age_bucket_years"]
        keys = pd.DataFrame(
            {
                "make": data.train["make"].astype("object"),
                "model": data.train["model"].astype("object"),
                "age_bucket": _age_bucket(data.train["age_years"], width),
                "price_eur": data.train["price"].astype("float64"),
            }
        )
        # `groupby` drops a null key, which is exactly the rule this baseline
        # needs: a row that cannot key a level does not contribute to it.
        grouped = [
            keys.groupby(["make", "model", "age_bucket"], observed=True, dropna=True)["price_eur"]
            .agg(["median", "size"])
            .reset_index()
            .assign(level="bucket"),
            keys.groupby(["make"], observed=True, dropna=True)["price_eur"]
            .agg(["median", "size"])
            .reset_index()
            .assign(level="make", model=None, age_bucket=np.nan),
            pd.DataFrame(
                {
                    "level": ["global"],
                    "make": pd.array([None], dtype="object"),
                    "model": pd.array([None], dtype="object"),
                    "age_bucket": [np.nan],
                    "median": [float(keys["price_eur"].median())],
                    "size": [len(keys)],
                }
            ),
        ]
        lookup = pd.concat(grouped, ignore_index=True)[
            ["level", "make", "model", "age_bucket", "median", "size"]
        ].rename(columns={"median": "price_eur", "size": "n_rows"})
        return cls(metadata, data.space, _tidy_lookup(lookup))

    def _predict_log_price(self, aligned: pd.DataFrame) -> np.ndarray:
        keys = pd.DataFrame(
            {
                "make": aligned["make"].astype("object"),
                "model": aligned["model"].astype("object"),
                "age_bucket": _age_bucket(aligned["age_years"], self._width),
            }
        )
        merged = keys.merge(self._buckets, on=["make", "model", "age_bucket"], how="left")
        if len(merged) != len(keys):
            raise ModelError(
                f"the lookup table of {self.variant!r} holds duplicate keys, so a row matched "
                f"more than one median ({len(merged)} matches for {len(keys)} row(s))"
            )
        price = merged["price_eur"].to_numpy(dtype="float64")
        by_make = keys["make"].map(self._makes).to_numpy(dtype="float64")
        price = np.where(np.isnan(price), by_make, price)
        return np.log(np.where(np.isnan(price), self._global, price))

    def _save_payload(self, directory: Path) -> None:
        self.lookup.to_parquet(directory / _LOOKUP_FILE, index=False)

    @classmethod
    def _load_payload(
        cls, directory: Path, metadata: dict, space: FeatureSpace
    ) -> "MedianBaselineModel":
        return cls(metadata, space, pd.read_parquet(directory / _LOOKUP_FILE))


def _age_bucket(age_years: pd.Series, width: float) -> pd.Series:
    """`age_years` as the index of the `width`-year band it falls in.

    Null where the age is, so the bucket key misses and the prediction falls
    back to the make. Float rather than an integer dtype for exactly that
    reason, and because it is a merge key on both sides of the lookup.
    """
    return np.floor(pd.to_numeric(age_years, errors="coerce").astype("float64") / width)


def _tidy_lookup(lookup: pd.DataFrame) -> pd.DataFrame:
    """The lookup in one fixed order, with the dtypes Parquet will hand back.

    Sorted rather than left in `groupby` order so that two fits on the same rows
    write byte-identical bytes (NFR-06), and typed explicitly because the
    prediction path merges against these columns.
    """
    ordered = lookup.assign(_level=lookup["level"].map(_LOOKUP_LEVELS.index)).sort_values(
        ["_level", "make", "model", "age_bucket"], na_position="last", kind="stable"
    )
    return (
        ordered.drop(columns="_level")
        .astype(
            {
                "level": "str",
                "make": "str",
                "model": "str",
                "age_bucket": "float64",
                "price_eur": "float64",
                "n_rows": "int64",
            }
        )
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# B1: Ridge on log price
# --------------------------------------------------------------------------


def _missing_as_level(frame: pd.DataFrame) -> pd.DataFrame:
    """Categoricals as strings, with absence as a level under a name of its own.

    Not a workaround: `OneHotEncoder` has handled missing values since
    scikit-learn 1.1 and would fit without this, making them a category called
    `nan` - measured, for `category` dtype and for `object` holding `np.nan`,
    `None` or `pd.NA` alike. Two reasons to name the level anyway. B1 exists to
    be read, and `make___missing__` says what a coefficient is for where
    `make_nan` reads like a defect. And the encoding then does not rest on a
    library behaviour that arrived in a minor release and could change in
    another, which for the one estimator whose job is to be a stable reference is
    worth a five-line transformer.

    A module-level function rather than a lambda, because it is pickled with the
    fitted pipeline.
    """
    return frame.astype("object").where(frame.notna(), MISSING_LEVEL).astype("str")


class RidgeModel(Model):
    """Ladder step 2: the interpretable depreciation baseline, on log price.

    A `ColumnTransformer` in front of `Ridge`, fitted on `log_price`. Two
    encoding choices are explained here because both look, at a glance,
    like a rule of this project being broken.

    **One-hot, confined to B1.** EDN-02's "no one-hot" is a statement
    about the tree family and the reason it was chosen. Ridge is linear: it has
    no native categorical handling, and the alternatives were target encoding,
    which leaks the target without cross-fitting, or dropping the categoricals,
    which would leave the baseline unable to tell a Porsche from a Dacia and so
    destroy its purpose. `handle_unknown="infrequent_if_exist"` is what makes an
    unseen level an answer rather than an error (EDN-18): it sends the level to the
    infrequent group where `min_frequency` produced one, and encodes it as all
    zeros where it did not. On the real snapshot that is 2 of the 8 categoricals
    with a group and six without, so both branches are live and both answer.
    `min_category_rows` is an absolute row count rather than a share
    because a share changes what it means as the training set grows.

    **Mean fill plus a per-feature indicator, which is not imputation.**
    Ridge cannot consume NaN, so the numeric branch is a union of a
    mean-filled, scaled copy and `MissingIndicator(features="all")`. With an
    indicator for every feature the encoding is information-preserving: a masked
    row contributes `beta * mean + gamma`, and `gamma` absorbs the offset, so the
    fill is a numerically neutral placeholder rather than a guess at the value
    (EDN-15). `features="all"` is not the default: scikit-learn emits an indicator
    only for features that were missing *at fit time*, so a column that happens to
    be complete in the training split would be silently mean-filled with no
    indicator when SC-06 masks it, and the criterion would be measuring its own
    imputation. It is insurance rather than a measured fix as configured, because
    none of `basic`'s 8 numeric features is complete on the real snapshot - but 136
    of `extended`'s 147 are, so a Ridge on `extended` would lose 136 indicators to
    the default.
    """

    estimator_name = "ridge"
    libraries = ("numpy", "pandas", "scikit-learn", "joblib")
    hyperparameters = ("alpha", "min_category_rows")

    def __init__(self, metadata: dict, space: FeatureSpace, pipeline: Pipeline | None) -> None:
        super().__init__(metadata, space)
        self.pipeline = pipeline

    @classmethod
    def fit(
        cls, metadata: dict, data: TrainingData, *, seed: int, num_threads: int
    ) -> "RidgeModel":
        """Fit the pipeline. Neither the seed nor the threads are read.

        Nothing in the pipeline consumes randomness, and `solver="lsqr"` is
        single-threaded by construction, which is the point of choosing it.
        """
        tuning = cls._tuning(metadata)
        # Through an instance, because the column split and the alignment are
        # the model's own and have to be the same at fit and at predict time.
        model = cls(metadata, data.space, None)
        pipeline = _ridge_pipeline(
            categorical=list(model.categorical_columns),
            numeric=list(model.numeric_columns),
            alpha=tuning["alpha"],
            min_category_rows=tuning["min_category_rows"],
        )
        pipeline.fit(model._align(data.train), data.train["log_price"].to_numpy(dtype="float64"))
        model.pipeline = pipeline
        return model

    def _predict_log_price(self, aligned: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("the pipeline exists after `fit` or `load`")
        return np.asarray(self.pipeline.predict(aligned), dtype="float64")

    def _save_payload(self, directory: Path) -> None:
        # joblib rather than a hand-written format: the fitted state is the
        # category lists, the imputer means, the scaler statistics and the
        # infrequent-level groupings, and none of that is worth re-serialising.
        joblib.dump(self.pipeline, directory / _PIPELINE_FILE)

    @classmethod
    def _load_payload(cls, directory: Path, metadata: dict, space: FeatureSpace) -> "RidgeModel":
        return cls(metadata, space, joblib.load(directory / _PIPELINE_FILE))


def _ridge_pipeline(
    *, categorical: list[str], numeric: list[str], alpha: float, min_category_rows: int
) -> Pipeline:
    return Pipeline(
        [
            (
                "encode",
                ColumnTransformer(
                    [
                        (
                            "categorical",
                            make_pipeline(
                                FunctionTransformer(
                                    _missing_as_level, feature_names_out="one-to-one"
                                ),
                                OneHotEncoder(
                                    handle_unknown="infrequent_if_exist",
                                    min_frequency=min_category_rows,
                                    sparse_output=True,
                                ),
                            ),
                            categorical,
                        ),
                        (
                            "numeric",
                            FeatureUnion(
                                [
                                    (
                                        "filled",
                                        make_pipeline(
                                            # `keep_empty_features`, so an
                                            # all-missing column keeps its place
                                            # and the matrix shape does not
                                            # depend on which columns the
                                            # training rows happened to fill.
                                            SimpleImputer(
                                                strategy="mean", keep_empty_features=True
                                            ),
                                            StandardScaler(),
                                        ),
                                    ),
                                    ("missing", MissingIndicator(features="all")),
                                ]
                            ),
                            numeric,
                        ),
                    ],
                    remainder="drop",
                ),
            ),
            # `lsqr`, not the default `auto`: `auto` picks a solver by the shape
            # and sparsity of the matrix, and a dense one goes through BLAS,
            # whose reduction order depends on the thread count. scipy's LSQR is
            # iterative and single-threaded, so nothing about it can vary with
            # the machine (NFR-06).
            #
            # Insurance documented by the libraries rather than a measured fix,
            # and worth being exact about: the coefficients came out
            # byte-identical across separate processes at 1 and 8 BLAS threads
            # with `auto` as well, on this data. Keeping `lsqr` costs nothing and
            # removes the dependency; it was not shown to be necessary here.
            ("ridge", Ridge(alpha=alpha, solver="lsqr")),
        ]
    )


# --------------------------------------------------------------------------
# Ladder steps 3 and 4: LightGBM
# --------------------------------------------------------------------------


class LightGBMModel(Model):
    """The main model: gradient-boosted trees on log price, native categoricals.

    Categoricals reach LightGBM as the integer codes of the contract's levels,
    with `categorical_feature` naming them, rather than as pandas `category`
    columns. Both routes work, and the explicit one is chosen because LightGBM's
    own `pandas_categorical` round trip is a second mapping between the level
    list and the codes: with the codes computed here from the contract, a one-row
    request is numbered by exactly the code table the training frame was. A
    negative code is LightGBM's missing value, and `Schema.conform` produces one
    for both an absent and an unseen level (EDN-18).

    The settings that are not in params.yaml are modelling choices rather than
    knobs a sweep should touch. `objective="regression"` is L2 on log price,
    because the log transform already handles the multiplicative error structure.
    `metric="l1"` for early stopping, because the absolute error in log space
    *is* the symmetric relative error, so the stopping point lines up with MdAPE,
    the metric the gate reads, instead of with a squared error nothing reports.
    """

    estimator_name = "lightgbm"
    libraries = ("numpy", "pandas", "lightgbm")
    hyperparameters = ("learning_rate", "num_leaves", "n_estimators", "early_stopping_rounds")
    #: `min_child_samples`, the fewest training rows a leaf may hold, is a knob of
    #: the search of issue #88 (EDN-73), because many `model` values occur only
    #: once and a leaf that small can fit a single listing. Optional, at
    #: LightGBM's own default of 20, so a params block that leaves it out fits
    #: exactly the trees it fitted before the key existed - `booster.txt` records
    #: `min_data_in_leaf: 20` either way - and `params.yaml` names it only once a
    #: search has adopted another value. `reg_lambda`, the L2 penalty that shrinks
    #: a leaf of n rows by about n / (n + lambda), is optional the same way at
    #: LightGBM's 0.0 for the search of issue #117. `booster.txt` records it as
    #: `lambda_l2`, an alias never set here, because LightGBM warns when both are.
    optional_hyperparameters = MappingProxyType({"min_child_samples": 20, "reg_lambda": 0.0})
    #: The estimator the copies were measured on (EDN-82).
    supports_blanked_copies = True

    def __init__(
        self,
        metadata: dict,
        space: FeatureSpace,
        booster: lightgbm.Booster | None,
        intervals: Mapping[str, lightgbm.Booster] | None = None,
    ):
        super().__init__(metadata, space)
        self.booster = booster
        #: The three quantile boosters of UC2's price range, by `INTERVAL_QUANTILES`
        #: role, or `None` for a variant that names no `intervals` (EDN-85).
        self.intervals = None if intervals is None else dict(intervals)
        #: The OpenMP threads each prediction may use; LightGBM's `0` is one per
        #: core. `booster.txt` records the training's `num_threads`, but
        #: `Booster.predict` reads only its own arguments, so the setting lives
        #: here. Batch scoring keeps the default; the API sets 1
        #: (`recommenditos.api.serving.PREDICTION_THREADS`).
        self.prediction_threads = 0

    @staticmethod
    def regressor(
        tuning: Mapping, *, seed: int, num_threads: int, **objective
    ) -> lightgbm.LGBMRegressor:
        """The estimator every LightGBM fit here starts from, with `objective` on top.

        One constructor for the point model and the quantile models, so the
        settings below hold for all of them and the quantile models differ only
        in what they are fitted to minimise.
        """
        return lightgbm.LGBMRegressor(
            **objective,
            first_metric_only=True,
            # `deterministic` and `force_row_wise` together stop LightGBM from
            # choosing its histogram construction by data size and thread count.
            # Documented insurance rather than a measured fix, and worth being
            # exact about: with both flags off the trees are still identical
            # across 1, 2, 4 and 8 threads, so they are not what makes the result
            # thread-independent here. They do change which trees are built, so
            # they are not free of consequence; they are kept because they are
            # what LightGBM documents for this purpose and what would hold if
            # `num_threads` were raised on data where it does matter.
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
            random_state=seed,
            # Never left at `-1`: the thread count is written into `booster.txt`,
            # so a default would make the artefact depend on the core count of
            # whoever ran `dvc repro` (NFR-06).
            n_jobs=num_threads,
            **tuning,
        )

    @classmethod
    def fit(
        cls, metadata: dict, data: TrainingData, *, seed: int, num_threads: int
    ) -> "LightGBMModel":
        tuning = cls._tuning(metadata)
        rounds = tuning.pop("early_stopping_rounds")
        model = cls(metadata, data.space, None)
        estimator = cls.regressor(
            tuning, seed=seed, num_threads=num_threads, objective="regression", metric="l1"
        )
        # `eval_X`/`eval_y` rather than `eval_set`, which lightgbm 4.7 deprecates.
        estimator.fit(
            model._as_codes(model._align(data.train)),
            data.train["log_price"].to_numpy(dtype="float64"),
            eval_X=model._as_codes(model._align(data.validation)),
            eval_y=data.validation["log_price"].to_numpy(dtype="float64"),
            eval_names=[_EVAL_NAME],
            categorical_feature=list(model.categorical_columns),
            callbacks=[lightgbm.early_stopping(rounds, first_metric_only=True, verbose=False)],
        )
        # One score per round the fit ran, because the curve is recorded for every
        # round and only the booster is truncated.
        boosting_rounds = len(estimator.evals_result_[_EVAL_NAME]["l1"])
        # `best_iteration_` is the best round *inside the budget*, whether or not
        # early stopping fired: lightgbm 4.7's callback reports it on the last
        # round too. So it cannot say which of the two ended the fit, and a curve
        # that wobbled near the end of a binding budget records a few trees less
        # than the budget, which reads exactly like early stopping. It is 0 only
        # when the callback is disabled (`early_stopping_rounds: 0`), and then
        # `save_model(num_iteration=0)` would write a booster with no trees.
        best_iteration = int(estimator.best_iteration_ or estimator.booster_.num_trees())
        # Fired means the fit ran the full patience past its best round. Not
        # "stopped short of `n_estimators`": the patience can run out on the very
        # last round of the budget, and that fit was ended by early stopping.
        early_stopped = rounds > 0 and boosting_rounds - best_iteration >= rounds
        metadata["training"].update(
            best_iteration=best_iteration,
            boosting_rounds=boosting_rounds,
            early_stopped=early_stopped,
        )
        # Only with early stopping on: with `early_stopping_rounds: 0` the budget is
        # the tree count by design, and there is no patience to report.
        if rounds > 0 and not early_stopped:
            logger.warning(
                f"{metadata['variant']}: the n_estimators budget of {tuning['n_estimators']:,} "
                f"ended the fit, not early stopping. The validation L1 was still at its best "
                f"within the last {rounds} round(s), so the model kept {best_iteration:,} "
                f"trees and was still improving: raise n_estimators until early stopping "
                f"decides it."
            )
        model.booster = estimator.booster_
        return model

    def log_price_contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Each row's log price split into one SHAP value per feature (FR-08, EDN-11).

        LightGBM's own TreeSHAP export, `predict(pred_contrib=True)`, rather than
        the `shap` package, which stays out of the serving runtime. One column
        per feature, in `features` order, and `BASE_LOG_PRICE`, the booster's
        mean log price over its training rows; a categorical is one column, as it
        reaches the booster as one native categorical feature. A row sums to the
        booster's log price before `predict_eur` bounds it to the training range.
        """
        if self.booster is None:
            raise RuntimeError("the booster exists after `fit` or `load`")
        contributions = self.booster.predict(
            self._as_codes(self._align(frame)),
            pred_contrib=True,
            num_threads=self.prediction_threads,
        )
        return pd.DataFrame(
            np.asarray(contributions, dtype="float64"),
            index=frame.index,
            columns=[*self.features, BASE_LOG_PRICE],
        )

    def _as_codes(self, aligned: pd.DataFrame) -> pd.DataFrame:
        coded = aligned.copy()
        for name in self.categorical_columns:
            coded[name] = coded[name].cat.codes.astype("int32")
        return coded

    def _predict_log_price(self, aligned: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("the booster exists after `fit` or `load`")
        return np.asarray(
            self.booster.predict(self._as_codes(aligned), num_threads=self.prediction_threads),
            dtype="float64",
        )

    @property
    def has_intervals(self) -> bool:
        record = self.metadata.get("intervals") or {}
        return self.intervals is not None and record.get("offsets") is not None

    def predict_interval_eur(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The lower booster minus its offset and the upper one plus its, in euros.

        This is MAPIE's conformalized quantile regression at prediction time
        (`ConformalizedQuantileRegressor.predict_interval` with its default
        asymmetric correction), written out so that serving does not need MAPIE:
        both offsets are constants of the calibration, computed by MAPIE in
        `intervals.py` and stored in `model.json`. A test checks the two agree on
        the same rows. Nothing is clipped or reordered, for the same reason: the
        served bounds are MAPIE's bounds, and a bound clipped to the training
        price range would change the coverage SC-05 measured.

        `frame` goes through `_align`, like `predict_eur`, so a request with
        fields left out is encoded the way the interval models' training and
        calibration copies were (`Model.mask_absent`).
        """
        boosters = self.intervals
        if boosters is None or not self.has_intervals:
            return super().predict_interval_eur(frame)
        aligned = self._align(frame)
        if aligned.empty:
            return pd.DataFrame(
                {column: np.empty(0, dtype="float64") for column in INTERVAL_COLUMNS.values()},
                index=frame.index,
            )
        coded = self._as_codes(aligned)
        log_price = {
            name: np.asarray(
                booster.predict(coded, num_threads=self.prediction_threads), dtype="float64"
            )
            for name, booster in boosters.items()
        }
        offsets = self.metadata["intervals"]["offsets"]
        log_price["lower"] = log_price["lower"] - offsets["lower"]
        log_price["upper"] = log_price["upper"] + offsets["upper"]
        return pd.DataFrame(
            {INTERVAL_COLUMNS[name]: np.exp(log_price[name]) for name in INTERVAL_QUANTILES},
            index=frame.index,
        )

    def attach_intervals(self, boosters: Mapping[str, lightgbm.Booster], record: dict) -> None:
        """Make the price-range boosters and their calibration record part of the bundle."""
        missing = sorted(set(INTERVAL_QUANTILES) - set(boosters))
        if missing:
            raise ModelError(
                f"the price range of {self.variant!r} needs the {', '.join(INTERVAL_QUANTILES)} "
                f"boosters and was given none for {', '.join(missing)}."
            )
        self.intervals = {name: boosters[name] for name in INTERVAL_QUANTILES}
        self.metadata["intervals"] = record

    def _save_payload(self, directory: Path) -> None:
        # LightGBM's own text format, not a pickle of the sklearn wrapper: it
        # survives a LightGBM upgrade, it is readable, and EDN-11's SHAP export
        # needs a `Booster` rather than the wrapper.
        #
        # `num_iteration` is stated rather than left to default, and it is
        # currently inert: lightgbm 4.7 already truncates the booster when early
        # stopping fires, so the object holds exactly `best_iteration` trees and
        # saving it with any bound writes the same file. Stated anyway, because
        # the property the API depends on is that the file *is* the early-stopped
        # model and needs no iteration argument at predict time, and that should
        # not rest on a truncation nothing here asked for.
        if self.booster is None:
            raise RuntimeError("the booster exists after `fit` or `load`")
        self.booster.save_model(
            directory / _BOOSTER_FILE, num_iteration=self.metadata["training"]["best_iteration"]
        )
        if self.intervals is not None:
            iterations = self.metadata["intervals"]["best_iteration"]
            for name, booster in self.intervals.items():
                booster.save_model(
                    directory / _INTERVAL_FILES[name], num_iteration=iterations[name]
                )

    @classmethod
    def _load_payload(
        cls, directory: Path, metadata: dict, space: FeatureSpace
    ) -> "LightGBMModel":
        # A bundle without an `intervals` record is a point model only, as every
        # bundle was before EDN-85, so `/predict` loads either.
        intervals = (
            {
                name: lightgbm.Booster(model_file=str(directory / _INTERVAL_FILES[name]))
                for name in INTERVAL_QUANTILES
            }
            if metadata.get("intervals")
            else None
        )
        return cls(
            metadata,
            space,
            lightgbm.Booster(model_file=str(directory / _BOOSTER_FILE)),
            intervals,
        )


# --------------------------------------------------------------------------
# What an input field is, for a model and a request contract
#
# Here rather than in `evaluate.py` because two stages ask: `evaluate` masks the
# optional fields for SC-06, and a fit blanks them in the copies it adds
# (EDN-82). One derivation, so the two cannot disagree about what is optional.
# --------------------------------------------------------------------------


def input_field_columns(model: Model, request: RequestContract) -> dict[str, tuple[str, ...]]:
    """Every API input field of this model's feature set, and the columns it produced.

    SC-06 masks *input fields*, not feature columns, because FR-01 is a statement
    about a request: one equipment field becomes many multi-hot columns, so a
    column-wise sweep would mask a single equipment item, which is not a request
    the API can make; and `registration_date` becomes the derived `age_years`, so
    a naive set difference against the required fields would make the most
    important feature in the model look optional.

    This function answers only *which fields exist*. What each one expands to is
    `Model.columns_of`, so there is one derivation of "which columns did masking
    this field mask" rather than two that can disagree. Which columns an equipment
    field produced is what the training rows decided, so neither half can come
    from params.yaml.
    """
    derived = dict(request.field_to_feature)
    conflicting = sorted(set(derived) & set(model.features))
    if conflicting:
        raise ValueError(
            f"request.input_field_to_feature maps {', '.join(conflicting)} to another column "
            f"while {model.variant!r} also consumes a column of that name. One input field "
            f"would then claim two unrelated columns and masking it would mask both."
        )
    source_of = {
        equipment_feature_name(source, item): source
        for source, items in model.space.vocabulary.equipment.items()
        for item in items
    }
    field_of_derived = {feature: field for field, feature in derived.items()}

    fields: list[str] = []
    for column in model.features:
        claims = {
            claim
            for claim in (source_of.get(column), field_of_derived.get(column))
            if claim is not None
        }
        if len(claims) > 1:
            raise ValueError(
                f"column {column!r} is claimed by two input fields ({', '.join(sorted(claims))}), "
                f"so masking either would mask a column the other owns. "
                f"request.input_field_to_feature names a column an equipment field produced."
            )
        field = claims.pop() if claims else column
        if field not in fields:
            fields.append(field)

    absent = [field for field in request.required_fields if field not in fields]
    if absent:
        raise ValueError(
            f"{model.variant!r} consumes no column for the required input field(s) "
            f"{', '.join(absent)}. FR-01 refuses a request that omits one, so the required "
            f"list and the trained model disagree about what a valuation needs; either the "
            f"feature set or request.required_input_fields is wrong."
        )
    # The seam expands a field into its columns; this function only renames the
    # one field whose column has a different name first. `columns_of` knows the
    # equipment vocabulary and nothing about the rename, so asking it for
    # `registration_date` raises - the model consumes `age_years`. The rename is
    # the contract's business and the expansion is the model's, and keeping them
    # in that order is what stops this from becoming a second expansion.
    return {field: model.columns_of(derived.get(field, field)) for field in fields}


def optional_input_fields(
    fields: Mapping[str, tuple[str, ...]], request: RequestContract
) -> tuple[str, ...]:
    """The input fields FR-01 leaves optional, which is what SC-06 masks (EDN-15)."""
    return tuple(sorted(set(fields) - set(request.required_fields)))


def seam_names(fields: Mapping[str, tuple[str, ...]], request: RequestContract) -> dict[str, str]:
    """Each request field under the name `Model.columns_of` knows it by.

    The same rename `input_field_columns` applies, in the same direction and from
    the same contract: the seam knows the equipment vocabulary and not the
    request, so it cannot answer for `registration_date`.
    """
    return {field: request.field_to_feature.get(field, field) for field in fields}


def optional_seam_fields(model: Model, request: RequestContract) -> tuple[str, ...]:
    """Every optional input field of `model`, as `Model.mask_absent` takes them.

    Omitting all of them is the request that carries only what FR-01 requires:
    SC-06's all-at-once scenario, and the `all` copy of a training row.
    """
    fields = input_field_columns(model, request)
    seam = seam_names(fields, request)
    return tuple(seam[field] for field in optional_input_fields(fields, request))


# --------------------------------------------------------------------------
# Blanked copies of the training rows (EDN-82)
# --------------------------------------------------------------------------

#: The keys of a variant's `blanked_copies` block: how many copies of every
#: training row the fit adds with all optional fields omitted, and how many
#: with a random subset omitted. A key left out is no copy of that kind.
BLANKED_COPY_KINDS: tuple[str, ...] = ("all", "random")

#: The stream the random copies are drawn from, beside the project seed, so that
#: they do not share draws with anything else the seed feeds. It is the number
#: of the issue that measured the copies, and it is what makes the copies of a
#: pipeline fit the ones that measurement fitted on.
_BLANKING_STREAM = 116


def with_blanked_copies(
    seam: Model,
    frame: pd.DataFrame,
    fields: Sequence[str],
    *,
    copies: Mapping[str, int],
    seed: int,
) -> pd.DataFrame:
    """`frame`, followed by copies of its rows with some of `fields` omitted.

    `copies` is the number of copies per kind, as `blanked_copy_counts` reads a
    variant's block: `all` and `random`, and a kind left out is none.

    Why a fit wants them: every optional field is sometimes empty in the
    listings, but almost no listing leaves all of them empty at once, so a
    request carrying only the fields FR-01 requires lies outside what the model
    saw, and its error there is the model's own doing rather than the price of
    the missing information (EDN-82). A copy shows the model that request.

    An `all` copy omits every field, which is exactly that request. A `random`
    copy omits a random subset: its size is drawn uniformly from one to all of
    the fields and the fields at random, so every level of sparsity is covered
    equally. Each copy keeps every other column of its row, the target included.

    "Omitted" is `Model.mask_absent`'s meaning of it, cell by cell, so a copy
    is a row the API could be sent. The rows of `frame` come first and are not
    changed; the index of the result is a fresh range.

    The draws come from the project seed and the point model's own stream; the
    interval models draw theirs through `with_blanked_copies_drawn`.
    """
    return with_blanked_copies_drawn(
        seam, frame, fields, copies=copies, rng=np.random.default_rng([seed, _BLANKING_STREAM])
    )


def with_blanked_copies_drawn(
    seam: Model,
    frame: pd.DataFrame,
    fields: Sequence[str],
    *,
    copies: Mapping[str, int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """`with_blanked_copies` with the random draws taken from `rng`.

    For the copies of the interval models (EDN-85), which draw their training,
    validation and calibration copies from streams of their own
    (`INTERVAL_STREAMS`), so that no two sets of copies share a draw.
    """
    fields = tuple(fields)
    n_rows, n_fields = len(frame), len(fields)
    frames = [frame]
    for _ in range(copies.get("all", 0)):
        frames.append(seam.mask_absent(frame, fields))
    for _ in range(copies.get("random", 0)):
        sizes = rng.integers(1, n_fields + 1, size=n_rows)
        # The rank of a row's uniform draws is a random order of its fields, so
        # "rank below the size" picks exactly `size` of them.
        rank = rng.random((n_rows, n_fields)).argsort(axis=1).argsort(axis=1)
        frames.append(seam.mask_absent(frame, fields, where=rank < sizes[:, None]))
    if len(frames) == 1:
        return frame
    return pd.concat(frames, ignore_index=True)


def blanked_copy_counts(variant: str, settings: Mapping) -> dict[str, int]:
    """The `blanked_copies` block of a variant as a count per kind, zero where unnamed.

    A variant that does not name the block adds no copy, so its fit is the one
    it was before the block existed. A key the fit does not know, and a count
    that is not a whole number of copies, are refused rather than read as zero:
    a copy somebody asked for and nobody added would otherwise be found in a
    metric, if at all.
    """
    block = settings.get("blanked_copies", {})
    counts = dict.fromkeys(BLANKED_COPY_KINDS, 0)
    problems = []
    if not isinstance(block, Mapping):
        problems.append(f"is {block!r}, not a mapping")
    else:
        unknown = sorted(set(block) - set(BLANKED_COPY_KINDS), key=str)
        if unknown:
            problems.append(f"sets {', '.join(map(str, unknown))}, which no fit reads")
        for kind in BLANKED_COPY_KINDS:
            count = block.get(kind, 0)
            # `bool` is an `int` in Python, and `all: true` is not a count.
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                problems.append(f"sets {kind} to {count!r}, which is not a whole number of copies")
            else:
                counts[kind] = count
    if problems:
        raise ModelError(
            f"the blanked_copies block of variant {variant!r} {' and '.join(problems)}. It "
            f"reads exactly: {', '.join(BLANKED_COPY_KINDS)}, each the number of copies of "
            f"every training row the fit adds (0 or more)."
        )
    return counts


# --------------------------------------------------------------------------
# Fitting and loading a variant
# --------------------------------------------------------------------------

#: Every estimator params.yaml may name, by the value of its `estimator` key.
ESTIMATORS: dict[str, type[Model]] = {
    cls.estimator_name: cls for cls in (MedianBaselineModel, RidgeModel, LightGBMModel)
}


def fit_variant(
    variant: str, settings: dict, data: TrainingData, *, seed: int, num_threads: int
) -> Model:
    """Fit `variant` as params.yaml describes it. The only compute worth measuring.

    Everything else about the stage - reading the matrices, opening the MLflow
    run, writing the bundle - happens around this call, so issue #38 can wrap
    exactly this with an `EmissionsTracker` and measure the fit alone.
    """
    estimator = settings["estimator"]
    if estimator not in ESTIMATORS:
        raise ModelError(
            f"variant {variant!r} names estimator {estimator!r}, which this module does not "
            f"implement. Known estimators: {', '.join(sorted(ESTIMATORS))}."
        )
    cls = ESTIMATORS[estimator]
    schema = data.space.schema
    copies = blanked_copy_counts(variant, settings)
    metadata = {
        "variant": variant,
        "estimator": estimator,
        "feature_set": settings["feature_set"],
        # Two lists, never one: the matrix carries the label beside the inputs,
        # so a consumer that took a single list at its word would fit the target
        # on itself.
        "features": list(schema.feature_names),
        "targets": list(schema.target_names),
        "params": dict(settings["params"]),
        # The copies of its training rows the fit added, and the fields it
        # omitted in them: none and no field for a variant that names no
        # `blanked_copies` (EDN-82). Filled below, once the fields are known.
        "blanked_copies": {**copies, "fields": []},
        "seed": seed,
        "num_threads": num_threads,
        "training": {
            # The listings the variant was fitted on, and the rows its estimator
            # saw, which are the same number unless the fit added copies.
            "n_train_rows": len(data.train),
            "n_fit_rows": len(data.train),
            "n_validation_rows": len(data.validation),
            # The bound `predict_eur` applies, in the unit a reader thinks in.
            "price_min_eur": float(data.train["price"].min()),
            "price_max_eur": float(data.train["price"].max()),
            # Filled after the fit, below. Declared here so that the record has
            # the same shape for every variant and a reader does not have to know
            # which estimator writes which key.
            "train_l1_log_price": None,
            "validation_l1_log_price": None,
            # Written by the estimators that boost, and null for the two that do
            # not: the trees the model kept, the rounds the fit ran, and whether
            # early stopping or the `n_estimators` budget ended it.
            "best_iteration": None,
            "boosting_rounds": None,
            "early_stopped": None,
            # FR-04's scope, carried in the metadata because the API answers the
            # scope check from the model it serves rather than from a file it has
            # to be pointed at separately.
            "supported_makes": list(data.supported_makes),
        },
        "versions": _versions(*cls.libraries),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    model = cls.fit(
        metadata,
        _with_blanked_copies(cls, data, metadata, copies=copies, seed=seed),
        seed=seed,
        num_threads=num_threads,
    )
    # Measured on the model as it will be served, bound included, rather than
    # read out of the estimator's own fitting log. Only LightGBM keeps such a
    # log, so recomputing is what makes the number mean the same thing for all
    # four variants. On the listings as they are, also for a fit that added
    # copies of them, so the number stays comparable between the two.
    metadata["training"]["train_l1_log_price"] = _l1_log_price(model, data.train)
    metadata["training"]["validation_l1_log_price"] = _l1_log_price(model, data.validation)
    return model


def _with_blanked_copies(
    cls: type[Model], data: TrainingData, metadata: dict, *, copies: Mapping[str, int], seed: int
) -> TrainingData:
    """`data` with the copies the variant asks for appended to its training rows.

    The validation rows are never copied: early stopping watches them as they
    are, and a copy of a training row is a training row, so no seller crosses
    the split (EDN-14). The fields a copy omits and the rows the estimator
    sees are written into `metadata`, which is the bundle's record.
    """
    if not any(copies.values()):
        return data
    variant = metadata["variant"]
    if not cls.supports_blanked_copies:
        supporting = sorted(
            name for name, each in ESTIMATORS.items() if each.supports_blanked_copies
        )
        raise ModelError(
            f"variant {variant!r} names blanked_copies, and its estimator "
            f"{cls.estimator_name!r} does not take them: it would count every copy of a "
            f"listing as a listing of its own. Estimators that do: {', '.join(supporting)}."
        )
    if data.request is None:
        raise ModelError(
            f"variant {variant!r} adds blanked copies of its training rows, and which fields "
            f"a copy may omit is the request contract's to say. The fit was not given one: "
            f"pass `request=RequestContract.from_params(params)` to `train.read_matrices`, "
            f"or set `TrainingData.request`."
        )
    # A handle on the seam before any estimator exists: `mask_absent` and
    # `columns_of` read the feature space alone.
    seam = Model(metadata, data.space)
    fields = optional_seam_fields(seam, data.request)
    if not fields:
        raise ModelError(
            f"variant {variant!r} adds blanked copies of its training rows, and its feature "
            f"set {metadata['feature_set']!r} has no optional input field to omit in them."
        )
    train = with_blanked_copies(seam, data.train, fields, copies=copies, seed=seed)
    metadata["blanked_copies"]["fields"] = list(fields)
    metadata["training"]["n_fit_rows"] = len(train)
    return replace(data, train=train)


# --------------------------------------------------------------------------
# The quantile models of UC2's price range (EDN-85)
#
# Fitted here with LightGBM alone, so that serving and its check
# (`tests/test_serving.py`) can build and read them without a pipeline package.
# Their calibration needs MAPIE and lives in `intervals.py`, which only the
# `train` stage imports.
# --------------------------------------------------------------------------

#: The keys of a variant's `intervals` block.
INTERVAL_KEYS: tuple[str, ...] = ("confidence_level", "blanked_copies")

#: The streams the interval models' copies are drawn from, one per split, so
#: that the training, validation and calibration copies share no draw with each
#: other or with the point model's copies.
INTERVAL_STREAMS: Mapping[str, int] = MappingProxyType(
    {"train": 1371, "validation": 1372, "calibration": 1373}
)


def interval_settings(variant: str, settings: Mapping) -> dict | None:
    """A variant's `intervals` block, checked, or `None` when it names none.

    `confidence_level` is the nominal coverage of the range (0.9 for SC-05), and
    `blanked_copies` reads like the point model's block: how many copies of each
    training, validation and calibration row carry every field but the ones
    FR-02 requires left out (`all`) and how many a random subset (`random`).
    """
    block = settings.get("intervals")
    if block is None:
        return None
    problems = []
    if not isinstance(block, Mapping):
        raise ModelError(f"the intervals block of variant {variant!r} is not a mapping: {block!r}")
    unknown = sorted(set(block) - set(INTERVAL_KEYS), key=str)
    missing = sorted(set(INTERVAL_KEYS) - set(block))
    if unknown:
        problems.append(f"sets {', '.join(map(str, unknown))}, which no fit reads")
    if missing:
        problems.append(f"does not set {', '.join(missing)}")
    level = block.get("confidence_level")
    if "confidence_level" in block and not (isinstance(level, float) and 0.0 < level < 1.0):
        problems.append(f"sets confidence_level to {level!r}, which is not a share in (0, 1)")
    if settings.get("estimator") != LightGBMModel.estimator_name:
        problems.append(
            f"belongs to estimator {settings.get('estimator')!r}, and only "
            f"{LightGBMModel.estimator_name!r} fits the quantile models"
        )
    if problems:
        raise ModelError(
            f"the intervals block of variant {variant!r} {' and '.join(problems)}. It reads "
            f"exactly: {', '.join(INTERVAL_KEYS)}."
        )
    copies = blanked_copy_counts(variant, {"blanked_copies": block["blanked_copies"]})
    return {"confidence_level": level, "blanked_copies": copies}


def interval_quantiles(confidence_level: float) -> dict[str, float]:
    """The quantile each booster is fitted at, in the order MAPIE requires them."""
    return {
        "lower": (1 - confidence_level) / 2,
        "upper": (1 + confidence_level) / 2,
        "median": 0.5,
    }


def price_range_seam_fields(model: Model, request: RequestContract) -> tuple[str, ...]:
    """Every input field a `/price-range` request may leave out, as `mask_absent` takes them.

    All fields of the feature set except the ones FR-02 requires, which is `make`
    alone: what the interval models' `all` copies omit, and what P1 and a
    make-only request are drawn from.
    """
    fields = input_field_columns(model, request)
    unknown = sorted(set(request.price_range_required_fields) - set(fields))
    if unknown or not request.price_range_required_fields:
        raise ModelError(
            f"request.price_range_required_input_fields is "
            f"{list(request.price_range_required_fields)}, and {model.variant!r} consumes "
            f"{', '.join(fields)}. FR-02 requires `make`, so the list names at least one field "
            f"and only fields of the feature set."
        )
    seam = seam_names(fields, request)
    required = set(request.price_range_required_fields)
    return tuple(seam[field] for field in fields if field not in required)


def fit_quantile_estimators(
    model: LightGBMModel,
    data: TrainingData,
    chosen: Mapping,
    *,
    seed: int,
    num_threads: int,
) -> tuple[dict[str, lightgbm.LGBMRegressor], dict]:
    """The three quantile models of the price range, and the record of their fit.

    `chosen` is the variant's `intervals` block as `interval_settings` reads it.

    Fitted on the training rows plus their copies with fields left out, and
    early-stopped on the validation rows plus copies drawn the same way, because
    the validation rows have to look like the requests the models are for. The
    hyperparameters are the point model's, and only the objective differs: the
    pinball loss at each quantile, which is also what early stopping watches.

    The calibration record (`offsets`) stays empty: MAPIE fills it in
    `intervals.py`, and until it does the bundle `has_intervals` is false.
    """
    if data.request is None:
        raise ModelError(
            f"the price range of {model.variant!r} omits the fields FR-02 does not require, "
            f"and which those are is the request contract's to say. Set `TrainingData.request`."
        )
    fields = price_range_seam_fields(model, data.request)
    copies = chosen["blanked_copies"]
    confidence_level = chosen["confidence_level"]
    train, validation = (
        with_blanked_copies_drawn(
            model,
            frame,
            fields,
            copies=copies,
            rng=np.random.default_rng([seed, INTERVAL_STREAMS[split]]),
        )
        for split, frame in (("train", data.train), ("validation", data.validation))
    )
    tuning = LightGBMModel._tuning(model.metadata)
    rounds = tuning.pop("early_stopping_rounds")
    quantiles = interval_quantiles(confidence_level)
    train_x = model._as_codes(model._align(train))
    validation_x = model._as_codes(model._align(validation))
    estimators = {}
    for name, alpha in quantiles.items():
        estimator = LightGBMModel.regressor(
            tuning,
            seed=seed,
            num_threads=num_threads,
            objective="quantile",
            alpha=alpha,
            metric="quantile",
        )
        estimator.fit(
            train_x,
            train["log_price"].to_numpy(dtype="float64"),
            eval_X=validation_x,
            eval_y=validation["log_price"].to_numpy(dtype="float64"),
            eval_names=[_EVAL_NAME],
            categorical_feature=list(model.categorical_columns),
            callbacks=[lightgbm.early_stopping(rounds, first_metric_only=True, verbose=False)],
        )
        estimators[name] = estimator
    record = {
        "confidence_level": confidence_level,
        "quantiles": quantiles,
        "required_fields": list(data.request.price_range_required_fields),
        "blanked_copies": {**copies, "fields": list(fields)},
        "n_fit_rows": len(train),
        "n_validation_rows": len(validation),
        "best_iteration": {
            name: int(each.best_iteration_ or each.booster_.num_trees())
            for name, each in estimators.items()
        },
        # Filled by `intervals.conformalize`.
        "offsets": None,
        "calibration": None,
    }
    return estimators, record


def _l1_log_price(model: Model, frame: pd.DataFrame) -> float:
    """The mean absolute error in log space, which is the metric fitting reads.

    In log space rather than in euros on purpose: `train` must not put a second
    implementation of MdAPE next to `evaluate`'s, because two numbers in one
    MLflow run that can disagree are worse than one number.
    """
    return float((model.predict_log_price(frame) - frame["log_price"]).abs().mean())


def load_model(path: Path) -> Model:
    """The model in `path`, or a failure naming what is wrong with the bundle.

    `path` is a `models/<variant>` directory. There is no default and no
    fallback: a caller that does not know where its model is has a bug, and
    silently loading some other variant would produce metrics for the wrong one.
    """
    record = path / MODEL_FILE
    if not record.exists():
        raise ModelError(
            f"{record} does not exist, so {path} is not a model bundle. `dvc repro train` "
            f"writes one directory per variant of params.yaml's train.variants."
        )
    try:
        metadata = json.loads(record.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ModelError(f"{record} is not readable JSON ({error})") from error

    estimator = metadata.get("estimator")
    if estimator not in ESTIMATORS:
        raise ModelError(
            f"{record} was written by estimator {estimator!r}, which this module does not "
            f"implement. Known estimators: {', '.join(sorted(ESTIMATORS))}."
        )
    space = FeatureSpace.load(path, name=f"features-{metadata['feature_set']}")
    model = ESTIMATORS[estimator]._load_payload(path, metadata, space)
    _warn_about_library_versions(model)
    return model


def _warn_about_library_versions(model: Model) -> None:
    """Say so when a payload was written by another version of its own library.

    A warning rather than a failure: a pickled scikit-learn estimator usually
    loads across a minor release, and refusing would make a routine upgrade stop
    the pipeline. Silence would be worse than either, because the symptom of a
    payload read by the wrong version is a changed number, not an error.
    """
    recorded = model.metadata.get("versions", {})
    running = _versions(*type(model).libraries)
    moved = {
        name: (was, running[name])
        for name, was in recorded.items()
        if name in running and running[name] != was
    }
    if moved:
        listed = ", ".join(f"{name} {was} -> {now}" for name, (was, now) in sorted(moved.items()))
        logger.warning(
            f"{model.variant!r} was trained with {listed}. Its predictions may differ from "
            f"the metrics recorded for it; retrain to make the two comparable."
        )
