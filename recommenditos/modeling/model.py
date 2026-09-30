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

`predict_interval_eur` is deliberately **absent**: SC-05's conformalized
quantile intervals are UC2 work, the `calibration` split is already held for
them, and `evaluate` reports SC-05 as null precisely because this attribute does
not exist. Adding it here as a stub that raises would make that check pass and
the criterion look measured.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version as distribution_version
import json
from pathlib import Path
import platform

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

from recommenditos.data.build_features import FeatureSpace
from recommenditos.schema import FEATURE_SPACE_FILE, Schema

#: The name every prediction Series carries, so a frame that has been joined
#: with predictions says which column is which without a convention.
PREDICTION_NAME = "predicted_price_eur"

#: The metadata record, and the only file `load_model` opens before it knows
#: which estimator it is loading.
MODEL_FILE = "model.json"

_BOOSTER_FILE = "booster.txt"
_PIPELINE_FILE = "pipeline.joblib"
_LOOKUP_FILE = "lookup.parquet"

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
class TrainingData:
    """The frames a fit sees, the contract they were written against, and the scope.

    One object rather than five arguments, so that `fit_variant` stays the single
    call issue #38's energy measurement has to wrap.

    `supported_makes` is here rather than beside the parameters because it is a
    property of these rows: `train` restricts both frames to the makes the API
    serves (EDN-48), and the list has to reach the model bundle so that the API
    answers FR-04's scope check from the model it is serving.
    """

    space: FeatureSpace
    train: pd.DataFrame
    validation: pd.DataFrame
    supported_makes: tuple[str, ...]


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

    def __init__(self, metadata: dict, space: FeatureSpace) -> None:
        self.metadata = metadata
        self.space = space
        self.input_schema: Schema = space.schema.features(name=f"{metadata['feature_set']}-input")
        self._check_the_bundle_agrees_with_itself()
        training = metadata["training"]
        # In log space, because that is where the bound is applied. Kept as euros
        # in the metadata, so a reader - and #39's report of how many predictions
        # sit on a bound - sees the number in the unit it means.
        self._log_bounds = (
            float(np.log(training["price_min_eur"])),
            float(np.log(training["price_max_eur"])),
        )

    def _check_the_bundle_agrees_with_itself(self) -> None:
        """Refuse a bundle whose record and feature space describe different models.

        The two files are written together and restored together, so this can
        only happen to a hand-edited or half-restored bundle. It is worth a check
        because the symptom otherwise is a prediction from a model encoding its
        input by one level order and scoring it by another, with no error.
        """
        declared = tuple(self.metadata["features"])
        if declared != self.input_schema.names:
            raise ModelError(
                f"{MODEL_FILE} lists {len(declared)} feature(s) and the {FEATURE_SPACE_FILE} "
                f"beside it describes {len(self.input_schema.names)}. The two halves of the "
                f"bundle disagree about what this model consumes, so its codes cannot be "
                f"trusted; retrain the variant rather than editing either file."
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

    # -- prediction ---------------------------------------------------------

    def predict_eur(self, frame: pd.DataFrame) -> pd.Series:
        """`frame`'s rows as a price in euros, one per row, in `frame`'s order.

        Float64, named `PREDICTION_NAME`, finite, strictly positive and inside
        the training price range, with the same index as `frame`. Extra columns
        are ignored; a missing one raises and names itself. `frame` is not
        mutated: the alignment works on a copy.

        A missing value is passed through unimputed and the estimator decides
        what to do with it (EDN-15). What "missing" means for a column whose
        domain already represents absence is the caller's business and is fixed
        by EDN-23: an omitted assertion flag is `False`, an omitted equipment
        list is empty, everything else is absent. The three flag columns are
        non-nullable in the contract, so passing `None` for one of them fails
        here rather than quietly becoming a third state.
        """
        aligned = self._align(frame)
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
        and nullable `boolean` columns become float64, because an absent
        equipment list has to reach the estimators as NaN rather than as False.
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
        (directory / MODEL_FILE).write_text(
            json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8"
        )
        # A copy of the `features` stage's own artefact rather than a second,
        # model-shaped rendering of the same facts. It is what `load_model`,
        # `read_feature_space` and the API already know how to read, and one file
        # cannot disagree with itself about which levels a code means.
        space = {
            "schema": self.space.schema.to_dicts(),
            "vocabulary": self.space.vocabulary.to_dict(),
        }
        (directory / FEATURE_SPACE_FILE).write_text(
            json.dumps(space, indent=2) + "\n", encoding="utf-8"
        )
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
        parameter somebody set and nothing used.
        """
        given = dict(metadata["params"])
        expected = set(cls.hyperparameters)
        unknown = sorted(set(given) - expected)
        missing = sorted(expected - set(given))
        if unknown or missing:
            problems = []
            if missing:
                problems.append(f"does not set {', '.join(missing)}")
            if unknown:
                problems.append(f"sets {', '.join(unknown)}, which this estimator does not read")
            raise ModelError(
                f"the params block of variant {metadata['variant']!r} "
                f"{' and '.join(problems)}. {cls.estimator_name!r} reads exactly: "
                f"{', '.join(cls.hyperparameters)}."
            )
        return given


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
    the log of the stored median and `predict_eur` exponentiates it again, which
    is a float round trip and therefore exact to within one unit in the last
    place rather than bit-exact.

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
        self._width = float(metadata["params"]["age_bucket_years"])
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
    """Categoricals as strings, with absence as a level of its own.

    A module-level function rather than a lambda because it is pickled with the
    fitted pipeline. `OneHotEncoder` has no missing-value handling of its own, so
    without this an absent value would fail the fit; with it, absence gets a
    coefficient like any other level.
    """
    return frame.astype("object").where(frame.notna(), MISSING_LEVEL).astype("str")


class RidgeModel(Model):
    """Ladder step 2: the interpretable depreciation baseline, on log price.

    A `ColumnTransformer` in front of `Ridge`, fitted on `log_price`. Two
    encoding decisions are recorded in the EDN because both look, at a glance,
    like a rule of this project being broken.

    **One-hot, confined to B1 (EDN-50).** EDN-02's "no one-hot" is a statement
    about the tree family and the reason it was chosen. Ridge is linear: it has
    no native categorical handling, and the alternatives were target encoding,
    which leaks the target without cross-fitting, or dropping the categoricals,
    which would leave the baseline unable to tell a Porsche from a Dacia and so
    destroy its purpose. `handle_unknown="infrequent_if_exist"` is what makes an
    unseen level an answer rather than an error (EDN-18), and it needs
    `min_frequency` to have a group to send that level to - which is why
    `min_category_rows` is a parameter and why it is an absolute row count
    rather than a share: a share changes what it means as the training set grows.

    **Mean fill plus a per-feature indicator, which is not imputation
    (EDN-51).** Ridge cannot consume NaN, so the numeric branch is a union of a
    mean-filled, scaled copy and `MissingIndicator(features="all")`. With an
    indicator for every feature the encoding is information-preserving: a masked
    row contributes `beta * mean + gamma`, and `gamma` absorbs the offset, so the
    fill is a numerically neutral placeholder rather than a guess at the value
    (EDN-15). `features="all"` is load-bearing and not a default: scikit-learn
    emits an indicator only for features that were missing *at fit time*, so a
    column that happens to be complete in the training split would be silently
    mean-filled with no indicator when SC-06 masks it, and the criterion would be
    measuring its own imputation.
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
            # `lsqr`, not the default `auto`: `auto` picks a dense Cholesky when
            # it can, which goes through BLAS, whose reduction order changes with
            # the thread count. LSQR is scipy's iterative solver, runs
            # single-threaded and is therefore the same on any machine (NFR-06).
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

    def __init__(self, metadata: dict, space: FeatureSpace, booster: lightgbm.Booster | None):
        super().__init__(metadata, space)
        self.booster = booster

    @classmethod
    def fit(
        cls, metadata: dict, data: TrainingData, *, seed: int, num_threads: int
    ) -> "LightGBMModel":
        tuning = cls._tuning(metadata)
        rounds = tuning.pop("early_stopping_rounds")
        model = cls(metadata, data.space, None)
        estimator = lightgbm.LGBMRegressor(
            objective="regression",
            metric="l1",
            first_metric_only=True,
            # `deterministic` and `force_row_wise` together stop LightGBM from
            # choosing its histogram construction by data size and thread count.
            # Documented insurance rather than a measured fix: at this data size
            # the trees came out identical across 1, 4 and 8 threads with the
            # flags off as well. They cost nothing and they are what keeps the
            # promise if `num_threads` is ever raised.
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
        # `best_iteration_` is 0 when early stopping never fired, and
        # `save_model(num_iteration=0)` would write a booster with no trees.
        metadata["training"]["best_iteration"] = int(
            estimator.best_iteration_ or estimator.booster_.num_trees()
        )
        model.booster = estimator.booster_
        return model

    def _as_codes(self, aligned: pd.DataFrame) -> pd.DataFrame:
        coded = aligned.copy()
        for name in self.categorical_columns:
            coded[name] = coded[name].cat.codes.astype("int32")
        return coded

    def _predict_log_price(self, aligned: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.booster.predict(self._as_codes(aligned)), dtype="float64")

    def _save_payload(self, directory: Path) -> None:
        # LightGBM's own text format, not a pickle of the sklearn wrapper: it
        # survives a LightGBM upgrade, it is readable, and EDN-11's SHAP export
        # needs a `Booster` rather than the wrapper. Saved at the early-stopped
        # iteration, so the file *is* the model and predict needs no extra
        # argument.
        self.booster.save_model(
            directory / _BOOSTER_FILE, num_iteration=self.metadata["training"]["best_iteration"]
        )

    @classmethod
    def _load_payload(
        cls, directory: Path, metadata: dict, space: FeatureSpace
    ) -> "LightGBMModel":
        return cls(metadata, space, lightgbm.Booster(model_file=str(directory / _BOOSTER_FILE)))


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
        "seed": seed,
        "num_threads": num_threads,
        "training": {
            "n_train_rows": len(data.train),
            "n_validation_rows": len(data.validation),
            # The bound `predict_eur` applies, in the unit a reader thinks in.
            "price_min_eur": float(data.train["price"].min()),
            "price_max_eur": float(data.train["price"].max()),
            # Filled after the fit, below. Declared here so that the record has
            # the same shape for every variant and a reader does not have to know
            # which estimator writes which key.
            "train_l1_log_price": None,
            "validation_l1_log_price": None,
            # FR-04's scope, carried in the metadata because the API answers the
            # scope check from the model it serves rather than from a file it has
            # to be pointed at separately.
            "supported_makes": list(data.supported_makes),
        },
        "versions": _versions(*cls.libraries),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    model = cls.fit(metadata, data, seed=seed, num_threads=num_threads)
    # Measured on the model as it will be served, bound included, rather than
    # read out of the estimator's own fitting log. Only LightGBM keeps such a
    # log, so recomputing is what makes the number mean the same thing for all
    # four variants.
    metadata["training"]["train_l1_log_price"] = _l1_log_price(model, data.train)
    metadata["training"]["validation_l1_log_price"] = _l1_log_price(model, data.validation)
    return model


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
