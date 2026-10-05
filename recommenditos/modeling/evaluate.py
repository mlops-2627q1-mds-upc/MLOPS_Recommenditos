"""`evaluate` stage: the metrics artefacts, the per-segment breakdown and the SC gate.

The course demo has no equivalent - it asserts one threshold inside
`tests/test_model.py`. We need artefacts, because NFR-01 gates deployment on
the success criteria and EDN-12 makes promotion a human decision that someone
has to be able to read a result off.

Every metric here is a **fraction in [0, 1], never a percentage**, matching the
thresholds in `params.yaml` (`0.09`, not `9`). Only the log lines multiply by 100.

Three outputs:

- `metrics.json`, the DVC `metrics` file: the gate verdict NFR-01 reads, plus
  the headline numbers of each variant. Deliberately narrow, because
  `dvc metrics show` flattens nested JSON into one column per leaf. This shape
  has 29 leaves, 9 plus 5 per variant, of which it renders the ones that are not
  null - 28 while no variant is deployable - and that is already past the width a
  plain terminal table keeps, so read it with `dvc metrics show --md`. Anything
  more per variant belongs in the per-variant record below. `dvc metrics diff` is
  long-format and lists only the leaves that changed, which is what compares two
  model versions across a merge and is the promotion reviewer's checklist under
  EDN-12; there a null renders as `-`, so a leaf becoming non-null is visible.
  Two of those leaves are not measurements: `data_source` is the resolved
  `download.source` and each variant's `estimator` is the model actually fitted.
  They cost five columns so that nobody can read a number off this file without
  seeing what produced it - a stub on generated data once wrote the same MdAPE
  under all four variant names here, and the artefact said nothing about it.
- `reports/metrics/<variant>.json`, the full record per variant: every criterion
  with its measured value, the per-segment table and the masking sweep.
- `reports/metrics/segments.csv` and `reports/metrics/masked-inputs.csv`, the
  same two tables across all variants in one file each, which is what the report
  and its LaTeX tables cite.

The stage reports on the test rows the API would answer, which is the makes the
model was fitted on (EDN-48). `features` has already restricted the test matrix
to them (EDN-67), so this stage checks rather than filters: it takes the list out
of the model's own metadata rather than reading `supported_makes.json` again,
refuses a test matrix holding or encoded over any other make, and checks the
per-make segment levels against the same list. A filter here would only ever
remove rows from a matrix of another run, and would then report on a population
chosen by whichever run the model came from.

Two structural rules keep a criterion honest, both in code that fails rather
than in a convention somebody has to remember:

- A segment that conditions on the target is reported and can never gate
  (`TARGET_CONDITIONED_SEGMENTS`, `criterion_segments`).
- The partial-input scenario P1 masks *required* fields, so it can never reach
  SC-06 (`sc05_scenarios`, and the refusal in `evaluate_sc06`).

Each variant's metrics are appended to the MLflow run `train@<variant>` created,
rather than logged to a run of this stage's own. One run per variant then holds
the hyperparameters, the artefact, the energy figures and the verdict, which is
what makes the four comparable in one table.
"""

import csv
from dataclasses import dataclass, replace
from itertools import pairwise
import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from recommenditos.config import (
    METRICS_FILE,
    MODELS_DIR,
    PARAMS_FILE,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)
from recommenditos.data.build_features import (
    FeatureSpace,
    check_served_makes_only,
    equipment_feature_name,
)
from recommenditos.modeling.model import Model, load_model
from recommenditos.pipeline import load_params, read_frame
from recommenditos.tracking import resume_run

#: Every criterion NFR-01 gates on. A variant is deployable only when all six
#: pass, so a `None` here blocks the gate rather than being ignored.
CRITERIA: tuple[str, ...] = ("sc01", "sc02", "sc03", "sc04", "sc05", "sc06")

#: The "close enough" bands problem-spec section 6 reports a share for.
_CLOSE_ENOUGH_BANDS = {"within_10pct": 0.10, "within_20pct": 0.20}

_PERCENT = 100.0

#: Segments that condition on the target. Reported for the report's fairness
#: section; never eligible for a criterion, because SC-04 would otherwise reward
#: a model for being accurate where the price already gives the answer.
TARGET_CONDITIONED_SEGMENTS: frozenset[str] = frozenset({"price_bucket"})

#: The matrix column a bucketed segment is cut from, and the `evaluate` key
#: holding its lower edges. A segment named after a column of the matrix needs no
#: entry here; these two are the derived ones.
_BUCKETED_SEGMENTS: dict[str, tuple[str, str]] = {
    "age_bucket": ("age_years", "age_bucket_edges"),
    "price_bucket": ("price", "price_bucket_edges"),
}

#: The segment whose levels have to be makes the API answers (EDN-48, FR-04).
MAKE_SEGMENT = "make"

#: The level a row gets when the value being segmented on is absent, and when it
#: is below the first bucket edge. Two levels rather than one, because a null
#: `registration_date` and a listing registered after the reference date are
#: different data-quality findings, and folding a negative age into the youngest
#: bucket would hide the second one entirely (EDN-22 removes it in `preprocess`).
MISSING_SEGMENT_LEVEL = "(missing)"
INVALID_SEGMENT_LEVEL = "(invalid)"

_PLACEHOLDER_LEVELS: frozenset[str] = frozenset({MISSING_SEGMENT_LEVEL, INVALID_SEGMENT_LEVEL})

#: Why a reported segment level does not enter SC-04. An empty string means it
#: does; every other row says which rule kept it out, so nothing is dropped
#: silently from the table the report prints.
EXCLUDED_TARGET_CONDITIONED = "target_conditioned"
EXCLUDED_ABSENT_REQUIRED_FIELD = "absence_of_a_required_field"
EXCLUDED_TOO_FEW_ROWS = "below_sc04_min_segment_rows"

#: The masking scenario that leaves out every optional field at once, and the
#: partial-input scenario of SC-05. Parenthesised so neither can collide with an
#: input field name in the same column of the exported table.
ALL_OPTIONAL_FIELDS = "(all optional fields)"
P1_SCENARIO = "(P1: make, model, registration date, mileage)"

#: The method a model gains when the UC2 conformal intervals are built, checked
#: with `hasattr` so SC-05 starts being measured when one exists rather than when
#: somebody remembers to edit this stage. `recommenditos.modeling.model` states
#: why it is deliberately absent today.
INTERVAL_METHOD = "predict_interval_eur"

#: The two tables the report cites, one file across all variants each.
SEGMENTS_FILE = "segments.csv"
MASKED_INPUTS_FILE = "masked-inputs.csv"

#: Their columns, in the order the files carry them. A fixed header rather than
#: whatever the row dicts happen to hold, for two different reasons: a row
#: carrying a key this list does not name raises from `csv.DictWriter`, and the
#: order is what a positional reader - a LaTeX table, a spreadsheet - depends on,
#: so it is asserted as a line in `tests/test_pipeline.py`. The other direction
#: is not caught here: a key this list names and a row omits is written as an
#: empty cell, which is the same thing an absent value looks like.
_SEGMENT_FIELDS: tuple[str, ...] = (
    "variant",
    "segment",
    "level",
    "n",
    "mdape",
    "within_10pct",
    "within_20pct",
    "mae_eur",
    "mape",
    "counts_toward_sc04",
    "excluded_because",
)
_MASKED_INPUT_FIELDS: tuple[str, ...] = (
    "variant",
    "criterion",
    "field",
    "n_columns_masked",
    "mdape",
    "mdape_ratio",
    "threshold",
    "passed",
)

app = typer.Typer()


# --------------------------------------------------------------------------
# The metric layer
# --------------------------------------------------------------------------


def relative_error(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    """The absolute percentage error of every row, as a fraction.

    The one quantity MdAPE, the two band shares and MAPE are all derived from,
    computed once per scenario.

    The denominator is the absolute actual price: with a signed one a negative
    price would make a 10 % error read as a perfect prediction. Prices cannot be
    negative once the range filter runs, but this function is also what the API
    and the drift job will use, on data no filter has seen.

    Four guards, because the alternative to each is a plausible-looking number:

    - A misaligned index is aligned into NaN by the subtraction rather than
      refused, and pandas then computes the median over the overlap while the
      band shares divide by the union. Measured on a 10-row frame with the
      prediction indexed 5..14: MdAPE 0.05 with `within_20pct` 0.3333.
    - A NaN prediction is skipped by `median` and `mean` and counted by the band
      shares, so the two halves of the record use different denominators.
      Measured with 5 of 10 predictions NaN: MdAPE unchanged at 0.05 while
      `within_20pct` halves to 0.5. A model that failed on half its rows would
      report an unchanged primary metric.
    - A NaN actual is the same failure on the other side.
    - A zero price makes the ratio infinite.
    """
    if actual.empty:
        raise ValueError("cannot compute metrics on an empty set")
    if not actual.index.equals(predicted.index):
        raise ValueError(
            f"the actual prices and the predictions are indexed differently "
            f"({len(actual)} and {len(predicted)} row(s)), so pandas would align them into "
            f"NaN rather than compare them row for row. Both come from one frame, so this "
            f"is a reindex or a reset somebody did between predicting and scoring."
        )
    _refuse_nulls(actual, "actual price")
    _refuse_nulls(predicted, "prediction")
    if (actual == 0).any():
        raise ValueError(
            f"cannot compute a percentage error against a price of 0, at row position(s) "
            f"{_positions(actual == 0)}. `preprocess.price_min_eur` makes it impossible in "
            f"the pipeline, so this is a frame that never passed through it."
        )
    return (predicted - actual).abs() / actual.abs()


def _refuse_nulls(values: pd.Series, what: str) -> None:
    if values.isna().any():
        raise ValueError(
            f"{int(values.isna().sum())} of {len(values)} {what}(s) are missing, at row "
            f"position(s) {_positions(values.isna())}. `median` and `mean` would skip them "
            f"while the band shares counted them, so the metrics would be computed over two "
            f"different denominators."
        )


def _positions(mask: pd.Series, limit: int = 5) -> str:
    where = [int(position) for position in mask.to_numpy().nonzero()[0]]
    return f"{where[:limit]}{'' if len(where) <= limit else ', ...'}"


def point_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """MdAPE, the +/-10 % and +/-20 % shares, MAE and MAPE (problem-spec 6).

    The band shares are inclusive at the edge: an error of exactly 10 % counts as
    within 10 %. The median of an even number of rows is the mean of the two
    middle errors, which is pandas' convention and is pinned by a hand-computed
    case in `tests/test_model.py` so nobody "fixes" it to a lower-middle one.
    """
    relative = relative_error(actual, predicted)
    return {
        "mdape": float(relative.median()),
        **{name: float((relative <= band).mean()) for name, band in _CLOSE_ENOUGH_BANDS.items()},
        "mae_eur": float(((predicted - actual).abs()).mean()),
        "mape": float(relative.mean()),
    }


# --------------------------------------------------------------------------
# The per-segment breakdown
# --------------------------------------------------------------------------


def bucket_labels(edges: "list[float] | tuple[float, ...]") -> list[str]:
    """The label of each bucket the lower `edges` describe, last one open ended.

    Generated from the edges rather than written out, so that editing
    `evaluate.age_bucket_edges` moves the boundaries and the labels together.
    """
    return [f"{low}-{high}" for low, high in pairwise(edges)] + [f"over {edges[-1]}"]


def bucket_levels(values: pd.Series, edges: "list[float] | tuple[float, ...]") -> pd.Series:
    """`values` as the label of the bucket each falls in, as plain strings.

    Left-closed and right-open, so a value exactly on an edge belongs to the
    bucket above it and the buckets partition the line.
    """
    labels = bucket_labels(edges)
    cut = pd.cut(values, bins=[*edges, np.inf], right=False, labels=labels, ordered=True)
    level = cut.astype("object")
    # `pd.cut` returns NaN both for a null value and for one below the first
    # edge, and the two are different findings, so they get different levels.
    level[cut.isna()] = MISSING_SEGMENT_LEVEL
    level[values < edges[0]] = INVALID_SEGMENT_LEVEL
    return level.astype("str")


def segment_levels(values: pd.Series, *, missing: str = MISSING_SEGMENT_LEVEL) -> pd.Series:
    """`values` as plain string level labels, with an explicit level for a null.

    Not `astype("str")`: measured on pandas 3, that leaves a float NaN among the
    strings (`['PrivateSeller', 'Dealer', nan]`), which raises `TypeError` as
    soon as anything sorts or groups the levels.
    """
    return values.astype("object").where(values.notna(), missing).astype("str")


def segment_level_column(frame: pd.DataFrame, segment: str, params: dict) -> pd.Series:
    """The level of `segment` for every row of `frame`."""
    if segment in _BUCKETED_SEGMENTS:
        source, edges_key = _BUCKETED_SEGMENTS[segment]
        return bucket_levels(frame[source], params["evaluate"][edges_key])
    return segment_levels(frame[segment])


def criterion_segments(params: dict) -> tuple[str, ...]:
    """The segments SC-04 may gate on, refusing the two kinds it must not.

    Both refusals are the reason this function exists rather than the caller
    simply reading `evaluate.segments`:

    - A target-conditioned segment would let a model pass SC-04 by being accurate
      where the price already gives the answer, so adding one fails the stage.
    - SC-04 reports the `"(missing)"` level of a segment but excludes it from the
      criterion, because FR-01 refuses a request that omits a required field with
      a 422, so such a level cannot occur at serving time at all. That exclusion
      is only correct while every segment is a required input field, so a segment
      over an optional one is refused rather than silently excluded.
    """
    segments = tuple(params["evaluate"]["segments"])
    conditioned = sorted(TARGET_CONDITIONED_SEGMENTS.intersection(segments))
    if conditioned:
        raise ValueError(
            f"evaluate.segments names {', '.join(conditioned)}, which condition on the price "
            f"SC-04 measures the error against, so a criterion over them would reward a model "
            f"for being accurate where the target already gives the answer. They are reported "
            f"in {SEGMENTS_FILE} and may not gate."
        )
    required = set(params["evaluate"]["required_input_fields"])
    optional = [name for name in segments if _input_field_behind(name, params) not in required]
    if optional:
        raise ValueError(
            f"evaluate.segments names {', '.join(optional)}, which FR-01 does not require. "
            f"SC-04 excludes a {MISSING_SEGMENT_LEVEL!r} level because a request omitting a "
            f"required field is refused; for an optional field that absence is a case the API "
            f"answers, so excluding it would hide exactly the rows worth looking at. Decide "
            f"what the criterion does with it before adding the segment."
        )
    return segments


def _input_field_behind(segment: str, params: dict) -> str:
    """The API input field a segment is cut from, which is not always its name.

    `age_bucket` is cut from `age_years`, which FR-01 asks for as
    `registration_date`; the mapping is read from params rather than repeated.
    """
    column = _BUCKETED_SEGMENTS[segment][0] if segment in _BUCKETED_SEGMENTS else segment
    derived = params["evaluate"]["input_field_to_feature"]
    return next((field for field, feature in derived.items() if feature == column), column)


def segment_rows(
    levels: "dict[str, pd.Series]",
    actual: pd.Series,
    predicted: pd.Series,
    *,
    variant: str,
    min_rows: int,
) -> "list[dict]":
    """One row per level of each segment: its metrics, and whether SC-04 counts it.

    `levels` maps a segment name to that segment's level for every row, which is
    what `segment_level_column` produces. Grouping plain strings rather than a
    categorical is what makes an empty level impossible: `groupby` yields only
    the levels that occur, so no row is ever built over zero rows.

    Every row carries `n` and, when it does not count toward SC-04, the rule that
    kept it out. Nothing is filtered away, because a reader of the fairness
    section has to be able to see that a make had 129 test rows rather than infer
    it from an absence.
    """
    rows = []
    for segment, level_of in levels.items():
        for level, members in level_of.groupby(level_of):
            index = members.index
            rows.append(
                {
                    "variant": variant,
                    "segment": segment,
                    "level": str(level),
                    "n": len(index),
                    **point_metrics(actual.loc[index], predicted.loc[index]),
                    **_sc04_eligibility(segment, str(level), len(index), min_rows=min_rows),
                }
            )
    return sorted(rows, key=lambda row: -row["n"])


def _sc04_eligibility(segment: str, level: str, n: int, *, min_rows: int) -> dict:
    """Whether one level enters SC-04, and which rule kept it out if it does not.

    Reading the segment's own name rather than taking a flag, so that a
    target-conditioned segment is excluded whoever builds its rows.

    The order is the priority: a target-conditioned segment is out whatever its
    size, and the absence of a required field is out even when it clears the row
    minimum, because it is a data-quality defect the Great Expectations suite
    owns rather than a population the deployed API can be asked about.
    """
    if segment in TARGET_CONDITIONED_SEGMENTS:
        because = EXCLUDED_TARGET_CONDITIONED
    elif level in _PLACEHOLDER_LEVELS:
        because = EXCLUDED_ABSENT_REQUIRED_FIELD
    elif n < min_rows:
        because = EXCLUDED_TOO_FEW_ROWS
    else:
        because = ""
    return {"counts_toward_sc04": because == "", "excluded_because": because}


# --------------------------------------------------------------------------
# What an input field is, and what its absence looks like
# --------------------------------------------------------------------------


def input_field_columns(model: Model, params: dict) -> "dict[str, tuple[str, ...]]":
    """Every API input field of this model's feature set, and the columns it produced.

    SC-06 masks *input fields*, not feature columns, because FR-01 is a statement
    about a request: one equipment field becomes many multi-hot columns, so a
    column-wise sweep would mask a single equipment item, which is not a request
    the API can make; and `registration_date` becomes the derived `age_years`, so
    a naive set difference against `required_input_fields` would make the most
    important feature in the model look optional.

    This function answers only *which fields exist*. What each one expands to is
    `Model.columns_of`, so there is one derivation of "which columns did masking
    this field mask" rather than two that can disagree. Which columns an equipment
    field produced is what the training rows decided, so neither half can come
    from params.yaml.
    """
    derived = dict(params["evaluate"]["input_field_to_feature"])
    conflicting = sorted(set(derived) & set(model.features))
    if conflicting:
        raise ValueError(
            f"evaluate.input_field_to_feature maps {', '.join(conflicting)} to another column "
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
        claims = {source_of.get(column), field_of_derived.get(column)} - {None}
        if len(claims) > 1:
            raise ValueError(
                f"column {column!r} is claimed by two input fields ({', '.join(sorted(claims))}), "
                f"so masking either would mask a column the other owns. "
                f"evaluate.input_field_to_feature names a column an equipment field produced."
            )
        field = claims.pop() if claims else column
        if field not in fields:
            fields.append(field)

    absent = [
        field for field in params["evaluate"]["required_input_fields"] if field not in fields
    ]
    if absent:
        raise ValueError(
            f"{model.variant!r} consumes no column for the required input field(s) "
            f"{', '.join(absent)}. FR-01 refuses a request that omits one, so the required "
            f"list and the trained model disagree about what a valuation needs; either the "
            f"feature set or evaluate.required_input_fields is wrong."
        )
    # The seam expands a field into its columns; this function only renames the
    # one field whose column has a different name first. `columns_of` knows the
    # equipment vocabulary and nothing about `input_field_to_feature`, so asking
    # it for `registration_date` raises - the model consumes `age_years`. The
    # rename is params' business and the expansion is the model's, and keeping
    # them in that order is what stops this from becoming a second expansion.
    return {field: model.columns_of(derived.get(field, field)) for field in fields}


def optional_input_fields(fields: "dict[str, tuple[str, ...]]", params: dict) -> tuple[str, ...]:
    """The input fields FR-01 leaves optional, which is what SC-06 masks (EDN-15)."""
    required = set(params["evaluate"]["required_input_fields"])
    return tuple(sorted(set(fields) - required))


# --------------------------------------------------------------------------
# The masking sweep
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskingScenario:
    """One request with something left out: which criterion asks for it, and what it hides.

    `field` is what the report calls the scenario and `fields` is what
    `Model.mask_absent` is handed, which differ for a combination of fields and
    for the one request field whose column has another name. `n_columns` is
    carried because the exported table reports it and the expansion is already
    known here.

    `criterion` is what keeps SC-05's P1 out of SC-06 structurally: the two
    scenario lists are built by different functions and `evaluate_sc06` refuses a
    row that did not come from its own.
    """

    criterion: str
    field: str
    fields: tuple[str, ...]
    n_columns: int


def sc06_scenarios(
    fields: "dict[str, tuple[str, ...]]", optional: "tuple[str, ...]", params: dict
) -> "tuple[MaskingScenario, ...]":
    """Each optional input field left out on its own, and then all of them at once."""
    seam = _seam_names(fields, params)
    scenarios = [
        MaskingScenario("sc06", field, (seam[field],), len(fields[field])) for field in optional
    ]
    if optional:
        scenarios.append(
            MaskingScenario(
                "sc06",
                ALL_OPTIONAL_FIELDS,
                tuple(seam[field] for field in optional),
                sum(len(fields[field]) for field in optional),
            )
        )
    return tuple(scenarios)


def _seam_names(fields: "dict[str, tuple[str, ...]]", params: dict) -> "dict[str, str]":
    """Each request field under the name `Model.columns_of` knows it by.

    The same rename `input_field_columns` applies, in the same direction and from
    the same params key: the seam knows the equipment vocabulary and not the
    request, so it cannot answer for `registration_date`.
    """
    derived = params["evaluate"]["input_field_to_feature"]
    return {field: derived.get(field, field) for field in fields}


def sc05_scenarios(
    fields: "dict[str, tuple[str, ...]]", params: dict
) -> "tuple[MaskingScenario, ...]":
    """The partial-input scenario P1 of SC-05: only the fields it names are given.

    Its point metrics are the half of SC-05 that can be measured without the
    intervals, and they must not reach SC-06: P1 leaves out required fields, so
    its inflation is a large number about a request FR-01 refuses rather than the
    degradation SC-06 bounds.
    """
    given = set(params["evaluate"]["sc05_p1_fields"])
    unknown = sorted(given - set(fields))
    if unknown:
        raise ValueError(
            f"evaluate.sc05_p1_fields names {', '.join(unknown)}, which this feature set has "
            f"no input field for, so P1 would silently be a different scenario than the one "
            f"problem-spec section 8 defines."
        )
    seam = _seam_names(fields, params)
    left_out = sorted(set(fields) - given)
    return (
        MaskingScenario(
            "sc05",
            P1_SCENARIO,
            tuple(seam[field] for field in left_out),
            sum(len(fields[field]) for field in left_out),
        ),
    )


def masking_sweep(
    model: Model,
    frame: pd.DataFrame,
    scenarios: "tuple[MaskingScenario, ...]",
    *,
    full_mdape: float,
) -> "list[dict]":
    """One re-prediction per scenario, and how far each inflates MdAPE.

    The whole test set per scenario, not a subsample: measured on the real
    snapshot the full extended sweep of 25 scenarios over 19,665 rows costs
    seconds, which is two orders of magnitude below NFR-10's budget.
    """
    if full_mdape <= 0:
        raise ValueError(
            f"the full-input MdAPE is {full_mdape}, so the ratio SC-06 bounds has no "
            f"denominator. A median error of 0 means the model reproduces at least half the "
            f"test prices exactly, which is a leak or a test set built from training rows, "
            f"not a model to measure the masking of."
        )
    rows = []
    for scenario in scenarios:
        masked = model.mask_absent(frame, scenario.fields)
        metrics = point_metrics(frame["price"], model.predict_eur(masked))
        rows.append(
            {
                "variant": model.variant,
                "criterion": scenario.criterion,
                "field": scenario.field,
                "n_columns_masked": scenario.n_columns,
                "mdape": metrics["mdape"],
                "mdape_ratio": metrics["mdape"] / full_mdape,
            }
        )
    return rows


def masked_input_rows(sweep: "list[dict]", criteria: dict) -> "list[dict]":
    """The sweep as the report's table, with SC-06's verdict on the rows it judges.

    P1 carries no verdict. It masks required fields, so the SC-06 threshold does
    not apply to it, and an empty cell says that where a `False` would read as a
    criterion this model missed.
    """
    threshold = criteria["sc06_masked_mdape_ratio_max"]
    return [
        {
            **row,
            "threshold": threshold if row["criterion"] == "sc06" else None,
            "passed": row["mdape_ratio"] <= threshold if row["criterion"] == "sc06" else None,
        }
        for row in sweep
    ]


# --------------------------------------------------------------------------
# The six criteria
# --------------------------------------------------------------------------


def evaluate_sc04(rows: "list[dict]", criteria: dict, *, segments: "tuple[str, ...]") -> dict:
    """SC-04: the worst MdAPE among the segment levels large enough to judge.

    When no level reaches the row minimum the criterion is **not measured**, not
    met. "Every level satisfies P" is vacuously true over an empty set, and
    reporting that as a pass would say the model had been checked where it had
    not been.
    """
    foreign = sorted({row["segment"] for row in rows} - set(segments))
    if foreign:
        raise ValueError(
            f"SC-04 was handed rows for segment(s) {', '.join(foreign)}, which "
            f"criterion_segments did not return. A segment that may not gate is reported in "
            f"{SEGMENTS_FILE} and must not reach this function."
        )
    minimum = criteria["sc04_min_segment_rows"]
    qualifying = [row for row in rows if row["counts_toward_sc04"]]
    counts = {
        "sc04_min_segment_rows": minimum,
        "sc04_n_segment_levels": len(rows),
        "sc04_n_qualifying": len(qualifying),
    }
    if not qualifying:
        largest = max(rows, key=lambda row: row["n"], default=None)
        where = (
            "there is no segment level at all"
            if largest is None
            else f"the largest is {largest['segment']}={largest['level']} at {largest['n']}"
        )
        return {
            **counts,
            "sc04_measured": None,
            "sc04_passed": None,
            "sc04_worst_segment": None,
            "sc04_note": (
                f"no segment level reaches {minimum} test rows ({where}), so SC-04 is not "
                f"measured rather than vacuously met"
            ),
        }
    worst = max(qualifying, key=lambda row: row["mdape"])
    return {
        **counts,
        "sc04_measured": worst["mdape"],
        "sc04_passed": worst["mdape"] <= criteria["sc04_segment_mdape_max"],
        "sc04_worst_segment": {key: worst[key] for key in ("segment", "level", "n", "mdape")},
        "sc04_note": "",
    }


def interval_metrics(model: Model, frames: "dict[str, pd.DataFrame]") -> "dict | None":
    """Each scenario's interval coverage and mean relative width, or `None`.

    `None` when the model exposes no interval method, which is the state of the
    first delivery. A capability check rather than a hard-coded null, so the
    criterion starts being measured when a model gains the method.

    Coverage is the share of rows whose price falls inside the interval. The mean
    relative width is reported and is not a pass criterion: `params.yaml` sets no
    threshold for it, correctly, because there is no reference value for how wide
    a useful interval is.
    """
    if not hasattr(model, INTERVAL_METHOD):
        return None
    measured = {}
    for name, frame in frames.items():
        interval = getattr(model, INTERVAL_METHOD)(frame)
        price = frame["price"]
        inside = (interval["lower_eur"] <= price) & (price <= interval["upper_eur"])
        width = (interval["upper_eur"] - interval["lower_eur"]) / model.predict_eur(frame)
        measured[name] = {
            "coverage": float(inside.mean()),
            "mean_relative_width": float(width.mean()),
        }
    return measured


def evaluate_sc05(intervals: "dict | None", criteria: dict) -> dict:
    """SC-05: whether the nominal 90 % intervals cover the share they claim.

    Unmeasurable in the first delivery, and that **blocks** the gate rather than
    passing it: no model is deployable before the intervals exist and their
    coverage has been checked. Reported as a structured status rather than a
    prose note, so a reader can tell "not measured" from "measured and failed"
    without parsing English, and `metrics.json` names the outstanding criterion.

    `sc05_measured` is the coverage furthest from the middle of the band, which
    is the scenario the criterion turns on; every scenario's numbers are kept.
    """
    if intervals is None:
        return {
            "sc05_measured": None,
            "sc05_passed": None,
            "sc05_status": "not_measured",
            "sc05_capability_checked": INTERVAL_METHOD,
            "sc05_reason": (
                f"no model exposes {INTERVAL_METHOD}: the UC2 conformal intervals, and the "
                f"calibration split held for them, are a later ticket, so interval coverage "
                f"cannot be measured by any means"
            ),
            "sc05_coverage": None,
            "sc05_worst_scenario": None,
        }
    low, high = criteria["sc05_coverage_min"], criteria["sc05_coverage_max"]
    centre = (low + high) / 2
    worst = max(intervals, key=lambda name: abs(intervals[name]["coverage"] - centre))
    return {
        "sc05_measured": intervals[worst]["coverage"],
        "sc05_passed": all(low <= entry["coverage"] <= high for entry in intervals.values()),
        "sc05_status": "measured",
        "sc05_capability_checked": INTERVAL_METHOD,
        "sc05_reason": "",
        "sc05_coverage": intervals,
        "sc05_worst_scenario": worst,
    }


def evaluate_sc06(sweep: "list[dict]", criteria: dict) -> dict:
    """SC-06: the worst MdAPE inflation over the masking sweep.

    The worst over the whole sweep including the all-at-once scenario, because
    the criterion is "with each masked on its own, and with all of them masked at
    once", which is one bound over every one of those requests.
    """
    foreign = sorted({row["field"] for row in sweep if row["criterion"] != "sc06"})
    if foreign:
        raise ValueError(
            f"SC-06 was handed the scenario(s) {', '.join(foreign)}, which mask a required "
            f"input field. That is SC-05's P1: FR-01 refuses a request omitting a required "
            f"field, so its inflation is not the degradation SC-06 bounds."
        )
    if not sweep:
        return {
            "sc06_measured": None,
            "sc06_passed": None,
            "sc06_n_scenarios": 0,
            "sc06_worst_field": None,
            "sc06_note": (
                "this feature set has no input field FR-01 leaves optional, so there is "
                "nothing to mask and the criterion is not measured"
            ),
        }
    worst = max(sweep, key=lambda row: row["mdape_ratio"])
    return {
        "sc06_measured": worst["mdape_ratio"],
        "sc06_passed": worst["mdape_ratio"] <= criteria["sc06_masked_mdape_ratio_max"],
        "sc06_n_scenarios": len(sweep),
        "sc06_worst_field": worst["field"],
        "sc06_note": "",
    }


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GateInput:
    """Everything NFR-01's verdict for one variant is computed from.

    One record rather than six arguments, so that a test can flip one measurement
    and leave the rest, and so that the inputs of a criterion are named in one
    place instead of spread over a call signature.
    """

    metrics: dict
    baseline_mdape: "float | None"
    segments: "list[dict]"
    criterion_segments: "tuple[str, ...]"
    sweep: "list[dict]"
    intervals: "dict | None"


def evaluate_gate(measured: GateInput, criteria: dict) -> dict:
    """Each success criterion as flat `<id>_measured` / `<id>_passed` keys.

    `<id>_passed` is `True`, `False` or `None`, and `None` means the criterion was
    not measured, never that the model missed it. `gate_passed` is `all()` over
    the six, so a `None` blocks; `_check_every_criterion_is_decided` is what makes
    that a checked property rather than a convention.
    """
    mdape = measured.metrics["mdape"]
    within = measured.metrics["within_20pct"]
    baseline = measured.baseline_mdape
    # `not baseline` would read a perfect baseline as a missing one.
    improvement = None if baseline is None or baseline == 0 else (baseline - mdape) / baseline
    record = {
        "sc01_measured": mdape,
        "sc01_passed": mdape <= criteria["sc01_mdape_max"],
        "sc02_measured": within,
        "sc02_passed": within >= criteria["sc02_within_20pct_min"],
        "sc03_measured": improvement,
        "sc03_passed": None
        if improvement is None
        else improvement >= criteria["sc03_mdape_improvement_over_baseline_min"],
        **evaluate_sc04(measured.segments, criteria, segments=measured.criterion_segments),
        **evaluate_sc05(measured.intervals, criteria),
        **evaluate_sc06(measured.sweep, criteria),
    }
    _check_every_criterion_is_decided(record)
    return record


def _check_every_criterion_is_decided(record: dict) -> None:
    """Refuse a record that leaves a criterion out or decides it with a non-bool.

    `gate_passed` is `all(record[f"{c}_passed"] for c in CRITERIA)`, so a
    criterion simply absent from the record raises `KeyError` instead of blocking,
    and a truthy value of another type would pass. This is the machine-checked
    version of what `CRITERIA`'s comment says.
    """
    undecided = [name for name in CRITERIA if f"{name}_passed" not in record]
    if undecided:
        raise ValueError(
            f"the gate record decides nothing for {', '.join(undecided)}. NFR-01 gates on all "
            f"{len(CRITERIA)}, and a missing key would make the verdict raise rather than block."
        )
    wrong = {
        f"{name}_passed": record[f"{name}_passed"]
        for name in CRITERIA
        if record[f"{name}_passed"] is not True
        and record[f"{name}_passed"] is not False
        and record[f"{name}_passed"] is not None
    }
    if wrong:
        raise ValueError(
            f"a criterion verdict has to be True, False or None (not measured), and "
            f"{wrong} is none of those. A truthy value of another type would pass the gate."
        )


def gate_passed(record: dict) -> bool:
    """Whether every criterion of NFR-01 was measured and met.

    `all()` over a `None` is `False`, which is how an unmeasured criterion blocks.
    Written this way rather than as `not any(... is False ...)`, which would read
    an unmeasured criterion as met.
    """
    return all(record[f"{name}_passed"] for name in CRITERIA)


def gate_blocked_by(record: dict) -> str:
    """The criteria that stop this variant, a miss told apart from a non-measurement.

    One string rather than two lists, because this is a `dvc metrics diff` row,
    and it is the most informative line a promotion reviewer reads (EDN-12).
    """
    failed = [name for name in CRITERIA if record[f"{name}_passed"] is False]
    unmeasured = [name for name in CRITERIA if record[f"{name}_passed"] is None]
    parts = []
    if failed:
        parts.append(f"{','.join(failed)} failed")
    if unmeasured:
        parts.append(f"{','.join(unmeasured)} unmeasured")
    return "; ".join(parts)


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------


def _evaluate_variant(
    variant: str, input_dir: Path, models_dir: Path, params: dict
) -> "tuple[dict, GateInput]":
    """One variant's whole measurement: the record it writes, and its gate inputs.

    The gate itself runs afterwards, in one pass over every variant, because
    SC-03 needs the baseline variant's MdAPE and that is not known until all four
    have been measured.
    """
    model = load_model(models_dir / variant)
    feature_set = model.feature_set
    # The contract travels with the matrices, because the equipment multi-hot
    # columns and the category levels are whatever the training rows decided.
    space = FeatureSpace.load(input_dir / feature_set, name=f"features-{feature_set}")
    test = read_frame(input_dir / feature_set / "test.parquet", space.schema)
    # Taken from the model's own metadata rather than from `supported_makes.json`,
    # so the evaluated population is checked against the one the model was
    # fitted on (EDN-48) and the two cannot be read from different `split` runs.
    check_served_makes_only(
        test, model.metadata["training"]["supported_makes"], name=f"{variant} test"
    )
    predicted = model.predict_eur(test)
    metrics = point_metrics(test["price"], predicted)

    criteria = params["evaluate"]["success_criteria"]
    segments = criterion_segments(params)
    rows = segment_rows(
        {name: segment_level_column(test, name, params) for name in segments},
        test["price"],
        predicted,
        variant=variant,
        min_rows=criteria["sc04_min_segment_rows"],
    )
    check_make_levels_are_served(
        rows, supported=model.metadata["training"]["supported_makes"], variant=variant
    )
    # A second call to the same builder, concatenated only into the report table.
    # The rows SC-04 reads are the ones above, so a target-conditioned level does
    # not exist in the object the criterion sees, and `_sc04_eligibility` would
    # refuse to count one even if it did.
    reported = [
        *rows,
        *segment_rows(
            {
                name: segment_level_column(test, name, params)
                for name in sorted(TARGET_CONDITIONED_SEGMENTS)
            },
            test["price"],
            predicted,
            variant=variant,
            min_rows=criteria["sc04_min_segment_rows"],
        ),
    ]

    fields = input_field_columns(model, params)
    optional = optional_input_fields(fields, params)
    sweep = masking_sweep(
        model, test, sc06_scenarios(fields, optional, params), full_mdape=metrics["mdape"]
    )
    p1 = sc05_scenarios(fields, params)
    partial = masking_sweep(model, test, p1, full_mdape=metrics["mdape"])
    record = {
        "variant": variant,
        # What actually produced these numbers, so a committed metrics file
        # cannot be read as a result of something it was not: the estimator that
        # was fitted, and the data source `download` resolved to.
        "estimator": model.estimator,
        "data_source": params["download"]["source"],
        "feature_set": feature_set,
        "n_test_rows": len(test),
        "mlflow_run_id": model.metadata["mlflow"]["run_id"],
        **metrics,
        "criterion_segments": list(segments),
        "n_optional_input_fields": len(optional),
        "optional_input_fields": list(optional),
        "segments": reported,
        "masked_inputs": masked_input_rows([*sweep, *partial], criteria),
    }
    gate_input = GateInput(
        metrics=metrics,
        baseline_mdape=None,
        segments=rows,
        criterion_segments=segments,
        sweep=sweep,
        # Both scenarios SC-05 names, because the criterion asks for the coverage
        # of a full request and of P1. The frame is masked by the same helper the
        # sweep above uses, so the two halves of SC-05 describe one request.
        intervals=interval_metrics(
            model,
            {
                "full": test,
                P1_SCENARIO: model.mask_absent(test, p1[0].fields),
            },
        ),
    )
    return record, gate_input


def check_make_levels_are_served(
    rows: "list[dict]", *, supported: "list[str] | tuple[str, ...]", variant: str
) -> None:
    """Refuse a per-make segment level naming a make the API answers with a 422.

    EDN-48's third obligation. `features` restricts the test rows before they
    are cut and `check_served_makes_only` checks the matrix, so this is the check
    on what the stage actually produced: without it, the report would print an
    error figure for a make FR-04 rejects, which is worse than printing nothing
    at all.
    """
    unserved = sorted(
        {row["level"] for row in rows if row["segment"] == MAKE_SEGMENT} - set(supported)
    )
    if unserved:
        raise ValueError(
            f"the per-make segments of {variant!r} cover {', '.join(unserved)}, which the model "
            f"was not fitted on and FR-04 rejects with a 422. The test rows have to be "
            f"restricted to the model's own supported makes before they are cut (EDN-48)."
        )


def _write_table(path: Path, rows: "list[dict]", fields: "tuple[str, ...]") -> None:
    """One CSV with a fixed header, which is what a LaTeX table is written against."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} row(s) to {path}.")


def _segment_table(records: "dict[str, dict]", segments: "tuple[str, ...]") -> "list[dict]":
    """Every variant's segment rows in one table, biggest segment first.

    Sorted by variant, then by the order `evaluate.segments` declares with the
    target-conditioned ones last, then by row count descending, so the report's
    eye lands on the segments that carry the population.
    """
    order = [*segments, *sorted(TARGET_CONDITIONED_SEGMENTS)]
    variants = list(records)
    rows = [row for record in records.values() for row in record["segments"]]
    return sorted(
        rows,
        key=lambda row: (variants.index(row["variant"]), order.index(row["segment"]), -row["n"]),
    )


def gate_summary(records: "dict[str, dict]", *, data_source: str) -> dict:
    """`metrics.json`: the verdict NFR-01 reads, and the headline of each variant.

    Every leaf here is a column of `dvc metrics show` and a candidate row of
    `dvc metrics diff`, so the shape is the promotion reviewer's checklist under
    EDN-12 rather than an arbitrary summary, and it is kept to 29 leaves.
    `dvc metrics show` omits a null leaf rather than showing it empty, so
    `deployable_variant` appears as a column only once a variant is deployable;
    `dvc metrics diff` renders it as `-` and shows the transition.

    `data_source` is required rather than defaulted, because a default would be
    the one value that could silently misdescribe a run.
    """
    passing = [name for name, record in records.items() if record["gate_passed"]]
    # Variants where nothing that could be measured failed. The leaf that makes
    # this artefact useful while SC-05 is unmeasurable: `n_variants_passing` is 0
    # for a reason that says nothing about the model, and this says how close the
    # candidates are. It is not the gate and is not one of its inputs:
    # `gate_passed` is a leaf of its own, computed from the six verdicts alone,
    # and `criteria_not_measured` beside it names what is still outstanding.
    measurable = [
        name
        for name, record in records.items()
        if not any(record[f"{each}_passed"] is False for each in CRITERIA)
    ]
    outstanding = sorted(
        {
            each
            for record in records.values()
            for each in CRITERIA
            if record[f"{each}_passed"] is None
        }
    )
    best = min(records.values(), key=lambda record: record["mdape"])
    # NFR-01 gates the deployment of *a model*, so the variant this artefact puts
    # forward has to be one that met all six criteria. The lowest MdAPE overall
    # is reported separately, because it is what a reader looks for first and it
    # would be confusing to omit it.
    deployable = min(
        (records[name] for name in passing), key=lambda record: record["mdape"], default=None
    )
    return {
        "gate_passed": bool(passing),
        "n_variants": len(records),
        "n_variants_passing": len(passing),
        "n_variants_passing_measurable": len(measurable),
        "criteria_not_measured": ",".join(outstanding),
        # Which data every number below was measured on, so `dvc metrics show`
        # alone tells a real result from one produced on the generated stand-in.
        "data_source": data_source,
        "deployable_variant": None if deployable is None else deployable["variant"],
        "best_variant": best["variant"],
        "best_mdape": best["mdape"],
        # Five leaves per variant, so `dvc metrics show --md` stays a table a
        # person can read and every row of `dvc metrics diff` says something.
        # `estimator` is one of them for the same reason `data_source` is above:
        # the variant *name* says which rung of the ladder was meant, and only
        # this says which model answered for it.
        "variants": {
            name: {
                "estimator": record["estimator"],
                "mdape": record["mdape"],
                "within_20pct": record["within_20pct"],
                "gate_passed": record["gate_passed"],
                "gate_blocked_by": record["gate_blocked_by"],
            }
            for name, record in records.items()
        },
    }


def _append_to_the_training_run(record: dict) -> None:
    """The verdict on the run `train@<variant>` opened, not on a run of this stage's.

    A context manager per variant because this stage covers all four in one
    process, and a run left open would collect the next variant's metrics.
    """
    with resume_run(record["mlflow_run_id"]) as run:
        run.log_metrics(
            {
                **{name: record[name] for name in ("mdape", "within_10pct", "within_20pct")},
                "mae_eur": record["mae_eur"],
                "mape": record["mape"],
                **{f"{each}_measured": record[f"{each}_measured"] for each in CRITERIA},
            }
        )
        run.set_tags(
            {
                "gate_passed": str(record["gate_passed"]),
                "gate_blocked_by": record["gate_blocked_by"],
                "sc04_worst_segment": json.dumps(record["sc04_worst_segment"]),
                "sc06_worst_field": str(record["sc06_worst_field"]),
            }
        )


@app.command()
def main(
    input_dir: Path = PROCESSED_DATA_DIR / "features",
    models_dir: Path = MODELS_DIR,
    metrics_dir: Path = REPORTS_DIR / "metrics",
    summary_path: Path = METRICS_FILE,
    params_path: Path = PARAMS_FILE,
):
    params = load_params(params_path)
    criteria = params["evaluate"]["success_criteria"]

    records, inputs = {}, {}
    for variant in params["train"]["variants"]:
        records[variant], inputs[variant] = _evaluate_variant(
            variant, input_dir, models_dir, params
        )
    baseline_mdape = records.get(params["evaluate"]["baseline_variant"], {}).get("mdape")

    metrics_dir.mkdir(parents=True, exist_ok=True)
    for variant, record in records.items():
        record.update(
            evaluate_gate(replace(inputs[variant], baseline_mdape=baseline_mdape), criteria)
        )
        record["gate_passed"] = gate_passed(record)
        record["gate_blocked_by"] = gate_blocked_by(record)
        path = metrics_dir / f"{variant}.json"
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        logger.info(
            f"{variant}: MdAPE {record['mdape'] * _PERCENT:.1f} %, gate "
            f"{'passed' if record['gate_passed'] else 'blocked by ' + record['gate_blocked_by']}"
            f" -> {path}."
        )
        _append_to_the_training_run(record)

    _write_table(
        metrics_dir / SEGMENTS_FILE,
        _segment_table(records, criterion_segments(params)),
        _SEGMENT_FIELDS,
    )
    _write_table(
        metrics_dir / MASKED_INPUTS_FILE,
        [row for record in records.values() for row in record["masked_inputs"]],
        _MASKED_INPUT_FIELDS,
    )
    summary = gate_summary(records, data_source=params["download"]["source"])
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.success(
        f"Gate on {summary['data_source']} data: "
        f"{summary['n_variants_passing']}/{summary['n_variants']} variants pass; "
        f"{summary['n_variants_passing_measurable']} meet every criterion that could be "
        f"measured (outstanding: {summary['criteria_not_measured'] or 'none'})."
    )


if __name__ == "__main__":
    app()
