"""Whether blanked-out optional fields in training let `lgbm-extended` hold up on sparse requests (EDN-82).

The measurement behind issue #116. `lgbm-extended` is the more accurate model on a complete request
and collapses on a sparse one, which is why `lgbm-basic` is the candidate (EDN-62). This script
asks whether training on copies of the listings with optional fields blanked out removes the
collapse, and what it costs on a complete request. It measures and applies a rule; it changes
nothing in `params.yaml`, `dvc.lock`, `models/` or `metrics.json`.

**Everything below was fixed before the first fit**, and the commit that added this file holds no
results.

Configurations. Every one is the `train` stage's own fit (`read_matrices`, `fit_variant`, the
committed hyperparameters, seed and thread count); the only thing that differs is the training
frame. A blanked copy of a training row is that row with some optional fields set to what
`Model.mask_absent` gives an omitted field (EDN-23, EDN-61), so training, SC-06 and the API share
one definition of "not sent". The copies are appended to the training rows, which stay as they are.

- `basic` and `extended`: the two committed variants, refitted unchanged. Their validation
  predictions are checked against the committed bundles, so the script's fit is shown to be the
  pipeline's before anything is read from a new one.
- `extended+all`: one copy per row with all 23 optional fields blanked, which is the request that
  carries only the fields FR-01 requires.
- `extended+random`: one copy per row with a random subset blanked. The size of the subset is drawn
  uniformly from 1 to 23 and the fields at random, so every level of sparsity is covered equally.
- `extended+all+random`: one copy of each kind.
- `extended+all+3random`: one `all` copy and three `random` copies, to see whether more copies help.
- `basic+all+random`: the same scheme on the candidate's own 6 optional fields. Reported only: it
  says whether the candidate itself gains, and what the sparse end can reach at all.

What is read. The `train` and `validation` matrices and the validation frame's seller groups.
Never `test` or `calibration`: the test split is read once, by `dvc repro`, if a configuration is
adopted (docs/docs/pipeline.md, "Tuning a hyperparameter"). The validation rows are never copied
or changed on disk, early stopping watches them as they are, and a copy of a training row is a
training row, so the seller grouping of the split is kept.

Scenarios, on all 8,859 validation rows and identical for every configuration:

- the request levels: 0, 2, 5, 10, 15, 20 and 23 of the extended set's 23 optional fields are
  given and the rest blanked, the given fields drawn at random per row with the committed seed.
  Level 0 is the required-only request and level 23 the complete one. A basic model is scored on
  the same requests, so it sees whichever of its 6 optional fields a request gives. "Given" means
  not blanked here: a field the listing itself leaves empty stays empty.
- SC-06 as `evaluate` computes it, on validation rows: each optional field blanked on its own, and
  all at once.

Differences are taken against `basic`, the candidate, by the paired bootstrap of
`recommenditos/modeling/tune.py`: 2,000 draws resampled by seller group, the same draws for every
configuration, with per-point and simultaneous (max-statistic) 95 % intervals. The metric is
MdAPE, because SC-06 and the candidate decision are stated in it; validation L1 in log price is
reported beside it.

The rule. A blanked `lgbm-extended` configuration qualifies only if all four hold on validation:

1. SC-06 passes: every single-field ratio and the required-only ratio are at most 1.5.
2. On the required-only request it is not measurably worse than `basic`: the per-point interval of
   its MdAPE difference does not lie entirely above zero. Per-point because it is the narrower
   interval, so the stricter test of "not worse".
3. On the complete request it is measurably better than `basic`: the simultaneous interval of its
   MdAPE difference lies entirely below zero. Simultaneous because six configurations are compared.
4. Early stopping ended its fit, and the fit took less than NFR-10's 15 minutes.

Of those that qualify, the one with the lowest mean MdAPE over the seven request levels is the
measured recommendation. Adopting it is not this script's decision: it changes the candidate
(EDN-62) and is taken by the team.

Also reported, deciding nothing: the better of the two committed models per request level, which
is what serving both models could reach at best, since a real routing rule sees only how many
fields a request gives.

Every fit is measured with CodeCarbon as `train` measures one and logged as its own run in the
MLflow experiment `recommenditos-price-tuning`, tagged `sweep=116-blanked-fields` and
`sweep_role=point`; one `sweep_role=summary` run carries the verdict, this script's output and
CodeCarbon's rows. The sweep id is used once, as for a search (EDN-78).

Run from the repository root after `dvc pull`, from a clean commit so the runs are not `git_dirty`,
with the output written outside the repository and copied in afterwards (about 15 minutes):

    uv run python reports/analysis/blanked_fields.py > /tmp/blanked_fields_results.txt
    cp /tmp/blanked_fields_results.txt reports/analysis/blanked_fields_results.txt

`--smoke` runs the whole path on 4,000 training rows and 40 trees with tracking off, to check the
code. Its numbers mean nothing and are evidence of nothing.
"""

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile

from loguru import logger
import numpy as np
import pandas as pd

from recommenditos.config import MODELS_DIR, PARAMS_FILE, PROCESSED_DATA_DIR
from recommenditos.data.build_features import read_supported_makes
from recommenditos.modeling import evaluate, tune
from recommenditos.modeling.energy import EnergySettings, FitMeasurement, measure
from recommenditos.modeling.model import Model, TrainingData, fit_variant, load_model
from recommenditos.modeling.train import log_fit, logged_params, read_matrices
from recommenditos.pipeline import load_params
from recommenditos.provenance import input_tags
from recommenditos.tracking import DisabledRun, Run

SWEEP = "116-blanked-fields"
RESULTS_FILE = "blanked_fields_results.txt"
RECORD_FILE = "blanked_fields.json"
EMISSIONS_FILE = "emissions.csv"

#: How many of the extended set's optional fields a request gives. Fixed before the first fit.
GIVEN_LEVELS = (0, 2, 5, 10, 15, 20, 23)
#: NFR-10's budget for fitting the final configuration, in seconds.
FIT_BUDGET_SECONDS = 15 * 60
SMOKE_ROWS, SMOKE_TREES = 4_000, 40
_PERCENT = 100


@dataclass(frozen=True)
class Configuration:
    """One training frame: a variant, and how many blanked copies of each row it adds."""

    name: str
    variant: str
    #: Copies of every training row with every optional field blanked.
    all_copies: int = 0
    #: Copies with a random subset blanked, its size uniform from 1 to all of them.
    random_copies: int = 0

    @property
    def is_committed(self) -> bool:
        return self.all_copies == 0 and self.random_copies == 0


BASIC, EXTENDED = "lgbm-basic", "lgbm-extended"
CANDIDATE = "basic"
CONFIGURATIONS = (
    Configuration(CANDIDATE, BASIC),
    Configuration("extended", EXTENDED),
    Configuration("extended+all", EXTENDED, all_copies=1),
    Configuration("extended+random", EXTENDED, random_copies=1),
    Configuration("extended+all+random", EXTENDED, all_copies=1, random_copies=1),
    Configuration("extended+all+3random", EXTENDED, all_copies=1, random_copies=3),
    Configuration("basic+all+random", BASIC, all_copies=1, random_copies=1),
)


# --------------------------------------------------------------------------------------------
# Blanking: per row, through the seam's own definition of an omitted field
# --------------------------------------------------------------------------------------------


def blank(frame: pd.DataFrame, model: Model, fields: tuple[str, ...], mask: np.ndarray) -> pd.DataFrame:
    """`frame` with field `fields[j]` of row `i` set to an omitted field wherever `mask[i, j]`.

    The values come from `Model.mask_absent`, which blanks a field for every row; this only picks,
    row by row, between the listing's value and that one. So a row blanked here is exactly a row
    of `mask_absent` in the columns it blanks, and the listing's own row in the others.
    """
    if mask.shape != (len(frame), len(fields)):
        raise ValueError(f"the mask is {mask.shape} for {len(frame)} rows and {len(fields)} fields")
    absent = model.mask_absent(frame, fields)
    blanked = frame.copy()
    for index, field in enumerate(fields):
        keep = pd.Series(~mask[:, index], index=frame.index)
        for column in model.columns_of(field):
            blanked[column] = frame[column].where(keep, absent[column])
    changed = [name for name in frame.columns if blanked[name].dtype != frame[name].dtype]
    if changed:
        raise TypeError(f"blanking changed the dtype of {changed}, so the rows left the contract")
    return blanked


def subsets(rng: np.random.Generator, sizes: np.ndarray, n_fields: int) -> np.ndarray:
    """A rows x fields mask with exactly `sizes[i]` fields of row `i` set, chosen at random."""
    rank = rng.random((len(sizes), n_fields)).argsort(axis=1).argsort(axis=1)
    return rank < sizes[:, None]


def check_blank(frame: pd.DataFrame, model: Model, fields: tuple[str, ...]) -> None:
    """The two identities `blank` has to satisfy, on the frame it is about to be used on."""
    nothing = np.zeros((len(frame), len(fields)), dtype=bool)
    pd.testing.assert_frame_equal(blank(frame, model, fields, nothing), frame)
    pd.testing.assert_frame_equal(blank(frame, model, fields, ~nothing), model.mask_absent(frame, fields))


def training_frame(
    configuration: Configuration, train: pd.DataFrame, model: Model, fields: tuple[str, ...], seed: int
) -> pd.DataFrame:
    """The training rows, followed by the configuration's blanked copies of them."""
    rng = np.random.default_rng([seed, 116])
    n_rows, n_fields = len(train), len(fields)
    frames = [train]
    for _ in range(configuration.all_copies):
        frames.append(blank(train, model, fields, np.ones((n_rows, n_fields), dtype=bool)))
    for _ in range(configuration.random_copies):
        sizes = rng.integers(1, n_fields + 1, size=n_rows)
        frames.append(blank(train, model, fields, subsets(rng, sizes, n_fields)))
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else train


def request_masks(n_rows: int, n_fields: int, seed: int) -> dict[int, np.ndarray]:
    """Per request level, which of the extended set's optional fields each row leaves out."""
    rng = np.random.default_rng([seed, 116, 1])
    return {
        given: subsets(rng, np.full(n_rows, n_fields - given), n_fields) for given in GIVEN_LEVELS
    }


# --------------------------------------------------------------------------------------------
# One configuration: fitted, measured, tracked and scored
# --------------------------------------------------------------------------------------------


@dataclass
class Result:
    configuration: Configuration
    training: dict
    n_train_rows: int
    cost: FitMeasurement
    #: Per request level: the absolute percentage error and the absolute log error of every row.
    ape: dict[int, np.ndarray]
    log_error: dict[int, np.ndarray]
    sc06: list[dict]
    run_id: str | None

    def mdape(self, given: int) -> float:
        return float(np.median(self.ape[given]))

    @property
    def mean_mdape(self) -> float:
        return float(np.mean([self.mdape(given) for given in GIVEN_LEVELS]))

    @property
    def worst_single(self) -> dict:
        singles = [row for row in self.sc06 if row["field"] != evaluate.ALL_OPTIONAL_FIELDS]
        return max(singles, key=lambda row: row["mdape_ratio"])

    @property
    def required_only_ratio(self) -> float:
        (row,) = [row for row in self.sc06 if row["field"] == evaluate.ALL_OPTIONAL_FIELDS]
        return float(row["mdape_ratio"])


@dataclass(frozen=True)
class Variant:
    """What the configurations of one variant share: its matrices, its fields, its seam."""

    name: str
    data: TrainingData
    #: The committed bundle. Its feature space is the matrices', so it is the seam the training
    #: frames are blanked through before any model of this script exists.
    committed: Model
    #: The optional fields, under the names the seam knows them by: the fields of SC-06's
    #: all-at-once scenario, so "all blanked" here is that scenario by construction.
    optional: tuple[str, ...]
    scenarios: tuple[evaluate.MaskingScenario, ...]
    #: Where each of this variant's optional fields sits among the extended set's.
    position: tuple[int, ...]


def score(model: Model, variant: Variant, masks: dict[int, np.ndarray]) -> tuple[dict, dict, list]:
    """The per-row errors of every request level, and the gate's own SC-06 sweep."""
    validation = variant.data.validation
    price = validation["price"]
    log_price = validation["log_price"].to_numpy(dtype="float64")
    ape, log_error = {}, {}
    for given, mask in masks.items():
        request = blank(validation, model, variant.optional, mask[:, list(variant.position)])
        predicted = model.predict_eur(request)
        ape[given] = evaluate.relative_error(price, predicted).to_numpy(dtype="float64")
        log_error[given] = np.abs(np.log(predicted.to_numpy(dtype="float64")) - log_price)
    full = evaluate.point_metrics(price, model.predict_eur(validation))["mdape"]
    sc06 = evaluate.masking_sweep(model, validation, variant.scenarios, full_mdape=full)
    # The level scenarios and the gate's own sweep are two routes to the same two requests.
    (required_only,) = [row for row in sc06 if row["field"] == evaluate.ALL_OPTIONAL_FIELDS]
    assert np.isclose(np.median(ape[GIVEN_LEVELS[-1]]), full, rtol=0, atol=1e-12)
    assert np.isclose(np.median(ape[0]), required_only["mdape"], rtol=0, atol=1e-12)
    return ape, log_error, sc06


def run_configuration(
    configuration: Configuration,
    variant: Variant,
    masks: dict[int, np.ndarray],
    params: dict,
    *,
    workdir: Path,
    smoke: bool,
) -> Result:
    settings = params["train"]["variants"][configuration.variant]
    train = variant.data.train
    if smoke:
        settings = {**settings, "params": {**settings["params"], "n_estimators": SMOKE_TREES}}
        train = train.iloc[:SMOKE_ROWS]
    data = TrainingData(
        space=variant.data.space,
        train=training_frame(configuration, train, variant.committed, variant.optional, params["seed"]),
        validation=variant.data.validation,
        supported_makes=variant.data.supported_makes,
    )
    tags = {
        "sweep": SWEEP,
        "sweep_role": "point",
        "point": configuration.name,
        "is_reference": str(configuration.name == CANDIDATE).lower(),
        "variant": configuration.variant,
    }
    with _run(f"{SWEEP}-{configuration.name}", smoke=smoke) as run:
        # Before the fit, so the run of a fit that fails still names its sweep and point.
        run.set_tags(tags if smoke else {**tags, **input_tags(f"train@{configuration.variant}")})
        model, cost = measure(
            lambda: fit_variant(
                configuration.variant,
                settings,
                data,
                seed=params["seed"],
                num_threads=params["train"]["num_threads"],
            ),
            name=f"{SWEEP} {configuration.name}",
            settings=EnergySettings.from_params(params["train"]["energy"]),
            output_file=workdir / EMISSIONS_FILE,
        )
        if configuration.is_committed and not smoke:
            _check_it_is_the_committed_model(model, variant)
        ape, log_error, sc06 = score(model, variant, masks)
        result = Result(
            configuration, model.metadata["training"], len(data.train), cost, ape, log_error, sc06, run.run_id
        )
        log_fit(
            run,
            model,
            cost=cost,
            params={
                **logged_params(configuration.variant, data, params),
                **cost.params(),
                "blank.all_copies": configuration.all_copies,
                "blank.random_copies": configuration.random_copies,
                "blank.optional_fields": len(variant.optional),
            },
            tags=tags,
        )
        run.log_metrics(
            {
                "validation_mdape": result.mdape(GIVEN_LEVELS[-1]),
                "validation_mdape_required_only": result.mdape(0),
                "validation_mdape_mean_over_levels": result.mean_mdape,
                **{f"validation_mdape_given_{given}": result.mdape(given) for given in GIVEN_LEVELS},
                "validation_sc06_required_only_ratio": result.required_only_ratio,
                "validation_sc06_worst_single_ratio": result.worst_single["mdape_ratio"],
            }
        )
    logger.warning(
        f"{configuration.name}: {len(data.train):,} rows, best round "
        f"{result.training['best_iteration']}, {cost.fit_seconds:.0f} s, MdAPE complete "
        f"{result.mdape(GIVEN_LEVELS[-1]):.2%}, required only {result.mdape(0):.2%}"
    )
    return result


def _check_it_is_the_committed_model(model: Model, variant: Variant) -> None:
    """The refit of a committed variant has to predict what its committed bundle predicts."""
    validation = variant.data.validation
    refit = model.predict_eur(validation).to_numpy()
    committed = variant.committed.predict_eur(validation).to_numpy()
    if not np.allclose(refit, committed, rtol=1e-9, atol=0):
        raise RuntimeError(
            f"the refit of {variant.name} does not reproduce the committed bundle's validation "
            f"predictions (largest relative difference {np.max(np.abs(refit / committed - 1)):.3g}), "
            f"so this script's fit is not the pipeline's. Are models/ and the matrices from one "
            f"`dvc pull`, and are the library versions the lock's?"
        )


def _run(name: str, *, smoke: bool):
    """A run that must open, as a search's does (EDN-78); a smoke run logs nothing."""
    if smoke:
        return nullcontext(DisabledRun())
    return tune._tracked(tune.TUNING_EXPERIMENT, name)


# --------------------------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------------------------


def differences(results: dict[str, Result], given: int, groups: np.ndarray, seed: int, errors: str):
    """Every configuration against the candidate on one request level, by the search's bootstrap."""
    statistic, plain = (
        (tune.median_error, np.median) if errors == "ape" else (tune.mean_error, np.mean)
    )
    per_row = {name: getattr(result, errors)[given] for name, result in results.items()}
    _, comparisons = tune.compare(per_row, CANDIDATE, statistic, plain, tune.Bootstrap(groups, seed))
    return {comparison.point: comparison for comparison in comparisons}


def verdict(results: dict[str, Result], sparse: dict, complete: dict, threshold: float) -> dict:
    checks = {}
    for name, result in results.items():
        configuration = result.configuration
        if configuration.variant != EXTENDED or configuration.is_committed:
            continue
        checks[name] = {
            "sc06_passes": bool(
                result.worst_single["mdape_ratio"] <= threshold
                and result.required_only_ratio <= threshold
            ),
            "required_only_not_worse_than_candidate": bool(sparse[name].per_point[0] <= 0),
            "complete_better_than_candidate": bool(complete[name].simultaneous[1] < 0),
            "early_stopped_within_budget": bool(
                result.training["early_stopped"] and result.cost.fit_seconds < FIT_BUDGET_SECONDS
            ),
        }
    qualifying = [name for name, passed in checks.items() if all(passed.values())]
    chosen = min(qualifying, key=lambda name: results[name].mean_mdape) if qualifying else None
    return {"checks": checks, "qualifying": qualifying, "recommended": chosen}


# --------------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------------


def pct(value: float, digits: int = 2) -> str:
    return f"{value * _PERCENT:.{digits}f} %"


def pp(value: float) -> str:
    return f"{value * _PERCENT:+.2f}"


def interval(pair: tuple[float, float]) -> str:
    return f"[{pp(pair[0])}, {pp(pair[1])}]"


def title(lines: list[str], text: str) -> None:
    lines.extend(["", "=" * 110, text, "=" * 110])


def report(results: dict[str, Result], comparisons: dict, outcome: dict, context: dict) -> list[str]:
    names = list(results)
    width = max(len(name) for name in names) + 2
    lines = [
        "# reports/analysis/blanked_fields.py on the real snapshot" + ("  [SMOKE RUN, NOT EVIDENCE]" if context["smoke"] else ""),
        f"issue #116, EDN-82; MLflow experiment {tune.TUNING_EXPERIMENT}, sweep {SWEEP}; seed {context['seed']}",
        f"train {context['n_train']:,} rows, validation {context['n_validation']:,} rows in "
        f"{context['n_groups']:,} seller groups; the test and calibration splits are not read",
        f"optional fields: {context['n_optional'][EXTENDED]} for lgbm-extended, "
        f"{context['n_optional'][BASIC]} for lgbm-basic",
    ]

    title(lines, "1. What was fitted")
    lines.append(
        f"{'configuration':{width}}{'train rows':>11}{'best round':>12}{'early stopped':>15}"
        f"{'fit s':>8}{'fit CPU s':>11}{'energy Wh':>11}"
    )
    for name, result in results.items():
        lines.append(
            f"{name:{width}}{result.n_train_rows:>11,}{result.training['best_iteration']:>12,}"
            f"{'yes' if result.training['early_stopped'] else 'NO':>15}"
            f"{result.cost.fit_seconds:>8.1f}{result.cost.fit_cpu_seconds:>11.1f}"
            f"{result.cost.metrics()['energy_kwh'] * 1000:>11.4f}"
        )
    lines.append(
        "The two committed variants are refitted unchanged; their validation predictions equal the "
        "committed bundles'."
    )

    title(
        lines,
        f"2. Validation MdAPE by how many of the {GIVEN_LEVELS[-1]} optional fields a request gives",
    )
    header = "".join(f"{given:>9}" for given in GIVEN_LEVELS)
    lines.append(f"{'configuration':{width}}{header}{'mean':>9}")
    lines.append(f"{'':{width}}{'required':>9}{'':>{9 * (len(GIVEN_LEVELS) - 2)}}{'complete':>9}")
    for name, result in results.items():
        cells = "".join(f"{pct(result.mdape(given)):>9}" for given in GIVEN_LEVELS)
        lines.append(f"{name:{width}}{cells}{pct(result.mean_mdape):>9}")
    routed = [min(results[CANDIDATE].mdape(given), results["extended"].mdape(given)) for given in GIVEN_LEVELS]
    lines.append(
        f"{'best committed':{width}}" + "".join(f"{pct(value):>9}" for value in routed)
        + f"{pct(float(np.mean(routed))):>9}"
    )
    lines.append(
        "'best committed' is the better of basic and extended per column: the most that serving both "
        "models could reach."
    )

    title(lines, "3. The same in validation L1 of the log price (reported, decides nothing)")
    lines.append(f"{'configuration':{width}}{header}")
    for name, result in results.items():
        cells = "".join(f"{np.mean(result.log_error[given]):>9.4f}" for given in GIVEN_LEVELS)
        lines.append(f"{name:{width}}{cells}")

    title(lines, f"4. SC-06 on the validation split (bound {context['threshold']}x)")
    lines.append(
        f"{'configuration':{width}}{'complete':>10}{'worst single field':>34}{'required only':>22}{'passes':>8}"
    )
    for name, result in results.items():
        worst = result.worst_single
        single = f"{worst['field']} {worst['mdape_ratio']:.3f}x"
        required = f"{pct(result.mdape(0))} {result.required_only_ratio:.3f}x"
        passes = max(worst["mdape_ratio"], result.required_only_ratio) <= context["threshold"]
        lines.append(
            f"{name:{width}}{pct(result.mdape(GIVEN_LEVELS[-1])):>10}{single:>34}{required:>22}"
            f"{'yes' if passes else 'NO':>8}"
        )

    title(
        lines,
        "5. MdAPE difference to basic, the candidate, in percentage points "
        "(paired bootstrap by seller group, 2,000 draws, 95 %)",
    )
    for label, key in (("required-only request", "sparse"), ("complete request", "complete")):
        lines.append(f"{label}:")
        lines.append(f"  {'configuration':{width}}{'difference':>11}{'per-point':>20}{'simultaneous':>20}")
        for name, comparison in comparisons[key].items():
            lines.append(
                f"  {name:{width}}{pp(comparison.difference):>11}{interval(comparison.per_point):>20}"
                f"{interval(comparison.simultaneous):>20}"
            )

    title(lines, "6. What blanking costs on a complete request, against extended as committed")
    lines.append(f"{'configuration':{width}}{'MdAPE':>9}{'difference pp':>15}{'L1 log price':>14}{'difference':>12}")
    reference = results["extended"]
    full = GIVEN_LEVELS[-1]
    for name, result in results.items():
        if result.configuration.variant != EXTENDED:
            continue
        lines.append(
            f"{name:{width}}{pct(result.mdape(full)):>9}{pp(result.mdape(full) - reference.mdape(full)):>15}"
            f"{np.mean(result.log_error[full]):>14.4f}"
            f"{np.mean(result.log_error[full]) - np.mean(reference.log_error[full]):>+12.4f}"
        )

    title(lines, "7. The rule fixed before the fits")
    columns = ("SC-06", "sparse: not worse", "complete: better", "early stop, < 15 min")
    lines.append(f"{'configuration':{width}}" + "".join(f"{column:>22}" for column in columns) + f"{'qualifies':>11}")
    for name, passed in outcome["checks"].items():
        cells = "".join(f"{'yes' if value else 'NO':>22}" for value in passed.values())
        lines.append(f"{name:{width}}{cells}{'yes' if all(passed.values()) else 'NO':>11}")
    if outcome["recommended"] is None:
        lines.append("No blanked configuration qualifies.")
    else:
        lines.append(
            f"Qualifying: {', '.join(outcome['qualifying'])}. Lowest mean MdAPE over the levels: "
            f"{outcome['recommended']}. Adopting it is the team's decision (EDN-62), not this script's."
        )
    totals = [result.cost for result in results.values()]
    lines.append("")
    lines.append(
        f"totals: {len(totals)} fits, {sum(cost.fit_seconds for cost in totals):.1f} s of fitting, "
        f"{sum(cost.metrics()['energy_kwh'] for cost in totals) * 1000:.4f} Wh, "
        f"{sum(cost.metrics()['emissions_kg_co2eq'] for cost in totals) * 1e6:.2f} mg CO2eq"
    )
    return lines


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------


def load_variant(name: str, params: dict, extended_optional: tuple[str, ...] | None) -> Variant:
    settings = params["train"]["variants"][name]
    supported = read_supported_makes(PROCESSED_DATA_DIR)
    data = read_matrices(PROCESSED_DATA_DIR / "features", settings["feature_set"], supported)
    committed = load_model(MODELS_DIR / name)
    if committed.features != data.space.schema.feature_names:
        raise RuntimeError(f"models/{name} and its matrices describe different feature spaces")
    fields = evaluate.input_field_columns(committed, params)
    scenarios = evaluate.sc06_scenarios(fields, evaluate.optional_input_fields(fields, params), params)
    everything = scenarios[-1]
    if everything.field != evaluate.ALL_OPTIONAL_FIELDS:
        raise RuntimeError("SC-06's last scenario is expected to be the all-at-once one")
    optional = everything.fields
    reference = extended_optional or optional
    return Variant(name, data, committed, optional, scenarios, tuple(reference.index(f) for f in optional))


def main(smoke: bool) -> None:
    # The stages log through loguru; only warnings are kept, on stderr, beside this script's own.
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    params = load_params(PARAMS_FILE)
    seed = params["seed"]
    threshold = params["evaluate"]["success_criteria"]["sc06_masked_mdape_ratio_max"]

    extended = load_variant(EXTENDED, params, None)
    basic = load_variant(BASIC, params, extended.optional)
    variants = {EXTENDED: extended, BASIC: basic}
    if len(extended.optional) != GIVEN_LEVELS[-1]:
        raise RuntimeError(f"the request levels assume {GIVEN_LEVELS[-1]} optional fields, not {len(extended.optional)}")
    if not np.array_equal(extended.data.validation["price"].to_numpy(), basic.data.validation["price"].to_numpy()):
        raise RuntimeError("the two validation matrices do not hold the same rows in the same order")
    for variant in variants.values():
        check_blank(variant.data.validation, variant.committed, variant.optional)

    groups = tune.validation_seller_groups(PROCESSED_DATA_DIR, extended.data, extended.data.supported_makes)
    masks = request_masks(len(extended.data.validation), len(extended.optional), seed)
    if not smoke:
        tune._require_tracking_configured()
        tune._refuse_a_tracked_sweep_id(SWEEP, tune.TUNING_EXPERIMENT)

    with tempfile.TemporaryDirectory() as directory:
        workdir = Path(directory)
        results = {
            configuration.name: run_configuration(
                configuration, variants[configuration.variant], masks, params, workdir=workdir, smoke=smoke
            )
            for configuration in CONFIGURATIONS
        }
        comparisons = {
            "sparse": differences(results, 0, groups, seed, "ape"),
            "complete": differences(results, GIVEN_LEVELS[-1], groups, seed, "ape"),
        }
        outcome = verdict(results, comparisons["sparse"], comparisons["complete"], threshold)
        context = {
            "smoke": smoke,
            "seed": seed,
            "threshold": threshold,
            "n_train": len(extended.data.train) if not smoke else SMOKE_ROWS,
            "n_validation": len(extended.data.validation),
            "n_groups": len(np.unique(groups)),
            "n_optional": {name: len(variant.optional) for name, variant in variants.items()},
        }
        lines = report(results, comparisons, outcome, context)
        print("\n".join(lines))
        if smoke:
            return
        (workdir / RESULTS_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")
        record = {
            "sweep": SWEEP,
            "given_levels": list(GIVEN_LEVELS),
            "verdict": outcome,
            "configurations": {
                name: {
                    "variant": result.configuration.variant,
                    "all_copies": result.configuration.all_copies,
                    "random_copies": result.configuration.random_copies,
                    "mlflow_run_id": result.run_id,
                    "n_train_rows": result.n_train_rows,
                    "best_iteration": result.training["best_iteration"],
                    "early_stopped": result.training["early_stopped"],
                    "fit_seconds": result.cost.fit_seconds,
                    "energy_kwh": result.cost.metrics()["energy_kwh"],
                    "validation_mdape_by_given": {str(g): result.mdape(g) for g in GIVEN_LEVELS},
                    "validation_l1_by_given": {str(g): float(np.mean(result.log_error[g])) for g in GIVEN_LEVELS},
                    "sc06": result.sc06,
                }
                for name, result in results.items()
            },
            "mdape_difference_to_candidate": {
                key: {
                    name: {
                        "difference": comparison.difference,
                        "per_point": comparison.per_point,
                        "simultaneous": comparison.simultaneous,
                    }
                    for name, comparison in table.items()
                }
                for key, table in comparisons.items()
            },
        }
        (workdir / RECORD_FILE).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        with tune._tracked(tune.TUNING_EXPERIMENT, f"{SWEEP}-summary") as run:
            run.set_tags(
                {
                    "sweep": SWEEP,
                    "sweep_role": "summary",
                    "verdict": "none qualifies" if outcome["recommended"] is None else "qualifies",
                    "recommended": outcome["recommended"] or "none",
                    "reference_point": CANDIDATE,
                }
            )
            run.log_params({"seed": seed, "given_levels": str(GIVEN_LEVELS), "bootstrap.draws": tune.BOOTSTRAP_DRAWS})
            run.log_metrics(
                {
                    "n_points": float(len(results)),
                    "total_fit_seconds": sum(result.cost.fit_seconds for result in results.values()),
                    "total_energy_kwh": sum(result.cost.metrics()["energy_kwh"] for result in results.values()),
                    "n_qualifying": float(len(outcome["qualifying"])),
                }
            )
            run.log_artifacts(workdir, artifact_path=tune.ARTIFACT_PATH)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv[1:])
