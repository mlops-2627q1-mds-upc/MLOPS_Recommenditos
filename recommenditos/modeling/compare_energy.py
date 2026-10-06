"""`compare-energy` stage: the ladder's training energy against its test error (#38).

The course demo stops at logging a number per run. The four variants of the
experiment ladder differ one rung at a time, so the number can answer a question:
whether a rung earned what it cost to train. This stage puts the two halves side
by side, one row per variant, as `reports/figures/energy-vs-error.csv` and the
figure drawn from it, `reports/figures/energy-vs-error.png`.

It is a stage rather than a notebook cell so that the table and the figure are
built from the pipeline's artefacts and nothing else, and `dvc repro` rebuilds
them whenever a model, its energy record or its metrics change. It is a stage of
its own rather than part of `evaluate` so that a change to how the figure is
drawn reruns this and not the evaluation.

Every row is joined on identifiers rather than on file names. The energy row is
the CodeCarbon run id the model's own `model.json` recorded, so a CSV left over
from another fit cannot be read as this model's cost; and the metrics are refused
if `evaluate` recorded a different MLflow run than the model carries, which is
what a stale `reports/metrics` would look like.

**How to read the energy column.** On every machine we train on, CodeCarbon has no
read access to an energy counter and estimates (`recommenditos/modeling/energy.py`
says how): a constant 10 W for the RAM plus the CPU's TDP times the share of the
machine's CPUs the fit kept busy. That makes the energy
of a fit its duration times 13 to 18 W on the i5-10210U (the fits use more than one
thread although `train.num_threads` is 1), so the ratios between rows follow the
fit-time ratios closely without equalling them, and a fit that shared the
machine with other work is charged for the extra seconds it took. The table
carries `cpu_power_method`, the fit times and the machine's CPU load during the
fit beside the energy, so it says all of that itself.
"""

import csv
import json
from pathlib import Path

from loguru import logger
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import typer

from recommenditos.config import FIGURES_DIR, MODELS_DIR, PARAMS_FILE, REPORTS_DIR
from recommenditos.modeling.energy import EMISSIONS_DIR, read_record
from recommenditos.modeling.evaluate import ALL_OPTIONAL_FIELDS
from recommenditos.modeling.model import load_model
from recommenditos.pipeline import load_params

#: The table's columns, in the order the file carries them, which is what a LaTeX
#: table written against it reads positionally.
COLUMNS: tuple[str, ...] = (
    "variant",
    "estimator",
    "feature_set",
    "n_features",
    "n_train_rows",
    "fit_seconds",
    "fit_cpu_seconds",
    "energy_wh",
    "cpu_energy_wh",
    "ram_energy_wh",
    "emissions_mg_co2eq",
    "cpu_power_method",
    "cpu_tdp_w",
    "country_iso_code",
    "machine_cpu_utilization_pct",
    "mdape",
    "within_20pct",
    "mdape_required_fields_only",
)

TABLE_FILE = "energy-vs-error.csv"
FIGURE_FILE = "energy-vs-error.png"

_WH_PER_KWH = 1000.0
_MG_PER_KG = 1e6
_PERCENT = 100.0

# The figure's colours: the first two categorical slots of a palette checked
# for colour-vision deficiency, text and grid in neutral inks.
_MARK = "#2a78d6"
_MARK_SECONDARY = "#eb6834"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRID = "#e4e3df"

app = typer.Typer()


class ComparisonError(ValueError):
    """The artefacts of one variant do not describe the same fit."""


def variant_row(variant: str, models_dir: Path, emissions_dir: Path, metrics_dir: Path) -> dict:
    """One variant's cost and error, joined on the identifiers the artefacts carry."""
    model = load_model(models_dir / variant)
    fit = model.metadata.get("fit")
    if fit is None:
        raise ComparisonError(
            f"{models_dir / variant} records no `fit` block, so it was trained before its "
            f"energy was measured. Retrain it: `dvc repro train@{variant}`."
        )
    record = read_record(emissions_dir / f"{variant}.csv", fit["codecarbon_run_id"])
    metrics = json.loads((metrics_dir / f"{variant}.json").read_text(encoding="utf-8"))
    if metrics["mlflow_run_id"] != model.metadata["mlflow"]["run_id"]:
        raise ComparisonError(
            f"{metrics_dir / variant}.json was measured on MLflow run "
            f"{metrics['mlflow_run_id']!r} and the model in {models_dir / variant} belongs to "
            f"{model.metadata['mlflow']['run_id']!r}, so the error and the energy would describe "
            f"two different fits. Rerun `evaluate`."
        )
    return {
        "variant": variant,
        "estimator": model.estimator,
        "feature_set": model.feature_set,
        "n_features": len(model.features),
        "n_train_rows": model.metadata["training"]["n_train_rows"],
        "fit_seconds": fit["seconds"],
        "fit_cpu_seconds": fit["cpu_seconds"],
        "energy_wh": float(record["energy_consumed"]) * _WH_PER_KWH,
        "cpu_energy_wh": float(record["cpu_energy"]) * _WH_PER_KWH,
        "ram_energy_wh": float(record["ram_energy"]) * _WH_PER_KWH,
        "emissions_mg_co2eq": float(record["emissions"]) * _MG_PER_KG,
        "cpu_power_method": fit["cpu_power_method"],
        "cpu_tdp_w": fit["cpu_tdp_w"],
        "country_iso_code": record["country_iso_code"],
        "machine_cpu_utilization_pct": float(record["cpu_utilization_percent"]),
        "mdape": metrics["mdape"],
        "within_20pct": metrics["within_20pct"],
        "mdape_required_fields_only": _required_fields_only(metrics),
    }


def _required_fields_only(metrics: dict) -> float | None:
    """MdAPE on a request carrying only the fields FR-01 requires.

    That is SC-06's all-at-once scenario, and it belongs in an efficiency table
    because it is the request on which the extended feature set's extra columns
    are all absent: what a rung costs to train is spent whether or not a caller
    sends what it was trained on. `None` for a feature set with no optional field.
    """
    for row in metrics["masked_inputs"]:
        if row["criterion"] == "sc06" and row["field"] == ALL_OPTIONAL_FIELDS:
            return row["mdape"]
    return None


def write_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} row(s) to {path}.")


def draw_figure(path: Path, rows: list[dict]) -> None:
    """Energy on a log axis against MdAPE, two points per variant.

    A log axis because the ladder spans two orders of magnitude of training cost,
    from a group-by to a thousand trees. Each variant is drawn twice, joined by a
    thin line: its MdAPE on a fully described car, and on a request carrying only
    the fields FR-01 requires. The second is what decided the candidate (EDN-62),
    and it is the one an efficiency question has to see, because training energy
    spent on optional columns is spent whether or not a caller sends them. Two
    series, so a legend; each pair is labelled with its variant once.

    The subtitle states the method, because a reader of the picture alone would
    otherwise take the x axis for a meter reading.

    A bare `Figure` rather than `pyplot`, so drawing it touches no global state
    and needs no backend chosen for a process that has no display.
    """
    figure = Figure(figsize=(6.4, 3.8), dpi=200)
    axis = figure.subplots()
    energy = [row["energy_wh"] for row in rows]
    full = [row["mdape"] * _PERCENT for row in rows]
    required = [
        None
        if row["mdape_required_fields_only"] is None
        else row["mdape_required_fields_only"] * _PERCENT
        for row in rows
    ]
    for x, low, high in zip(energy, full, required, strict=True):
        if high is not None:
            axis.plot([x, x], [low, high], color=_GRID, linewidth=2, zorder=2)
    marks = {"s": 48, "edgecolors": "white", "linewidths": 2, "zorder": 3}
    axis.scatter(energy, full, color=_MARK, label="every field given", **marks)
    shown = [(x, y) for x, y in zip(energy, required, strict=True) if y is not None]
    axis.scatter(
        [x for x, _ in shown],
        [y for _, y in shown],
        color=_MARK_SECONDARY,
        label="only the fields FR-01 requires",
        **marks,
    )
    for row, x, low, high in zip(rows, energy, full, required, strict=True):
        # B0 reads three columns, all required, so its two points are one: the same
        # predictions and so exactly the same MdAPE. The label says so rather than
        # leaving a reader to look for the hidden point.
        same = high == low
        axis.annotate(
            f"{row['variant']} (both)" if same else row["variant"],
            (x, max(low, high or low)),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=_INK,
        )
    axis.set_xscale("log")
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_xlabel("Training energy of the fit, Wh (log scale)", fontsize=8, color=_INK_SECONDARY)
    axis.set_ylabel("Test MdAPE, %", fontsize=8, color=_INK_SECONDARY)
    methods = sorted({str(row["cpu_power_method"]) for row in rows})
    axis.set_title(
        "Training energy against test error across the experiment ladder\n"
        f"CodeCarbon estimate ({', '.join(methods)} CPU, 10 W RAM floor), not a meter reading",
        fontsize=8,
        color=_INK,
        loc="left",
    )
    axis.legend(fontsize=7, frameon=False, loc="upper right", labelcolor=_INK_SECONDARY)
    axis.grid(True, which="major", color=_GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=7, colors=_INK_SECONDARY)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(_GRID)
    axis.margins(x=0.2, y=0.18)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    # No `Software` entry, so the bytes depend on the data and the matplotlib
    # version, not on anything else about the machine that drew them.
    figure.savefig(path, metadata={"Software": None})
    logger.info(f"Wrote {path}.")


@app.command()
def main(
    models_dir: Path = MODELS_DIR,
    emissions_dir: Path = EMISSIONS_DIR,
    metrics_dir: Path = REPORTS_DIR / "metrics",
    output_dir: Path = FIGURES_DIR,
    params_path: Path = PARAMS_FILE,
):
    params = load_params(params_path)
    rows = [
        variant_row(variant, models_dir, emissions_dir, metrics_dir)
        for variant in params["train"]["variants"]
    ]
    write_table(output_dir / TABLE_FILE, rows)
    draw_figure(output_dir / FIGURE_FILE, rows)
    for row in rows:
        logger.info(
            f"{row['variant']}: {row['energy_wh']:.4f} Wh, {row['emissions_mg_co2eq']:.2f} mg "
            f"CO2eq, {row['fit_seconds']:.2f} s, MdAPE {row['mdape'] * _PERCENT:.2f} %."
        )


if __name__ == "__main__":
    app()
