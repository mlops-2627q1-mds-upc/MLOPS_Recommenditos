"""`compare-energy` stage: the ladder's training energy against its test error (#38).

The course demo stops at logging a number per run. The four variants of the
experiment ladder differ one rung at a time, so the number can answer a question:
whether a rung earned what it cost to train. This stage puts the two halves side
by side, one row per variant, as `reports/figures/energy-vs-error.csv` and the
figure drawn from it, `reports/figures/energy-vs-error.pdf`.

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
of a fit approximately its duration times 13 to 18 W on the i5-10210U (the fits use more than one
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
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
import typer

from recommenditos import plots
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
FIGURE_FILE = f"energy-vs-error{plots.FIGURE_SUFFIX}"

_WH_PER_KWH = 1000.0
_MG_PER_KG = 1e6
_PERCENT = 100.0

#: The figure's size in inches, which is the size the report prints it at: two
#: thirds of the text width.
_FIGURE_SIZE_IN = (4.2, 3.0)

#: Closer than this, in percentage points, the dot sits inside the ring and the
#: line between the two is hidden, so the pair's label says how far apart they are.
_HIDDEN_BELOW_PP = 0.3
#: How far a mark reaches from its centre, in points, for a label to stay clear of.
_MARK_REACH_PT = 3.6
_POINTS_PER_INCH = 72.0

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


def pair_label(variant: str, low: float, high: float | None) -> str:
    """What a pair of points is called: its variant, and what its marks cannot show.

    B0 reads three columns, all required, so its two points are one: the same
    predictions and so exactly the same MdAPE. A variant trained to do without
    its optional fields has two points so close that the dot sits inside the
    ring. In both cases the pair has no visible line, and the label says why
    rather than leaving a reader to look for a missing mark.
    """
    if high is None:
        return variant
    if high == low:
        return f"{variant} (both)"
    if abs(high - low) < _HIDDEN_BELOW_PP:
        return f"{variant} ({abs(high - low):.2f} points apart)"
    return variant


def _label_pairs(axis: Axes, pairs: list[tuple[str, float, float, float | None]]) -> None:
    """Name each pair over its upper mark, where the name crosses no mark and no line.

    Centred over the mark first. Two variants of similar training cost stand
    side by side, and a centred name would then run into the neighbour's line,
    so the name is next tried ending at its mark and then starting at it. With
    no free place it is centred, as a name on a line is still better than none.
    The layout has to be final when this runs, because it measures the text.
    """
    renderer = plots.renderer(axis)
    reach = _MARK_REACH_PT * axis.figure.dpi / _POINTS_PER_INCH
    taken = []
    for _, x, low, high in pairs:
        ends = [axis.transData.transform((x, y)) for y in (low, high) if y is not None]
        taken += [
            Bbox.from_extents(cx - reach, cy - reach, cx + reach, cy + reach) for cx, cy in ends
        ]
        if len(ends) > 1:
            taken.append(Bbox.from_extents(ends[0][0] - 1, ends[0][1], ends[0][0] + 1, ends[1][1]))
    for text, x, low, high in pairs:
        places = (("center", 0), ("right", 4), ("left", -4), ("center", 0))
        for attempt, (alignment, shift_pt) in enumerate(places):
            label = axis.annotate(
                text,
                (x, max(low, high or low)),
                xytext=(shift_pt, 7),
                textcoords="offset points",
                ha=alignment,
                fontsize=plots.LABEL_PT,
                color=plots.INK,
            )
            box = label.get_window_extent(renderer)
            last = attempt == len(places) - 1
            if last or not any(box.overlaps(other) for other in taken):
                taken.append(box)
                break
            label.remove()


def figure_of(rows: list[dict]) -> Figure:
    """Energy on a log axis against MdAPE, two points per variant.

    A log axis because the ladder spans two orders of magnitude of training cost,
    from a group-by to a thousand trees. Each variant is drawn twice, joined by a
    thin line: its MdAPE on a fully described car, and on a request carrying only
    the fields FR-01 requires. The second is what decided the candidate (EDN-62),
    and it is the one an efficiency question has to see, because training energy
    spent on optional columns is spent whether or not a caller sends them. Two
    series, so a legend, above the axes; each pair is labelled with its variant
    once (`pair_label`, `_label_pairs`).

    The two series differ in shape as well as in colour, a filled dot and an open
    ring, because the two colours print as the same grey. The ring is the larger
    of the two, so where both conditions give the same error the dot shows inside
    it and the reader sees that the two coincide.

    The x axis names the method, because a reader of the picture alone would
    otherwise take it for a meter reading. `plots` says why the figure has no
    title and is drawn at the size it is printed at.
    """
    figure = plots.new_figure(*_FIGURE_SIZE_IN)
    # The names of the pairs are placed by their measured size, which needs a
    # canvas that can render.
    FigureCanvasAgg(figure)
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
            axis.plot([x, x], [low, high], color=plots.MUTED, linewidth=1.2, zorder=2)
    shown = [(x, y) for x, y in zip(energy, required, strict=True) if y is not None]
    axis.scatter(
        [x for x, _ in shown],
        [y for _, y in shown],
        s=46,
        facecolors="white",
        edgecolors=plots.ACCENT,
        linewidths=1.6,
        zorder=3,
        label="only the required fields",
    )
    axis.scatter(
        energy,
        full,
        s=30,
        color=plots.MARK,
        edgecolors="white",
        linewidths=1.0,
        zorder=4,
        label="every field given",
    )
    axis.set_xscale("log")
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    methods = sorted({str(row["cpu_power_method"]) for row in rows})
    axis.set_xlabel(
        "Training energy of the fit, Wh (log scale)\n"
        f"CodeCarbon's {', '.join(methods)} estimate, not a meter reading",
        fontsize=plots.AXIS_PT,
        color=plots.INK_SECONDARY,
    )
    axis.set_ylabel("Test MdAPE, %", fontsize=plots.AXIS_PT, color=plots.INK_SECONDARY)
    # Above the axes, not inside them: a frameless legend in the data area puts
    # its two handles among the points, where they read as a fifth variant. The
    # filled dot first, as it is the condition the other one is compared with.
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        handles[::-1],
        labels[::-1],
        fontsize=plots.LABEL_PT,
        frameon=False,
        labelcolor=plots.INK_SECONDARY,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncols=2,
        borderaxespad=0.3,
        borderpad=0.0,
        handletextpad=0.2,
        columnspacing=1.2,
    )
    plots.style_axes(axis)
    axis.margins(x=0.2, y=0.18)
    figure.tight_layout()
    _label_pairs(
        axis,
        [
            (pair_label(row["variant"], low, high), x, low, high)
            for row, x, low, high in zip(rows, energy, full, required, strict=True)
        ],
    )
    return figure


def draw_figure(path: Path, rows: list[dict]) -> None:
    plots.save_figure(figure_of(rows), path)
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
