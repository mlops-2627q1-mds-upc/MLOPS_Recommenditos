"""`profile-data` stage: how the raw listings spread over makes and countries.

The dataset card's first finding is that the snapshot is concentrated: a few
makes and one country hold most of the rows. That is what the scope rules answer
(EDN-04, EDN-05) and what the holdout country is measured against (EDN-03). This
stage counts both dimensions on the raw file and writes them as
`reports/figures/data-composition.csv` and the figure drawn from it,
`reports/figures/data-composition.pdf`.

It reads the raw file, not the cleaned one, because the scope rules were decided
on what the publisher delivered: the shares here are the ones the dataset card
states. It reads the two columns it counts and nothing else, so the stage never
holds the 75 columns of the file, the personal data among them.

The table carries every level, the missing value included. The figure shows at
most `_MAX_BARS` bars per dimension and folds the rest into one, because the tail
of the makes is seventeen bars nobody could tell apart at this size.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
import typer

from recommenditos import plots
from recommenditos.config import FIGURES_DIR, PARAMS_FILE, RAW_DATA_DIR
from recommenditos.pipeline import load_params

#: The table's columns, in the order the file carries them.
COLUMNS: tuple[str, ...] = ("dimension", "level", "listings", "share")

TABLE_FILE = "data-composition.csv"
FIGURE_FILE = f"data-composition{plots.FIGURE_SUFFIX}"

#: The raw columns counted, each with what a panel calls it and several of it.
DIMENSIONS: dict[str, tuple[str, str]] = {
    "make": ("Make", "makes"),
    "country_code": ("Country", "countries"),
}

#: What a row without a value is called in the table. It is not drawn.
MISSING = "(missing)"

_PERCENT = 100.0
_MAX_BARS = 8

# The layout, in inches at print size. Both panels have the same bar area and
# the same scale, so a bar's length means the same share in either.
_TOP_IN = 0.24
_BOTTOM_IN = 0.03
_ROW_IN = 0.165
_BAR_AREA_IN = 1.7
_PANEL_LEFT_IN: tuple[float, ...] = (1.02, 4.06)
#: A bar's thickness as a share of the space its row has.
_BAR_THICKNESS = 0.62

app = typer.Typer()


@dataclass(frozen=True)
class Bar:
    """One bar of a panel: its label, its share of all rows and what kind it is."""

    label: str
    share: float
    #: `"level"`, `"held out"` for the holdout country, `"rest"` for the folded tail.
    kind: str = "level"


def composition(frame: pd.DataFrame) -> list[dict]:
    """One row per level of each dimension, the largest first, as shares of all rows."""
    rows = []
    for column in DIMENSIONS:
        counts = frame[column].astype("object").fillna(MISSING).value_counts()
        ordered = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
        rows.extend(
            {
                "dimension": column,
                "level": str(level),
                "listings": int(listings),
                "share": int(listings) / len(frame),
            }
            for level, listings in ordered
        )
    return rows


def bars_of(rows: list[dict], dimension: str, holdout_country: str) -> list[Bar]:
    """The bars of one panel: the largest levels, then the rest of them as one bar.

    A dimension with more levels than fit keeps one bar for the rest, so the
    bars still add up to the rows that have a value. The holdout country is
    never part of that rest: it keeps its own bar however small it is, because
    marking it is half of what the panel is for. The missing value is not a
    level and gets no bar.
    """
    levels = [row for row in rows if row["dimension"] == dimension and row["level"] != MISSING]
    shown = levels if len(levels) <= _MAX_BARS else levels[: _MAX_BARS - 1]
    held_out = [
        row for row in levels if dimension == "country_code" and row["level"] == holdout_country
    ]
    if held_out and held_out[0] not in shown:
        shown = [*shown[:-1], held_out[0]]
    bars = []
    for row in shown:
        name = plots.country_name(row["level"]) if dimension == "country_code" else row["level"]
        bars.append(
            Bar(f"{name}, held out", row["share"], "held out")
            if row in held_out
            else Bar(name, row["share"])
        )
    rest = [row for row in levels if row not in shown]
    if rest:
        label = f"{len(rest)} other {DIMENSIONS[dimension][1]}"
        bars.append(Bar(label, sum(row["share"] for row in rest), "rest"))
    return bars


def write_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} row(s) to {path}.")


def _draw_panel(axis: Axes, title: str, bars: list[Bar], longest: float) -> None:
    colours = {"level": plots.MARK, "held out": plots.ACCENT, "rest": plots.MUTED}
    positions = range(len(bars))
    axis.barh(
        positions,
        [bar.share * _PERCENT for bar in bars],
        height=_BAR_THICKNESS,
        color=[colours[bar.kind] for bar in bars],
    )
    for position, bar in zip(positions, bars, strict=True):
        axis.annotate(
            f"{bar.share * _PERCENT:.1f} %",
            (bar.share * _PERCENT, position),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=plots.LABEL_PT,
            color=plots.INK_SECONDARY,
            annotation_clip=False,
        )
    axis.set_xlim(0, longest * _PERCENT)
    axis.set_ylim(len(bars) - 0.5, -0.5)
    axis.set_yticks(positions, [bar.label for bar in bars])
    axis.set_xticks([])
    axis.tick_params(axis="y", length=0, labelsize=plots.LABEL_PT, labelcolor=plots.INK)
    for side in ("top", "right", "bottom"):
        axis.spines[side].set_visible(False)
    # The bars' common baseline, and the only line of the panel.
    axis.spines["left"].set_color(plots.BASELINE)
    axis.set_title(title, loc="left", fontsize=plots.AXIS_PT, color=plots.INK_SECONDARY, pad=5)


def figure_of(rows: list[dict], holdout_country: str) -> Figure:
    """Two panels of horizontal bars, makes and countries, on one scale.

    Bars because the reader compares magnitudes across names, and horizontal
    because the names are long. Every bar carries its share at the tip, so the
    panels need neither an axis nor a grid. The scale is linear from zero and the
    same in both panels, so the largest make and the largest country can be
    compared by length.

    One colour for every level: the bars are one series, and colouring them by
    size would say twice what their length says. Two bars differ, and their
    labels say why, so the figure does not need its colours to be read: the
    holdout country, and the bar that stands for the levels too small to draw.
    """
    panels = {dimension: bars_of(rows, dimension, holdout_country) for dimension in DIMENSIONS}
    longest = max((bar.share for bars in panels.values() for bar in bars), default=1.0)
    most = max((len(bars) for bars in panels.values()), default=1)
    height_in = _TOP_IN + _BOTTOM_IN + most * _ROW_IN
    figure = plots.new_figure(plots.TEXT_WIDTH_IN, height_in)
    for left_in, (dimension, bars) in zip(_PANEL_LEFT_IN, panels.items(), strict=True):
        axis = figure.add_axes(
            (
                left_in / plots.TEXT_WIDTH_IN,
                (height_in - _TOP_IN - len(bars) * _ROW_IN) / height_in,
                _BAR_AREA_IN / plots.TEXT_WIDTH_IN,
                len(bars) * _ROW_IN / height_in,
            )
        )
        _draw_panel(axis, DIMENSIONS[dimension][0], bars, longest)
    return figure


def draw_figure(path: Path, rows: list[dict], holdout_country: str) -> None:
    plots.save_figure(figure_of(rows, holdout_country), path)
    logger.info(f"Wrote {path}.")


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "listings.parquet",
    output_dir: Path = FIGURES_DIR,
    params_path: Path = PARAMS_FILE,
):
    params = load_params(params_path)
    frame = pd.read_parquet(input_path, columns=list(DIMENSIONS))
    rows = composition(frame)
    write_table(output_dir / TABLE_FILE, rows)
    draw_figure(output_dir / FIGURE_FILE, rows, params["split"]["holdout_country"])
    for dimension, (title, _) in DIMENSIONS.items():
        largest = next(row for row in rows if row["dimension"] == dimension)
        logger.info(
            f"{title}: {sum(row['dimension'] == dimension for row in rows)} level(s) over "
            f"{len(frame):,} listings, the largest {largest['level']} with "
            f"{largest['share'] * _PERCENT:.1f} %."
        )


if __name__ == "__main__":
    app()
