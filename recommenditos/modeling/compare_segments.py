"""`compare-segments` stage: the candidate's test error in every level SC-04 checks.

SC-04 is the fairness criterion: the MdAPE may not exceed its bound in any level
of make, country, seller type, fuel category or age band that has enough test
rows. `evaluate` measures every level and the gate reads the worst one, which is
a single number. This stage draws all of them on one axis against the bound, as
`reports/figures/error-by-segment.pdf`, so a reader sees whether a miss is one
level or a pattern and how far the worst level is from the rest.

It is a stage for the reason `compare-energy` is one: the figure is built from
`evaluate`'s record and the bound in `params.yaml` and from nothing else, and
`dvc repro` redraws it whenever either changes. It writes no table, because
`reports/metrics/segments.csv` already is the table of what it draws.

**Which variant.** The one `report.candidate_variant` names. `metrics.json` names
a variant only once one passes the gate, so until then the candidate is a
decision somebody took (EDN-62) and is stated as a parameter.

**What is not drawn.** The levels SC-04 does not check: those under
`sc04_min_segment_rows` test rows, and the price buckets, which condition on the
target. They are in `segments.csv` with the reason. A figure of them would show
medians of a few dozen rows beside medians of thousands.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from loguru import logger
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox
import typer

from recommenditos import plots
from recommenditos.config import FIGURES_DIR, PARAMS_FILE, REPORTS_DIR
from recommenditos.pipeline import load_params

FIGURE_FILE = f"error-by-segment{plots.FIGURE_SUFFIX}"

#: What a segment dimension is called in a row label, one of it and several.
_DIMENSION_NAMES: dict[str, tuple[str, str]] = {
    "make": ("make", "makes"),
    "country_code": ("country", "countries"),
    "seller_type": ("seller type", "seller types"),
    "fuel_category": ("fuel type", "fuel types"),
    "age_bucket": ("age band", "age bands"),
}
_SELLER_NAMES: dict[str, str] = {"Dealer": "dealers", "PrivateSeller": "private sellers"}

_PERCENT = 100.0
_POINTS_PER_INCH = 72.0

# The layout, in inches at print size: fixed margins rather than `tight_layout`,
# so the plot area does not move with the length of a label.
_LEFT_IN = 0.9
_RIGHT_IN = 0.1
_TOP_IN = 0.22
_BOTTOM_IN = 0.45
_ROW_IN = 0.33

#: The area of a dot in points squared, as `scatter` takes it, and the diameter
#: that goes with it including its white ring.
_DOT_AREA = 26.0
_DOT_PT = 5.6
#: How far a dot is moved off its row's line when it would cover another, and
#: the lanes tried in order. One lane each way is what fits between two rows
#: while the dots of two neighbouring rows still read as two groups.
_LANE_PT = 6.1
_LANES: tuple[int, ...] = (0, 1, -1)
#: The gap between a dot and its label, and between a label and the bound line.
_LABEL_GAP_PT = 6.0
_LINE_GAP_PT = 3.0

app = typer.Typer()


@dataclass(frozen=True)
class Level:
    """One segment level SC-04 checks: its dimension, its name and its test error."""

    segment: str
    level: str
    n: int
    mdape_pct: float

    @property
    def name(self) -> str:
        """What a label calls the level."""
        if self.segment == "country_code":
            return plots.country_name(self.level)
        if self.segment == "age_bucket":
            return f"{self.level} years"
        if self.segment == "seller_type":
            return _SELLER_NAMES.get(self.level, self.level)
        return self.level


def checked_levels(record: dict) -> list[Level]:
    """The levels of a variant's metrics record that count toward SC-04."""
    return [
        Level(row["segment"], row["level"], row["n"], row["mdape"] * _PERCENT)
        for row in record["segments"]
        if row["counts_toward_sc04"]
    ]


def rows_of(levels: list[Level]) -> list[list[Level]]:
    """The levels by dimension, the dimension with the worst level first.

    Within a row the levels are in ascending order of their error, so the last
    one is the row's worst. Ties are broken by name, so the order never depends
    on the order of the record.
    """
    by_segment: dict[str, list[Level]] = {}
    for level in sorted(levels, key=lambda level: (level.mdape_pct, level.level)):
        by_segment.setdefault(level.segment, []).append(level)
    return sorted(by_segment.values(), key=lambda row: (-row[-1].mdape_pct, row[-1].segment))


def lanes(positions_pt: list[float]) -> list[int]:
    """A lane per dot of one row, so that no dot covers another.

    `positions_pt` are the dots' horizontal positions in points, ascending. The
    dots are placed from the right, so the row's worst level, which is the one a
    reader looks for and the one that is labelled, stays on the row's line. A dot
    takes the first lane of `_LANES` in which it is at least a dot's width from
    every dot already there. With every lane taken it goes on the row's line,
    where it overlaps: the row label states how many dots the row has.
    """
    last_in_lane: dict[int, float] = {}
    assigned = []
    for position in reversed(positions_pt):
        free = [
            lane
            for lane in _LANES
            if lane not in last_in_lane or last_in_lane[lane] - position >= _DOT_PT
        ]
        lane = free[0] if free else _LANES[0]
        last_in_lane[lane] = position
        assigned.append(lane)
    return assigned[::-1]


def labelled(rows: list[list[Level]], bound_pct: float) -> list[Level]:
    """The levels that get a label: the worst of every row, and the runner-up.

    The runner-up is the worst level within the bound when another level is above
    it, because the distance between the two is what tells a single miss from a
    criterion that is missed everywhere.
    """
    chosen = [row[-1] for row in rows]
    within = [level for row in rows for level in row if level.mdape_pct <= bound_pct]
    above = any(level.mdape_pct > bound_pct for row in rows for level in row)
    if above and within:
        runner_up = max(within, key=lambda level: (level.mdape_pct, level.level))
        if runner_up not in chosen:
            chosen.append(runner_up)
    return chosen


def _row_label(row: list[Level]) -> str:
    one, several = _DIMENSION_NAMES.get(row[0].segment, (row[0].segment.replace("_", " "),) * 2)
    return f"{len(row)} {one if len(row) == 1 else several}"


def _value(level: Level, bound_pct: float) -> str:
    """The level's error as a label prints it.

    One decimal, and a second one for a level so close to the bound that one
    would print the bound itself: a dot marked as above the bound and labelled
    with the bound's own figure reads as a mistake.
    """
    text = f"{level.mdape_pct:.1f}"
    if text == f"{bound_pct:.1f}" and level.mdape_pct != bound_pct:
        text = f"{level.mdape_pct:.2f}"
    return f"{text} %"


#: Where a label can go relative to its dot: the offset of its anchor in units of
#: `_LABEL_GAP_PT`, and how the text is aligned on that anchor.
_PLACES: dict[str, tuple[tuple[int, int], str, str]] = {
    "right": ((1, 0), "left", "center"),
    "left": ((-1, 0), "right", "center"),
    "over": ((0, 1), "center", "bottom"),
    "under": ((0, -1), "center", "top"),
}


def _label(axis: Axes, level: Level, y: float, bound_pct: float, taken: list[Bbox]) -> None:
    """Label a dot at the first place where the label is in nobody's way.

    A label stays on its dot's side of the bound line, because the dots are read
    against that line, stays inside the axes, and overlaps nothing in `taken`:
    the boxes of every dot and of the labels placed before it, in pixels. Its
    own box is added to them. Right of the dot is where a reader looks first, so
    that is tried first.

    A level within the bound is then tried left of its dot, which works for one
    that stands apart from its row, then beside it with its value on a second
    line, which needs no more width than the name, then under and over it, and
    only after all that without its value: the runner-up's value is the number
    the text compares the miss with. With no room at all it stays unlabelled.

    A level above the bound is tried over its dot and under it, then left of it,
    and is in primary ink. It is the level the figure is about, so with no free
    place it is labelled over its dot all the same.
    """
    renderer = axis.figure.canvas.get_renderer()
    frame = axis.get_window_extent(renderer)
    line_px = axis.transData.transform((bound_pct, y))[0]
    gap_px = _points(axis, _LINE_GAP_PT)
    above = level.mdape_pct > bound_pct
    low_px, high_px = (line_px + gap_px, frame.x1) if above else (frame.x0, line_px - gap_px)
    full = f"{level.name} {_value(level, bound_pct)}"
    if above:
        candidates = [(full, "right"), (full, "over"), (full, "under"), (full, "left")]
    else:
        stacked = f"{level.name}\n{_value(level, bound_pct)}"
        candidates = [(full, "right"), (full, "left"), (stacked, "right"), (stacked, "left")]
        candidates += [
            (full, "under"),
            (full, "over"),
            (level.name, "right"),
            (level.name, "left"),
        ]

    def place(text: str, where: str):
        (dx, dy), ha, va = _PLACES[where]
        label = axis.annotate(
            text,
            (level.mdape_pct, y),
            xytext=(dx * _LABEL_GAP_PT, dy * _LABEL_GAP_PT),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=plots.LABEL_PT,
            color=plots.INK if above else plots.INK_SECONDARY,
            linespacing=1.1,
        )
        if dx == 0:
            # Centred on the dot, then moved sideways into the room it has.
            box = label.get_window_extent(renderer)
            shift_px = max(0.0, low_px - box.x0) - max(0.0, box.x1 - high_px)
            label.xyann = (shift_px * _POINTS_PER_INCH / axis.figure.dpi, dy * _LABEL_GAP_PT)
        return label, label.get_window_extent(renderer)

    for text, where in candidates:
        label, box = place(text, where)
        inside = low_px <= box.x0 and box.x1 <= high_px and box.y0 >= frame.y0
        if inside and not any(box.overlaps(other) for other in taken):
            taken.append(box)
            return
        label.remove()
    if above:
        taken.append(place(full, "over")[1])


def _points(axis: Axes, points: float) -> float:
    """A length in points as display pixels of the figure `axis` is on."""
    return points * axis.figure.dpi / _POINTS_PER_INCH


def figure_of(levels: list[Level], bound_pct: float, criterion: str = "SC-04") -> Figure:
    """One row per segment dimension, one dot per level, the bound as a line.

    A strip plot rather than a bar per level: the reader has to find one value
    among many and compare all of them with one bound, which a shared axis shows
    in well under half the height 26 labelled bars would take. The axis starts at zero
    although dots do not need it to, so that a level's distance to the bound is
    in proportion to its error.

    The levels differ widely in size, from 541 to 15,797 test rows on the real
    snapshot, and are drawn as equal dots, because the criterion treats them
    equally: it is missed by one level whatever its share of the rows. The
    caption has to say how many rows a miss stands for.

    Colour is redundant by design. A level above the bound is orange, and it is
    also the only kind of dot right of the bound line and the only one labelled
    in primary ink, so the figure reads the same printed in greyscale.

    Only the worst level of a row is named, and the runner-up (`labelled`), so
    with misses in several rows the figure shows how many there are and names
    one per row.

    Without any checked level, which is what a fixture too small for SC-04
    gives, the figure says so instead of showing an empty axis.
    """
    rows = rows_of(levels)
    height_in = _TOP_IN + _BOTTOM_IN + max(len(rows), 1) * _ROW_IN
    figure = plots.new_figure(plots.TEXT_WIDTH_IN, height_in)
    # Labels are placed by their measured width, which needs a canvas that can
    # render; the widths are the font's, so they hold for the PDF this is saved as.
    FigureCanvasAgg(figure)
    width_in = plots.TEXT_WIDTH_IN - _LEFT_IN - _RIGHT_IN
    axis = figure.add_axes(
        (
            _LEFT_IN / plots.TEXT_WIDTH_IN,
            _BOTTOM_IN / height_in,
            width_in / plots.TEXT_WIDTH_IN,
            (height_in - _TOP_IN - _BOTTOM_IN) / height_in,
        )
    )
    worst = max((level.mdape_pct for level in levels), default=bound_pct)
    upper = max(1.35 * bound_pct, 1.1 * worst)
    axis.set_xlim(0, upper)
    axis.set_ylim(max(len(rows), 1) - 0.5, -0.5)
    points_per_unit = width_in * _POINTS_PER_INCH / upper
    lane_step = _LANE_PT / (_ROW_IN * _POINTS_PER_INCH)

    positions: dict[Level, float] = {}
    for y, row in enumerate(rows):
        offsets = lanes([level.mdape_pct * points_per_unit for level in row])
        for level, lane in zip(row, offsets, strict=True):
            positions[level] = y - lane * lane_step
    for colour, above in ((plots.MARK, False), (plots.ACCENT, True)):
        shown = [level for level in positions if (level.mdape_pct > bound_pct) is above]
        axis.scatter(
            [level.mdape_pct for level in shown],
            [positions[level] for level in shown],
            s=_DOT_AREA,
            color=colour,
            edgecolors="white",
            linewidths=1.0,
            zorder=3,
        )

    # Solid and in primary ink, against a grid one step off the page: the one
    # line of the plot that means something. Its label is in secondary ink like
    # the axis text, so the label of a level above the bound, which ends up
    # beside it, is not read as part of it.
    axis.axvline(bound_pct, color=plots.INK, linewidth=1.0, zorder=2)
    bound_label = axis.annotate(
        f"{criterion} bound {bound_pct:g} %",
        (bound_pct, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=plots.LABEL_PT,
        color=plots.INK_SECONDARY,
    )
    # What a label may not overlap: the bound's label, every dot, and then every
    # label before it.
    reach_px = _points(axis, _DOT_PT / 2)
    taken = [bound_label.get_window_extent(figure.canvas.get_renderer())]
    taken += [
        Bbox.from_extents(x - reach_px, dot_y - reach_px, x + reach_px, dot_y + reach_px)
        for x, dot_y in (
            axis.transData.transform((level.mdape_pct, y)) for level, y in positions.items()
        )
    ]
    for level in labelled(rows, bound_pct):
        _label(axis, level, positions[level], bound_pct, taken)
    if not rows:
        # In the room left of the bound line, so the line does not strike it out.
        axis.text(
            bound_pct / 2,
            0,
            "No segment level has enough test rows",
            ha="center",
            va="center",
            fontsize=plots.LABEL_PT,
            color=plots.INK_SECONDARY,
        )

    axis.set_yticks(range(len(rows)), [_row_label(row) for row in rows])
    axis.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    axis.set_xlabel(
        "Median absolute percentage error (MdAPE) on the test split, %",
        fontsize=plots.AXIS_PT,
        color=plots.INK_SECONDARY,
    )
    plots.style_axes(axis, grid="x")
    axis.tick_params(axis="y", length=0, labelcolor=plots.INK)
    axis.spines["left"].set_visible(False)
    return figure


def draw_figure(path: Path, levels: list[Level], bound_pct: float) -> None:
    plots.save_figure(figure_of(levels, bound_pct), path)
    logger.info(f"Wrote {path}.")


@app.command()
def main(
    metrics_dir: Path = REPORTS_DIR / "metrics",
    output_dir: Path = FIGURES_DIR,
    params_path: Path = PARAMS_FILE,
):
    params = load_params(params_path)
    variant = params["report"]["candidate_variant"]
    bound_pct = params["evaluate"]["success_criteria"]["sc04_segment_mdape_max"] * _PERCENT
    record = json.loads((metrics_dir / f"{variant}.json").read_text(encoding="utf-8"))
    levels = checked_levels(record)
    draw_figure(output_dir / FIGURE_FILE, levels, bound_pct)
    above = [level for level in levels if level.mdape_pct > bound_pct]
    logger.info(
        f"{variant}: {len(levels)} level(s) checked against {bound_pct:g} %, {len(above)} above "
        f"it{''.join(f'; {level.name} {level.mdape_pct:.2f} % (n {level.n})' for level in above)}."
    )


if __name__ == "__main__":
    app()
