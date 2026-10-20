"""The house style of the report's figures, shared by every stage that draws one.

The figures are drawn by pipeline stages (`compare-energy`, `compare-segments`,
`profile-data`) so that each is built from the artefacts it shows and `dvc repro`
redraws it when they change. This module is what makes them look like one set:
the colours, the font sizes and the axis treatment are defined once, here.

**Colour.** Two hues and neutral inks. Blue is the regular case and orange the
one case a figure is about: the request with only the required fields, the level
above the bound, the country held out. The pair was checked for colour-vision
deficiency (worst pair under protanopia and deuteranopia well clear of the
threshold) but prints as two similar greys, so no figure may rely on the hue
alone: the orange mark also differs in shape or position, or carries a label.
Text is never drawn in a data colour.

**Size.** A figure is drawn at the size the report prints it, so the font sizes
below are the sizes on paper, where the report's own text is 11 pt. It carries no
title: the caption of whatever prints it does, and a title inside the picture
would be scaled with it.

**Format.** PDF, so the report prints lines and text as vectors at any zoom. The
PNG the first figure was saved as carried an alpha channel, which pdfLaTeX turns
into a soft mask that some viewers render as a grey box behind the plot.

A bare `Figure` rather than `pyplot`, so drawing touches no global state and
needs no backend chosen for a process that has no display.
"""

from pathlib import Path
from typing import Literal

import matplotlib
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

#: The regular case, and the one case the figure is about.
MARK = "#2a78d6"
ACCENT = "#eb6834"
#: Text: labels a reader has to find, and the axis text around them.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
#: Lines that connect or separate marks without being data themselves.
MUTED = "#898781"
GRID = "#e4e3df"
BASELINE = "#c3c2b7"

#: The report's text width, 16 cm on A4 with 2.5 cm margins, in inches.
TEXT_WIDTH_IN = 6.3
#: Font sizes in points: axis titles, and tick and mark labels.
AXIS_PT = 9
LABEL_PT = 8

#: The suffix of every figure file, for the stages and the tests that name one.
FIGURE_SUFFIX = ".pdf"

#: What the datasets' ISO codes are called in a label.
COUNTRY_NAMES: dict[str, str] = {
    "AT": "Austria",
    "BE": "Belgium",
    "DE": "Germany",
    "ES": "Spain",
    "FR": "France",
    "IT": "Italy",
    "LU": "Luxembourg",
    "NL": "Netherlands",
}


def country_name(code: str) -> str:
    """The country's name, or the code itself for one this module does not know."""
    return COUNTRY_NAMES.get(code, code)


def new_figure(width_in: float, height_in: float) -> Figure:
    """An empty figure of the size it will be printed at."""
    return Figure(figsize=(width_in, height_in))


def renderer(axis: Axes) -> RendererBase:
    """The renderer that measures text on `axis`'s figure, which must have an Agg canvas.

    The figures here are bare `Figure`s with a `FigureCanvasAgg` attached, the one
    canvas that can say how large a piece of text will be before it is saved.
    """
    canvas = axis.figure.canvas
    if not isinstance(canvas, FigureCanvasAgg):
        raise TypeError(f"text can only be measured on an Agg canvas, not on {type(canvas)}")
    return canvas.get_renderer()


def style_axes(axis: Axes, *, grid: Literal["x", "y", "both"] = "both") -> None:
    """Recessive axes: a solid hairline grid, no box, secondary ink on the ticks.

    `grid` names the axis whose ticks get gridlines: `"x"`, `"y"` or `"both"`.
    """
    axis.grid(True, which="major", axis=grid, color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(which="both", labelsize=LABEL_PT, colors=INK_SECONDARY)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)


def save_figure(figure: Figure, path: Path) -> None:
    """Write `figure` as a PDF whose bytes depend only on what was drawn.

    No creation date, so a redraw of the same data with the same matplotlib is
    the same file and `dvc status` stays quiet. TrueType outlines (font type 42)
    rather than matplotlib's default Type 3, which viewers render unhinted and
    whose text cannot be searched or copied.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        figure.savefig(path, format="pdf", metadata={"CreationDate": None})
