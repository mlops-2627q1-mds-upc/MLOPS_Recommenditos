"""Draw the drift, error, and coverage of each window of the ES replay on the VM.

Reads `reports/monitoring/vm-es-replay-2026-11-04.json`, the drift job's
`report --all --json` after the replay of 2026-11-04, and writes
`reports/figures/es-replay-windows.pdf`, which the report includes from its copy
in `reports/latex/figures/`.

The data is a record of one run on the VM, not a pipeline artefact, so the
figure is drawn here and not by a stage. The thresholds are the ones the model
alerts compute in `deploy/prometheus/rules/model-performance.yml`, at each
window's own label count.

    uv run python reports/analysis/es_replay_figure.py
    cp reports/figures/es-replay-windows.pdf reports/latex/figures/
"""

import json
import math
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg

from recommenditos.plots import (
    ACCENT,
    AXIS_PT,
    INK,
    INK_SECONDARY,
    LABEL_PT,
    MARK,
    MUTED,
    TEXT_WIDTH_IN,
    new_figure,
    save_figure,
    style_axes,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "reports" / "monitoring" / "vm-es-replay-2026-11-04.json"
OUT = ROOT / "reports" / "figures" / "es-replay-windows.pdf"

#: The constants of deploy/prometheus/rules/model-performance.yml at the time of the replay
#: (2026-11-04); the rules file has since moved to the market baseline of #306 (K = 3.55).
MDAPE_C = 3.52
COVERAGE_Z = 2.576
FEATURES = 23


def mdape_ceiling(test_mdape: float, labelled: int) -> float:
    """ModelErrorHigh fires above this MdAPE."""
    return test_mdape * (1 + MDAPE_C / math.sqrt(labelled))


def coverage_floor(nominal: float, ranges: int) -> float:
    """ModelCoverageLow fires below this coverage."""
    return nominal - COVERAGE_Z * math.sqrt(nominal * (1 - nominal) / ranges) - 0.5 / ranges


def reference(axis: Axes, y: float, text: str, *, dashed: bool, above: bool = True) -> None:
    """A horizontal reference line, labelled at its left end, over the control windows."""
    axis.axhline(y, color=INK if dashed else MUTED, linewidth=0.9, linestyle=(0, (4, 2)) if dashed else "-", zorder=1)
    axis.annotate(
        text,
        xy=(0.55, y),
        xytext=(0, 2 if above else -2),
        textcoords="offset points",
        ha="left",
        va="bottom" if above else "top",
        fontsize=LABEL_PT,
        color=INK_SECONDARY,
    )


def main() -> None:
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    windows = [r["window"] for r in rows]
    perf = [r["performance"] for r in rows]
    assert all(p["mdape_status"] == "ok" and p["coverage_status"] == "ok" for p in perf)
    assert len({p["test_mdape"] for p in perf}) == 1 and len({p["nominal_level"] for p in perf}) == 1
    assert len({p["labelled"] for p in perf}) == 1 and len({p["labelled_ranges"] for p in perf}) == 1

    ids = [w["window_id"] for w in windows]
    es = [w["es_share"] == 1.0 for w in windows]
    assert all(w["es_share"] in (0.0, 1.0) for w in windows), "a window mixes control and ES"
    drifted = [w["drifted_features"] for w in windows]
    mdape = [100 * p["mdape"] for p in perf]
    coverage = [100 * p["coverage"] for p in perf]
    test_mdape = 100 * perf[0]["test_mdape"]
    nominal = perf[0]["nominal_level"]
    ceiling = 100 * mdape_ceiling(perf[0]["test_mdape"], perf[0]["labelled"])
    floor = 100 * coverage_floor(nominal, perf[0]["labelled_ranges"])
    print(f"MdAPE ceiling {ceiling:.2f} %, coverage floor {floor:.2f} %")

    figure = new_figure(TEXT_WIDTH_IN, 2.15)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    figure.subplots_adjust(left=0.065, right=0.99, bottom=0.2, top=0.86, wspace=0.32)

    panels = [
        (axes[0], drifted, "Drifted features, of 23", (0, FEATURES)),
        (axes[1], mdape, "Real MdAPE (%)", (0, 12)),
        (axes[2], coverage, "Range coverage (%)", (80, 100)),
    ]
    last_control = max(i for i, e in zip(ids, es, strict=True) if not e)
    for axis, values, title, limits in panels:
        style_axes(axis, grid="y")
        axis.set_title(title, fontsize=AXIS_PT, color=INK, loc="left", pad=4)
        axis.set_ylim(*limits)
        axis.set_xlim(0.4, 10.6)
        axis.set_xticks(ids)
        axis.set_xlabel("Window of 1,000 requests", fontsize=LABEL_PT, color=INK_SECONDARY)
        axis.axvline(last_control + 0.5, color=MUTED, linewidth=0.8, zorder=1)
        control_x = [i for i, e in zip(ids, es, strict=True) if not e]
        es_x = [i for i, e in zip(ids, es, strict=True) if e]
        axis.scatter(control_x, [v for v, e in zip(values, es, strict=True) if not e], s=24, marker="o", color=MARK, edgecolors="white", linewidths=0.6, zorder=3)
        axis.scatter(es_x, [v for v, e in zip(values, es, strict=True) if e], s=26, marker="s", color=ACCENT, edgecolors="white", linewidths=0.6, zorder=3)
        top = limits[1]
        axis.text(3, top, "control", ha="center", va="top", fontsize=LABEL_PT, color=INK_SECONDARY)
        axis.text(8, top, "ES", ha="center", va="top", fontsize=LABEL_PT, color=INK_SECONDARY)

    axes[0].set_yticks([0, 5, 10, 15, 20])
    reference(axes[1], ceiling, f"alert above\n{ceiling:.2f}", dashed=True)
    reference(axes[1], test_mdape, f"test {test_mdape:.2f}", dashed=False, above=False)
    reference(axes[2], 100 * nominal, f"nominal {100 * nominal:.0f}", dashed=False)
    reference(axes[2], floor, f"alert below\n{floor:.2f}", dashed=True, above=False)

    save_figure(figure, OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
