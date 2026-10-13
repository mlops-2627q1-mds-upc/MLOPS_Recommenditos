"""The figures the report prints, at the seams of the stages that draw them.

`compare-segments` and `profile-data` each turn an artefact into a figure, and
what can go wrong is in the steps before the drawing: which rows are shown, in
which order, what is folded, and where a label ends up. These tests feed those
steps constructed inputs, where the right answer is known, and ask the finished
figure where its marks and its labels are rather than comparing pixels.

That the stages run on the pipeline's own artefacts, and that `dvc.yaml` wires
them to those, is in `tests/test_pipeline.py`. That the copies the LaTeX build
reads are the stages' outputs is in `tests/test_committed_outputs.py`.
"""

from itertools import pairwise
import json

from matplotlib.colors import to_hex
from matplotlib.figure import Figure
import pandas as pd
import pytest
from tests.conftest import params_override

from recommenditos import plots
from recommenditos.data import profile_data
from recommenditos.data.profile_data import Bar, bars_of, composition
from recommenditos.modeling import compare_energy, compare_segments
from recommenditos.modeling.compare_segments import (
    Level,
    checked_levels,
    figure_of,
    labelled,
    lanes,
    rows_of,
)

BOUND = 15.0

#: A record shaped like the candidate's on the real snapshot: one level above
#: the bound, a runner-up well below it, and two levels within 0.01 points.
LEVELS = [
    Level("age_bucket", "0-1", 4037, 5.55),
    Level("age_bucket", "6-10", 3996, 6.93),
    Level("age_bucket", "10-20", 2910, 10.21),
    Level("age_bucket", "over 20", 784, 16.77),
    Level("make", "BMW", 7219, 7.24),
    Level("make", "Alfa Romeo", 1300, 7.23),
    Level("make", "Volvo", 714, 9.23),
    Level("seller_type", "Dealer", 15797, 6.45),
    Level("seller_type", "PrivateSeller", 3868, 8.57),
    Level("country_code", "DE", 8030, 6.37),
    Level("country_code", "FR", 1058, 8.42),
]


def label_texts(figure: Figure) -> list[str]:
    return [text.get_text() for text in figure.axes[0].texts]


def label_boxes(figure: Figure) -> dict:
    """Every label of the figure with the box it takes on the page, in pixels."""
    renderer = figure.canvas.get_renderer()
    return {text.get_text(): text.get_window_extent(renderer) for text in figure.axes[0].texts}


def bound_px(figure: Figure, bound: float = BOUND) -> float:
    return figure.axes[0].transData.transform((bound, 0))[0]


# --------------------------------------------------------------------------
# The house style
# --------------------------------------------------------------------------


def test_a_figure_drawn_twice_is_the_same_file(tmp_path):
    # `dvc.lock` records a figure by its bytes, so a redraw of unchanged data
    # must not look like a change. The creation date is what would make it one,
    # and two draws within the same second would not show it, so the file is
    # also asked directly.
    paths = [tmp_path / "first.pdf", tmp_path / "second.pdf"]
    for path in paths:
        plots.save_figure(figure_of(LEVELS, BOUND), path)

    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert paths[0].read_bytes().startswith(b"%PDF")
    assert b"CreationDate" not in paths[0].read_bytes()


# --------------------------------------------------------------------------
# compare-segments: which levels, in which order
# --------------------------------------------------------------------------


def test_only_the_levels_sc04_checks_are_drawn():
    record = {
        "segments": [
            {
                "segment": "make",
                "level": "BMW",
                "n": 7219,
                "mdape": 0.0724,
                "counts_toward_sc04": True,
            },
            # Too few rows, and a price bucket: both are in the record with the
            # reason, and neither is a level the criterion is decided on.
            {
                "segment": "make",
                "level": "Honda",
                "n": 177,
                "mdape": 0.1054,
                "counts_toward_sc04": False,
            },
            {
                "segment": "price_bucket",
                "level": "0-5000",
                "n": 755,
                "mdape": 0.2216,
                "counts_toward_sc04": False,
            },
        ]
    }

    assert checked_levels(record) == [Level("make", "BMW", 7219, pytest.approx(7.24))]


def test_the_dimension_with_the_worst_level_is_the_first_row():
    rows = rows_of(LEVELS)

    assert [row[0].segment for row in rows] == [
        "age_bucket",
        "make",
        "seller_type",
        "country_code",
    ]
    # Ascending within a row, so its last level is its worst.
    for row in rows:
        assert [level.mdape_pct for level in row] == sorted(level.mdape_pct for level in row)


def test_the_order_of_the_rows_does_not_depend_on_the_order_of_the_record():
    assert rows_of(LEVELS) == rows_of(LEVELS[::-1])


def test_a_country_without_a_known_name_is_labelled_with_its_code():
    # A new snapshot can bring a country this project has no name for. Its dot
    # is still labelled, with what the data calls it.
    levels = [Level("country_code", "DE", 8030, 6.37), Level("country_code", "PT", 900, 8.4)]

    assert "PT 8.4 %" in label_texts(figure_of(levels, BOUND))


# --------------------------------------------------------------------------
# compare-segments: dots that would cover each other
# --------------------------------------------------------------------------


def test_no_two_dots_of_a_lane_cover_each_other():
    positions = [100.0, 101.0, 103.0, 120.0, 121.0]

    assigned = lanes(positions)

    by_lane: dict[int, list[float]] = {}
    for position, lane in zip(positions, assigned, strict=True):
        by_lane.setdefault(lane, []).append(position)
    for lane, taken in by_lane.items():
        gaps = [later - earlier for earlier, later in pairwise(taken)]
        assert all(gap >= compare_segments._DOT_PT for gap in gaps), (lane, taken)


def test_the_worst_level_of_a_row_stays_on_the_rows_line():
    # The rightmost dot is the one that is labelled and the one a reader looks
    # for, so it is the one that may not be moved off the line.
    assert lanes([100.0, 101.0])[-1] == 0
    assert lanes([100.0, 101.0])[0] != 0


def test_dots_apart_from_each_other_all_stay_on_the_line():
    assert lanes([10.0, 40.0, 90.0]) == [0, 0, 0]


def test_a_dot_with_no_free_lane_goes_on_the_rows_line():
    # Four levels on one value and three lanes: the fourth overlaps on the line
    # instead of being pushed into the next row.
    assert lanes([50.0, 50.0, 50.0, 50.0]) == [0, -1, 1, 0]


# --------------------------------------------------------------------------
# compare-segments: what the marks say
# --------------------------------------------------------------------------


def dots_of(figure: Figure) -> dict[str, list[float]]:
    """The x value of every dot of the figure, by the colour it is drawn in."""
    by_colour: dict[str, list[float]] = {}
    for collection in figure.axes[0].collections:
        colour = to_hex(collection.get_facecolor()[0])
        by_colour.setdefault(colour, []).extend(float(x) for x, _ in collection.get_offsets())
    return by_colour


def test_every_level_is_drawn_once_at_its_error():
    drawn = dots_of(figure_of(LEVELS, BOUND))

    assert sorted(x for xs in drawn.values() for x in xs) == pytest.approx(
        sorted(level.mdape_pct for level in LEVELS)
    )


def test_only_the_levels_above_the_bound_are_in_the_accent_colour():
    drawn = dots_of(figure_of(LEVELS, BOUND))

    assert drawn[plots.ACCENT] == pytest.approx([16.77])
    assert max(drawn[plots.MARK]) <= BOUND


def test_a_level_exactly_on_the_bound_meets_the_criterion():
    # SC-04 is "at most", so the bound itself is within it.
    drawn = dots_of(figure_of([Level("make", "BMW", 7219, BOUND)], BOUND))

    assert drawn[plots.MARK] == [BOUND]
    assert not drawn.get(plots.ACCENT)


def test_the_bound_is_drawn_where_the_parameters_put_it():
    (line,) = figure_of(LEVELS, 12.5).axes[0].lines

    assert list(line.get_xdata()) == [12.5, 12.5]


def test_a_level_just_above_the_bound_is_not_labelled_with_the_bounds_own_figure():
    # 15.04 % rounds to "15.0 %", which beside "bound 15 %" and an orange dot
    # reads as a mistake in the figure.
    levels = [Level("make", "BMW", 7219, 7.24), Level("make", "Volvo", 714, 15.04)]

    assert "Volvo 15.04 %" in label_texts(figure_of(levels, BOUND))


# --------------------------------------------------------------------------
# compare-segments: what is labelled, and where
# --------------------------------------------------------------------------


def test_the_worst_level_of_every_row_and_the_runner_up_are_labelled():
    chosen = labelled(rows_of(LEVELS), BOUND)

    assert {level.level for level in chosen} == {
        "over 20",
        "Volvo",
        "PrivateSeller",
        "FR",
        "10-20",
    }


def test_no_runner_up_is_labelled_when_every_level_is_within_the_bound():
    within = [level for level in LEVELS if level.mdape_pct <= BOUND]

    chosen = labelled(rows_of(within), BOUND)

    # One label per row, the worst of each.
    assert {level.level for level in chosen} == {"10-20", "Volvo", "PrivateSeller", "FR"}


def test_a_runner_up_that_is_already_its_rows_worst_is_labelled_once():
    levels = [
        Level("age_bucket", "0-1", 4037, 5.55),
        Level("age_bucket", "over 20", 784, 16.77),
        Level("make", "Volvo", 714, 9.23),
    ]

    chosen = labelled(rows_of(levels), BOUND)

    assert sorted(level.level for level in chosen) == ["Volvo", "over 20"]


def test_the_figure_names_the_level_above_the_bound_and_the_bound():
    texts = label_texts(figure_of(LEVELS, BOUND))

    assert "over 20 years 16.8 %" in texts
    assert "SC-04 bound 15 %" in texts
    assert "10-20 years 10.2 %" in texts


def test_no_label_crosses_the_bound_line():
    figure = figure_of(LEVELS, BOUND)
    line = bound_px(figure)

    for text, box in label_boxes(figure).items():
        assert box.x1 <= line or box.x0 >= line, f"{text!r} lies across the bound line"


def test_the_label_of_a_level_above_the_bound_is_on_its_side_of_the_line():
    figure = figure_of(LEVELS, BOUND)

    assert label_boxes(figure)["over 20 years 16.8 %"].x0 > bound_px(figure)


def test_every_label_stays_on_the_page():
    figure = figure_of(LEVELS, BOUND)
    width, height = figure.canvas.get_width_height()

    for text, box in label_boxes(figure).items():
        assert box.x0 >= 0 and box.x1 <= width, f"{text!r} leaves the figure sideways"
        assert box.y0 >= 0 and box.y1 <= height, f"{text!r} leaves the figure vertically"


def test_a_label_with_no_room_before_the_bound_line_goes_left_of_its_dot():
    # The case the figure will show once the candidate passes: the worst level
    # just under the bound, apart from the others. It is the level a reader most
    # wants named, and the line it stands at leaves no room on its right.
    levels = [
        Level("age_bucket", "0-1", 4037, 5.55),
        Level("age_bucket", "over 20", 784, 14.6),
    ]
    figure = figure_of(levels, BOUND)

    box = label_boxes(figure)["over 20 years 14.6 %"]

    assert box.x1 < figure.axes[0].transData.transform((14.6, 0))[0]


def test_a_label_too_long_for_one_line_keeps_its_value_on_a_second_line():
    # The runner-up of the committed model: at 10.4 % its name and value no
    # longer fit between its dot and the bound line. The value is the number the
    # report compares the miss with, so it moves to a second line and stays.
    levels = [level for level in LEVELS if level.level != "10-20"]
    levels.append(Level("age_bucket", "10-20", 2910, 10.36))
    figure = figure_of(levels, BOUND)

    box = label_boxes(figure)["10-20 years\n10.4 %"]

    assert box.x0 > figure.axes[0].transData.transform((10.36, 0))[0]
    assert box.x1 < bound_px(figure)


def test_a_label_with_no_free_place_for_its_value_keeps_its_name():
    # Too close to the line for its value, even on a second line, a neighbour
    # on its left, one in the lane over it, and no row under it: the name alone
    # still fits on the right.
    levels = [
        Level("make", "Audi", 2431, 9.9),
        Level("make", "Volvo", 714, 10.5),
        Level("make", "BMW", 7219, 10.55),
    ]

    texts = label_texts(figure_of(levels, 12.0))

    assert "BMW" in texts
    assert not [text for text in texts if "BMW" in text and "%" in text]


def test_a_label_that_fits_nowhere_is_left_out():
    # Hard against the bound line, with a neighbour on its left and one in the
    # lane over it, in the only row there is: no place is free even for the name.
    levels = [
        Level("seller_type", "Dealer", 15797, 14.2),
        Level("seller_type", "Other", 900, 14.85),
        Level("seller_type", "PrivateSeller", 3868, 14.9),
    ]

    texts = label_texts(figure_of(levels, BOUND))

    assert not [text for text in texts if "private sellers" in text]


def test_no_label_covers_the_dot_of_another_level():
    figure = figure_of(LEVELS, BOUND)
    axis = figure.axes[0]
    dots = [
        axis.transData.transform(point)
        for collection in axis.collections
        for point in collection.get_offsets()
    ]
    assert len(dots) == len(LEVELS)

    for text, box in label_boxes(figure).items():
        covered = [(x, y) for x, y in dots if box.contains(x, y)]
        assert not covered, f"{text!r} covers a dot"


def test_a_label_above_the_bound_goes_under_its_dot_when_a_neighbour_is_over_it():
    # Two misses of one row on almost one value: the second is moved to the lane
    # over the first, which is where the label would go. Right of the dot is the
    # edge of the axes, so the label goes under.
    levels = [
        Level("age_bucket", "0-1", 4037, 5.0),
        Level("age_bucket", "over 20", 784, 19.6),
        Level("make", "BMW", 7219, 6.0),
        Level("make", "Audi", 2431, 19.0),
        Level("make", "Volvo", 714, 19.05),
        Level("seller_type", "Dealer", 15797, 6.0),
        Level("seller_type", "PrivateSeller", 3868, 8.0),
    ]
    figure = figure_of(levels, BOUND)
    axis = figure.axes[0]
    dots = [
        axis.transData.transform(point)
        for collection in axis.collections
        for point in collection.get_offsets()
    ]
    volvo = max(dots, key=lambda dot: (dot[0] <= axis.transData.transform((19.04, 0))[0], -dot[0]))

    box = label_boxes(figure)["Volvo 19.1 %"]

    assert box.y1 < volvo[1], "the label is not under its dot"
    assert not [(x, y) for x, y in dots if box.contains(x, y)]


def test_no_two_labels_overlap():
    # Misses in two rows on one value: one label goes under its dot and the
    # other over its own, towards each other.
    levels = [
        Level("age_bucket", "10-20", 2910, 19.0),
        Level("age_bucket", "over 20", 784, 19.05),
        Level("make", "BMW", 7219, 6.0),
        Level("make", "Volvo", 714, 19.0),
    ]

    boxes = list(label_boxes(figure_of(levels, BOUND)).values())

    for index, box in enumerate(boxes):
        assert not [other for other in boxes[index + 1 :] if box.overlaps(other)]


def test_a_level_above_the_bound_is_named_even_where_no_place_is_free():
    # Three misses on almost one value at the right edge and a fourth on their
    # left: no room right of the dot, neighbours just over and just under it,
    # and a dot where a label on its left would go. The level the figure is
    # about is named all the same.
    levels = [
        Level("make", "Suzuki", 824, 18.2),
        Level("make", "Audi", 2431, 19.0),
        Level("make", "BMW", 7219, 19.05),
        Level("make", "Volvo", 714, 19.1),
    ]

    assert "Volvo 19.1 %" in label_texts(figure_of(levels, BOUND))


def test_a_record_with_no_checked_level_says_so_instead_of_drawing_an_empty_axis():
    # What the test fixture gives: too few test rows for any level to count.
    figure = figure_of([], BOUND)

    box = label_boxes(figure)["No segment level has enough test rows"]

    assert box.x1 < bound_px(figure), "the bound line strikes the message out"


def test_levels_above_the_bound_in_several_rows_are_each_named_clear_of_every_dot():
    # Not today's figure, which has one miss, but the one a worse model draws:
    # a label over its dot must not land on the dots of the row above.
    levels = [
        Level("age_bucket", "0-1", 4037, 5.55),
        Level("age_bucket", "over 20", 784, 16.77),
        Level("make", "BMW", 7219, 7.24),
        Level("make", "Audi", 2431, 16.2),
        Level("make", "Volvo", 714, 17.3),
        Level("seller_type", "Dealer", 15797, 6.45),
        Level("seller_type", "PrivateSeller", 3868, 15.9),
    ]
    figure = figure_of(levels, BOUND)
    axis = figure.axes[0]
    dots = [
        axis.transData.transform(point)
        for collection in axis.collections
        for point in collection.get_offsets()
    ]
    boxes = label_boxes(figure)

    for name in ("Volvo 17.3 %", "over 20 years 16.8 %", "private sellers 15.9 %"):
        assert boxes[name].x0 > bound_px(figure), name
        assert not [(x, y) for x, y in dots if boxes[name].contains(x, y)], name


def test_the_axis_reaches_past_the_worst_level_however_far_it_is_above_the_bound():
    figure = figure_of(
        [Level("make", "BMW", 7219, 48.0), Level("make", "Audi", 2431, 12.0)], BOUND
    )

    assert figure.axes[0].get_xlim()[1] > 48.0
    assert figure.axes[0].get_xlim()[0] == 0


def test_the_stage_draws_the_variant_the_parameters_name(tmp_path, params):
    # `metrics.json` names no candidate while no variant passes the gate, so
    # the parameter decides, and a variant nobody named is not read at all.
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    segments = [
        {"segment": "make", "level": "BMW", "n": 900, "mdape": 0.07, "counts_toward_sc04": True}
    ]
    (metrics / "chosen.json").write_text(json.dumps({"segments": segments}), encoding="utf-8")
    (metrics / "other.json").write_text("not a record", encoding="utf-8")
    params_path = params_override(tmp_path, params, report={"candidate_variant": "chosen"})

    compare_segments.main(metrics, tmp_path / "figures", params_path)

    figure = tmp_path / "figures" / compare_segments.FIGURE_FILE
    assert figure.read_bytes().startswith(b"%PDF")


# --------------------------------------------------------------------------
# profile-data: what is counted, and what is folded
# --------------------------------------------------------------------------


def frame_of(makes: dict[str, int], countries: dict[str | None, int]) -> pd.DataFrame:
    """A raw frame with the given listings per make and per country."""
    make_column = [make for make, listings in makes.items() for _ in range(listings)]
    country_column = [code for code, listings in countries.items() for _ in range(listings)]
    assert len(make_column) == len(country_column)
    return pd.DataFrame({"make": make_column, "country_code": country_column})


def test_every_level_is_counted_as_a_share_of_all_rows():
    frame = frame_of({"BMW": 6, "Audi": 3, "Volvo": 1}, {"DE": 5, "ES": 4, None: 1})

    rows = composition(frame)

    assert rows == [
        {"dimension": "make", "level": "BMW", "listings": 6, "share": 0.6},
        {"dimension": "make", "level": "Audi", "listings": 3, "share": 0.3},
        {"dimension": "make", "level": "Volvo", "listings": 1, "share": 0.1},
        {"dimension": "country_code", "level": "DE", "listings": 5, "share": 0.5},
        {"dimension": "country_code", "level": "ES", "listings": 4, "share": 0.4},
        # A listing without a country is in the table, so its shares add up to one.
        {"dimension": "country_code", "level": profile_data.MISSING, "listings": 1, "share": 0.1},
    ]


def test_levels_of_the_same_size_are_ordered_by_name():
    frame = frame_of({"Volvo": 2, "Audi": 2}, {"DE": 4})

    assert [row["level"] for row in composition(frame)[:2]] == ["Audi", "Volvo"]


def test_a_dimension_with_more_levels_than_fit_folds_the_smallest_into_one_bar():
    makes = {f"make {index:02d}": 20 - index for index in range(12)}
    frame = frame_of(makes, {"DE": sum(makes.values())})

    bars = bars_of(composition(frame), "make", "ES")

    assert len(bars) == profile_data._MAX_BARS
    assert bars[-1].label == "5 other makes"
    assert bars[-1].kind == "rest"
    # The bars still stand for every row that has a make.
    assert sum(bar.share for bar in bars) == pytest.approx(1.0)


def test_a_dimension_whose_levels_all_fit_is_drawn_without_a_rest():
    countries = {code: 10 for code in ("AT", "BE", "DE", "ES", "FR", "IT", "LU", "NL")}
    frame = frame_of({"BMW": 80}, countries)

    bars = bars_of(composition(frame), "country_code", "ES")

    assert len(bars) == 8
    assert all(bar.kind != "rest" for bar in bars)


def test_the_holdout_country_is_marked_and_named():
    frame = frame_of({"BMW": 10}, {"DE": 6, "ES": 4})

    bars = bars_of(composition(frame), "country_code", "ES")

    assert bars == [Bar("Germany", 0.6), Bar("Spain, held out", 0.4, "held out")]


def test_a_missing_value_gets_no_bar():
    frame = frame_of({"BMW": 10}, {"DE": 9, None: 1})

    bars = bars_of(composition(frame), "country_code", "ES")

    assert [bar.label for bar in bars] == ["Germany"]


def test_the_holdout_country_keeps_its_bar_however_small_it_is():
    # With more countries than bars the smallest are folded into one. Folding
    # the holdout country with them would drop the mark the panel exists for.
    countries = {f"C{index}": 30 - index for index in range(9)} | {"ES": 1}
    frame = frame_of({"BMW": sum(countries.values())}, countries)

    bars = bars_of(composition(frame), "country_code", "ES")

    assert len(bars) == profile_data._MAX_BARS
    assert [bar.kind for bar in bars].count("held out") == 1
    assert bars[-1].label == "3 other countries"
    assert sum(bar.share for bar in bars) == pytest.approx(1.0)


def test_a_bar_is_as_long_as_its_share_and_only_the_holdout_is_in_the_accent_colour():
    frame = frame_of({"BMW": 6, "Audi": 4}, {"DE": 7, "ES": 3})

    make_panel, country_panel = profile_data.figure_of(composition(frame), "ES").axes

    assert [bar.get_width() for bar in make_panel.patches] == pytest.approx([60.0, 40.0])
    assert [bar.get_width() for bar in country_panel.patches] == pytest.approx([70.0, 30.0])
    colours = [
        to_hex(bar.get_facecolor()) for bar in (*make_panel.patches, *country_panel.patches)
    ]
    assert colours == [plots.MARK, plots.MARK, plots.MARK, plots.ACCENT]
    # One scale for both panels, so a bar's length means the same share in either.
    assert make_panel.get_xlim() == country_panel.get_xlim() == (0.0, 70.0)


def test_the_stage_reads_the_two_columns_it_counts_and_no_other(tmp_path, params, monkeypatch):
    # The raw file holds the sellers' personal data, and a stage that counts
    # makes and countries has no reason to load it. The file here has such a
    # column, and the read is watched for what it asks for.
    raw = tmp_path / "raw" / "listings.parquet"
    raw.parent.mkdir()
    frame = frame_of({"BMW": 6, "Audi": 4}, {"DE": 7, "ES": 3})
    frame.assign(seller_company_name="a dealer").to_parquet(raw, index=False)
    asked_for = []
    read_parquet = pd.read_parquet

    def watched(path, **options):
        asked_for.append(options.get("columns"))
        return read_parquet(path, **options)

    monkeypatch.setattr(profile_data.pd, "read_parquet", watched)

    profile_data.main(raw, tmp_path / "figures", params_override(tmp_path, params))

    assert asked_for == [list(profile_data.DIMENSIONS)]
    table = (tmp_path / "figures" / profile_data.TABLE_FILE).read_text(encoding="utf-8")
    assert table.splitlines()[0] == ",".join(profile_data.COLUMNS)
    assert "make,BMW,6,0.6" in table.splitlines()
    figure = tmp_path / "figures" / profile_data.FIGURE_FILE
    assert figure.read_bytes().startswith(b"%PDF")


# --------------------------------------------------------------------------
# compare-energy: the two conditions differ in shape, not only in colour
# --------------------------------------------------------------------------


def energy_row(variant: str, energy_wh: float, mdape: float, required_only: float | None) -> dict:
    return {
        "variant": variant,
        "energy_wh": energy_wh,
        "mdape": mdape,
        "mdape_required_fields_only": required_only,
        "cpu_power_method": "cpu_load",
    }


def test_the_two_conditions_of_the_energy_figure_can_be_told_apart_without_colour():
    # Blue and orange print as the same grey. A filled dot and an open ring do not.
    rows = [energy_row("b0", 0.001, 0.12, 0.12), energy_row("lgbm-basic", 0.05, 0.068, 0.083)]

    axis = compare_energy.figure_of(rows).axes[0]

    marks = {collection.get_label(): collection for collection in axis.collections}
    filled, ring = marks["every field given"], marks["only the required fields"]
    assert to_hex(filled.get_facecolor()[0]) == plots.MARK
    assert to_hex(ring.get_facecolor()[0]) == "#ffffff"
    assert to_hex(ring.get_edgecolor()[0]) == plots.ACCENT
    # The ring is the larger mark, so where both conditions coincide, as for
    # `b0`, the dot shows inside it.
    assert ring.get_sizes()[0] > filled.get_sizes()[0]
    assert filled.get_zorder() > ring.get_zorder()
    assert [float(y) for _, y in ring.get_offsets()] == pytest.approx([12.0, 8.3])
    assert [float(y) for _, y in filled.get_offsets()] == pytest.approx([12.0, 6.8])
