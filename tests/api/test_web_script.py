"""The pure functions of the web page's script, run in Node (#269).

The page has no build step and no JavaScript test runner, so these tests import
`app.js` as an ES module in Node and print what its exported functions return.
Without Node on the machine they are skipped; GitHub's runners have it.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

from recommenditos.api.webui import WEB_DIR

NODE = shutil.which("node")
# Skipped on a machine without Node, but never in CI, where a missing Node would
# silently drop the page's script from the tests.
pytestmark = pytest.mark.skipif(
    NODE is None and not os.environ.get("CI"), reason="Node is not installed"
)


import time
import warnings
TIMES: list[str] = []


@pytest.fixture(scope="module", autouse=True)
def _probe():
    for _ in range(3):
        t = time.perf_counter()
        subprocess.run([NODE, "--version"], capture_output=True, check=False, timeout=120)
        TIMES.append(f"version={time.perf_counter() - t:.2f}")
    yield
    warnings.warn("NODE-TIMING " + " ".join(TIMES), stacklevel=1)


def run(expression: str) -> Any:
    """The JSON value of `expression`, evaluated with app.js's exports as `app`."""
    assert NODE is not None, "CI is set, but Node is not installed"
    script = (
        f"const app = await import({json.dumps((WEB_DIR / 'app.js').as_uri())});\n"
        f"console.log(JSON.stringify({expression}));"
    )
    t = time.perf_counter()
    result = subprocess.run(
        [NODE, "--input-type=module", "-e", script],
        capture_output=True,
        # Node writes UTF-8; Windows would decode it with its ANSI code page.
        encoding="utf-8",
        timeout=30,
        check=False,
        cwd=Path(__file__).parent,
    )
    TIMES.append(f"{time.perf_counter() - t:.2f}")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_a_problem_without_an_instance_names_no_reference():
    """nginx's own 503 names no instance; the page must not show "null" or "undefined"."""
    nginx = {
        "type": "about:blank",
        "title": "Service Unavailable",
        "status": 503,
        "detail": "The API is not answering; try again in a few seconds.",
    }
    shown = run(f"app.describeProblem(503, {json.dumps(nginx)})")
    assert shown == {
        "title": "The price service is not answering.",
        "detail": "Your inputs are kept; try again in a minute.",
        "retry": True,
    }
    assert run("app.describeProblem(0, null)") == shown
    failed = {
        **nginx,
        "status": 500,
        "title": "Internal Server Error",
        "instance": "urn:uuid:ab-12",
    }
    assert run(f"app.describeProblem(500, {json.dumps(failed)})")["detail"] == (
        "Try again; if it repeats, quote request ab-12."
    )
    without = run("app.describeProblem(500, {})")["detail"]
    assert "null" not in without
    assert "undefined" not in without


def test_a_422_the_form_cannot_map_names_its_field():
    problem = {"errors": [{"pointer": "#/cylinders", "detail": "Too many."}]}
    assert run(f"app.describeProblem(422, {json.dumps(problem)})")["detail"] == (
        "The service did not accept cylinders: Too many."
    )


def test_signed_numbers_use_a_real_minus_sign():
    assert run(
        "[app.signedPercent(-21.74), app.signedPercent(9.2),"
        " app.signedPercent(0.01), app.signedPercent(34.4, 0)]"
    ) == [
        "\u221221.7%",
        "+9.2%",
        "0.0%",
        "+34%",
    ]


def test_close_circles_are_stacked_into_rows():
    assert run("app.stackRows([10, 14, 30, 31, 32], 8)") == [0, 1, 0, 1, 2]


def test_the_option_lists_and_limits_are_read_from_the_openapi_schema():
    properties = {
        "make": {"type": "string", "enum": ["Audi", "BMW"]},
        "drive_train": {"anyOf": [{"type": "string", "enum": ["4WD"]}, {"type": "null"}]},
        "gears": {"anyOf": [{"type": "integer", "minimum": 0, "maximum": 12}, {"type": "null"}]},
        "model": {"type": "string"},
    }
    found = run(
        f"[app.allowedValues({json.dumps(properties)}, 'make'),"
        f" app.allowedValues({json.dumps(properties)}, 'drive_train'),"
        f" app.allowedValues({json.dumps(properties)}, 'model'),"
        f" app.limits({json.dumps(properties)}, 'gears')]"
    )
    assert found == [["Audi", "BMW"], ["4WD"], None, {"min": 0, "max": 12, "integer": True}]


def test_a_422_is_mapped_to_its_fields_with_the_allowed_values():
    problem = {
        "errors": [
            {"pointer": "#/make", "detail": "'Ford' is not supported.", "allowed": ["Audi"]},
            {"pointer": "#/power_kw", "detail": "Too small."},
            {"pointer": "#/power_kw", "detail": "A second fault of the same field."},
            {"pointer": "#", "detail": "The body as a whole."},
        ]
    }
    assert run(f"app.fieldErrors({json.dumps(problem)})") == {
        "make": "'Ford' is not supported. Choose one of: Audi.",
        "power_kw": "Too small.",
    }


def test_the_waterfall_multiplies_the_effects_from_the_base_to_the_estimate():
    explanation = {
        "base_price_eur": 38389.96,
        "top_features": [
            {"field": "power_kw", "effect_pct": -21.7},
            {"field": "mileage_km_raw", "effect_pct": 9.2},
        ],
        "other_features_effect_pct": -4.3,
    }
    steps = run(f"app.waterfallSteps({json.dumps(explanation)})")
    expected = 38389.96 * (1 - 0.217) * (1 + 0.092) * (1 - 0.043)
    assert steps["total"] == pytest.approx(expected)
    assert [step["field"] for step in steps["steps"]] == ["power_kw", "mileage_km_raw", None]
    for before, after in zip(steps["steps"], steps["steps"][1:], strict=False):
        assert after["from"] == pytest.approx(before["to"])


def test_the_range_scale_holds_the_range_and_every_listing():
    scale = run("app.rangeScale(20000, 30000, [18000, 35000])")
    assert scale["min"] < 18000
    assert scale["max"] > 35000
    assert run("app.rangeScale(20000, 30000, [])")["min"] >= 0


def test_a_listing_is_titled_by_its_make_and_model_alone():
    titles = run(
        "[app.listingTitle({make: 'BMW', model: '320', model_version: 'd Touring M Sport'}),"
        " app.listingTitle({make: 'BMW', model: null, model_version: '320d'})]"
    )
    assert titles == ["BMW 320", "BMW"]


def test_the_sellers_description_is_quoted_as_written_or_left_out():
    words = run(
        "[app.sellerDescription({model: '320', model_version: ' d/Virtual-Cockpit/62.000KM '}),"
        " app.sellerDescription({model: '320', model_version: null}),"
        " app.sellerDescription({model: '320', model_version: '  '})]"
    )
    assert words == ["\u201cd/Virtual-Cockpit/62.000KM\u201d", "", ""]


def test_a_description_cut_at_the_data_limit_ends_in_an_ellipsis():
    cut = "x" * 50
    words = run(
        f"[app.sellerDescription({{model_version: '{cut}'}}),"
        f" app.sellerDescription({{model_version: '{cut[:49]}'}}), app.VERSION_MAX_LENGTH]"
    )
    assert words == ["\u201c" + cut + "\u2026\u201d", "\u201c" + cut[:49] + "\u201d", 50]
