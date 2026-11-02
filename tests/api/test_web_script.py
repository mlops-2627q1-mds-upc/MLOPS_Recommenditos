"""The pure functions of the web page's script, run in Node (#269).

The page has no build step and no JavaScript test runner, so these tests import
`app.js` as an ES module in Node and print what its exported functions return.
Without Node on the machine they are skipped; GitHub's runners have it.
"""

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest

from recommenditos.api.webui import WEB_DIR

NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="Node is not installed")


def run(expression: str) -> Any:
    """The JSON value of `expression`, evaluated with app.js's exports as `app`."""
    assert NODE is not None
    script = (
        f"const app = await import({json.dumps((WEB_DIR / 'app.js').as_uri())});\n"
        f"console.log(JSON.stringify({expression}));"
    )
    result = subprocess.run(
        [NODE, "--input-type=module", "-e", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        cwd=Path(__file__).parent,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_a_problem_without_an_instance_has_no_reference():
    """nginx's own 503 names no instance; the page must not show "null"."""
    nginx = {
        "type": "about:blank",
        "title": "Service Unavailable",
        "status": 503,
        "detail": "The API is not answering; try again in a few seconds.",
    }
    shown = run(f"app.describeProblem(503, {json.dumps(nginx)})")
    assert shown == {
        "title": "The service is busy or restarting",
        "detail": nginx["detail"],
        "reference": None,
    }
    api = {**nginx, "status": 500, "title": "Internal Server Error", "instance": "urn:uuid:ab-12"}
    assert run(f"app.describeProblem(500, {json.dumps(api)})")["reference"] == "ab-12"
    assert run("app.describeProblem(502, null)")["detail"] == "The service answered 502."


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
