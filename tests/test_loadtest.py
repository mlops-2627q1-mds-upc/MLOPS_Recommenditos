"""The load test's planning and measuring code (`tools/loadtest/`, issue #143).

The run itself needs a stack and is documented in docs/docs/load-testing.md;
these tests hold the parts that turn what a run collects into its figures.
"""

import random

import pandas as pd
import pytest
from tools.loadtest import measure
from tools.loadtest.bodies import to_body
from tools.loadtest.plan import build_plan
from tools.loadtest.run import seconds

from recommenditos.api.metrics import ApiMetrics
from recommenditos.api.schemas.common import ModelInfo
from recommenditos.api.schemas.predict import ValuationRequest

LISTING = {
    "make": "Volkswagen",
    "model": "Golf",
    "registration_date": "2019-06",
    "mileage_km_raw": 45000.0,
    "power_kw": 110.0,
    "fuel_category": "Gasoline",
    "transmission": "Manual",
    "country_code": "DE",
    "body_type": "Compact",
    "seller_type": "Dealer",
    "nr_doors": 5,
}


def _openapi(**schemas: dict) -> dict:
    """An OpenAPI document with one POST path per schema, as FastAPI writes it."""
    paths = {}
    components = {}
    for name, schema in schemas.items():
        path = "/v1/" + name.replace("_", "-")
        ref = {"$ref": f"#/components/schemas/{name}"}
        paths[path] = {"post": {"requestBody": {"content": {"application/json": {"schema": ref}}}}}
        components[name] = schema
    return {"paths": paths, "components": {"schemas": components}}


def test_the_plan_calls_only_the_endpoints_the_target_serves():
    predict = {"properties": {field: {} for field in LISTING}, "required": ["make", "model"]}
    price_range = {"properties": {"make": {}, "registration_date": {}}, "required": ["make"]}

    plan = build_plan(_openapi(predict=predict, price_range=price_range), [LISTING])

    assert [e.path for e in plan.endpoints] == ["/v1/predict", "/v1/price-range"]
    assert set(plan.skipped) == {"/v1/comparables"}
    # A body is cut down to the fields the endpoint defines, which it would
    # otherwise refuse.
    assert plan.endpoints[1].bodies == ({"make": "Volkswagen", "registration_date": "2019-06"},)


def test_a_body_without_a_required_field_is_not_used_for_that_endpoint():
    predict = {"properties": {field: {} for field in LISTING}, "required": list(LISTING)}
    comparables = {"properties": {"make": {}, "vin": {}}, "required": ["vin"]}

    plan = build_plan(_openapi(predict=predict, comparables=comparables), [LISTING])

    assert [e.path for e in plan.endpoints] == ["/v1/predict"]
    assert "requires" in plan.skipped["/v1/comparables"]


def test_a_target_without_a_valuation_endpoint_stops_the_run():
    with pytest.raises(ValueError, match="none of the valuation endpoints"):
        build_plan({"paths": {}}, [LISTING])


def test_the_mix_follows_the_weights():
    schema = {"properties": {field: {} for field in LISTING}, "required": []}
    plan = build_plan(_openapi(predict=schema, price_range=schema, comparables=schema), [LISTING])
    rng = random.Random(0)

    paths = [plan.pick(rng)[0] for _ in range(5000)]

    assert paths.count("/v1/predict") / len(paths) == pytest.approx(0.6, abs=0.03)


def test_a_processed_listing_becomes_a_valid_predict_body():
    listing = {
        **LISTING,
        "registration_date": pd.Timestamp("2019-06-01"),
        "nr_doors": 5.0,
        "weight_kg": "1,355 kg",
        "equipment_comfort": "['Armrest', 'Cruise control']",
        "drive_train": float("nan"),
        "price": 18500.0,
    }

    body = to_body(listing)

    assert body is not None
    assert body["registration_date"] == "2019-06"
    assert body["nr_doors"] == 5
    assert body["weight_kg"] == 1355
    assert body["equipment_comfort"] == ["Armrest", "Cruise control"]
    assert "drive_train" not in body and "price" not in body
    ValuationRequest.model_validate(body)


def test_a_listing_without_a_required_field_gives_no_body():
    assert to_body({**LISTING, "power_kw": float("nan")}) is None


def test_the_shares_within_the_targets_come_from_the_difference_of_two_scrapes():
    metrics = ApiMetrics(ModelInfo(variant="lgbm-basic", version="abc", trained_at=None))
    metrics.observe_request("POST", "/v1/predict", 200, 0.5)  # before the run
    before = measure.latency_buckets(metrics.exposition().decode())
    for latency in (0.05, 0.15, 0.25, 0.35):
        metrics.observe_request("POST", "/v1/predict", 200, latency)
    metrics.observe_request("POST", "/v1/predict", 422, 0.01)
    metrics.observe_request("GET", "/health", 200, 0.01)

    shares = measure.latency_shares(before, measure.latency_buckets(metrics.exposition().decode()))

    assert shares["/v1/predict"] == {"answers": 5, "within_200ms": 0.6, "within_300ms": 0.8}
    assert shares["/health"]["answers"] == 1


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("157.3MiB / 15.4GiB|12.50%\n", (157.3 * 2**20, 12.5)),
        ("1.02GiB / 4GiB|180.2%", (1.02 * 2**30, 180.2)),
        ("900kB / 1GB|0.00%", (900e3, 0.0)),
    ],
)
def test_docker_stats_lines_are_read_in_bytes_and_percent(line, expected):
    memory, cpu = measure.parse_docker_stats(line)
    assert memory == pytest.approx(expected[0])
    assert cpu == expected[1]


def test_locust_statistics_are_read_per_endpoint(tmp_path):
    columns = (
        "Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,"
        "Min Response Time,Max Response Time,Average Content Size,Requests/s,Failures/s,"
        "50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%"
    )
    stats = tmp_path / "run_stats.csv"
    stats.write_text(
        f"{columns}\n"
        "POST,/v1/predict,1200,6,45,52,20,900,1500,20.0,0.1,45,50,55,60,80,120,200,300,800,900,900\n"
        ",Aggregated,1200,6,45,52,20,900,1500,20.0,0.1,45,50,55,60,80,120,200,300,800,900,900\n",
        encoding="utf-8",
    )

    read = measure.read_locust_stats(stats)

    assert read["/v1/predict"].p95_ms == 120
    assert read["/v1/predict"].error_rate == pytest.approx(0.005)
    assert read["Aggregated"].requests == 1200


def test_the_log_check_passes_only_with_one_record_per_answer():
    assert measure.check_log(["a", "b"], ["b", "a"]).exactly_once
    missing = measure.check_log(["a", "b"], ["a"])
    assert (missing.missing, missing.exactly_once) == (1, False)
    unexpected = measure.check_log(["a"], ["a", "c"])
    assert (unexpected.unexpected, unexpected.exactly_once) == (1, False)
    duplicated = measure.check_log(["a"], ["a", "a"])
    assert (duplicated.duplicated, duplicated.exactly_once) == (1, False)


@pytest.mark.parametrize(
    ("text", "expected"), [("90", 90), ("90s", 90), ("5m", 300), ("1h30m", 5400)]
)
def test_durations_are_read_in_locust_format(text, expected):
    assert seconds(text) == expected
