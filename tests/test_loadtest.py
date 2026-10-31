"""The load test's planning and measuring code (`tools/loadtest/`, issue #143).

The run itself needs a stack and is documented in docs/docs/load-testing.md;
these tests hold the parts that turn what a run collects into its figures.
"""

import json
import random

import pandas as pd
import pytest
from tools.loadtest import measure
from tools.loadtest.arrivals import Arrivals
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
    rng = random.Random(0)  # noqa: S311 - a reproducible draw, no secret

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
    metrics.observe_request("POST", "/v1/predict", 200, 0.1)  # before the run, within both
    before = measure.latency_buckets(metrics.exposition().decode())
    for latency in (0.05, 0.15, 0.28, 0.35):
        metrics.observe_request("POST", "/v1/predict", 200, latency)
    metrics.observe_request("POST", "/v1/predict", 422, 0.01)
    metrics.observe_request("GET", "/health", 200, 0.01)

    shares = measure.latency_shares(before, measure.latency_buckets(metrics.exposition().decode()))

    assert shares["/v1/predict"] == {"answers": 5, "within_250ms": 0.6, "within_300ms": 0.8}
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


def test_locust_statistics_are_read_per_endpoint_and_in_total(tmp_path):
    # The shape of Locust's `--json-file` output, keys of `response_times` as strings.
    entries = [
        {"name": "/v1/predict", "method": "POST", "num_requests": 100, "num_failures": 1,
         "response_times": {"40": 50, "60": 45, "200": 4, "900": 1}},
        {"name": "/v1/price-range", "method": "POST", "num_requests": 20, "num_failures": 0,
         "response_times": {"30": 20}},
    ]  # fmt: skip
    path = tmp_path / "run-locust.json"
    path.write_text(json.dumps(entries), encoding="utf-8")

    read = measure.read_locust_json(path, seconds=10)

    predict = read["/v1/predict"]
    assert (predict.p50_ms, predict.p95_ms, predict.p99_ms) == (40, 60, 200)
    assert predict.error_rate == pytest.approx(0.01)
    assert predict.rate_per_s == 10
    assert (read["Aggregated"].requests, read["Aggregated"].failures) == (120, 1)
    assert read["Aggregated"].rate_per_s == 12


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


def _send_times(users: int, rate: float, duration: float, answer: float, seed: int) -> list[float]:
    """When `users` users started together at 0 send, each answer taking `answer` seconds."""
    rng = random.Random(seed)  # noqa: S311 - simulated send times, no secret
    sends = []
    for _ in range(users):
        arrivals = Arrivals(rate / users, rng)
        now = 0.0
        while (now := now + arrivals.next_wait(now)) < duration:
            sends.append(now)
            now += answer
    return sorted(sends)


def test_users_started_together_do_not_send_together():
    # NFR-03's 20 requests/s from 80 users that Locust starts within a second:
    # with a fixed gap all 80 first requests went out within that second, and
    # again every 4 s.
    rng = random.Random(7)  # noqa: S311 - simulated send times, no secret
    firsts = [Arrivals(20 / 80, rng).next_wait(0.0) for _ in range(80)]
    assert max(firsts) - min(firsts) > 4
    assert sum(first < 1 for first in firsts) < 40

    sends = _send_times(users=80, rate=20, duration=60, answer=0.1, seed=7)
    per_tenth = [0] * 600
    for send in sends:
        per_tenth[int(send * 10)] += 1
    # 2 requests per 100 ms on average; a burst of all users would put 80 in one.
    assert max(per_tenth) <= 10


def test_the_users_together_keep_the_total_rate():
    sends = _send_times(users=80, rate=20, duration=60, answer=0.1, seed=11)
    assert len(sends) / 60 == pytest.approx(20, rel=0.05)


def test_a_user_sends_at_once_after_an_answer_slower_than_its_gap():
    arrivals = Arrivals(rate=1000.0, rng=random.Random(3))  # noqa: S311
    gaps = random.Random(3)  # noqa: S311 - the same draws, to know the gaps
    first, _, third = (gaps.expovariate(1000.0) for _ in range(3))
    assert arrivals.next_wait(0.0) == first
    late = first + 5.0
    assert arrivals.next_wait(late) == 0.0
    # The next gap counts from the late send, not from the send it planned, so
    # the user does not catch up with a burst after a slow answer.
    assert arrivals.next_wait(late) == pytest.approx(third)


@pytest.mark.parametrize(
    ("written", "kept"),
    [(b'{"a": 1}', b'{"a": 1}\n'), (b'{"a": 1}\n', b'{"a": 1}\n'), (b"", b"")],
)
def test_the_locust_json_ends_with_one_newline(tmp_path, written, kept):
    path = tmp_path / "run-locust.json"
    path.write_bytes(written)
    measure.end_with_newline(path)
    assert path.read_bytes() == kept


@pytest.mark.parametrize(
    ("image", "written"),
    [
        (f"ghcr.io/o/api:{'902de35a' * 5}", "ghcr.io/o/api:902de35a902d"),
        ("recommenditos-api:latest", "recommenditos-api:latest"),
        ("ghcr.io/o/api:902de35", "ghcr.io/o/api:902de35"),
    ],
)
def test_a_full_commit_sha_tag_is_shortened_in_the_report(image, written):
    assert measure.short_image(image) == written
