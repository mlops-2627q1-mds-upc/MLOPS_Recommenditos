"""`GET /metrics`: requests, latency, errors and ML signals for Prometheus (FR-11).

Every test here builds its own app, so its counters start at zero and the
figures it reads are its own requests' only.
"""

from collections.abc import Iterator
from pathlib import Path

from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
import pytest

from recommenditos.api.app import create_app
from recommenditos.api.metrics import (
    MULTIPROC_DIR_ENV_VAR,
    UNMATCHED_PATH,
    prepare_multiprocess_dir,
)
from recommenditos.api.middleware import MAX_BODY_BYTES
from recommenditos.api.serving import ServedModel, bundle_version

PREDICT = "/v1/predict"


def scrape(client: TestClient) -> dict[tuple[str, frozenset], float]:
    """Every sample `/metrics` returns, by name and label set."""
    response = client.get("/metrics")
    assert response.status_code == 200, response.text
    return {
        (sample.name, frozenset(sample.labels.items())): sample.value
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
    }


def value(samples: dict, name: str, **labels: str) -> float:
    """The sample `name` with exactly `labels`, or 0 if the series does not exist."""
    return samples.get((name, frozenset(labels.items())), 0.0)


@pytest.fixture
def fresh(bundle: Path) -> Iterator[TestClient]:
    """A client of an app of its own, with every metric at zero."""
    with TestClient(create_app(bundle), raise_server_exceptions=False) as client:
        yield client


@pytest.mark.req("FR-11")
def test_fr11_metrics_count_requests_per_route_template_and_status(
    fresh: TestClient, listing: dict
):
    for _ in range(3):
        assert fresh.post(PREDICT, json=listing).status_code == 200
    assert fresh.post(PREDICT, json={**listing, "make": "Trabant"}).status_code == 422
    assert fresh.get("/health").status_code == 200

    samples = scrape(fresh)

    ok = {"method": "POST", "path": PREDICT, "status": "200"}
    refused = {"method": "POST", "path": PREDICT, "status": "422"}
    assert value(samples, "http_requests_total", **ok) == 3
    assert value(samples, "http_requests_total", **refused) == 1
    assert value(samples, "http_requests_total", method="GET", path="/health", status="200") == 1
    assert value(samples, "http_request_errors_total", **refused) == 1
    assert ("http_request_errors_total", frozenset(ok.items())) not in samples
    assert value(samples, "http_request_duration_seconds_count", **ok) == 3
    assert value(samples, "http_request_duration_seconds_sum", **ok) > 0


@pytest.mark.req("FR-11")
def test_fr11_the_path_label_is_the_route_template_never_the_raw_path(fresh: TestClient):
    """An unknown path is one series whatever it is, so a client cannot grow the label set."""
    for path in ("/cars/1", "/cars/2", "/v1/predict/extra"):
        assert fresh.get(path).status_code == 404

    samples = scrape(fresh)

    paths = {dict(labels)["path"] for name, labels in samples if name == "http_requests_total"}
    assert paths == {UNMATCHED_PATH}
    unmatched = {"method": "GET", "path": UNMATCHED_PATH, "status": "404"}
    assert value(samples, "http_requests_total", **unmatched) == 3
    assert value(samples, "http_request_errors_total", **unmatched) == 3


@pytest.mark.req("FR-11")
def test_fr11_answers_given_before_routing_are_counted_under_their_route(fresh: TestClient):
    """A body refused by its `Content-Length` never reaches the router, yet is a /v1/predict."""
    response = fresh.post(
        PREDICT, content=b"{}", headers={"Content-Length": str(MAX_BODY_BYTES + 1)}
    )
    assert response.status_code == 413

    samples = scrape(fresh)

    assert (
        value(samples, "http_request_errors_total", method="POST", path=PREDICT, status="413") == 1
    )


@pytest.mark.req("FR-11")
def test_fr11_an_unexpected_failure_is_counted_as_the_500_the_client_gets(
    fresh: TestClient, listing: dict, monkeypatch: pytest.MonkeyPatch
):
    def fail(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ServedModel, "value", fail)
    assert fresh.post(PREDICT, json=listing).status_code == 500
    monkeypatch.undo()

    samples = scrape(fresh)

    assert (
        value(samples, "http_request_errors_total", method="POST", path=PREDICT, status="500") == 1
    )


@pytest.mark.req("FR-11")
def test_fr11_the_latency_histogram_has_bounds_at_nfr02s_targets(fresh: TestClient):
    assert fresh.get("/health").status_code == 200

    samples = scrape(fresh)

    bounds = {
        dict(labels)["le"]
        for name, labels in samples
        if name == "http_request_duration_seconds_bucket"
    }
    assert {"0.2", "0.3"} <= bounds


@pytest.mark.req("FR-11")
def test_fr11_metrics_hold_the_predicted_prices_the_warnings_and_the_model(
    fresh: TestClient, listing: dict, served: ServedModel, bundle: Path
):
    prices = [fresh.post(PREDICT, json=listing).json()["price_eur"] for _ in range(2)]
    warned = fresh.post(PREDICT, json={**listing, "country_code": "ES"})
    prices.append(warned.json()["price_eur"])

    samples = scrape(fresh)

    assert value(samples, "recommenditos_predicted_price_eur_count") == 3
    assert value(samples, "recommenditos_predicted_price_eur_sum") == pytest.approx(sum(prices))
    assert value(samples, "recommenditos_prediction_warnings_total", field="country_code") == 1
    assert (
        value(
            samples,
            "recommenditos_model_info",
            variant=served.info.variant,
            version=bundle_version(bundle),
        )
        == 1
    )


@pytest.mark.req("FR-11", "FR-07")
def test_fr11_price_ranges_have_metrics_of_their_own(fresh: TestClient, listing: dict):
    make = {"make": listing["make"]}
    answers = [fresh.post("/v1/price-range", json=make).json() for _ in range(2)]
    answers.append(fresh.post("/v1/price-range", json={**make, "country_code": "ES"}).json())

    samples = scrape(fresh)

    assert value(samples, "recommenditos_price_range_price_eur_count") == 3
    assert value(samples, "recommenditos_price_range_price_eur_sum") == pytest.approx(
        sum(answer["price_eur"] for answer in answers)
    )
    widths = [(a["upper_eur"] - a["lower_eur"]) / a["price_eur"] for a in answers]
    assert value(samples, "recommenditos_price_range_relative_width_sum") == pytest.approx(
        sum(widths)
    )
    assert value(samples, "recommenditos_price_range_warnings_total", field="country_code") == 1
    assert value(samples, "recommenditos_predicted_price_eur_count") == 0


@pytest.mark.req("FR-11")
def test_fr11_metrics_are_served_in_prometheus_text_format(fresh: TestClient):
    response = fresh.get("/metrics")

    assert response.headers["content-type"].startswith("text/plain; version=")
    assert "# TYPE recommenditos_model_info gauge" in response.text


@pytest.mark.req("FR-11")
def test_fr11_the_multiprocess_directory_is_emptied_of_metric_files_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    directory = tmp_path / "prometheus"
    directory.mkdir()
    (directory / "counter_123.db").write_bytes(b"stale")
    (directory / "keep.txt").write_text("not ours")
    monkeypatch.setenv(MULTIPROC_DIR_ENV_VAR, str(directory))

    assert prepare_multiprocess_dir() == directory
    assert sorted(path.name for path in directory.iterdir()) == ["keep.txt"]


@pytest.mark.req("FR-11")
def test_fr11_an_unwritable_multiprocess_directory_stops_the_service_with_its_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(MULTIPROC_DIR_ENV_VAR, str(tmp_path))
    monkeypatch.setattr("os.access", lambda *_args: False)

    with pytest.raises(PermissionError, match=MULTIPROC_DIR_ENV_VAR):
        prepare_multiprocess_dir()


@pytest.mark.req("FR-11")
def test_fr11_without_the_variable_nothing_is_prepared(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(MULTIPROC_DIR_ENV_VAR, raising=False)

    assert prepare_multiprocess_dir() is None
