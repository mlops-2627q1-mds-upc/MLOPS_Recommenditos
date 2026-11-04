"""The replay sends what a client would and reports each price once (#227).

The API is an `httpx.MockTransport` that answers like the real one: a valuation
with a `request_id`, a feedback with 201. The listings are six of each real
sample (tests/fixtures/), some of them without optional fields such as `gears`.
"""

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
import gzip
import json
from pathlib import Path
import shutil
from typing import Any
import uuid

import httpx
import pytest

from recommenditos_monitoring import __main__ as cli
from recommenditos_monitoring.replay import (
    Phase,
    Replay,
    ReplayError,
    Summary,
    format_summary,
    phases_of,
    read_samples,
    request_body,
)

FIXTURES = Path(__file__).parent / "fixtures"
BASE_URL, FEEDBACK_URL = "http://proxy:8080", "http://api:8000"
#: Stands in for the API key of the stack.
STAND_IN = "stand-in-for-the-stack-s-own"
#: Four days before a month completes: 11 whole months after 2025-11-08.
SENT_AT = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)

type Handler = Callable[[httpx.Request], httpx.Response]


def records(name: str) -> list[dict[str, Any]]:
    lines = (FIXTURES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    """The fixtures as the stage writes them: gzipped JSON lines and the manifest."""
    for name in ("control", "es"):
        with gzip.open(tmp_path / f"{name}.jsonl.gz", "wb") as sample:
            sample.write((FIXTURES / f"{name}.jsonl").read_bytes())
    shutil.copy(FIXTURES / "manifest.json", tmp_path / "manifest.json")
    return tmp_path


class FakeAPI:
    """Answers valuations with a fresh request ID and stores feedback, recording every request."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        #: The request ID each valuation was answered with, by its position in `requests`.
        self.request_ids: dict[int, str] = {}
        #: Answers to send first, per path, before the normal ones.
        self.scripted: dict[str, list[httpx.Response | Exception]] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        queued = self.scripted.get(request.url.path)
        if queued:
            answer = queued.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        if request.url.path == "/v1/feedback":
            return httpx.Response(201, json=json.loads(request.content))
        request_id = str(uuid.uuid4())
        self.request_ids[len(self.requests) - 1] = request_id
        return httpx.Response(200, json={"request_id": request_id, "price_eur": 1.0})

    def to(self, path: str) -> list[httpx.Request]:
        return [request for request in self.requests if request.url.path == path]

    def bodies(self, path: str) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.to(path)]


class FakeTime:
    """A monotonic clock that advances only when the replay sleeps."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def api() -> FakeAPI:
    return FakeAPI()


@pytest.fixture
def fake_time() -> FakeTime:
    return FakeTime()


@pytest.fixture
def replay(api: FakeAPI, fake_time: FakeTime) -> Iterator[Replay]:
    with httpx.Client(transport=httpx.MockTransport(api)) as client:
        yield Replay(
            client=client,
            base_url=BASE_URL,
            feedback_url=FEEDBACK_URL,
            api_key=STAND_IN,
            reference_date="2025-11-08",
            rate=5.0,
            clock=fake_time.clock,
            sleep=fake_time.sleep,
            now=lambda: SENT_AT,
        )


def run(replay: Replay, *phases: tuple[str, list[dict[str, Any]]]) -> Summary:
    summary = Summary(rate=replay.rate)
    replay.run(list(phases), summary)
    return summary


def test_each_listing_is_sent_as_the_sample_holds_it_with_the_registration_shifted(
    replay: Replay, api: FakeAPI
):
    control = records("control")

    run(replay, ("control", control))

    sent = [
        json.loads(request.content)
        for request in api.requests
        if request.url.path != "/v1/feedback"
    ]
    assert len(sent) == len(control)
    for body, record in zip(sent, control, strict=True):
        expected = dict(record["body"])
        year, month = map(int, expected["registration_date"].split("-"))
        month += 11
        expected["registration_date"] = f"{year + (month - 1) // 12}-{(month - 1) % 12 + 1:02d}"
        assert body == expected


def test_a_field_the_listing_lacks_stays_unsent(replay: Replay, api: FakeAPI):
    control = records("control")
    lacking = [record for record in control if "gears" not in record["body"]]
    assert lacking, "the fixture has a listing without gears"

    run(replay, ("control", control))

    sent = [json.loads(request.content) for request in api.requests[::2]]
    assert sum("gears" not in body for body in sent) == len(lacking)
    assert all(None not in body.values() for body in sent)


def test_the_source_records_are_not_changed(replay: Replay):
    control = records("control")

    run(replay, ("control", control))

    assert control == records("control")


@pytest.mark.parametrize(
    ("sent_at", "expected"),
    [
        # 11 whole months: the 12th completes at 00:00 UTC on 2026-11-08.
        (datetime(2026, 11, 7, 23, 59, tzinfo=UTC), "2022-01"),
        (datetime(2026, 11, 8, 0, 0, tzinfo=UTC), "2022-02"),
        # Before a month completes in UTC, whatever the sender's time zone says.
        (datetime.fromisoformat("2026-11-08T00:30:00+01:00"), "2022-01"),
        (datetime(2025, 11, 8, tzinfo=UTC), "2021-02"),
    ],
)
def test_the_registration_moves_by_the_whole_months_since_the_reference_date(
    sent_at: datetime, expected: str
):
    body = {"make": "BMW", "registration_date": "2021-02"}

    assert request_body(body, sent_at, "2025-11-08")["registration_date"] == expected


def test_the_listings_alternate_between_the_two_priced_endpoints_through_the_proxy(
    replay: Replay, api: FakeAPI
):
    run(replay, ("control", records("control")), ("es", records("es")))

    valuations = [request for request in api.requests if request.url.path != "/v1/feedback"]
    assert [request.url.path for request in valuations] == ["/v1/predict", "/v1/price-range"] * 6
    assert {request.url.host for request in valuations} == {"proxy"}
    assert {request.headers["X-Client"] for request in valuations} == {"replay"}
    assert all("X-API-Key" not in request.headers for request in valuations)


def test_each_answer_gets_the_listings_observed_price_with_its_request_id_and_the_key(
    replay: Replay, api: FakeAPI
):
    control = records("control")

    run(replay, ("control", control))

    # Each valuation is followed by its feedback, before the next listing is sent.
    feedback = api.requests[1::2]
    assert len(feedback) == len(control)
    for position, (request, record) in enumerate(zip(feedback, control, strict=True)):
        assert request.url == f"{FEEDBACK_URL}/v1/feedback"
        assert request.headers["X-API-Key"] == STAND_IN
        assert json.loads(request.content) == {
            "request_id": api.request_ids[2 * position],
            "observed_price_eur": record["observed_price_eur"],
        }


def test_the_request_id_comes_from_the_answer(api: FakeAPI, replay: Replay):
    given = "0b9f3c1e-6c0a-4b5e-9d1f-2a3b4c5d6e7f"
    api.scripted["/v1/predict"] = [httpx.Response(200, json={"request_id": given})]

    run(replay, ("control", records("control")[:1]))

    assert api.bodies("/v1/feedback")[0]["request_id"] == given


def test_a_refused_valuation_gets_no_feedback_and_is_counted(api: FakeAPI, replay: Replay):
    api.scripted["/v1/predict"] = [httpx.Response(422, json={"title": "Unprocessable"})]

    summary = run(replay, ("control", records("control")[:2]))

    assert summary.phases[0].answered == {"422": 1, "200": 1}
    assert summary.phases[0].feedback == {"201": 1}
    assert not summary.ok


def test_a_valuation_no_route_handled_is_sent_again(
    api: FakeAPI, replay: Replay, fake_time: FakeTime
):
    api.scripted["/v1/predict"] = [
        httpx.ConnectError("refused"),
        httpx.Response(503, json={}),
        httpx.Response(502, text="Bad Gateway"),
    ]

    summary = run(replay, ("control", records("control")[:1]))

    assert len(api.to("/v1/predict")) == 4
    assert summary.phases[0].answered == {"200": 1}
    assert summary.phases[0].feedback == {"201": 1}
    assert fake_time.sleeps == [0.5, 1.0, 2.0]
    assert summary.ok


def test_a_valuation_that_may_have_been_logged_is_not_sent_twice(api: FakeAPI, replay: Replay):
    """A read timeout can come after the API logged the answer, so a retry could log it twice."""
    api.scripted["/v1/predict"] = [httpx.ReadTimeout("slow"), httpx.Response(500, json={})]

    summary = run(replay, ("control", records("control")[:3]))

    assert len(api.to("/v1/predict")) == 2
    assert summary.phases[0].answered == {"ReadTimeout": 1, "500": 1, "200": 1}
    assert len(api.to("/v1/feedback")) == 1


def test_a_valuation_that_keeps_failing_is_given_up_after_the_retries(
    api: FakeAPI, replay: Replay
):
    api.scripted["/v1/predict"] = [httpx.Response(503, json={})] * 4

    summary = run(replay, ("control", records("control")[:1]))

    assert len(api.to("/v1/predict")) == 4
    assert summary.phases[0].answered == {"503": 1}
    assert api.to("/v1/feedback") == []


def test_a_failed_feedback_is_sent_again_since_a_repeated_one_is_safe(
    api: FakeAPI, replay: Replay
):
    api.scripted["/v1/feedback"] = [
        httpx.ReadTimeout("slow"),
        httpx.Response(500, json={}),
        httpx.Response(200, json={}),
    ]

    summary = run(replay, ("control", records("control")[:1]))

    bodies = api.bodies("/v1/feedback")
    assert len(bodies) == 3
    assert len({json.dumps(body) for body in bodies}) == 1
    assert summary.phases[0].feedback == {"200": 1}
    assert summary.ok


@pytest.mark.parametrize("status", [404, 409])
def test_a_feedback_the_api_refuses_is_counted_not_retried(
    api: FakeAPI, replay: Replay, status: int
):
    api.scripted["/v1/feedback"] = [httpx.Response(status, json={})]

    summary = run(replay, ("control", records("control")[:2]))

    assert len(api.to("/v1/feedback")) == 2
    assert summary.phases[0].feedback == {str(status): 1, "201": 1}
    assert not summary.ok
    assert f"{status}: 1" in format_summary(summary)


def test_a_refused_key_stops_the_run(api: FakeAPI, replay: Replay):
    api.scripted["/v1/feedback"] = [httpx.Response(401, json={})]
    summary = Summary(rate=5.0)

    with pytest.raises(ReplayError, match="401"):
        replay.run([("control", records("control"))], summary)

    assert summary.phases[0].sent == 1
    assert summary.phases[0].feedback == {"401": 1}


def test_listings_are_sent_at_the_rate(replay: Replay, fake_time: FakeTime):
    replay.rate = 2.0

    summary = run(replay, ("control", records("control")))

    # Six listings at 2/s: the last one 2.5 s after the first, the API answering at once.
    assert fake_time.sleeps == [0.5] * 5
    assert summary.finished - summary.started == pytest.approx(2.5)


def test_the_summary_counts_each_phase_and_the_total(api: FakeAPI, replay: Replay):
    api.scripted["/v1/price-range"] = [httpx.Response(422, json={})]

    summary = run(replay, ("control", records("control")[:4]), ("es", records("es")[:2]))

    assert [(phase.name, phase.sent) for phase in summary.phases] == [("control", 4), ("es", 2)]
    text = format_summary(summary)
    lines = text.splitlines()
    assert lines[0].split() == ["phase", "sent", "answered", "feedback"]
    assert lines[1].split() == ["control", "4", "200:", "3,", "422:", "1", "201:", "3"]
    assert lines[2].split() == ["es", "2", "200:", "2", "201:", "2"]
    assert lines[3].split() == ["total", "6", "200:", "5,", "422:", "1", "201:", "5"]
    assert lines[4].startswith("duration ")
    assert "target 5" in lines[4]
    assert lines[5] == "result: some listings were not priced or stored"


def test_a_phase_without_answers_shows_a_dash():
    summary = Summary(rate=1.0, phases=[Phase("control")])

    assert format_summary(summary).splitlines()[1].split() == ["control", "0", "-", "-"]


def test_the_phases_take_the_first_listings_of_each_sample(data_dir: Path):
    phases = phases_of(data_dir, control=4, then="es", then_count=None)

    assert [name for name, _ in phases] == ["control", "es"]
    control, es = (list(listings) for _, listings in phases)
    assert control == records("control")[:4]
    assert es == records("es")
    assert list(read_samples(data_dir / "es.jsonl.gz")) == records("es")


def test_no_control_listings_means_only_the_es_phase(data_dir: Path):
    phases = phases_of(data_dir, control=0, then="es", then_count=2)

    assert [(name, len(list(listings))) for name, listings in phases] == [("es", 2)]


@pytest.fixture
def command_line(
    data_dir: Path, api: FakeAPI, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, FakeAPI]:
    """The command line with the key set and the fake API behind its client."""
    monkeypatch.setenv(cli.API_KEY_VARIABLE, STAND_IN)
    monkeypatch.setattr(cli.httpx, "Client", answer_with(api))
    return data_dir, api


def answer_with(api: FakeAPI) -> Callable[..., httpx.Client]:
    """`httpx.Client` for the command line, with the fake API as its transport."""
    real = httpx.Client

    def client(**options: Any) -> httpx.Client:
        return real(transport=httpx.MockTransport(api), **options)

    return client


def test_the_command_line_runs_control_then_es_and_prints_the_summary(
    data_dir: Path, api: FakeAPI, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    monkeypatch.setenv(cli.API_KEY_VARIABLE, STAND_IN)
    monkeypatch.setattr(cli.httpx, "Client", answer_with(api))

    code = cli.main(
        ["replay", "--control", "3", "--then", "es", "--then-count", "2", "--rate", "100",
         "--data-dir", str(data_dir)]
    )  # fmt: skip

    out = capsys.readouterr().out
    assert code == 0, out
    assert len(api.to("/v1/feedback")) == 5
    assert {request.url.host for request in api.to("/v1/feedback")} == {"api"}
    assert out.splitlines()[1].split()[:2] == ["control", "3"]
    assert out.splitlines()[2].split()[:2] == ["es", "2"]
    assert out.splitlines()[-1] == "result: ok"


def test_the_command_line_fails_when_a_listing_is_not_stored(
    data_dir: Path, api: FakeAPI, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(cli.API_KEY_VARIABLE, STAND_IN)
    monkeypatch.setattr(cli.httpx, "Client", answer_with(api))
    api.scripted["/v1/feedback"] = [httpx.Response(409, json={})]

    code = cli.main(["replay", "--control", "2", "--rate", "100", "--data-dir", str(data_dir)])

    assert code == 1


def test_the_command_line_stops_on_a_refused_key(
    data_dir: Path, api: FakeAPI, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    monkeypatch.setenv(cli.API_KEY_VARIABLE, "wrong")
    monkeypatch.setattr(cli.httpx, "Client", answer_with(api))
    api.scripted["/v1/feedback"] = [httpx.Response(401, json={})] * 10

    code = cli.main(["replay", "--control", "5", "--rate", "100", "--data-dir", str(data_dir)])

    assert code == 1
    assert len(api.to("/v1/predict")) == 1
    assert "refused the key" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["replay"], "nothing to send"),
        (["replay", "--control", "-1"], "0 or more"),
    ],
)
def test_the_command_line_refuses_a_run_that_sends_nothing(
    arguments: list[str], message: str, command_line: tuple[Path, FakeAPI], capsys
):
    data_dir, api = command_line

    assert cli.main([*arguments, "--data-dir", str(data_dir)]) == 2
    assert message in capsys.readouterr().err
    assert api.requests == []


def test_the_command_line_needs_the_key_before_it_sends_anything(
    data_dir: Path, api: FakeAPI, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv(cli.API_KEY_VARIABLE, raising=False)
    monkeypatch.setattr(cli.httpx, "Client", answer_with(api))

    assert cli.main(["replay", "--control", "2", "--data-dir", str(data_dir)]) == 2
    assert api.requests == []
