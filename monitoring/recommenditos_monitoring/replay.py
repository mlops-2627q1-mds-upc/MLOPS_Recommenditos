"""The replay of held-out listings against the API, with their observed prices (#227).

M6 has no real users (EDN-13), so the traffic that makes drift, error and
coverage visible is this replay. It sends listings of the versioned samples of
the `monitoring_samples` stage (`data/monitoring/`, baked into the image) to the
running API as a client would, and reports each listing's real price to
`/v1/feedback` (FR-10):

- first `--control N` listings of `control`, drawn from `train` at random and
  so i.i.d. with the drift reference: normal traffic, a window of which the
  drift job must not flag (NFR-11). The model was fitted on them, so their
  error and coverage are in-sample and optimistic;
- then, with `--then es`, the listings of the `ES` holdout (EDN-03), the market
  the model never saw, which the drift job must flag.

Each listing is sent with the body the sample holds, the fields its listing has
and nothing else, so the prediction log records what such a client sends
(#191). The listings alternate between `/v1/predict` and `/v1/price-range`,
both through the reverse proxy, as a client reaches them, and carry
`X-Client: replay`. `registration_date` is shifted forward by the whole months
from the sample's `reference_date` (2025-11-08, the day the model computes ages
on) to the send time, so a 2025 listing replayed in 2026 is as old as it was in
the snapshot (the rule is in `drift_features.py`).

After each answer, the listing's observed price goes to `/v1/feedback` with the
answer's `request_id` and the key in `X-API-Key`, straight to the API inside the
Compose network, because the proxy refuses the path from outside. The same price
again is answered 200, so a feedback is retried on any failure; a valuation is
retried only when it surely reached no route (no connection, or 502/503), so a
retry never logs a listing twice. That holds because the wait for an answer
(`--timeout`) is at most 50 s: nginx answers 503 itself after 60 s without the
API's answer, which the API may still log, and the replay's read timeout, which
is never retried, comes first. Listings are sent one at a time at `--rate`
per second, and a summary of what was sent and answered ends the run.

    python -m recommenditos_monitoring replay --control 5000 --then es --rate 5
"""

from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
import gzip
from itertools import islice
import json
from pathlib import Path
import sys
import time
from typing import Any, TextIO

import httpx

from recommenditos_monitoring.drift_features import months_since_reference, shifted_registration

#: The two priced endpoints, sent alternately, so both live panels move and each
#: drift window gets about as many ranges as point estimates.
ENDPOINTS: tuple[str, ...] = ("/v1/predict", "/v1/price-range")
FEEDBACK_PATH = "/v1/feedback"
#: The header the API records as `client`; `replay` is one of its KNOWN_CLIENTS.
CLIENT_HEADER, CLIENT = "X-Client", "replay"
API_KEY_HEADER = "X-API-Key"
#: The API's answers to a valuation that no route has handled, so that sending
#: it again cannot log the listing twice: the proxy found no upstream (502), or
#: the API could not serve or log it (503).
RETRIED_VALUATION_STATUSES = frozenset({502, 503})
OK, UNAUTHORIZED = 200, 401
#: The phases a run can switch to after the control listings.
THEN_SAMPLES = ("es",)


class ReplayError(Exception):
    """A run that cannot go on, such as a key the API refuses."""


@dataclass
class Phase:
    """The tally of one sample's listings."""

    name: str
    sent: int = 0
    #: The valuations by answer: an HTTP status, or the error that left none.
    answered: Counter[str] = field(default_factory=Counter)
    #: The feedback by answer, for the valuations answered 200.
    feedback: Counter[str] = field(default_factory=Counter)


@dataclass
class Summary:
    """What a run sent and what came back, by phase."""

    rate: float
    phases: list[Phase] = field(default_factory=list)
    started: float = 0.0
    finished: float = 0.0

    @property
    def ok(self) -> bool:
        """Whether every listing was priced and its price stored."""
        return all(
            set(phase.answered) <= {"200"} and set(phase.feedback) <= {"200", "201"}
            for phase in self.phases
        )


def read_samples(path: Path) -> Iterator[dict[str, Any]]:
    """The records of a sample file, one at a time: `listing_id`, `observed_price_eur`, `body`."""
    with gzip.open(path, "rt", encoding="utf-8") as lines:
        for line in lines:
            yield json.loads(line)


def reference_date_of(data_dir: Path) -> str:
    """The `reference_date` of the samples, from their manifest."""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    return str(manifest["reference_date"])


def request_body(
    body: Mapping[str, Any], sent_at: datetime, reference_date: str
) -> dict[str, Any]:
    """`body` as the replay sends it at `sent_at`: the same fields, the registration shifted."""
    sent = dict(body)
    if "registration_date" in sent:
        months = months_since_reference(sent_at, reference_date)
        sent["registration_date"] = shifted_registration(sent["registration_date"], months)
    return sent


@dataclass
class Replay:
    """Sends listings to an API and their prices to its feedback endpoint."""

    client: httpx.Client
    base_url: str
    feedback_url: str
    api_key: str
    reference_date: str
    rate: float
    retries: int = 3
    backoff_s: float = 0.5
    progress_every: int = 500
    out: TextIO = sys.stderr
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], datetime] = lambda: datetime.now(UTC)

    def run(
        self, phases: Sequence[tuple[str, Iterable[Mapping[str, Any]]]], summary: Summary
    ) -> None:
        """Replays each phase's records in order into `summary`, at `rate` listings per second."""
        summary.started = self.clock()
        sent = 0
        try:
            for name, records in phases:
                phase = Phase(name)
                summary.phases.append(phase)
                for record in records:
                    self._pace(summary.started, sent)
                    self.send(record, ENDPOINTS[sent % len(ENDPOINTS)], phase)
                    sent += 1
                    if phase.sent % self.progress_every == 0:
                        print(f"{name}: {phase.sent} listings sent", file=self.out, flush=True)
        finally:
            summary.finished = self.clock()

    def _pace(self, started: float, sent: int) -> None:
        wait = started + sent / self.rate - self.clock()
        if wait > 0:
            self.sleep(wait)

    def send(self, record: Mapping[str, Any], endpoint: str, phase: Phase) -> None:
        """Sends one listing to `endpoint` and, when it is priced, its observed price."""
        body = request_body(record["body"], self.now(), self.reference_date)
        phase.sent += 1
        answer = self._post(
            f"{self.base_url}{endpoint}",
            body,
            {CLIENT_HEADER: CLIENT},
            retry_on=RETRIED_VALUATION_STATUSES,
            retry_errors=(httpx.ConnectError, httpx.ConnectTimeout),
        )
        phase.answered[_status(answer)] += 1
        if not isinstance(answer, httpx.Response) or answer.status_code != OK:
            return
        feedback = self._post(
            f"{self.feedback_url}{FEEDBACK_PATH}",
            {
                "request_id": answer.json()["request_id"],
                "observed_price_eur": float(record["observed_price_eur"]),
            },
            {CLIENT_HEADER: CLIENT, API_KEY_HEADER: self.api_key},
            retry_on=frozenset(range(500, 600)),
            retry_errors=(httpx.TransportError,),
        )
        phase.feedback[_status(feedback)] += 1
        refused = isinstance(feedback, httpx.Response) and feedback.status_code == UNAUTHORIZED
        if refused:
            raise ReplayError(
                "/v1/feedback refused the key (401): RECOMMENDITOS_API_KEY is not the API's key"
            )

    def _post(
        self,
        url: str,
        body: Mapping[str, Any],
        headers: Mapping[str, str],
        retry_on: frozenset[int],
        retry_errors: tuple[type[httpx.TransportError], ...],
    ) -> httpx.Response | httpx.TransportError:
        """The answer to a POST, after up to `retries` more attempts; or the last error."""
        for attempt in range(self.retries + 1):
            last = attempt == self.retries
            try:
                response = self.client.post(url, json=body, headers=dict(headers))
            except retry_errors as error:
                if last:
                    return error
            except httpx.TransportError as error:
                return error
            else:
                if response.status_code not in retry_on or last:
                    return response
            self.sleep(self.backoff_s * 2**attempt)
        raise AssertionError("unreachable: the last attempt returns")


def _status(answer: httpx.Response | httpx.TransportError) -> str:
    if isinstance(answer, httpx.Response):
        return str(answer.status_code)
    return type(answer).__name__


def format_summary(summary: Summary) -> str:
    """The summary a run ends with, one line per phase and a total."""

    def counts(counter: Counter[str]) -> str:
        return ", ".join(f"{status}: {n}" for status, n in sorted(counter.items())) or "-"

    total = Phase("total")
    for phase in summary.phases:
        total.sent += phase.sent
        total.answered.update(phase.answered)
        total.feedback.update(phase.feedback)
    rows = [("phase", "sent", "answered", "feedback")]
    rows += [
        (phase.name, str(phase.sent), counts(phase.answered), counts(phase.feedback))
        for phase in [*summary.phases, total]
    ]
    widths = [max(len(row[column]) for row in rows) for column in range(3)]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row[:3], widths, strict=True))
        + "  "
        + row[3]
        for row in rows
    ]
    duration = summary.finished - summary.started
    achieved = total.sent / duration if duration > 0 else 0.0
    lines.append(f"duration {duration:.1f} s, {achieved:.2f} listings/s (target {summary.rate:g})")
    lines.append("result: " + ("ok" if summary.ok else "some listings were not priced or stored"))
    return "\n".join(lines)


def phases_of(
    data_dir: Path, control: int, then: str | None, then_count: int | None
) -> list[tuple[str, Iterable[Mapping[str, Any]]]]:
    """The phases of a run: `control` listings of the control sample, then the `then` sample."""
    phases: list[tuple[str, Iterable[Mapping[str, Any]]]] = []
    if control > 0:
        phases.append(("control", islice(read_samples(data_dir / "control.jsonl.gz"), control)))
    if then is not None:
        phases.append((then, islice(read_samples(data_dir / f"{then}.jsonl.gz"), then_count)))
    return phases
