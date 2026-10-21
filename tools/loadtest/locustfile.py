"""The load test of the valuation endpoints (NFR-02, NFR-03; issue #143).

Run it through `make loadtest`, which also takes the measurements around it
(`tools/loadtest/run.py`, docs/docs/load-testing.md). Directly, from the
repository root:

    uv run python -m locust -f tools/loadtest/locustfile.py --headless \
        --host http://localhost:8080 -u 20 -r 20 -t 1m --rate 20

At start it reads the target's `/openapi.json` and calls only the valuation
endpoints the target serves (`tools/loadtest/plan.py`). The total rate is
`--rate` requests per second, shared evenly by the `-u` users; a user waits
for its answer before it sends the next request, so a slow target gets fewer
requests rather than a growing queue. Every request with a 4xx or 5xx answer
counts as a failure.

`--request-ids` names a file that receives the `X-Request-ID` of every
successful valuation, one per line, for the prediction-log check of NFR-03.
"""

from http import HTTPStatus
import json
from pathlib import Path
import random
from typing import Any
from urllib.request import urlopen

from locust import HttpUser, events, task
from locust.env import Environment
from locust.user.wait_time import constant_throughput

from tools.loadtest.plan import Plan, build_plan

#: The request bodies `tools/loadtest/bodies.py` writes by default.
DEFAULT_BODIES = "data/interim/loadtest/bodies.jsonl"

REQUEST_ID_HEADER = "X-Request-ID"

#: The mix, made when the test starts, from the target's OpenAPI document.
_state: dict[str, Plan] = {}
_request_ids: list[str] = []


def _options(environment: Environment) -> Any:
    """The parsed command line, which Locust sets before any event below fires."""
    if environment.parsed_options is None:
        raise RuntimeError("run this file with the locust command")
    return environment.parsed_options


@events.init_command_line_parser.add_listener
def _arguments(parser: Any) -> None:
    parser.add_argument("--rate", type=float, default=20.0, help="Total requests per second")
    parser.add_argument("--bodies", default=DEFAULT_BODIES, help="JSON Lines of request bodies")
    parser.add_argument("--request-ids", default="", help="File for the successful request IDs")


@events.test_start.add_listener
def _plan_the_mix(environment: Environment, **_: Any) -> None:
    options = _options(environment)
    lines = Path(options.bodies).read_text(encoding="utf-8").splitlines()
    bodies = [json.loads(line) for line in lines if line.strip()]
    # The target the person running the test named with --host.
    url = f"{str(environment.host).rstrip('/')}/openapi.json"
    with urlopen(url, timeout=10) as answer:
        openapi = json.load(answer)
    plan = _state["plan"] = build_plan(openapi, bodies)
    for endpoint in plan.endpoints:
        print(f"loadtest: {endpoint.path} weight {endpoint.weight}, {len(endpoint.bodies)} bodies")
    for path, reason in plan.skipped.items():
        print(f"loadtest: {path} skipped, {reason}")


@events.request.add_listener
def _remember_the_id(name: str, response: Any, exception: Any, **_: Any) -> None:
    if (
        exception is None
        and response is not None
        and HTTPStatus.OK <= response.status_code < HTTPStatus.MULTIPLE_CHOICES
    ):
        request_id = response.headers.get(REQUEST_ID_HEADER)
        if request_id:
            _request_ids.append(request_id)


@events.test_stop.add_listener
def _write_the_ids(environment: Environment, **_: Any) -> None:
    path = _options(environment).request_ids
    if path:
        Path(path).write_text("".join(f"{i}\n" for i in _request_ids), encoding="utf-8")


class Seller(HttpUser):
    """A client of the valuation endpoints, at its share of `--rate`."""

    def on_start(self) -> None:
        self.rng = random.Random()
        options = _options(self.environment)
        self.pace = constant_throughput(options.rate / options.num_users)

    def wait_time(self) -> float:
        return self.pace(self)

    @task
    def value_a_listing(self) -> None:
        path, body = _state["plan"].pick(self.rng)
        self.client.post(path, json=body, name=path)
