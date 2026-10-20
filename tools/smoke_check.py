"""Check a running stack from outside, through its reverse proxy.

    uv run python -m tools.smoke_check http://localhost:8080      (make smoke)

It checks what a deployment has to get right and a unit test cannot see:

* `GET /health` answers 200, and the model version it reports is the hash
  `dvc.lock` records for that model's bundle (FR-12), so the image serves the
  model the checkout points to;
* the proxy's security headers are set and its `Server` header names no version;
* one valuation through `POST /v1/predict` returns a positive price;
* the internal endpoints, `/feedback` and `/metrics`, are refused by the proxy
  with a 403 (FR-10, EDN-13). The API itself would answer 404, 405 or 401, so
  only the proxy rule passes this check.

Each check prints one line. The exit code is 0 when every check passed and 1
otherwise, so a deployment job can run it as its last step (NFR-12, NFR-13).
The standard library does the HTTP; `dvc.lock` is read with PyYAML, which the
serving runtime already has.
"""

import argparse
from dataclasses import dataclass
from http import HTTPStatus
import json
from pathlib import Path
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

DEFAULT_LOCK = Path(__file__).resolve().parents[1] / "dvc.lock"

#: The paths the proxy must refuse, with the methods a client might try.
#: `/v1/feedback` because a client endpoint is versioned and #140 may mount the
#: feedback route under `/v1`.
REFUSED = (
    ("GET", "/feedback"),
    ("POST", "/feedback"),
    ("POST", "/v1/feedback"),
    ("GET", "/metrics"),
)

#: A listing with the ten fields FR-01 requires, values the served model accepts.
LISTING = {
    "make": "Volkswagen",
    "model": "Golf",
    "registration_date": "2019-06",
    "mileage_km_raw": 45000,
    "power_kw": 110,
    "fuel_category": "Gasoline",
    "transmission": "Manual",
    "country_code": "DE",
    "body_type": "Compact",
    "seller_type": "Dealer",
}

TIMEOUT_S = 10


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    def line(self) -> str:
        return f"{'PASS' if self.passed else 'FAIL'}  {self.name}: {self.detail}"


@dataclass(frozen=True)
class Response:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)

    def describe(self) -> str:
        """The status, or why there was none."""
        if self.status == 0:
            return f"no answer ({self.body.decode()})"
        return f"status {self.status}"


def bundle_hash(lock: Path, variant: str) -> str | None:
    """The MD5 `dvc.lock` records for `models/<variant>`, without DVC's `.dir` suffix."""
    stages = yaml.safe_load(lock.read_text(encoding="utf-8"))["stages"]
    for stage in stages.values():
        for out in stage.get("outs", []):
            if out["path"] == f"models/{variant}":
                return str(out["md5"]).removesuffix(".dir")
    return None


def fetch(base_url: str, method: str, path: str, body: Any = None) -> Response:
    """One request. An HTTP error is a response too, and no answer at all is status 0."""
    data = None if body is None else json.dumps(body).encode()
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = Request(base_url.rstrip("/") + path, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=TIMEOUT_S) as reply:
            return Response(reply.status, dict(reply.headers.items()), reply.read())
    except HTTPError as error:
        return Response(error.code, dict(error.headers.items()), error.read())
    except (URLError, OSError) as error:
        return Response(0, {}, str(getattr(error, "reason", error)).encode())


def check_health(health: Response, lock: Path) -> Check:
    name = "GET /health"
    if health.status != HTTPStatus.OK:
        return Check(name, False, f"{health.describe()}, expected 200")
    model = health.json().get("model", {})
    variant, version = model.get("variant"), model.get("version")
    expected = bundle_hash(lock, str(variant))
    if expected is None:
        return Check(name, False, f"{lock.name} records no models/{variant}")
    if version != expected:
        return Check(name, False, f"{variant} version {version}, {lock.name} records {expected}")
    return Check(name, True, f"200, {variant} version {version} as in {lock.name}")


def check_headers(health: Response) -> Check:
    expected = {"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY"}
    headers = {key.lower(): value for key, value in health.headers.items()}
    wrong = [
        f"{key}: {headers.get(key.lower())!r}"
        for key, value in expected.items()
        if headers.get(key.lower()) != value
    ]
    server = headers.get("server", "")
    if any(character.isdigit() for character in server):
        wrong.append(f"Server: {server!r} names a version")
    if "strict-transport-security" in headers:
        wrong.append("HSTS is set on plain HTTP")
    if wrong:
        return Check("Proxy headers", False, "; ".join(wrong))
    return Check("Proxy headers", True, "nosniff, DENY, no version in Server, no HSTS")


def check_predict(base_url: str) -> Check:
    name = "POST /v1/predict"
    reply = fetch(base_url, "POST", "/v1/predict", LISTING)
    if reply.status != HTTPStatus.OK:
        return Check(name, False, f"{reply.describe()}: {reply.body[:200]!r}")
    price = reply.json().get("price_eur")
    if not isinstance(price, int | float) or price <= 0:
        return Check(name, False, f"price_eur is {price!r}")
    return Check(name, True, f"200, {LISTING['make']} {LISTING['model']} at {price} EUR")


def check_refused(base_url: str, method: str, path: str) -> Check:
    name = f"{method} {path}"
    reply = fetch(base_url, method, path, {} if method == "POST" else None)
    if reply.status != HTTPStatus.FORBIDDEN:
        return Check(name, False, f"{reply.describe()}, expected the proxy's 403")
    return Check(name, True, "403, refused by the proxy")


def run(base_url: str, lock: Path) -> list[Check]:
    """Every check against `base_url`, in order; only the first when nothing answers."""
    health = fetch(base_url, "GET", "/health")
    if health.status == 0:
        return [check_health(health, lock)]
    return [
        check_health(health, lock),
        check_headers(health),
        check_predict(base_url),
        *[check_refused(base_url, method, path) for method, path in REFUSED],
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url", help="the proxy's address, e.g. http://localhost:8080")
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK, help="the dvc.lock to compare")
    args = parser.parse_args(argv)
    checks = run(args.base_url, args.lock)
    for check in checks:
        print(check.line())
    failed = sum(not check.passed for check in checks)
    print(f"{len(checks) - failed} of {len(checks)} checks passed against {args.base_url}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
