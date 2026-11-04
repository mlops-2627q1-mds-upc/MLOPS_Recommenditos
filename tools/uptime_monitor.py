"""Set up the external uptime check of the deployed API, and export its figure (issue #226).

    set -a; . .env; set +a        # BETTERSTACK_API_TOKEN, the team's Uptime API token
    uv run python -m tools.uptime_monitor apply
    uv run python -m tools.uptime_monitor apply --dry-run
    uv run python -m tools.uptime_monitor export --from 2026-11-05 --to 2026-12-15
    uv run python -m tools.uptime_monitor incidents --monitor-id <id>
    uv run python -m tools.uptime_monitor delete-monitor <id>

Better Stack (formerly Better Uptime) requests the API's public `/health` from
outside the VM, emails the account when it fails, and keeps the uptime figure
that NFR-05 promises to report (EDN-10). Grafana and Prometheus only see the VM
from inside, so they cannot tell that the VM itself is unreachable.

`apply` makes the account match `MONITOR` and `STATUS_PAGE` below: it creates
the monitor, the status page and the status page's entry for the monitor when
they are missing, and PATCHes only the attributes that differ, so a second run
changes nothing and says so. The configuration is therefore this file, reviewed
like code, and a lost account can be rebuilt by running it. `export` writes the
availability of the monitor over a date range, from the monitor SLA endpoint, to
a JSON file for the report. `incidents` lists the incidents of one monitor with
the timestamps that give the time to alert.

The token is read from the environment only and is never printed or logged:
every message that leaves this module passes through `redact`. Standard library
only; the HTTP layer is the `Api` class, which the tests replace.

What the API takes was read from https://betterstack.com/docs/uptime/api/ on
2026-11-04. The host `uptime.betterstack.com` is the one the account answered on
(the docs print `incidents.betterstack.com`, which serves the same API).
"""

import argparse
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Protocol
import urllib.error
import urllib.parse
import urllib.request

BASE_URL = "https://uptime.betterstack.com/api/v2"
TOKEN_VARIABLE = "BETTERSTACK_API_TOKEN"  # noqa: S105 - the name of the variable, not a secret
#: Seconds to wait for one API call.
API_TIMEOUT = 30

#: The API's public `/health`, which the Deploy workflow's smoke job also requests
#: from a GitHub-hosted runner (docs/docs/deployment.md). Plain HTTP, no TLS.
HEALTH_URL = "http://nattech.fib.upc.edu:40520/health"
DEFAULT_EXPORT = Path("reports/uptime/availability.json")

#: The monitor, as the create and update monitor calls take it. `status` is down
#: on any answer but a 2xx (a rehearsal monitor on a 403 opened an incident, see
#: docs/docs/deployment.md). 180 s is the shortest interval of the free
#: plan that the API accepted (the free plan advertises 3-minute checks), the
#: 15 s timeout is far above the API's answer time and below the interval, and
#: the confirmation period asks for the failure to last one more interval before
#: an incident opens, so one lost packet or one slow answer sends no email while
#: any outage longer than NFR-05's 30 s recovery still does. Two regions, one
#: near the VM and one across the Atlantic, because an incident needs the
#: failure from the regions that check. Email only, to the team's members.
MONITOR: dict[str, Any] = {
    "monitor_type": "status",
    "pronounceable_name": "Recommenditos API health",
    "url": HEALTH_URL,
    "http_method": "get",
    "check_frequency": 180,
    "request_timeout": 15,
    "confirmation_period": 180,
    "recovery_period": 180,
    "regions": ["eu", "us"],
    "email": True,
    "sms": False,
    "call": False,
    "push": False,
    "critical_alert": False,
}

#: The public status page. The subdomain is unique across Better Stack.
STATUS_PAGE: dict[str, Any] = {
    "company_name": "Recommenditos",
    "company_url": "https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos",
    "timezone": "Madrid",
    "subdomain": "recommenditos",
    "history": 90,
}

#: How the monitor appears on the status page.
RESOURCE: dict[str, Any] = {
    "resource_type": "Monitor",
    "public_name": "Price API",
    "widget_type": "history",
}


class Api(Protocol):
    """What the commands need from Better Stack; the tests replace it."""

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call `method` on `path` below the API root and return the decoded JSON."""
        ...


class UptimeError(Exception):
    """A failure the command line reports in one line, without a traceback."""


def redact(text: str, token: str | None) -> str:
    """`text` with every occurrence of the API token masked."""
    return text.replace(token, "***") if token else text


class BetterStack:
    """The Better Stack Uptime API over HTTPS with the team's bearer token."""

    def __init__(self, token: str, base_url: str = BASE_URL, opener: Any = None) -> None:
        self._token = token
        self._base_url = base_url.rstrip("/")
        self._open = opener or urllib.request.urlopen

    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        body: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self._base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(  # noqa: S310 - https URL built from BASE_URL
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with self._open(request, timeout=API_TIMEOUT) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            detail = redact(error.read().decode(errors="replace"), self._token)[:500]
            raise UptimeError(f"{method} {path} failed with HTTP {error.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            reason = redact(str(getattr(error, "reason", error)), self._token)
            raise UptimeError(f"{method} {path} failed: {reason}") from None
        if not payload:
            return {}
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise UptimeError(
                f"{method} {path} answered with {type(decoded).__name__}, not an object"
            )
        return decoded


def paginate(
    api: Api, path: str, params: Mapping[str, Any] | None = None
) -> Iterator[dict[str, Any]]:
    """Every item of a list endpoint, page by page (250 per page, the maximum)."""
    page = 1
    while True:
        answer = api.request("GET", path, params={**(params or {}), "page": page, "per_page": 250})
        yield from answer.get("data", [])
        if not (answer.get("pagination") or {}).get("next"):
            return
        page += 1


def differing(wanted: Mapping[str, Any], actual: Mapping[str, Any]) -> dict[str, Any]:
    """The attributes of `wanted` that `actual` does not hold.

    The API answers ids as numbers or strings and lists in an order of its own,
    so lists compare as sorted sets of strings and everything else as strings
    unless it is a boolean.
    """

    def same(a: Any, b: Any) -> bool:
        if isinstance(a, list) and isinstance(b, list):
            return sorted(map(str, a)) == sorted(map(str, b))
        if isinstance(a, bool) or isinstance(b, bool):
            return a is b
        return str(a) == str(b)

    return {key: value for key, value in wanted.items() if not same(value, actual.get(key))}


def ensure_monitor(
    api: Api, wanted: Mapping[str, Any], *, dry_run: bool = False
) -> tuple[str | None, str]:
    """Create the monitor on `wanted["url"]` or PATCH its drifted attributes.

    Returns its id (None on a dry run that would create it) and what happened.
    """
    found = [
        item
        for item in paginate(api, "/monitors", {"url": wanted["url"]})
        if item["attributes"].get("url") == wanted["url"]
    ]
    if len(found) > 1:
        ids = ", ".join(item["id"] for item in found)
        raise UptimeError(
            f"{len(found)} monitors check {wanted['url']} ({ids}); delete the extras first"
        )
    if not found:
        if dry_run:
            return None, "would create"
        created = api.request("POST", "/monitors", body=wanted)
        return str(created["data"]["id"]), "created"
    monitor_id = str(found[0]["id"])
    drift = differing(wanted, found[0]["attributes"])
    if not drift:
        return monitor_id, "unchanged"
    if not dry_run:
        api.request("PATCH", f"/monitors/{monitor_id}", body=drift)
    return monitor_id, f"{'would update' if dry_run else 'updated'} {', '.join(sorted(drift))}"


def ensure_status_page(
    api: Api, wanted: Mapping[str, Any], *, dry_run: bool = False
) -> tuple[str | None, str]:
    """Create the status page of `wanted["subdomain"]` or PATCH its drifted attributes."""
    found = [
        item
        for item in paginate(api, "/status-pages")
        if item["attributes"].get("subdomain") == wanted["subdomain"]
    ]
    if not found:
        if dry_run:
            return None, "would create"
        created = api.request("POST", "/status-pages", body=wanted)
        return str(created["data"]["id"]), "created"
    page_id = str(found[0]["id"])
    drift = differing(wanted, found[0]["attributes"])
    if not drift:
        return page_id, "unchanged"
    if not dry_run:
        api.request("PATCH", f"/status-pages/{page_id}", body=drift)
    return page_id, f"{'would update' if dry_run else 'updated'} {', '.join(sorted(drift))}"


def ensure_resource(
    api: Api,
    page_id: str | None,
    monitor_id: str | None,
    wanted: Mapping[str, Any],
    *,
    dry_run: bool = False,
) -> str:
    """Show the monitor on the status page, or fix how it is shown."""
    if page_id is None or monitor_id is None:
        return "would create"
    path = f"/status-pages/{page_id}/resources"
    found = [
        item
        for item in paginate(api, path)
        if item["attributes"].get("resource_type") == wanted["resource_type"]
        and str(item["attributes"].get("resource_id")) == monitor_id
    ]
    if not found:
        if not dry_run:
            api.request("POST", path, body={**wanted, "resource_id": monitor_id})
        return "would create" if dry_run else "created"
    drift = differing(wanted, found[0]["attributes"])
    if not drift:
        return "unchanged"
    if not dry_run:
        api.request("PATCH", f"{path}/{found[0]['id']}", body=drift)
    return f"{'would update' if dry_run else 'updated'} {', '.join(sorted(drift))}"


def public_url(subdomain: str) -> str:
    """Where a status page is served (the address the account shows for the subdomain)."""
    return f"https://{subdomain}.betteruptime.com"


def apply(api: Api, *, dry_run: bool = False) -> list[str]:
    """Make the account match `MONITOR`, `STATUS_PAGE` and `RESOURCE`; one line per object."""
    monitor_id, monitor_result = ensure_monitor(api, MONITOR, dry_run=dry_run)
    page_id, page_result = ensure_status_page(api, STATUS_PAGE, dry_run=dry_run)
    resource_result = ensure_resource(api, page_id, monitor_id, RESOURCE, dry_run=dry_run)
    return [
        (
            f"monitor {monitor_id or '-'} ({MONITOR['url']}, "
            f"every {MONITOR['check_frequency']} s): {monitor_result}"
        ),
        f"status page {page_id or '-'} ({public_url(STATUS_PAGE['subdomain'])}): {page_result}",
        f"status page entry for the monitor: {resource_result}",
    ]


def monitor_id_for(api: Api, url: str) -> str:
    """The id of the monitor that checks `url`."""
    for item in paginate(api, "/monitors", {"url": url}):
        if item["attributes"].get("url") == url:
            return str(item["id"])
    raise UptimeError(f"no monitor checks {url}; run `apply` first")


def export_availability(api: Api, monitor_id: str, start: str, end: str) -> dict[str, Any]:
    """The monitor's availability from `start` to `end` (YYYY-MM-DD), as the report states it."""
    sla = api.request("GET", f"/monitors/{monitor_id}/sla", params={"from": start, "to": end})
    monitor = api.request("GET", f"/monitors/{monitor_id}")["data"]["attributes"]
    return {
        "monitor_id": monitor_id,
        "monitor": monitor.get("pronounceable_name"),
        "url": monitor.get("url"),
        "check_frequency_s": monitor.get("check_frequency"),
        "regions": monitor.get("regions"),
        "from": start,
        "to": end,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "availability_percent": sla["data"]["attributes"]["availability"],
        "total_downtime_s": sla["data"]["attributes"]["total_downtime"],
        "number_of_incidents": sla["data"]["attributes"]["number_of_incidents"],
        "longest_incident_s": sla["data"]["attributes"]["longest_incident"],
        "average_incident_s": sla["data"]["attributes"]["average_incident"],
    }


def list_incidents(api: Api, monitor_id: str, start: str, end: str) -> list[dict[str, Any]]:
    """The incidents Better Stack opened for the monitor, with the timestamps of their alerts."""
    incidents = []
    for item in paginate(api, "/incidents", {"from": start, "to": end}):
        related = ((item.get("relationships") or {}).get("monitor") or {}).get("data") or {}
        if str(related.get("id")) == monitor_id:
            incidents.append({"id": item["id"], **item["attributes"]})
    return incidents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure and read the Better Stack uptime check."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    apply_cmd = commands.add_parser(
        "apply", help="create or update the monitor and the status page"
    )
    apply_cmd.add_argument(
        "--dry-run", action="store_true", help="say what would change, change nothing"
    )

    export_cmd = commands.add_parser(
        "export", help="write the monitor's availability over a date range"
    )
    export_cmd.add_argument("--from", dest="start", required=True, help="first day, YYYY-MM-DD")
    export_cmd.add_argument("--to", dest="end", required=True, help="last day, YYYY-MM-DD")
    export_cmd.add_argument(
        "--url", default=HEALTH_URL, help="the checked URL (default: the API's /health)"
    )
    export_cmd.add_argument("--out", type=Path, default=DEFAULT_EXPORT)

    incidents_cmd = commands.add_parser("incidents", help="list the monitor's incidents as JSON")
    incidents_cmd.add_argument("--monitor-id", required=True)
    incidents_cmd.add_argument("--from", dest="start", default="2026-11-01")
    incidents_cmd.add_argument("--to", dest="end", default=datetime.now(UTC).date().isoformat())

    create_cmd = commands.add_parser(
        "create-monitor",
        help="create a monitor on another URL with the settings of the /health monitor",
    )
    create_cmd.add_argument("--name", required=True)
    create_cmd.add_argument("--url", required=True)

    delete_cmd = commands.add_parser("delete-monitor", help="delete a monitor by id")
    delete_cmd.add_argument("monitor_id")
    return parser


def run(args: argparse.Namespace, api: Api) -> list[str]:
    """Run one command and return the lines to print."""
    if args.command == "apply":
        return apply(api, dry_run=args.dry_run)
    if args.command == "export":
        figure = export_availability(api, monitor_id_for(api, args.url), args.start, args.end)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(figure, indent=2) + "\n")
        return [
            (
                f"wrote {args.out}: {figure['availability_percent']} % available, "
                f"{figure['number_of_incidents']} incidents"
            )
        ]
    if args.command == "incidents":
        return [json.dumps(list_incidents(api, args.monitor_id, args.start, args.end), indent=2)]
    if args.command == "create-monitor":
        monitor_id, result = ensure_monitor(
            api, {**MONITOR, "pronounceable_name": args.name, "url": args.url}
        )
        return [f"monitor {monitor_id}: {result}"]
    if args.command == "delete-monitor":
        api.request("DELETE", f"/monitors/{args.monitor_id}")
        return [f"deleted monitor {args.monitor_id}"]
    raise UptimeError(f"unknown command {args.command}")


def main(argv: Sequence[str] | None = None, api: Api | None = None) -> int:
    args = build_parser().parse_args(argv)
    token = os.environ.get(TOKEN_VARIABLE)
    if api is None:
        if not token:
            print(
                f"{TOKEN_VARIABLE} is not set; load the team's Uptime API token from .env",
                file=sys.stderr,
            )
            return 1
        api = BetterStack(token)
    try:
        for line in run(args, api):
            print(redact(line, token))
    except UptimeError as error:
        print(f"error: {redact(str(error), token)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
