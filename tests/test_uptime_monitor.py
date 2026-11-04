"""The uptime check's setup script (tools/uptime_monitor.py, #226) against a fake Better Stack.

No test reaches the network: `FakeApi` keeps monitors, status pages and their
entries in memory and answers the paths the script calls, so what is checked is
the script's decisions (create or update, what it sends, what it prints) and not
Better Stack.
"""

from email.message import Message
import io
import json
import logging
from pathlib import Path
from typing import Any, Self
import urllib.error

import pytest
from tools import uptime_monitor as um

SENTINEL = "sentinel-value-the-tests-look-for"  # stands in for the API token


class FakeApi:
    """The slice of the Uptime API the script uses, kept in memory."""

    def __init__(self, per_page: int = 250) -> None:
        self.monitors: dict[str, dict[str, Any]] = {}
        self.pages: dict[str, dict[str, Any]] = {}
        self.resources: dict[str, dict[str, dict[str, Any]]] = {}
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.per_page = per_page
        self.incidents: list[dict[str, Any]] = []
        self._next_id = 100

    def _new_id(self) -> str:
        self._next_id += 1
        return str(self._next_id)

    def _listing(self, items: dict[str, dict[str, Any]], kind: str, params: Any) -> dict[str, Any]:
        rows: list[dict[str, Any]] = [
            {"id": i, "type": kind, "attributes": a} for i, a in items.items()
        ]
        if params and "url" in params:
            rows = [r for r in rows if params["url"] in str(r["attributes"].get("url"))]
        page = int(params.get("page", 1)) if params else 1
        chunk = rows[(page - 1) * self.per_page : page * self.per_page]
        more = page * self.per_page < len(rows)
        return {"data": chunk, "pagination": {"next": "https://next" if more else None}}

    def request(
        self, method: str, path: str, *, params: Any = None, body: Any = None
    ) -> dict[str, Any]:
        self.calls.append((method, path, dict(body) if body is not None else None))
        parts = path.strip("/").split("/")
        if parts[0] == "monitors":
            return self._monitors(method, parts, params, body)
        if parts[0] == "incidents":
            return {"data": self.incidents, "pagination": {"next": None}}
        if parts[0] == "status-pages":
            return self._status_pages(method, parts, params, body)
        raise AssertionError(f"unexpected call {method} {path}")

    def _monitors(self, method: str, parts: list[str], params: Any, body: Any) -> dict[str, Any]:
        if method == "GET" and len(parts) == 1:
            return self._listing(self.monitors, "monitor", params)
        if method == "POST":
            monitor_id = self._new_id()
            self.monitors[monitor_id] = {**body, "regions": list(reversed(body["regions"]))}
            return {"data": {"id": monitor_id, "attributes": self.monitors[monitor_id]}}
        monitor_id = parts[1]
        if method == "PATCH":
            self.monitors[monitor_id].update(body)
            return {"data": {"id": monitor_id, "attributes": self.monitors[monitor_id]}}
        if method == "DELETE":
            del self.monitors[monitor_id]
            return {}
        if method == "GET" and len(parts) == 3 and parts[2] == "sla":
            attributes = {
                "availability": 99.9876,
                "total_downtime": 120,
                "number_of_incidents": 1,
                "longest_incident": 120,
                "average_incident": 120,
            }
            return {"data": {"id": monitor_id, "type": "monitor_sla", "attributes": attributes}}
        if method == "GET":
            return {"data": {"id": monitor_id, "attributes": self.monitors[monitor_id]}}
        raise AssertionError(f"unexpected call {method} {parts}")

    def _status_pages(
        self, method: str, parts: list[str], params: Any, body: Any
    ) -> dict[str, Any]:
        if len(parts) == 1:
            if method == "GET":
                return self._listing(self.pages, "status_page", params)
            page_id = self._new_id()
            self.pages[page_id] = dict(body)
            self.resources[page_id] = {}
            return {"data": {"id": page_id, "attributes": self.pages[page_id]}}
        page_id = parts[1]
        if len(parts) == 2 and method == "PATCH":
            self.pages[page_id].update(body)
            return {"data": {"id": page_id, "attributes": self.pages[page_id]}}
        if len(parts) == 3 and method == "GET":
            return self._listing(self.resources[page_id], "status_page_resource", params)
        if len(parts) == 3 and method == "POST":
            resource_id = self._new_id()
            self.resources[page_id][resource_id] = {
                **body,
                "resource_id": int(body["resource_id"]),
            }
            return {
                "data": {"id": resource_id, "attributes": self.resources[page_id][resource_id]}
            }
        if len(parts) == 4 and method == "PATCH":
            self.resources[page_id][parts[3]].update(body)
            return {"data": {"id": parts[3], "attributes": self.resources[page_id][parts[3]]}}
        raise AssertionError(f"unexpected call {method} {parts}")

    def writes(self) -> list[tuple[str, str, dict[str, Any] | None]]:
        return [call for call in self.calls if call[0] != "GET"]


def test_the_first_run_creates_the_monitor_the_status_page_and_its_entry() -> None:
    api = FakeApi()

    lines = um.apply(api)

    assert [(method, path) for method, path, _ in api.writes()] == [
        ("POST", "/monitors"),
        ("POST", "/status-pages"),
        ("POST", f"/status-pages/{next(iter(api.pages))}/resources"),
    ]
    assert [line.rsplit(": ", 1)[1] for line in lines] == ["created", "created", "created"]


def test_the_monitor_is_created_as_the_issue_asks() -> None:
    api = FakeApi()

    um.apply(api)

    (monitor,) = api.monitors.values()
    assert monitor["url"] == "http://nattech.fib.upc.edu:40520/health"
    assert monitor["monitor_type"] == "expected_status_code"
    assert monitor["expected_status_codes"] == [200]
    assert monitor["check_frequency"] == 180
    assert monitor["request_timeout"] < monitor["check_frequency"]
    assert monitor["confirmation_period"] >= monitor["check_frequency"]
    assert monitor["email"] is True
    assert not (monitor["sms"] or monitor["call"] or monitor["push"])
    (entry,) = next(iter(api.resources.values())).values()
    assert entry["resource_type"] == "Monitor"
    assert entry["resource_id"] == int(next(iter(api.monitors)))


def test_a_second_run_changes_nothing() -> None:
    api = FakeApi()
    um.apply(api)
    api.calls.clear()

    lines = um.apply(api)

    assert api.writes() == []
    assert [line.rsplit(": ", 1)[1] for line in lines] == ["unchanged", "unchanged", "unchanged"]


def test_a_drifted_monitor_is_patched_with_the_drifted_attributes_only() -> None:
    api = FakeApi()
    um.apply(api)
    monitor_id = next(iter(api.monitors))
    api.monitors[monitor_id].update(check_frequency=600, email=False)
    api.calls.clear()

    lines = um.apply(api)

    assert api.writes() == [
        ("PATCH", f"/monitors/{monitor_id}", {"check_frequency": 180, "email": True})
    ]
    assert lines[0].endswith("updated check_frequency, email")


def test_a_drifted_status_page_and_entry_are_patched() -> None:
    api = FakeApi()
    um.apply(api)
    page_id = next(iter(api.pages))
    entry_id = next(iter(api.resources[page_id]))
    api.pages[page_id]["history"] = 7
    api.resources[page_id][entry_id]["public_name"] = "old"
    api.calls.clear()

    um.apply(api)

    assert api.writes() == [
        ("PATCH", f"/status-pages/{page_id}", {"history": 90}),
        ("PATCH", f"/status-pages/{page_id}/resources/{entry_id}", {"public_name": "Price API"}),
    ]


def test_a_dry_run_reports_the_changes_and_makes_none() -> None:
    api = FakeApi()

    lines = um.apply(api, dry_run=True)

    assert api.writes() == []
    assert all("would create" in line for line in lines)

    um.apply(api)
    api.monitors[next(iter(api.monitors))]["check_frequency"] = 600
    api.calls.clear()
    assert "would update check_frequency" in um.apply(api, dry_run=True)[0]
    assert api.writes() == []


def test_two_monitors_on_the_same_url_are_an_error_not_a_guess() -> None:
    api = FakeApi()
    um.apply(api)
    api.monitors["999"] = dict(next(iter(api.monitors.values())))

    with pytest.raises(um.UptimeError, match="2 monitors check"):
        um.apply(api)


def test_the_monitor_is_found_on_a_later_page() -> None:
    api = FakeApi(per_page=1)
    api.monitors["1"] = {"url": "http://other.example/health", "pronounceable_name": "other"}
    api.monitors["2"] = {"url": "http://other.example/health2", "pronounceable_name": "other 2"}
    um.apply(api)
    api.calls.clear()

    um.apply(api)

    assert api.writes() == []


def test_a_monitor_on_another_url_with_the_same_settings_for_the_rehearsal() -> None:
    api = FakeApi()
    um.apply(api)

    assert (
        um.main(["create-monitor", "--name", "r", "--url", "http://x.example/403"], api=api) == 0
    )

    assert len(api.monitors) == 2
    rehearsal = next(m for m in api.monitors.values() if m["url"].endswith("/403"))
    assert rehearsal["check_frequency"] == 180
    assert rehearsal["pronounceable_name"] == "r"


def test_export_writes_the_availability_of_the_health_monitor(tmp_path: Path) -> None:
    api = FakeApi()
    um.apply(api)
    out = tmp_path / "out" / "availability.json"

    code = um.main(
        ["export", "--from", "2026-11-05", "--to", "2026-12-15", "--out", str(out)], api=api
    )

    assert code == 0
    figure = json.loads(out.read_text())
    assert figure["availability_percent"] == 99.9876
    assert figure["total_downtime_s"] == 120
    assert (figure["from"], figure["to"]) == ("2026-11-05", "2026-12-15")
    assert figure["url"] == um.HEALTH_URL
    assert ("GET", f"/monitors/{figure['monitor_id']}/sla", None) in api.calls


def test_export_without_a_monitor_says_to_run_apply(capsys: pytest.CaptureFixture[str]) -> None:
    code = um.main(["export", "--from", "2026-11-05", "--to", "2026-11-06"], api=FakeApi())

    assert code == 1
    assert "run `apply` first" in capsys.readouterr().err


def test_incidents_lists_only_those_of_the_monitor(capsys: pytest.CaptureFixture[str]) -> None:
    api = FakeApi()
    api.incidents = [
        {
            "id": "1",
            "attributes": {"started_at": "t0"},
            "relationships": {"monitor": {"data": {"id": "7"}}},
        },
        {
            "id": "2",
            "attributes": {"started_at": "t1"},
            "relationships": {"monitor": {"data": {"id": "8"}}},
        },
        {"id": "3", "attributes": {"started_at": "t2"}, "relationships": {}},
    ]

    assert um.main(["incidents", "--monitor-id", "7"], api=api) == 0

    assert json.loads(capsys.readouterr().out) == [{"id": "1", "started_at": "t0"}]


def test_delete_monitor_deletes_by_id() -> None:
    api = FakeApi()
    um.apply(api)
    monitor_id = next(iter(api.monitors))

    assert um.main(["delete-monitor", monitor_id], api=api) == 0

    assert monitor_id not in api.monitors


class Response(io.BytesIO):
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def test_the_client_sends_the_bearer_token_and_the_json_body() -> None:
    seen = []

    def opener(request: Any, timeout: float) -> Response:
        seen.append((request, timeout))
        return Response(b'{"data": []}')

    client = um.BetterStack(SENTINEL, opener=opener)

    answer = client.request("POST", "/monitors", params={"page": 1}, body={"url": "u"})

    request, timeout = seen[0]
    assert answer == {"data": []}
    assert request.full_url == f"{um.BASE_URL}/monitors?page=1"
    assert request.get_method() == "POST"
    assert request.get_header("Authorization") == f"Bearer {SENTINEL}"
    assert json.loads(request.data) == {"url": "u"}
    assert timeout == um.API_TIMEOUT


def test_an_http_error_is_reported_without_the_token() -> None:
    def opener(request: Any, timeout: float) -> Response:
        raise urllib.error.HTTPError(
            request.full_url,
            403,
            "Forbidden",
            Message(),
            io.BytesIO(f'{{"errors": "bad {SENTINEL}"}}'.encode()),
        )

    client = um.BetterStack(SENTINEL, opener=opener)

    with pytest.raises(um.UptimeError) as error:
        client.request("GET", "/monitors")

    assert "HTTP 403" in str(error.value)
    assert SENTINEL not in str(error.value)


def test_the_token_never_reaches_output_or_logs(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Leaky(FakeApi):
        def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if method == "POST" and path == "/status-pages":
                raise um.UptimeError(f"POST {path} failed with HTTP 500: echoed Bearer {SENTINEL}")
            return super().request(method, path, **kwargs)

    monkeypatch.setenv(um.TOKEN_VARIABLE, SENTINEL)
    caplog.set_level(logging.DEBUG)

    assert um.main(["apply"], api=Leaky()) == 1
    assert um.main(["apply", "--dry-run"], api=FakeApi()) == 0

    captured = capsys.readouterr()
    assert SENTINEL not in captured.out + captured.err + caplog.text
    assert "***" in captured.err


def test_a_missing_token_is_reported_and_nothing_is_called(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(um.TOKEN_VARIABLE, raising=False)

    assert um.main(["apply"]) == 1

    assert um.TOKEN_VARIABLE in capsys.readouterr().err
