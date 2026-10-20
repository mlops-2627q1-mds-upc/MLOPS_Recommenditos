"""The deployment smoke check passes a correct stack and fails every broken one.

A deployment job trusts its exit code (NFR-13), so a check that passes a stack
whose proxy lets `/feedback` through, or whose image serves another model than
`dvc.lock` names, would hide exactly the failures it exists for. The stack is
played by a local HTTP server that answers like nginx in front of the API.
"""

from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import pytest
from tools import smoke_check
import yaml

VERSION = "2b53bd3487f97475ebf0e6bd268ba0f0"
PROXY_HEADERS = {"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Server": "nginx"}
REFUSED = (403, {"Content-Type": "application/problem+json"}, {"status": 403})


def healthy_stack() -> dict[tuple[str, str], tuple[int, dict[str, str], dict]]:
    """(method, path) -> (status, headers, JSON body) of a correctly deployed stack."""
    routes = {
        ("GET", "/health"): (
            200,
            PROXY_HEADERS,
            {"status": "ok", "model": {"variant": "lgbm-basic", "version": VERSION}},
        ),
        ("POST", "/v1/predict"): (200, PROXY_HEADERS, {"price_eur": 17890.5}),
    }
    routes.update({route: REFUSED for route in smoke_check.REFUSED})
    return routes


@pytest.fixture
def stack() -> Iterator[tuple[str, dict]]:
    """A server at a free local port and the routes it answers, editable per test."""
    routes = healthy_stack()

    class Handler(BaseHTTPRequestHandler):
        def _answer(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            status, headers, body = routes.get((self.command, self.path), (404, {}, {}))
            payload = json.dumps(body).encode()
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = _answer

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", routes
    server.shutdown()
    server.server_close()


@pytest.fixture
def lock(tmp_path: Path) -> Path:
    path = tmp_path / "dvc.lock"
    outs = [{"path": "models/lgbm-basic", "hash": "md5", "md5": f"{VERSION}.dir"}]
    path.write_text(
        yaml.safe_dump({"schema": "2.0", "stages": {"train@lgbm-basic": {"outs": outs}}}),
        encoding="utf-8",
        newline="\n",
    )
    return path


def run(url: str, lock: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = smoke_check.main([url, "--lock", str(lock)])
    return code, capsys.readouterr().out


def test_a_correct_stack_passes_every_check(stack, lock, capsys):
    url, _ = stack
    code, out = run(url, lock, capsys)
    assert code == 0, out
    assert out.count("PASS") == 3 + len(smoke_check.REFUSED)
    assert "FAIL" not in out


def test_feedback_answered_by_the_api_fails(stack, lock, capsys):
    url, routes = stack
    routes[("POST", "/feedback")] = (401, PROXY_HEADERS, {"status": 401})
    code, out = run(url, lock, capsys)
    assert code == 1
    assert "FAIL  POST /feedback: status 401, expected the proxy's 403" in out


def test_a_model_other_than_dvc_lock_names_fails(stack, lock, capsys):
    url, routes = stack
    routes[("GET", "/health")] = (
        200,
        PROXY_HEADERS,
        {"status": "ok", "model": {"variant": "lgbm-basic", "version": "0" * 32}},
    )
    code, out = run(url, lock, capsys)
    assert code == 1
    assert f"records {VERSION}" in out


def test_a_missing_header_or_a_server_version_fails(stack, lock, capsys):
    url, routes = stack
    headers = {"X-Content-Type-Options": "nosniff", "Server": "nginx/1.30.6"}
    routes[("GET", "/health")] = (200, headers, routes[("GET", "/health")][2])
    code, out = run(url, lock, capsys)
    assert code == 1
    assert "X-Frame-Options: None" in out
    assert "names a version" in out


def test_no_answer_fails(stack, lock, capsys):
    url, _ = stack
    code, out = run(url.rsplit(":", 1)[0] + ":9", lock, capsys)
    assert code == 1
    assert "no answer" in out
