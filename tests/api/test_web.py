"""The web page (#269): `/` for a browser, its assets, and what it relies on in the API."""

from html.parser import HTMLParser
import re

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from recommenditos.api.webui import (
    ASSET_CACHE_CONTROL,
    CONTENT_SECURITY_POLICY,
    WEB_DIR,
    WebPage,
    accepts_gzip,
    etag_matches,
    wants_html,
)

#: What Chrome and Firefox send when a person opens the address.
BROWSER_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"


class _Page(HTMLParser):
    """The asset URLs and the form fields of the page."""

    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []
        self.options: list[str] = []
        self.fields: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        for name in ("href", "src"):
            value = attributes.get(name)
            if value and value.startswith("assets/"):
                self.urls.append(value)
        if tag in {"input", "select"} and attributes.get("name"):
            self.fields.append(str(attributes["name"]))
        if tag == "select" and attributes.get("data-options"):
            self.options.append(str(attributes["data-options"]))


@pytest.fixture(scope="module")
def page(client: TestClient) -> _Page:
    response = client.get("/", headers={"Accept": BROWSER_ACCEPT})
    assert response.status_code == 200, response.text
    parsed = _Page()
    parsed.feed(response.text)
    return parsed


@pytest.mark.req("FR-16")
def test_a_browser_gets_the_page_and_every_other_client_the_json_index(client: TestClient):
    html = client.get("/", headers={"Accept": BROWSER_ACCEPT})
    assert html.headers["content-type"] == "text/html; charset=utf-8"
    assert "<form" in html.text
    assert "Accept" in html.headers["vary"]

    for accept in (None, "*/*", "application/json", "application/json, text/html;q=0.5"):
        headers = {"Accept": accept} if accept else {}
        response = client.get("/", headers=headers)
        assert response.headers["content-type"] == "application/json", accept
        assert response.json()["docs"] == "/docs"
        assert response.headers["vary"] == "Accept"


@pytest.mark.parametrize(
    ("accept", "expected"),
    [
        (BROWSER_ACCEPT, True),
        ("text/html", True),
        ("TEXT/HTML; q=0.8, application/json; q=0.8", True),
        ("*/*", False),
        ("", False),
        (None, False),
        ("application/json", False),
        ("text/html;q=0", False),
        ("text/html;q=0.4, application/json", False),
        ("text/html;q=nonsense", False),
    ],
)
def test_wants_html_follows_the_accept_header(accept: str | None, expected: bool):
    assert wants_html(accept) is expected


@pytest.mark.req("NFR-09")
def test_the_page_forbids_inline_code_framing_and_other_origins(client: TestClient):
    response = client.get("/", headers={"Accept": BROWSER_ACCEPT})
    assert response.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert "frame-ancestors 'none'" in CONTENT_SECURITY_POLICY
    assert "'unsafe-inline'" not in CONTENT_SECURITY_POLICY
    assert "'unsafe-eval'" not in CONTENT_SECURITY_POLICY
    assert response.headers["cache-control"] == "no-cache"
    # The CSP refuses inline scripts and styles, so the page must hold none.
    assert not re.search(r"<script(?![^>]*\bsrc=)", response.text)
    assert "style=" not in response.text
    assert "<style" not in response.text


def test_every_asset_the_page_names_is_served_and_cached_for_a_year(
    client: TestClient, page: _Page, app: FastAPI
):
    version = app.state.web.version
    css = client.get(f"/assets/{version}/app.css").text
    font_urls = [f"assets/{version}/{url}" for url in re.findall(r'url\("([^"]+)"\)', css)]
    assert font_urls, "the stylesheet names no font"
    for url in [*page.urls, *font_urls]:
        assert url.startswith(f"assets/{version}/"), url
        response = client.get("/" + url)
        assert response.status_code == 200, url
        assert response.headers["cache-control"] == ASSET_CACHE_CONTROL, url
        assert response.content, url


def test_an_asset_is_sent_compressed_and_answered_304_when_unchanged(
    client: TestClient, app: FastAPI
):
    url = f"/assets/{app.state.web.version}/app.js"
    plain = client.get(url, headers={"Accept-Encoding": "identity"})
    compressed = client.get(url, headers={"Accept-Encoding": "gzip"})
    assert "content-encoding" not in plain.headers
    assert compressed.headers["content-encoding"] == "gzip"
    assert int(compressed.headers["content-length"]) < len(plain.content)
    assert compressed.content == plain.content  # httpx decodes it

    again = client.get(url, headers={"If-None-Match": plain.headers["etag"]})
    assert again.status_code == 304
    assert again.content == b""


@pytest.mark.parametrize(
    "path",
    [
        "/assets/000000000000/app.js",
        "/assets/{version}/missing.js",
        "/assets/{version}/../app.py",
        "/assets/{version}/%2e%2e/app.py",
        "/assets/{version}/index.html",
    ],
)
def test_anything_but_a_current_asset_is_a_404_problem(
    client: TestClient, app: FastAPI, path: str
):
    response = client.get(path.format(version=app.state.web.version))
    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"


def test_the_version_changes_with_any_file(tmp_path):
    (tmp_path / "index.html").write_text('<link href="{{assets}}/a.css">')
    (tmp_path / "a.css").write_text("a {}")
    before = WebPage.load(tmp_path)
    (tmp_path / "a.css").write_text("a { color: red }")
    after = WebPage.load(tmp_path)
    assert before.version != after.version
    assert after.page.body == f'<link href="assets/{after.version}/a.css">'.encode()


def test_a_file_of_an_unknown_kind_fails_the_load(tmp_path):
    (tmp_path / "index.html").write_text('<link href="{{assets}}/a.css">')
    (tmp_path / "notes.md").write_text("#")
    with pytest.raises(ValueError, match=r"\.md"):
        WebPage.load(tmp_path)


def test_the_font_licences_ship_with_the_fonts():
    fonts = {path.stem for path in (WEB_DIR / "fonts").glob("*.woff2")}
    licences = {path.stem for path in (WEB_DIR / "fonts").glob("OFL-*.txt")}
    assert fonts == {"instrument-sans", "instrument-serif"}
    assert licences == {f"OFL-{font}" for font in fonts}


@pytest.mark.req("FR-03")
def test_every_list_of_the_form_is_an_enum_of_the_served_openapi(client: TestClient, page: _Page):
    """The page fills its selects from /openapi.json; each must find its values there."""
    properties = client.get("/openapi.json").json()["components"]["schemas"]["ValuationRequest"][
        "properties"
    ]
    assert page.options
    for field in page.options:
        branches = properties[field].get("anyOf", [properties[field]])
        assert any(branch.get("enum") for branch in branches), field


def test_the_form_sends_only_fields_of_the_request(client: TestClient, page: _Page):
    properties = client.get("/openapi.json").json()["components"]["schemas"]["ValuationRequest"][
        "properties"
    ]
    # The month and year selects make up `registration_date`.
    sent = {field for field in page.fields if not field.startswith("registration_")}
    assert sent <= set(properties)
    required = set(
        client.get("/openapi.json").json()["components"]["schemas"]["ValuationRequest"]["required"]
    )
    assert required - {"registration_date"} <= sent


def test_the_page_never_calls_the_feedback_endpoint():
    """The page shows estimates and sends no price back (#269, FR-10)."""
    in_a_string = re.compile(r"""["'`][^"'`\n]*feedback""")
    for path in WEB_DIR.rglob("*"):
        if path.suffix in {".html", ".js"}:
            assert not in_a_string.search(path.read_text()), path


@pytest.mark.parametrize(
    ("accept_encoding", "expected"),
    [
        ("gzip, deflate, br", True),
        ("br;q=1.0, gzip;q=0.8", True),
        ("*", True),
        ("gzip;q=0", False),
        ("*;q=0", False),
        ("identity", False),
        (None, False),
    ],
)
def test_accepts_gzip_follows_the_weights(accept_encoding: str | None, expected: bool):
    assert accepts_gzip(accept_encoding) is expected


@pytest.mark.parametrize(
    ("if_none_match", "expected"),
    [
        ('"abc"', True),
        ('W/"abc"', True),
        ('"x", "abc"', True),
        ("*", True),
        ('"x"', False),
        ("", False),
        (None, False),
    ],
)
def test_etag_matches_weak_tags_lists_and_any(if_none_match: str | None, expected: bool):
    assert etag_matches(if_none_match, '"abc"') is expected


def test_gzip_with_weight_zero_gets_the_plain_file(client: TestClient, app: FastAPI):
    url = f"/assets/{app.state.web.version}/app.js"
    response = client.get(url, headers={"Accept-Encoding": "gzip;q=0, identity"})
    assert "content-encoding" not in response.headers
