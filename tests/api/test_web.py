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
    font_urls = [
        f"assets/{version}/{url}" for url in re.findall(r'url\("(?!data:)([^"]+)"\)', css)
    ]
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


def _tokens(css: str, selector: str) -> dict[str, str]:
    block = re.search(rf"(?m)^\s*{re.escape(selector)} {{([^}}]*)}}", css)
    assert block, selector
    return dict(re.findall(r"--([a-z0-9-]+): ([^;]+);", block.group(1)))


def _rgb(value: str, ground: tuple[float, ...]) -> tuple[float, ...]:
    """A colour token as RGB, an `rgba()` composited over the colour it lies on."""
    if value.startswith("#"):
        return tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))
    *channels, alpha = (float(part) for part in re.findall(r"[\d.]+", value))
    return tuple(alpha * c + (1 - alpha) * g for c, g in zip(channels, ground, strict=True))


def _contrast(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    def luminance(rgb: tuple[float, ...]) -> float:
        linear = [
            c / 255 / 12.92 if c / 255 <= 0.04045 else ((c / 255 + 0.055) / 1.055) ** 2.4
            for c in rgb
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    high, low = sorted([luminance(a), luminance(b)], reverse=True)
    return (high + 0.05) / (low + 0.05)


#: Every pair the page uses, as (foreground, background, minimum): text 4.5:1, field borders,
#: marks, focus and the busy button's cue 3:1 (WCAG 2.2 AA; a busy button is disabled).
#: "field" is the field's fill over the form cell, where every field sits.
CONTRAST_PAIRS = [
    *[
        (text, ground, 4.5)
        for text in ("ink", "ink-2", "ink-3")
        for ground in ("ground", "form", "field")
    ],
    ("danger", "form", 4.5),
    ("danger", "field", 4.5),
    ("danger", "ground", 4.5),
    ("ink-2", "panel", 4.5),
    ("ink-3", "panel", 4.5),
    ("ink", "chip", 4.5),
    ("on-accent-busy", "accent-busy", 3.0),
    ("ink-2", "danger-tint", 4.5),
    ("danger", "danger-tint", 4.5),
    ("on-accent", "accent", 4.5),
    ("on-accent", "accent-hover", 4.5),
    ("band-text", "band-ink", 4.5),
    ("field-border", "field", 3.0),
    ("field-border", "form", 3.0),
    ("focus", "form", 3.0),
    ("focus", "ground", 3.0),
    ("band", "ground", 3.0),
    ("rule", "ground", 3.0),
    ("ink-2", "ground", 3.0),
]


#: The light tokens, the dark ones that follow the system, and the dark ones the toggle sets.
THEMES = [":root", ':root:not([data-theme="light"])', ':root[data-theme="dark"]']


def test_the_two_dark_blocks_hold_the_same_tokens():
    """The system's dark theme and the toggle's must not drift apart."""
    css = (WEB_DIR / "app.css").read_text(encoding="utf-8")
    assert _tokens(css, THEMES[1]) == _tokens(css, THEMES[2])


@pytest.mark.parametrize("theme", THEMES)
def test_the_colours_reach_aa_contrast_in_both_themes(theme: str):
    css = (WEB_DIR / "app.css").read_text(encoding="utf-8")
    tokens = {**_tokens(css, ":root"), **(_tokens(css, theme) if theme != ":root" else {})}
    ground = _rgb(tokens["ground"], (255, 255, 255))
    form = _rgb(tokens["form"], ground)
    colours = {name: value for name, value in tokens.items() if value.startswith(("#", "rgba"))}
    rgb = {
        name: _rgb(value, form if name == "surface" else ground) for name, value in colours.items()
    }
    rgb["field"] = _rgb(tokens["surface"], form)
    rgb["form"] = form
    rgb["band-text"] = _rgb(tokens["band-text"], _rgb(tokens["band-ink"], ground))
    rgb["on-accent-busy"] = _rgb(tokens["on-accent-busy"], rgb["accent-busy"])
    failing = [
        (fg, bg, round(_contrast(rgb[fg], rgb[bg]), 2))
        for fg, bg, minimum in CONTRAST_PAIRS
        if _contrast(rgb[fg], rgb[bg]) < minimum
    ]
    assert not failing


class _Foot(HTMLParser):
    """The ids inside the form's sticky button row."""

    def __init__(self) -> None:
        super().__init__()
        self.depth = 0
        self.inside: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if self.depth:
            self.depth += tag == "div"
            if attributes.get("id"):
                self.inside.append(str(attributes["id"]))
        elif tag == "div" and "form-foot" in (attributes.get("class") or "").split():
            self.depth = 1

    def handle_endtag(self, tag: str) -> None:
        if self.depth and tag == "div":
            self.depth -= 1


def _px(css: str, selector: str, prop: str) -> list[int]:
    block = re.search(rf"(?m)^{re.escape(selector)} {{([^}}]*)}}", css)
    assert block, selector
    value = re.search(rf"(?m)^\s*{prop}: ([^;]+);", block.group(1))
    assert value, f"{selector} {prop}"
    return [int(number) for number in re.findall(r"(-?\d+)px", value.group(1))]


def test_a_focused_field_scrolls_clear_of_the_sticky_button_row():
    """A field the Tab key reaches must not end up under the row stuck to the window's bottom.

    The row holds the buttons alone. With the error summary inside it, it grew to 137 px
    against 112 px of scroll padding, and after an empty submit at 1,440 x 700 the
    registration selects were half covered.
    """
    foot = _Foot()
    foot.feed((WEB_DIR / "index.html").read_text(encoding="utf-8"))
    assert "form-alert" not in foot.inside
    assert "submit" in foot.inside
    css = (WEB_DIR / "app.css").read_text(encoding="utf-8")
    padding = _px(css, ".form-foot", "padding")
    vertical = padding[0] + (padding[2] if len(padding) > 2 else padding[0])
    border = _px(css, ".form-foot", "border-top")[0]
    button = _px(css, ".button-primary,\n.button-quiet", "height")[0]
    assert _px(css, "html", "scroll-padding-bottom")[0] >= vertical + border + button + 16


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
            assert not in_a_string.search(path.read_text(encoding="utf-8")), path


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
