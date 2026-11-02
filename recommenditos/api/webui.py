"""The web page for people (#269): the files of `web/`, served from memory.

The page ships in the API image, so it reaches the VM with every deploy and
goes back with every rollback, through the same proxy as the API (EDN-96,
docs/docs/web-ui.md). nginx's `location /` passes every path but `/feedback` and
`/metrics` to the API, so the page needs no change to `deploy/nginx.conf`.

`WebPage.load` reads every file once, at startup, and holds it with its gzip
form, about 135 KB per worker process. The page itself is `index.html`, served
at `/` to a browser (`wants_html`); every other file is an asset under
`/assets/<version>/`, where `<version>` is a hash of all files, so a new page
gets new asset URLs and an asset can be cached for a year. `index.html` names
its assets with the placeholder `{{assets}}`, which `load` replaces with
`assets/<version>`. A path that is not one of the files is a 404: nothing is
looked up on disk per request.
"""

from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path

from starlette.requests import Request
from starlette.responses import Response

#: Where the files of the page are, in the package.
WEB_DIR = Path(__file__).with_name("web")

PAGE_FILE = "index.html"
ASSETS_PLACEHOLDER = "{{assets}}"
ASSETS_PREFIX = "/assets"

#: The media type of each kind of file the page may hold; any other suffix fails
#: `load`, so an unexpected file is noticed before it is served.
MEDIA_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
    ".txt": "text/plain; charset=utf-8",
}

#: Kinds of file worth compressing; woff2 is compressed already.
COMPRESSIBLE = frozenset({".html", ".css", ".js", ".svg", ".txt"})

#: The page loads scripts, styles and fonts from its own origin only, talks to
#: its own origin only, and may not be framed. No inline script or style.
CONTENT_SECURITY_POLICY = (
    "default-src 'none'; script-src 'self'; style-src 'self'; font-src 'self'; "
    "img-src 'self' data:; connect-src 'self'; base-uri 'none'; form-action 'none'; "
    "frame-ancestors 'none'"
)

PAGE_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "Referrer-Policy": "same-origin",
    # Revalidated on every load, so a deploy shows at once; the ETag keeps that
    # to a 304 when nothing changed.
    "Cache-Control": "no-cache",
}

#: An asset URL names its content, so it never changes.
ASSET_CACHE_CONTROL = "public, max-age=31536000, immutable"


@dataclass(frozen=True)
class WebFile:
    """One file of the page, as it is sent."""

    body: bytes
    gzipped: bytes | None
    media_type: str
    etag: str

    @classmethod
    def of(cls, body: bytes, suffix: str) -> "WebFile":
        media_type = MEDIA_TYPES.get(suffix)
        if media_type is None:
            raise ValueError(
                f"the web page holds a {suffix or 'suffix-less'} file, which it does not serve"
            )
        gzipped = None
        if suffix in COMPRESSIBLE:
            # mtime=0, so the same file always compresses to the same bytes.
            compressed = gzip.compress(body, compresslevel=9, mtime=0)
            gzipped = compressed if len(compressed) < len(body) else None
        return cls(body, gzipped, media_type, f'"{hashlib.sha256(body).hexdigest()[:20]}"')

    def response(self, request: Request, headers: dict[str, str]) -> Response:
        """This file for `request`: a 304 when the client holds it, gzip when it accepts that."""
        headers = {**headers, "ETag": self.etag, "Vary": "Accept-Encoding"}
        if request.headers.get("if-none-match") == self.etag:
            return Response(status_code=304, headers=headers)
        body = self.body
        if self.gzipped is not None and "gzip" in request.headers.get("accept-encoding", ""):
            body = self.gzipped
            headers["Content-Encoding"] = "gzip"
        return Response(body, media_type=self.media_type, headers=headers)


@dataclass(frozen=True)
class WebPage:
    """The page and its assets, read once."""

    version: str
    page: WebFile
    assets: dict[str, WebFile]

    @classmethod
    def load(cls, directory: Path = WEB_DIR) -> "WebPage":
        files = {
            path.relative_to(directory).as_posix(): path.read_bytes()
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        }
        if PAGE_FILE not in files:
            raise FileNotFoundError(f"no {PAGE_FILE} in {directory}")
        digest = hashlib.sha256()
        for name, body in files.items():
            digest.update(name.encode() + b"\0" + hashlib.sha256(body).digest())
        version = digest.hexdigest()[:12]
        html = files.pop(PAGE_FILE).decode()
        if ASSETS_PLACEHOLDER not in html:
            raise ValueError(f"{PAGE_FILE} names no asset with {ASSETS_PLACEHOLDER}")
        page = html.replace(ASSETS_PLACEHOLDER, f"{ASSETS_PREFIX.lstrip('/')}/{version}").encode()
        return cls(
            version=version,
            page=WebFile.of(page, ".html"),
            assets={name: WebFile.of(body, Path(name).suffix) for name, body in files.items()},
        )


def wants_html(accept: str | None) -> bool:
    """Whether a request with this `Accept` header asks for a page rather than JSON.

    A browser that opens an address lists `text/html` first; curl, httpx and
    other HTTP clients send `*/*` or nothing, and an API client that names
    `application/json` prefers it. So the page goes to a request that names
    `text/html` with a weight above zero and at least that of `application/json`.
    """
    weights: dict[str, float] = {}
    for item in (accept or "").split(","):
        media, *parameters = (part.strip() for part in item.split(";"))
        weight = 1.0
        for parameter in parameters:
            key, _, value = parameter.partition("=")
            if key.strip() == "q":
                try:
                    weight = float(value)
                except ValueError:
                    weight = 0.0
        weights[media.lower()] = max(weight, weights.get(media.lower(), 0.0))
    html = weights.get("text/html", 0.0)
    return html > 0 and html >= weights.get("application/json", 0.0)
