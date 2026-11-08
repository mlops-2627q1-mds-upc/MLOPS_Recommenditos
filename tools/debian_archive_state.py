"""The state of Debian's archive, the key of CD's cached apt layers (#314).

    python3 tools/debian_archive_state.py

Prints one line, such as
`trixie@2026-09-12T07:55:41Z trixie-security@2026-11-08T00:34:44Z`: the `Date`
of the Release file of each suite the runtime stages install from. Debian
writes a new Release file, with a new date, whenever it publishes into the
suite, about daily for trixie-security and with each point release for trixie.

CD passes the line as the build argument `DEBIAN_ARCHIVE_STATE`, which both
Dockerfiles declare above their apt line: a changed value makes BuildKit miss
the cache from there on, so the apt layer and those after it are rebuilt and
pick up a security fix in the next release, while a day without one keeps the
cache. When a Release file cannot be read, the line is today's date instead
(`today@2026-11-08`), so a Debian outage costs at most one rebuild a day and
never a release; a warning in the job says so. CD's step falls back the same
way when this script fails or runs longer than 60 s.

Standard library only, so CD runs it with the runner's Python.
"""

from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import http.client
import sys
import urllib.request

# The suites of the base image's /etc/apt/sources.list.d/debian.sources that
# carry security fixes, at the hosts it reads them from. trixie-updates, the
# third, is left out: on 2026-11-08 it held only tzdata and
# libdatetime-timezone-perl, neither of which the apt lines install, and its
# Date moves without a package changing, which would rebuild for nothing.
RELEASE_FILES = {
    "trixie": "https://deb.debian.org/debian/dists/trixie/InRelease",
    "trixie-security": "https://deb.debian.org/debian-security/dists/trixie-security/InRelease",
}
# Seconds per request; the files are about 40 to 150 kB and arrive in well under one.
TIMEOUT = 10


def release_date(text: str) -> datetime:
    """The `Date` field of a Release file, in UTC."""
    for line in text.splitlines():
        if line.startswith("Date: "):
            date = parsedate_to_datetime(line.removeprefix("Date: "))
            # "-0000" gives a naive datetime, which astimezone would read as
            # the runner's local time; RFC 5322 means UTC by it.
            return date.replace(tzinfo=UTC) if date.tzinfo is None else date.astimezone(UTC)
    raise ValueError("the Release file has no Date field")


def fetch(url: str) -> str:
    # S310: a fixed https URL of RELEASE_FILES, never one from input.
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:  # noqa: S310
        body: bytes = response.read()
    return body.decode("utf-8", errors="replace")


def archive_state(
    read: Callable[[str], str] = fetch, today: Callable[[], datetime] = lambda: datetime.now(UTC)
) -> tuple[str, str | None]:
    """The key, and the reason it fell back to today's date, if it did."""
    try:
        parts = [
            f"{suite}@{release_date(read(url)):%Y-%m-%dT%H:%M:%SZ}"
            for suite, url in RELEASE_FILES.items()
        ]
    # OSError covers timeouts and HTTP errors; HTTPException a response cut
    # short or malformed (IncompleteRead, BadStatusLine, LineTooLong).
    except (OSError, ValueError, http.client.HTTPException) as error:
        return f"today@{today():%Y-%m-%d}", f"{type(error).__name__}: {error}"
    return " ".join(parts), None


def main() -> int:
    key, failure = archive_state()
    if failure:
        print(
            f"::warning::Debian's Release files could not be read ({failure}); "
            "the apt layers are keyed on today's date instead.",
            file=sys.stderr,
        )
    print(key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
