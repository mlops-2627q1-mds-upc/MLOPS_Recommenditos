"""NFR-09: CD keys its cached apt layers on the state of Debian's archive (#314).

CI builds without a cache, so it cannot show that CD's cache is ever missed;
these tests check the two halves that make it so: the key, and that the key
reaches a build argument above the apt line of each image CD builds.
"""

from datetime import UTC, datetime
from pathlib import Path
import posixpath
import re

import pytest
from tools import debian_archive_state
from tools.debian_archive_state import archive_state, release_date
import yaml

PROJ_ROOT = Path(__file__).resolve().parents[1]
ARG = "DEBIAN_ARCHIVE_STATE"
# The head of trixie-security's InRelease as Debian served it on 2026-11-08.
IN_RELEASE = """-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA256

Origin: Debian
Label: Debian-Security
Suite: stable-security
Version: 13
Codename: trixie-security
Date: Sun, 08 Nov 2026 00:34:44 UTC
Valid-Until: Sun, 15 Nov 2026 00:34:44 UTC
"""
# The Dockerfiles with an apt line.
APT_DOCKERFILES = ["Dockerfile", "monitoring/Dockerfile"]


@pytest.mark.req("NFR-09")
def test_the_key_names_the_date_of_each_release_file():
    pages = {
        url: IN_RELEASE.replace("00:34:44", f"0{i}:00:00")
        for i, url in enumerate(debian_archive_state.RELEASE_FILES.values())
    }

    key, failure = archive_state(read=pages.__getitem__)

    assert failure is None
    assert key == "trixie@2026-11-08T00:00:00Z trixie-security@2026-11-08T01:00:00Z"


@pytest.mark.req("NFR-09")
def test_the_date_is_read_in_utc():
    assert release_date(IN_RELEASE.replace("UTC", "+0100")) == datetime(
        2026, 11, 7, 23, 34, 44, tzinfo=UTC
    )


@pytest.mark.req("NFR-09")
@pytest.mark.parametrize(
    "read",
    [
        pytest.param(lambda url: (_ for _ in ()).throw(TimeoutError("timed out")), id="timeout"),
        pytest.param(lambda url: "Origin: Debian\n", id="no-date-field"),
    ],
)
def test_an_unreadable_archive_falls_back_to_todays_date(read):
    key, failure = archive_state(
        read=read, today=lambda: datetime(2026, 11, 8, 23, 59, tzinfo=UTC)
    )

    assert key == "today@2026-11-08"
    assert failure


def final_stage(dockerfile: str) -> list[str]:
    """The instructions of the last stage, continuation lines joined."""
    text = re.sub(r"\\\n", " ", (PROJ_ROOT / dockerfile).read_text(encoding="utf-8"))
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    last_from = max(i for i, line in enumerate(lines) if line.startswith("FROM "))
    return lines[last_from + 1 :]


@pytest.mark.req("NFR-09")
@pytest.mark.parametrize("dockerfile", APT_DOCKERFILES)
def test_the_runtime_stage_declares_the_key_above_its_apt_line(dockerfile):
    # A build argument changes the cache key of the RUN instructions below its
    # declaration only, so above the apt line and not in an earlier stage,
    # where it would rebuild the dependencies too.
    stage = final_stage(dockerfile)
    apt = next(i for i, line in enumerate(stage) if line.startswith("RUN ") and "apt-get" in line)

    assert f"ARG {ARG}" in stage[:apt]


def dockerfile_of(build: dict[str, str]) -> str:
    """The Dockerfile of a build-push-action step, relative to the repository."""
    # posixpath, not os.path: the workflow's paths use "/" on every runner.
    return posixpath.normpath(build.get("file", posixpath.join(build["context"], "Dockerfile")))


@pytest.mark.req("NFR-09")
def test_cd_passes_the_key_to_every_image_with_an_apt_line():
    workflow = yaml.safe_load((PROJ_ROOT / ".github/workflows/cd.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["publish"]["steps"]
    sets_key = next(
        i for i, step in enumerate(steps) if "debian_archive_state.py" in step.get("run", "")
    )
    builds = {
        i: step["with"]
        for i, step in enumerate(steps)
        if step.get("uses", "").startswith("docker/build-push-action@")
    }
    with_apt = {
        i: build
        for i, build in builds.items()
        if any("apt-get" in line for line in final_stage(dockerfile_of(build)))
    }

    assert sorted(dockerfile_of(build) for build in with_apt.values()) == sorted(APT_DOCKERFILES)
    for i, build in with_apt.items():
        assert i > sets_key
        assert f"{ARG}=${{{{ env.{ARG} }}}}" in build.get("build-args", "").splitlines()
