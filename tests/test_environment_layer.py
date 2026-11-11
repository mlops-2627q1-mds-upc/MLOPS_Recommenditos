"""NFR-13: a release pulls the images' environment again only when the runtime changes (#346).

Each image installs its environment from a runtime-only export of its lock, so a
lock change that touches only dependency groups the image does not install
leaves the install step cached and the environment layer's digest unchanged.
CI builds without a cache, so it cannot show the cache hit itself; these tests
check the layout that makes it so: the install step reads nothing but the
export, and the export leaves out what the image does not install.
docs/docs/deployment.md#continuous-delivery has the measurement.
"""

from pathlib import Path
import re

import pytest

PROJ_ROOT = Path(__file__).resolve().parents[1]
# Each Dockerfile, and the flags that leave out the groups its image does not
# install, the same as the `uv sync` it replaced.
DOCKERFILES = {
    "Dockerfile": "--no-default-groups",
    "monitoring/Dockerfile": "--no-dev",
}


def stages(dockerfile: str) -> dict[str, list[str]]:
    """The instructions of each named stage, continuation lines joined and
    comments left out."""
    joined = re.sub(r"\\\n", " ", dockerfile)
    result: dict[str, list[str]] = {}
    current: list[str] = []
    for raw in joined.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        words = line.split()
        if words[0].upper() == "FROM":
            current = []
            if len(words) >= 4 and words[-2].upper() == "AS":
                result[words[-1].lower()] = current
        current.append(line)
    return result


@pytest.fixture(params=sorted(DOCKERFILES))
def dockerfile(request) -> tuple[dict[str, list[str]], str]:
    path = PROJ_ROOT / request.param
    return stages(path.read_text(encoding="utf-8")), DOCKERFILES[request.param]


@pytest.mark.req("NFR-13")
def test_the_install_reads_only_the_runtime_export(dockerfile):
    parsed, _ = dockerfile
    install = [line for line in parsed["dependencies"] if "pip sync" in line]

    assert len(install) == 1, "the dependencies stage installs with one `uv pip sync`"
    # Any file from the build context, uv.lock above all, would key the step on
    # the whole lock again.
    assert "from=runtime-lock,source=/pylock.toml" in install[0]
    assert "--require-hashes" in install[0]
    for line in parsed["dependencies"]:
        assert "uv.lock" not in line and "pyproject.toml" not in line, line
        assert not line.upper().startswith(("COPY", "ADD")) or "--from=uv " in line, line


@pytest.mark.req("NFR-13")
def test_the_export_leaves_out_what_the_image_does_not_install(dockerfile):
    parsed, groups = dockerfile
    export = [line for line in parsed["runtime-lock"] if "uv export" in line]

    assert len(export) == 1
    for flag in ("--locked", groups, "--no-emit-project", "--format pylock.toml"):
        assert flag in export[0], flag
    assert "--output-file /pylock.toml" in export[0]
