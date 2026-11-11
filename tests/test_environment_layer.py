"""The VM pulls the images' environment again only when the runtime changes (#346, EDN-106).

Each image installs its environment from a runtime-only export of its lock, so a
lock change that touches only dependency groups the image does not install
leaves the install step cached and the environment layer's digest unchanged.
CI builds without a cache, so it cannot show the cache hit itself; these tests
check the layout that makes it so: the install step reads nothing but the
export, the export leaves out what the image does not install, and the
environment is created as `uv sync` would create it from pyproject.toml.
docs/docs/deployment.md#continuous-delivery has the measurement.
"""

from pathlib import Path
import re
import shlex
import tomllib

import pytest

PROJ_ROOT = Path(__file__).resolve().parents[1]
# Each Dockerfile, its project, and the flag that leaves out the groups its
# image does not install, the same as the `uv sync` it replaced.
DOCKERFILES = {
    "Dockerfile": ("pyproject.toml", "--no-default-groups"),
    "monitoring/Dockerfile": ("monitoring/pyproject.toml", "--no-dev"),
}
# Flags that would bring a dependency group into the export.
GROUP_FLAGS = (
    "--group",
    "--all-groups",
    "--only-group",
    "--dev",
    "--only-dev",
    "--extra",
    "--all-extras",
)


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


def only(lines: list[str], text: str) -> list[str]:
    """The words of the one instruction that contains `text`."""
    matches = [line for line in lines if text in line]
    assert len(matches) == 1, f"one instruction with {text!r}, found {len(matches)}"
    return shlex.split(matches[0])


@pytest.fixture(params=sorted(DOCKERFILES))
def dockerfile(request) -> tuple[dict[str, list[str]], dict, str]:
    project, groups = DOCKERFILES[request.param]
    parsed = stages((PROJ_ROOT / request.param).read_text(encoding="utf-8"))
    pyproject = tomllib.loads((PROJ_ROOT / project).read_text(encoding="utf-8"))
    return parsed, pyproject, groups


def test_the_install_reads_only_the_runtime_export(dockerfile):
    parsed, _, _ = dockerfile
    install = only(parsed["dependencies"], "pip sync")

    # Any file from the build context, uv.lock above all, would key the step on
    # the whole lock again.
    mounts = [word for word in install if word.startswith("--mount=type=bind")]
    assert len(mounts) == 1, mounts
    options = dict(
        option.split("=", 1) for option in mounts[0].removeprefix("--mount=").split(",")
    )
    assert options["from"] == "runtime-lock"
    assert options["source"] == "/requirements.txt"
    assert "--require-hashes" in install
    assert install[-1] == options["target"]
    for line in parsed["dependencies"]:
        assert "uv.lock" not in line and "pyproject.toml" not in line, line
        assert not line.upper().startswith(("COPY", "ADD")) or "--from=uv " in line, line


def test_the_export_leaves_out_what_the_image_does_not_install(dockerfile):
    parsed, _, groups = dockerfile
    export = only(parsed["runtime-lock"], "uv export")

    for flag in ("--locked", groups, "--no-emit-project"):
        assert flag in export, flag
    assert export[export.index("--format") + 1] == "requirements.txt"
    assert export[export.index("--output-file") + 1] == "/requirements.txt"
    added = [word for word in export if word.split("=")[0] in GROUP_FLAGS]
    assert added == [], f"the export brings in {added}"


def test_the_environment_is_created_as_uv_sync_would(dockerfile):
    parsed, pyproject, _ = dockerfile
    venv = only(parsed["dependencies"], "uv venv")
    venv = venv[venv.index("venv") :]

    assert venv[venv.index("--python") + 1] == pyproject["project"]["requires-python"]
    assert venv[venv.index("--prompt") + 1] == pyproject["project"]["name"]
