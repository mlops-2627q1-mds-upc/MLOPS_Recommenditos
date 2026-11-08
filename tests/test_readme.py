"""The project tree in README.md names what the repository holds.

The lecturer reads the README when grading the project structure, and the first
report's feedback criticised a README that no longer matched the code
(references/lecturer-feedback.md). The prose cannot be checked, but the tree
can: every top-level directory Git tracks, every package of `recommenditos`,
and the module of every pipeline stage, named with its stage, and, the other way
round, every entry the tree lists exists.
"""

from pathlib import Path
import re
import subprocess

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def readme_tree() -> str:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Project organization", 1)[1]
    return section.split("```", 2)[1]


def tree_entries() -> list[tuple[Path, str]]:
    """Each entry of the tree as its path below the repository root, with its line.

    A line's depth is the column of its `── ` divided by the four columns a level
    indents, so `recommenditos/data/preprocess.py` is found from the lines above it.
    """
    entries: list[tuple[Path, str]] = []
    parents: list[str] = []
    for line in readme_tree().splitlines():
        match = re.match(r"^((?:│   |    )*)[├└]── (\S+)", line)
        if not match:
            continue
        depth = len(match[1]) // 4
        parents = [*parents[:depth], match[2]]
        entries.append((Path(*parents), line))
    assert entries, "README.md's project tree has no entries"
    return entries


def tree_lines(name: str) -> list[str]:
    """The lines of the tree whose entry is exactly `name`."""
    return [line for path, line in tree_entries() if path.name == name]


# Entries the tree lists although a clean checkout does not hold them: gitignored
# outputs of a DVC stage, which only exist after `dvc repro` or `dvc pull`.
GITIGNORED_DVC_OUTPUTS = {Path("gx")}  # `configure_gx`'s output (EDN-68)


def tracked_top_level_dirs() -> list[str]:
    dirs = subprocess.run(
        ["git", "-C", str(ROOT), "ls-tree", "-d", "--name-only", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    # Without this, a checkout Git cannot read would skip the test instead of failing it.
    assert dirs, "git ls-tree found no directories"
    return [d for d in dirs if not d.startswith(".")]


def stage_modules() -> dict[str, str]:
    """Each stage of dvc.yaml and the `recommenditos` module its command runs."""
    stages = yaml.safe_load((ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    modules = {}
    for name, stage in stages.items():
        command = stage.get("do", stage)["cmd"]
        match = re.search(r"-m (recommenditos\.[\w.]+)", command)
        assert match, f"stage {name} runs no recommenditos module: {command}"
        modules[name] = match[1]
    return modules


@pytest.mark.parametrize("directory", tracked_top_level_dirs())
def test_the_tree_names_every_top_level_directory(directory: str) -> None:
    assert tree_lines(directory), f"README.md's project tree does not name {directory}/"


@pytest.mark.parametrize(
    "package",
    sorted(p.parent.name for p in (ROOT / "recommenditos").glob("*/__init__.py")),
)
def test_the_tree_names_every_package(package: str) -> None:
    assert tree_lines(package), f"README.md's project tree does not name recommenditos/{package}/"


@pytest.mark.parametrize(("stage", "module"), sorted(stage_modules().items()))
def test_the_tree_names_every_stage_module_with_its_stage(stage: str, module: str) -> None:
    filename = module.rsplit(".", 1)[1] + ".py"
    lines = tree_lines(filename)
    assert lines, f"README.md's project tree does not name {filename}, the module of stage {stage}"
    assert any(f"`{stage}`" in line for line in lines), (
        f"README.md's project tree names {filename} but not its stage `{stage}`"
    )


@pytest.mark.parametrize(
    "path", sorted(p for p, _ in tree_entries() if p not in GITIGNORED_DVC_OUTPUTS), ids=str
)
def test_every_entry_of_the_tree_exists(path: Path) -> None:
    assert (ROOT / path).exists(), f"README.md's project tree lists {path}, which does not exist"
