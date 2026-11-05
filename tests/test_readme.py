"""The project tree in README.md names what the repository holds.

The lecturer reads the README when grading the project structure, and the first
report's feedback criticised a README that no longer matched the code
(references/lecturer-feedback.md). The prose cannot be checked, but the tree
can: every top-level directory Git tracks, every package of `recommenditos`,
and the module of every pipeline stage, named with its stage.
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


def tree_lines(name: str) -> list[str]:
    """The lines of the tree whose entry is `name`."""
    return [
        line for line in readme_tree().splitlines() if re.search(rf"── {re.escape(name)}\b", line)
    ]


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
