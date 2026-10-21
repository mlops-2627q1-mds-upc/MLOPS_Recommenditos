"""Reading `uv.lock`: which packages the serving runtime and each dependency group install.

Shared by `tests/test_serving.py`, which holds the serving environment to the
runtime set, and `tests/integration/`, which holds the built image to it. Only
the standard library, because `make test-serving` imports this module from an
environment that has nothing but the runtime and the `test` group.
"""

from collections.abc import Iterable
from pathlib import Path
import re
import tomllib

#: The project's own distribution, which the lock lists as the root.
PROJECT = "recommenditos"

LOCK_FILE = Path(__file__).resolve().parents[1] / "uv.lock"


def read_lock(path: Path = LOCK_FILE) -> dict[str, dict]:
    """Every package of the lock by its normalised name."""
    lock = tomllib.loads(path.read_text(encoding="utf-8"))
    return {normalise(package["name"]): package for package in lock["package"]}


def runtime_dependencies(lock: dict[str, dict]) -> list[dict]:
    """The project's `[project] dependencies`, the serving runtime."""
    return lock[PROJECT].get("dependencies", [])


def group_dependencies(lock: dict[str, dict], name: str) -> list[dict]:
    """The dependencies one dependency group declares."""
    return lock[PROJECT]["dev-dependencies"][name]


def closure(lock: dict[str, dict], roots: Iterable[dict]) -> set[str]:
    """The packages `roots` pull in, with the extras they ask for.

    Markers are not evaluated, so a Windows-only dependency counts as part of
    the closure on Linux too. That can only make a closure larger, which errs
    towards letting a package through rather than failing on one that uv would
    not have installed anyway.
    """
    names: set[str] = set()
    seen_extras: set[tuple[str, str]] = set()
    pending = list(roots)
    while pending:
        dependency = pending.pop()
        name = normalise(dependency["name"])
        package = lock[name]
        if name not in names:
            names.add(name)
            pending.extend(package.get("dependencies", []))
        for extra in dependency.get("extra", []):
            if (name, extra) not in seen_extras:
                seen_extras.add((name, extra))
                pending.extend(package.get("optional-dependencies", {}).get(extra, []))
    return names


def normalise(name: str) -> str:
    """A distribution name as PEP 503 compares it, which is how the lock spells it."""
    return re.sub(r"[-_.]+", "-", name).lower()


def runtime_closure(lock: dict[str, dict]) -> set[str]:
    """Every package the serving runtime installs, which is what the API image holds."""
    return closure(lock, runtime_dependencies(lock))


def group_closure(lock: dict[str, dict], name: str) -> set[str]:
    """Every package one dependency group installs."""
    return closure(lock, group_dependencies(lock, name))
