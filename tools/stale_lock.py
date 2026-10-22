"""Report the pipeline stages whose `dvc.lock` entry no longer matches the repository.

    python -m tools.stale_lock [--root <dir>]

`dvc.lock` records every dependency of a stage by its hash, so an edit to a file
a stage lists under `deps`, comments included, or to one of its parameters leaves
the stage stale until `dvc repro` re-runs it (docs/docs/pipeline.md, "Working on
a stage"). During M4 several such changes left the lock stale for days and no
check noticed (#183). This is the check CI runs on every pull request and every
push to `main` (#193, EDN-90). It only reports: a re-run takes 15-30 minutes and
logs new MLflow runs, which is too much to require for every edit of stage code.

The comparison is DVC's own: `dvc status --json`, filtered to what a clean
checkout can judge. CI has the git-tracked files but none of the data and models
that `dvc pull` fetches, so DVC reports those as deleted or modified. A
dependency counts only when git tracks it and, for a directory, when no output
`dvc.lock` records inside it is missing from git (`models/` holds the cached
model directories and only a git-tracked `.gitignore`). Every parameter counts,
because `params.yaml` is tracked. A changed command counts too.

It exits 0 whether or not a stage is stale. Under GitHub Actions it writes the
stale stages to the job summary and emits one warning annotation.
"""

import argparse
from collections.abc import Iterable, Mapping
import json
import os
from pathlib import Path, PurePath
import subprocess
import sys
from typing import Any

import yaml

# What `dvc status --json` prints: per stage, a list of strings such as
# "changed command" and of mappings such as {"changed deps": {path: change}}.
Status = Mapping[str, list[Any]]


def _in_git(path: str, tracked: frozenset[str]) -> bool:
    """Whether git tracks the file `path`, or at least one file under the directory `path`."""
    prefix = path.rstrip("/") + "/"
    return path in tracked or any(name.startswith(prefix) for name in tracked)


def _overlaps(a: str, b: str) -> bool:
    return a == b or a.startswith(b.rstrip("/") + "/") or b.startswith(a.rstrip("/") + "/")


def checkable(path: str, tracked: frozenset[str], outputs: Iterable[str]) -> bool:
    """Whether a clean checkout holds `path` as `dvc.lock` hashed it.

    True for a git-tracked file, and for a directory with git-tracked files in
    which no output of the pipeline is left to `dvc pull`.
    """
    if not _in_git(path, tracked):
        return False
    return all(_in_git(out, tracked) for out in outputs if _overlaps(path, out))


def stale_stages(
    status: Status, tracked: frozenset[str], outputs: Iterable[str]
) -> dict[str, list[str]]:
    """The stale stages of a `dvc status --json` result, each with what changed.

    `tracked` holds the git-tracked files and `outputs` every output path
    `dvc.lock` records, all relative to the repository root with `/` separators.
    """
    outputs = list(outputs)
    stale: dict[str, list[str]] = {}
    for stage, entries in status.items():
        changes: list[str] = []
        for entry in entries:
            if entry == "changed command":
                changes.append("command changed")
            if not isinstance(entry, dict):
                continue
            for raw_path, change in entry.get("changed deps", {}).items():
                path = PurePath(raw_path).as_posix()
                if isinstance(change, dict):
                    changes.extend(f"{path}: {key} ({what})" for key, what in change.items())
                elif checkable(path, tracked, outputs):
                    changes.append(f"{path} ({change})")
        if changes:
            stale[stage] = changes
    return stale


def lock_outputs(lock: Mapping[str, Any]) -> list[str]:
    """Every output path a parsed `dvc.lock` records."""
    stages = lock.get("stages") or {}
    return [out["path"] for stage in stages.values() for out in stage.get("outs", [])]


def collect(root: Path) -> dict[str, list[str]]:
    """Run DVC and git in `root` and return its stale stages."""
    result = subprocess.run(
        [sys.executable, "-m", "dvc", "status", "--json"],
        cwd=root,
        # Nothing of a CI run is worth reporting to DVC's usage statistics.
        env={**os.environ, "DVC_NO_ANALYTICS": "1"},
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dvc status failed in {root}:\n{result.stderr}{result.stdout}")
    status = json.loads(result.stdout or "{}")
    # S607: git from PATH, as every developer tool here.
    listing = subprocess.run(
        ["git", "ls-files", "-z"],  # noqa: S607
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tracked = frozenset(name for name in listing.split("\0") if name)
    lock_path = root / "dvc.lock"
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    return stale_stages(status, tracked, lock_outputs(lock or {}))


def summary(stale: Mapping[str, list[str]]) -> str:
    """The job summary in Markdown."""
    if not stale:
        return "## Pipeline lock\n\n`dvc.lock` matches the code and parameters of every stage.\n"
    rows = "\n".join(
        f"| `{stage}` | {'<br>'.join(f'`{change}`' for change in changes)} |"
        for stage, changes in stale.items()
    )
    return (
        "## Pipeline lock\n\n"
        f"{len(stale)} stage(s) differ from what `dvc.lock` records, so `dvc.lock` is stale.\n"
        "Run `dvc repro` and commit the new `dvc.lock` before the next delivery "
        '(docs/docs/pipeline.md, "Working on a stage"). This check does not block the merge.\n\n'
        "| Stage | What changed |\n| --- | --- |\n"
        f"{rows}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    stale = collect(args.root.resolve())
    text = summary(stale)
    print(text)
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as handle:
            handle.write(text)
    if stale and os.environ.get("GITHUB_ACTIONS") == "true":
        print(
            f"::warning file=dvc.lock,title=Stale dvc.lock::{len(stale)} stage(s) differ "
            f"from dvc.lock: {', '.join(stale)}. Re-run them with dvc repro; "
            "the job summary lists what changed."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
