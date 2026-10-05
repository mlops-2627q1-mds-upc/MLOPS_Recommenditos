"""Do the runs behind the committed models identify the committed state? (EDN-74, issue #79)

The population is the runs the committed pipeline points at: the run id each
`models/<variant>/model.json` records, for the bundles the committed `dvc.lock`
records. For each of them this reads the run back from the tracking server and
compares it with the lock committed at `HEAD`, the way a reader of the report
would:

- the bundle in the workspace is the one the lock records for `train@<variant>`,
  so the run id read from it belongs to the committed state;
- every `train.deps.<path>` tag equals the `md5` the lock records for that path
  under `train@<variant>`, and every `evaluate.deps.<path>` tag the one under
  `evaluate`, with no dependency missing and no tag extra;
- `git_dirty` and `evaluate.git_dirty` are `false`, and the two commits are
  named.

It also says, for a run from before the fix, what that run carried instead:
whether its `dvc_lock_md5` equals the digest of the committed lock.

Needs the tracking credentials in `.env`, `dvc pull` (or the run's own
`dvc repro`) for `models/`, and a clean tree, from the repository root:

    uv run python reports/analysis/run_provenance_check.py

`run_provenance_check_results.txt` holds its output before and after the
pipeline was re-run with the fix (2026-10-05).
"""

import hashlib
import json
import subprocess
import sys

from dvc.repo import Repo
import mlflow
import yaml

from recommenditos.config import MODELS_DIR, PROJ_ROOT


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(PROJ_ROOT), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def lock_inputs(lock: dict, stage: str, prefix: str) -> dict[str, str]:
    return {f"{prefix}.deps.{dep['path']}": dep["md5"] for dep in lock["stages"][stage]["deps"]}


def compare(tags: dict, expected: dict, prefix: str) -> list[str]:
    actual = {name: value for name, value in tags.items() if name.startswith(f"{prefix}.deps.")}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    differing = sorted(name for name in set(actual) & set(expected) if actual[name] != expected[name])
    matched = len(expected) - len(missing) - len(differing)
    print(f"    {prefix}.deps: {matched}/{len(expected)} equal the lock")
    problems = [f"{name} = {actual[name]}, the lock has {expected[name]}" for name in differing]
    if missing:
        problems.append(f"{len(missing)} of {len(expected)} {prefix}.deps tags missing")
    if extra:
        problems.append(f"{prefix}.deps tags for undeclared paths: {', '.join(extra)}")
    return problems


def main() -> int:
    head = git("rev-parse", "HEAD")
    changes = git("status", "--porcelain", "--untracked-files=all")
    lock_bytes = subprocess.run(
        ["git", "-C", str(PROJ_ROOT), "show", "HEAD:dvc.lock"], capture_output=True, check=True
    ).stdout
    lock = yaml.safe_load(lock_bytes)
    lock_md5 = hashlib.md5(lock_bytes).hexdigest()
    print(f"HEAD {head}, tree {'dirty' if changes else 'clean'}, MD5 of its dvc.lock {lock_md5}")
    if changes:
        print(changes)
        return 1

    client = mlflow.MlflowClient()
    problems: dict[str, list[str]] = {}
    with Repo(str(PROJ_ROOT)) as repo:
        for record_path in sorted(MODELS_DIR.glob("*/model.json")):
            variant = record_path.parent.name
            stage = f"train@{variant}"
            found = problems.setdefault(variant, [])
            bundle = repo.stage.get_target(stage).outs[0].get_hash().value
            recorded = lock["stages"][stage]["outs"][0]["md5"]
            if bundle != recorded:
                found.append(f"models/{variant} is {bundle}, the lock records {recorded}")
            run_id = json.loads(record_path.read_text(encoding="utf-8"))["mlflow"]["run_id"]
            run = client.get_run(run_id)
            tags = run.data.tags
            print(f"\n{variant}: run {run_id} ({run.info.run_name}, {run.info.status})")
            print(f"    dvc_stage {tags.get('dvc_stage')}")
            for prefix in ("", "evaluate."):
                commit, dirty = tags.get(f"{prefix}git_commit"), tags.get(f"{prefix}git_dirty")
                print(f"    {prefix}git_commit {commit}, {prefix}git_dirty {dirty}")
                if dirty != "false":
                    found.append(f"{prefix}git_dirty {dirty or 'missing'}")
            if "dvc_lock_md5" in tags:
                same = "equals" if tags["dvc_lock_md5"] == lock_md5 else "differs from"
                print(f"    dvc_lock_md5 {tags['dvc_lock_md5']} {same} the committed lock's")
            found += compare(tags, lock_inputs(lock, stage, "train"), "train")
            found += compare(tags, lock_inputs(lock, "evaluate", "evaluate"), "evaluate")

    print()
    for variant, found in problems.items():
        verdict = "OK" if not found else f"{len(found)} problem(s): " + "; ".join(found)
        print(f"{variant}: {verdict}")
    return 0 if not any(problems.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
