"""The promotion gate of FR-15: a pull request that moves `dvc.lock` needs a passing verdict.

    python -m tools.promotion_gate --base <git ref>

A model is promoted by merging a pull request that changes `dvc.lock` (EDN-08),
and from M5 on every such merge deploys (NFR-13). The gate of NFR-01 is computed
by the `evaluate` stage into the committed `metrics.json`, and nothing else reads
it: a pull request whose `metrics.json` says `"gate_passed": false` would pass
every other check. This is that check (EDN-12).

It does nothing, and succeeds, when `dvc.lock` is the same as at `--base`. That
is what lets the CI job be a required check on every pull request instead of
only on the ones that promote.

When `dvc.lock` changed, it fails unless all of these hold:

- `metrics.json` is the file the new `dvc.lock` records for the `evaluate` stage
  (same MD5). Otherwise the verdict is not the one of the model being promoted:
  a lock from a new run beside the metrics of an older one, or a hand edit.
- `gate_passed` is `true`.
- `deployable_variant` names the model being promoted.

The model being promoted is `report.candidate_variant` in `params.yaml`: the
variant put forward for release (EDN-62), which is the one the API will serve.
No code names a served variant yet, so this is the one place the project states
it. The gate reads `gate_passed` and `deployable_variant` and no criterion, so a
criterion added to the gate (SC-05) needs no change here.

Standard library plus PyYAML, which is a runtime dependency, and git on the PATH.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

PROJ_ROOT = Path(__file__).resolve().parents[1]
LOCK = "dvc.lock"
METRICS = "metrics.json"
EVALUATE_STAGE = "evaluate"


def lock_changed(base: str, root: Path) -> bool:
    """Whether `dvc.lock` at HEAD differs from the one at `base`.

    `git diff --quiet` exits 1 for a difference and 0 for none; anything else is
    git failing (an unknown ref, a shallow clone without the base), which must
    not read as "no change" and let a promotion through.
    """
    # S603, S607: git from PATH, as every developer tool here, with an argument list
    # this module builds and no shell.
    result = subprocess.run(  # noqa: S603
        ["git", "diff", "--quiet", base, "HEAD", "--", LOCK],  # noqa: S607
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"git could not compare {LOCK} with {base}: {result.stderr.strip()}")
    return result.returncode == 1


def promoted_variant(params: dict) -> str:
    """The variant a promotion puts forward: `report.candidate_variant`."""
    variant = (params.get("report") or {}).get("candidate_variant")
    if not isinstance(variant, str) or not variant:
        raise ValueError("params.yaml has no report.candidate_variant, so no model is named")
    return variant


def recorded_md5(lock: dict) -> str | None:
    """The MD5 `dvc.lock` records for `metrics.json` under the `evaluate` stage."""
    stage = (lock.get("stages") or {}).get(EVALUATE_STAGE) or {}
    for out in stage.get("outs") or []:
        if out.get("path") == METRICS:
            return out.get("md5")
    return None


def problems(metrics_bytes: bytes, lock: dict, variant: str) -> list[str]:
    """What stops this `metrics.json` from promoting `variant`; empty when nothing does."""
    found = []
    expected = recorded_md5(lock)
    actual = hashlib.md5(metrics_bytes, usedforsecurity=False).hexdigest()
    if expected is None:
        found.append(f"{LOCK} records no {METRICS} under the {EVALUATE_STAGE} stage")
    elif expected != actual:
        found.append(
            f"{METRICS} (md5 {actual}) is not the file {LOCK} records for the "
            f"{EVALUATE_STAGE} stage (md5 {expected}); run `dvc repro` and commit both"
        )
    try:
        metrics = json.loads(metrics_bytes)
    except ValueError as error:
        return [*found, f"{METRICS} is not valid JSON: {error}"]
    if not isinstance(metrics, dict):
        return [*found, f"{METRICS} is not a JSON object"]
    if "gate_passed" not in metrics:
        found.append(f"{METRICS} has no gate_passed field")
    elif metrics["gate_passed"] is not True:
        found.append(
            f"gate_passed is {json.dumps(metrics['gate_passed'])}: the gate of NFR-01 "
            "rejects the model, so it cannot be promoted"
        )
    if "deployable_variant" not in metrics:
        found.append(f"{METRICS} has no deployable_variant field")
    elif metrics["deployable_variant"] != variant:
        found.append(not_the_candidate(metrics, variant))
    return found


def not_the_candidate(metrics: dict, variant: str) -> str:
    """Why `deployable_variant` is not the candidate: it fails, or another passes better."""
    named = json.dumps(metrics["deployable_variant"])
    head = (
        f"deployable_variant is {named}, not {json.dumps(variant)}, "
        "the model report.candidate_variant in params.yaml puts forward"
    )
    record = (metrics.get("variants") or {}).get(variant)
    if isinstance(record, dict) and record.get("gate_passed") is True:
        return (
            f"{head}. The candidate passes the gate, but {named} passes it with the "
            "lower MdAPE: change report.candidate_variant (an EDN decision) "
            "or keep the candidate and say why"
        )
    return f"{head}, and the candidate itself fails the gate"


def check(base: str, root: Path = PROJ_ROOT) -> list[str]:
    """The problems with promoting at HEAD, relative to `base`; empty when it may merge."""
    if not lock_changed(base, root):
        return []
    try:
        variant = promoted_variant(yaml.safe_load((root / "params.yaml").read_text("utf-8")))
        lock = yaml.safe_load((root / LOCK).read_text("utf-8"))
        metrics_bytes = (root / METRICS).read_bytes()
    except (OSError, ValueError, yaml.YAMLError) as error:
        return [str(error)]
    return problems(metrics_bytes, lock, variant)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--base", required=True, help="git ref of the branch the change merges into"
    )
    args = parser.parse_args(argv)
    try:
        found = check(args.base)
    except RuntimeError as error:
        print(f"::error::{error}")
        return 2
    if found:
        print(f"{LOCK} changed, and the promotion gate does not pass:")
        for each in found:
            print(f"::error::{each}")
        return 1
    print(f"{LOCK} is unchanged, or the committed verdict passes the gate for the promoted model.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
