"""The evidence and the pull request link of a retrain on GitHub Actions (FR-15, #201).

    python -m tools.retrain_promotion tag-runs --base <git ref> --out <runs.json>
    python -m tools.retrain_promotion summary --base <git ref> --runs <runs.json> \
        --reason <text> --run-url <url> --repo-url <url> --branch <name> --out-dir <dir>

`.github/workflows/retrain.yml` runs `dvc repro` on a GitHub-hosted runner and
pushes the new `dvc.lock` to a branch. A person opens the promotion pull request
from that branch, which is EDN-12's approval step (EDN-92). This module writes
what the person decides on:

- `tag-runs` marks the MLflow runs the retrain trained as trained on a GitHub
  runner, and their energy figures as an estimate for that machine: CodeCarbon
  falls back to the TDP of the runner's server CPU, so they are not comparable
  with the laptop figures NFR-10 is judged on (EDN-92). It also writes the run
  links for `summary`. It is the one command here that needs the tracking server.
- `summary` writes the job summary: a link that opens the pull request with its
  title and a short body filled in, then the evidence: the model version, the
  gate verdict of NFR-01 as the Promotion gate check will see it, the metrics of
  every variant beside the committed ones, and NFR-06's comparison with them.

The model version is the MD5 `dvc.lock` records for `models/<candidate>`, which
is what `/health` reports (`recommenditos.api.serving.bundle_version`).
"""

import argparse
from collections.abc import Iterator
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from urllib.parse import urlencode

import yaml

from tools import promotion_gate

PROJ_ROOT = Path(__file__).resolve().parents[1]
LOCK = "dvc.lock"
METRICS = "metrics.json"

#: NFR-06: the same metrics within 0.1 percentage points.
TOLERANCE = 0.001

#: Leaves that record the run rather than its result, so they differ on every run.
RUN_RECORD_KEYS = {"mlflow_run_id"}

#: Above this a value is no share (see `outside_tolerance`).
SHARE_MAX = 1.5

#: The tags `tag-runs` sets; the values are filled in from the machine.
MACHINE_TAG = "trained_on"
ENERGY_TAG = "energy_estimate"

#: The longest reason the pull request link carries; the summary has it whole.
MAX_REASON_IN_LINK = 300

#: How many leaves outside the tolerance the summary lists before it only counts them.
MAX_LISTED = 15


def committed(base: str, path: str, root: Path = PROJ_ROOT) -> str | None:
    """The file at `path` as `base` has it; None when it does not exist there."""
    # S603, S607: git from PATH with an argument list this module builds, no shell.
    result = subprocess.run(  # noqa: S603
        ["git", "show", f"{base}:{path}"],  # noqa: S607
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def model_version(lock: dict, variant: str) -> str | None:
    """The MD5 `dvc.lock` records for `models/<variant>`, without DVC's `.dir` suffix."""
    stage = (lock.get("stages") or {}).get(f"train@{variant}") or {}
    for out in stage.get("outs") or []:
        if out.get("path") == f"models/{variant}":
            return str(out.get("md5", "")).removesuffix(".dir") or None
    return None


def flatten(value: object, prefix: str = "") -> Iterator[tuple[str, object]]:
    """Every leaf of a JSON value, keyed by its path (`segments[make=BMW].n`)."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from flatten(item, f"{prefix}[{label(item, index)}]")
    else:
        yield prefix, value


#: The fields that name an entry of a list in the metrics files, most specific first.
IDENTITY_FIELDS = (("segment", "level"), ("scenario", "make"), ("field",), ("make",))


def label(item: object, index: int) -> str:
    """What names a list entry: `make=Hyundai` for a segment, its position otherwise.

    A position is unreadable in a pull request, and it shifts when an entry is
    added, which would pair two different segments in the comparison.
    """
    if isinstance(item, dict):
        for fields in IDENTITY_FIELDS:
            if all(field in item for field in fields):
                return "=".join(str(item[field]) for field in fields)
    return str(index)


def is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def outside_tolerance(old: dict, new: dict) -> tuple[int, list[tuple[str, float, float]]]:
    """How many leaves differ, and the numeric ones that moved by more than NFR-06 allows.

    The tolerance is in percentage points, so it applies to the shares (MdAPE,
    within-20 %, coverage), not to counts or euro amounts; a leaf counts as a share
    when both values lie between 0 and 1.5, which keeps the MdAPE ratios of SC-06
    in and the row counts and errors in euro out.
    """
    old_leaves, new_leaves = dict(flatten(old)), dict(flatten(new))
    differing = 0
    over = []
    for key in sorted(old_leaves.keys() | new_leaves.keys()):
        before, after = old_leaves.get(key), new_leaves.get(key)
        if before == after or key.rsplit(".", 1)[-1] in RUN_RECORD_KEYS:
            continue
        differing += 1
        if (
            is_number(before)
            and is_number(after)
            and abs(before) <= SHARE_MAX  # type: ignore[arg-type]
            and abs(after) <= SHARE_MAX  # type: ignore[arg-type]
            and abs(after - before) > TOLERANCE  # type: ignore[operator]
        ):
            over.append((key, float(before), float(after)))  # type: ignore[arg-type]
    return differing, over


def percent(value: object) -> str:
    return f"{value * 100:.2f} %" if is_number(value) else "-"  # type: ignore[operator]


def verdict(value: object) -> str:
    return {True: "passes", False: "fails"}.get(value, "-")  # type: ignore[call-overload]


def link(runs: dict, variant: str) -> str:
    entry = runs.get(variant) or {}
    if entry.get("url"):
        return f"[{entry['run_id'][:8]}]({entry['url']})"
    return f"`{entry['run_id']}`" if entry.get("run_id") else "-"


def render(
    *,
    base: str,
    reason: str,
    run_url: str,
    runs: dict,
    root: Path | None = None,
) -> tuple[str, str]:
    """The title of the promotion pull request and the evidence for the workspace at `root`."""
    root = root or PROJ_ROOT
    params = yaml.safe_load((root / "params.yaml").read_text("utf-8"))
    candidate = promotion_gate.promoted_variant(params)
    variants = list((params.get("train") or {}).get("variants") or {})
    lock = yaml.safe_load((root / LOCK).read_text("utf-8"))
    old_lock_text = committed(base, LOCK, root)
    old_lock = yaml.safe_load(old_lock_text) if old_lock_text else {}
    version = model_version(lock, candidate) or "unknown"
    old_version = model_version(old_lock, candidate) or "none"
    metrics = json.loads((root / METRICS).read_text("utf-8"))
    old_metrics_text = committed(base, METRICS, root)
    old_metrics = json.loads(old_metrics_text) if old_metrics_text else {}
    problems = promotion_gate.check(base, root)

    gate = "passes" if not problems else "fails"
    title = f"chore: promote {candidate} model {version[:8]} from a retrain on GitHub Actions"
    lines = [
        f"Retrained by [this workflow run]({run_url}) from `{base}`.",
        "",
        f"**Why:** {reason.strip() or '(no reason given)'}",
        "",
        "## Model",
        "",
        f"- Promoted variant (`report.candidate_variant`): `{candidate}`",
        (
            f"- Model version (`dvc.lock` MD5 of `models/{candidate}`, what `/health` "
            f"reports): `{version}`, replacing `{old_version}`"
        ),
        "",
        "## Gate of NFR-01",
        "",
        f"The Promotion gate check **{gate}** for the promotion pull request.",
        (
            f"`gate_passed` is `{json.dumps(metrics.get('gate_passed'))}`, the deployable "
            f"variant is `{metrics.get('deployable_variant')}`, and "
            f"{metrics.get('n_variants_passing', '-')} of {metrics.get('n_variants', '-')} "
            "variants pass."
        ),
    ]
    if metrics.get("waived_criteria"):
        lines.append(f"Waived: {metrics['waived_criteria']}.")
    lines += [f"- {problem}" for problem in problems]
    lines += [
        "",
        "## Metrics",
        "",
        "Committed on the base, then this run.",
        "",
        "| Variant | MdAPE | Within 20 % | SC-05 coverage | Gate | MLflow run |",
        "|---|---|---|---|---|---|",
    ]
    for variant in variants:
        old = (old_metrics.get("variants") or {}).get(variant) or {}
        new = (metrics.get("variants") or {}).get(variant) or {}
        lines.append(
            f"| `{variant}` "
            f"| {percent(old.get('mdape'))} → {percent(new.get('mdape'))} "
            f"| {percent(old.get('within_20pct'))} → {percent(new.get('within_20pct'))} "
            f"| {percent(old.get('sc05_coverage'))} → {percent(new.get('sc05_coverage'))} "
            f"| {verdict(old.get('gate_passed'))} → {verdict(new.get('gate_passed'))} "
            f"| {link(runs, variant)} |"
        )
    lines += reproduction(base, variants, root)
    lines += [
        "",
        "## Energy",
        "",
        (
            "The emissions files and `energy-vs-error` are measured on the runner. CodeCarbon "
            "estimates them from the TDP of the runner's server CPU, so they are not "
            "comparable with the laptop figures NFR-10 is judged on (EDN-92); the MLflow runs "
            f"carry the tags `{MACHINE_TAG}` and `{ENERGY_TAG}`."
        ),
        "",
        "## After opening the pull request",
        "",
        (
            "CI runs on it, the Promotion gate included. Merging it promotes and deploys the "
            "model (EDN-08, EDN-12); nothing merges by itself. To drop the retrain, delete "
            "the branch instead."
        ),
    ]
    return title, "\n".join(lines) + "\n"


def compare_link(*, repo_url: str, branch: str, title: str, reason: str, run_url: str) -> str:
    """The link that opens the pull request from `branch`, its title and body filled in.

    The body is short and points to the run, whose summary holds the evidence: a
    URL much over 2,000 characters is not opened everywhere, and the evidence is
    several times that. Opening the pull request is the person's approval step,
    and a pull request a person opens starts CI, which one opened with the
    workflow's own token would not.
    """
    if len(reason) > MAX_REASON_IN_LINK:
        reason = reason[: MAX_REASON_IN_LINK - 3].rstrip() + "..."
    body = (
        f"Retrained by {run_url}.\n\n"
        f"**Why:** {reason.strip() or '(no reason given)'}\n\n"
        "The model version, the metrics before and after with their MLflow runs, the gate "
        f"verdict of NFR-01 and NFR-06's comparison are in the run's summary: {run_url}\n"
    )
    query = urlencode({"expand": "1", "title": title, "body": body})
    return f"{repo_url.rstrip('/')}/compare/main...{branch}?{query}"


def reproduction(base: str, variants: list[str], root: Path) -> list[str]:
    """NFR-06's comparison of the metrics files with the committed ones."""
    lines = [
        "",
        "## Reproduction (NFR-06)",
        "",
        (
            f"Every leaf of the metrics files against `{base}`; NFR-06 allows 0.1 percentage "
            "points on a share. Without a change to the code, the parameters or the data, "
            "this is the clean-clone drill: a difference is the machine, not the model."
        ),
        "",
        "| File | Leaves that differ | Shares outside 0.1 pp |",
        "|---|---|---|",
    ]
    over_all = []
    for path in [METRICS] + [f"reports/metrics/{variant}.json" for variant in variants]:
        old_text = committed(base, path, root)
        if old_text is None or not (root / path).exists():
            lines.append(f"| `{path}` | not on both sides | - |")
            continue
        differing, over = outside_tolerance(
            json.loads(old_text), json.loads((root / path).read_text("utf-8"))
        )
        lines.append(f"| `{path}` | {differing} | {len(over)} |")
        over_all += [(path, *each) for each in over]
    if over_all:
        lines += ["", "Outside the tolerance:", ""]
        lines += [
            f"- `{path}` `{key}`: {before:.4f} → {after:.4f}"
            for path, key, before, after in over_all[:MAX_LISTED]
        ]
        if len(over_all) > MAX_LISTED:
            lines.append(f"- and {len(over_all) - MAX_LISTED} more")
    return lines


def machine() -> str:
    """The runner, as specific as the job can name it without a secret."""
    cpu = "unknown CPU"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text("utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    # SIM112: `ImageOS` is the spelling the runner sets (`ubuntu24`).
    image = os.environ.get("ImageOS", platform.system())  # noqa: SIM112
    return f"GitHub-hosted runner {image}, {cpu}, {os.cpu_count()} vCPU"


def tag_runs(out: Path, base: str, root: Path | None = None) -> dict:
    """Tag the runs this retrain trained with the machine, and write every run's link to `out`.

    A variant whose run id is the one `base` commits was not retrained here: only a
    later stage reran, and `evaluate` appended to the run a laptop made. That run
    keeps its tags.
    """
    root = root or PROJ_ROOT
    # Imported here, so that `summary` neither loads MLflow nor prints its agent hint
    # into the job log; `recommenditos.tracking` silences the hint the same way.
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow  # noqa: PLC0415

    # From the environment, as the stages read it, and not `mlflow.get_tracking_uri()`,
    # which `recommenditos.tracking` points at a refusing sentinel when it is unset.
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").rstrip("/")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set, so there are no runs to tag")
    params = yaml.safe_load((root / "params.yaml").read_text("utf-8"))
    client = mlflow.MlflowClient(tracking_uri=uri)
    tags = {
        MACHINE_TAG: machine(),
        ENERGY_TAG: "runner: CodeCarbon at the runner CPU's TDP, not NFR-10's laptop reference",
    }
    runs = {}
    for variant in (params.get("train") or {}).get("variants") or {}:
        record = json.loads((root / "reports" / "metrics" / f"{variant}.json").read_text("utf-8"))
        run_id = record.get("mlflow_run_id")
        if not run_id:
            raise RuntimeError(f"{variant} has no MLflow run: was tracking on for the retrain?")
        run = client.get_run(run_id)
        old_text = committed(base, f"reports/metrics/{variant}.json", root)
        if not old_text or json.loads(old_text).get("mlflow_run_id") != run_id:
            for key, value in tags.items():
                client.set_tag(run_id, key, value)
        runs[variant] = {
            "run_id": run_id,
            "url": f"{uri}/#/experiments/{run.info.experiment_id}/runs/{run_id}",
        }
    out.write_text(json.dumps(runs, indent=2) + "\n", "utf-8")
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    tag = commands.add_parser("tag-runs", help="tag the variants' MLflow runs, write their links")
    tag.add_argument("--base", required=True, help="git ref of the committed outputs")
    tag.add_argument("--out", type=Path, required=True)
    report = commands.add_parser("summary", help="write the evidence and the pull request link")
    report.add_argument("--base", required=True, help="git ref of the committed outputs")
    report.add_argument("--runs", type=Path, help="the file `tag-runs` wrote")
    report.add_argument("--reason", default="")
    report.add_argument("--run-url", default="")
    report.add_argument("--repo-url", required=True, help="https://github.com/<owner>/<repo>")
    report.add_argument("--branch", required=True, help="the branch the retrain pushed")
    report.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "tag-runs":
        for variant, entry in tag_runs(args.out, args.base).items():
            print(f"{variant}: {entry['url']}")
        return 0
    runs = json.loads(args.runs.read_text("utf-8")) if args.runs and args.runs.exists() else {}
    title, evidence = render(base=args.base, reason=args.reason, run_url=args.run_url, runs=runs)
    link = compare_link(
        repo_url=args.repo_url,
        branch=args.branch,
        title=title,
        reason=args.reason,
        run_url=args.run_url,
    )
    summary = (
        f"## Promotion ready on `{args.branch}`\n\n"
        f"**[Open the promotion pull request]({link})** - opening it is the approval step "
        "of EDN-12 and starts CI.\n\n"
        f"Title: `{title}`\n\n{evidence}"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.md").write_text(summary, "utf-8")
    print(title)
    print(link)
    return 0


if __name__ == "__main__":
    sys.exit(main())
