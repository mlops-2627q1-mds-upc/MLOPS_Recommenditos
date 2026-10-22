"""What the summary of a retrain on GitHub Actions says, and the link it offers (FR-15, #201).

The workflow runs `tools.retrain_promotion summary` after committing the new outputs
on a branch, so these tests build the same situation in a throwaway repository:
`main` holds the committed run, the branch the retrained one.
"""

import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import parse_qs, urlsplit

import mlflow
import pytest
from tools.retrain_promotion import (
    ENERGY_TAG,
    MACHINE_TAG,
    compare_link,
    main,
    model_version,
    outside_tolerance,
    render,
    tag_runs,
)

CANDIDATE = "lgbm-basic"
VARIANTS = ("b0", CANDIDATE)


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
        + ["-c", "commit.gpgsign=false", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def write_run(root: Path, *, mdape: float, gate_passed: bool, model_md5: str) -> None:
    """The files a `dvc repro` commits, for a run whose candidate has `mdape`."""
    variants = {
        "b0": {"mdape": 0.12, "within_20pct": 0.70, "sc05_coverage": None, "gate_passed": False},
        CANDIDATE: {
            "mdape": mdape,
            "within_20pct": 0.90,
            "sc05_coverage": 0.91,
            "gate_passed": gate_passed,
        },
    }
    metrics = {
        "gate_passed": gate_passed,
        "n_variants": 2,
        "n_variants_passing": int(gate_passed),
        "deployable_variant": CANDIDATE if gate_passed else None,
        "variants": variants,
    }
    text = json.dumps(metrics, indent=2) + "\n"
    (root / "metrics.json").write_text(text, encoding="utf-8", newline="\n")
    metrics_md5 = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
    (root / "dvc.lock").write_text(
        "schema: '2.0'\nstages:\n"
        f"  train@{CANDIDATE}:\n    outs:\n    - path: models/{CANDIDATE}\n"
        f"      md5: {model_md5}.dir\n"
        "  evaluate:\n    outs:\n    - path: metrics.json\n"
        f"      md5: {metrics_md5}\n",
        encoding="utf-8",
        newline="\n",
    )
    (root / "params.yaml").write_text(
        f"train:\n  variants:\n    b0: {{}}\n    {CANDIDATE}: {{}}\n"
        f"report:\n  candidate_variant: {CANDIDATE}\n",
        encoding="utf-8",
        newline="\n",
    )
    per_variant = root / "reports" / "metrics"
    per_variant.mkdir(parents=True, exist_ok=True)
    for name in VARIANTS:
        record = {"variant": name, "mlflow_run_id": f"run-{model_md5}-{name}"} | variants[name]
        (per_variant / f"{name}.json").write_text(
            json.dumps(record) + "\n", encoding="utf-8", newline="\n"
        )


@pytest.fixture
def repo(tmp_path):
    """`main` holds a passing run of the candidate with an MdAPE of 6.70 %."""
    git(tmp_path, "init", "-q", "-b", "main")
    write_run(tmp_path, mdape=0.0670, gate_passed=True, model_md5="a" * 32)
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "committed run")
    git(tmp_path, "checkout", "-q", "-b", "retrain/run-1")
    return tmp_path


def retrain(root: Path, **run) -> None:
    write_run(root, **run)
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "retrain")


@pytest.mark.req("FR-15")
def test_the_model_version_is_the_hash_dvc_records_for_the_bundle():
    lock = {"stages": {"train@x": {"outs": [{"path": "models/x", "md5": "0123abcd.dir"}]}}}

    assert model_version(lock, "x") == "0123abcd"
    assert model_version(lock, "y") is None


@pytest.mark.req("NFR-06")
def test_only_a_share_that_moves_by_more_than_a_tenth_of_a_point_is_outside_the_tolerance():
    before = {"mdape": 0.0950, "within_10pct": 0.50, "mae_eur": 7499.0, "mlflow_run_id": "a"}
    after = {"mdape": 0.0955, "within_10pct": 0.51, "mae_eur": 7400.0, "mlflow_run_id": "b"}

    differing, over = outside_tolerance(before, after)

    # The run id records the run, the euro amount is no share, and the MdAPE moved
    # by 0.05 points: only within_10pct, 1 point, is outside.
    assert differing == 3
    assert over == [("within_10pct", 0.50, 0.51)]


@pytest.mark.req("NFR-06")
def test_a_segment_is_compared_with_the_same_segment_and_named_by_it():
    before = {"segments": [{"segment": "make", "level": "BMW", "within_10pct": 0.50}]}
    after = {
        "segments": [
            {"segment": "make", "level": "Audi", "within_10pct": 0.40},
            {"segment": "make", "level": "BMW", "within_10pct": 0.52},
        ]
    }

    _, over = outside_tolerance(before, after)

    assert ("segments[make=BMW].within_10pct", 0.50, 0.52) in over


@pytest.mark.req("FR-15", "NFR-01")
def test_the_pull_request_states_the_version_the_gate_and_the_metrics(repo):
    retrain(repo, mdape=0.0690, gate_passed=True, model_md5="b" * 32)
    runs = {CANDIDATE: {"run_id": "abcdef0123", "url": "https://mlflow.example/#/runs/abcdef0123"}}

    title, body = render(
        base="main", reason="Drift in country_code.", run_url="https://run", runs=runs, root=repo
    )

    assert title == f"chore: promote {CANDIDATE} model bbbbbbbb from a retrain on GitHub Actions"
    assert f"`{'b' * 32}`, replacing `{'a' * 32}`" in body
    assert "The Promotion gate check **passes**" in body
    assert "6.70 % → 6.90 %" in body
    assert "[abcdef01](https://mlflow.example/#/runs/abcdef0123)" in body
    assert "Drift in country_code." in body
    # 0.2 points on the candidate's MdAPE is outside NFR-06's tolerance.
    assert f"`reports/metrics/{CANDIDATE}.json` `mdape`: 0.0670 → 0.0690" in body


@pytest.mark.req("FR-15", "NFR-01")
def test_a_rejected_model_is_announced_as_failing_the_gate(repo):
    retrain(repo, mdape=0.1500, gate_passed=False, model_md5="c" * 32)

    _, body = render(base="main", reason="", run_url="https://run", runs={}, root=repo)

    assert "The Promotion gate check **fails**" in body
    assert "gate_passed is false" in body


@pytest.mark.req("FR-15")
def test_the_summary_links_to_a_prefilled_pull_request_and_holds_the_evidence(
    repo, monkeypatch, tmp_path_factory
):
    retrain(repo, mdape=0.0670, gate_passed=True, model_md5="d" * 32)
    out = tmp_path_factory.mktemp("promotion")
    monkeypatch.setattr("tools.retrain_promotion.PROJ_ROOT", repo)
    monkeypatch.chdir(repo)
    arguments = ["summary", "--base", "main", "--out-dir", str(out), "--reason", "test"]
    arguments += ["--repo-url", "https://github.com/o/r", "--branch", "retrain/run-7"]
    arguments += ["--run-url", "https://github.com/o/r/actions/runs/1"]

    assert main(arguments) == 0

    summary = (out / "summary.md").read_text("utf-8")
    link = summary.split("](", 1)[1].split(")", 1)[0]
    query = parse_qs(urlsplit(link).query)
    assert link.startswith("https://github.com/o/r/compare/main...retrain/run-7?")
    assert query["expand"] == ["1"]
    assert query["title"] == [
        f"chore: promote {CANDIDATE} model dddddddd from a retrain on GitHub Actions"
    ]
    assert "https://github.com/o/r/actions/runs/1" in query["body"][0]
    assert "## Reproduction (NFR-06)" in summary


@pytest.mark.req("FR-15")
def test_the_link_stays_short_whatever_the_reason():
    link = compare_link(
        repo_url="https://github.com/o/r",
        branch="retrain/run-7",
        title="chore: promote x",
        reason="a monitoring finding " * 200,
        run_url="https://github.com/o/r/actions/runs/1",
    )

    assert len(link) <= 2000
    assert parse_qs(urlsplit(link).query)["body"][0].count("...") == 1


@pytest.mark.req("FR-15")
def test_the_link_is_capped_on_its_encoded_length():
    # One emoji is twelve characters once encoded, so 300 of them alone pass 2,000.
    link = compare_link(
        repo_url="https://github.com/o/r",
        branch="retrain/run-7-1",
        title="chore: promote x",
        reason="\N{ROBOT FACE}" * 300,
        run_url="https://github.com/o/r/actions/runs/1",
    )

    assert len(link) <= 2000


@pytest.mark.req("FR-15", "NFR-10")
def test_the_summary_says_a_merge_replaces_the_laptop_energy_figures(repo):
    emissions = repo / "reports" / "emissions"
    emissions.mkdir(parents=True)
    for name in VARIANTS:
        (emissions / f"{name}.csv").write_text(f"{name},laptop\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "laptop energy")
    unchanged = render(base="retrain/run-1", reason="", run_url="u", runs={}, root=repo)[1]
    git(repo, "checkout", "-q", "-b", "retrain/run-2")
    (emissions / f"{CANDIDATE}.csv").write_text(f"{CANDIDATE},runner\n", encoding="utf-8")
    retrain(repo, mdape=0.0670, gate_passed=True, model_md5="e" * 32)

    _, body = render(base="retrain/run-1", reason="", run_url="u", runs={}, root=repo)

    assert "the committed energy figures stay as they are" in unchanged
    assert "replaces the committed energy table" in body
    assert f"runner estimates for `{CANDIDATE}`, mixed with the laptop rows" in body
    assert "NFR-10 caption must be updated" in body


def write_bundle(root: Path, variant: str, run_id: str) -> None:
    bundle = root / "models" / variant
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "model.json").write_text(json.dumps({"mlflow": {"run_id": run_id}}), "utf-8")


@pytest.mark.req("FR-15", "NFR-10")
def test_only_the_runs_retrained_on_the_runner_are_labelled_as_such(repo, monkeypatch):
    # A local store, run from the temporary repository so its artifacts land there.
    uri = f"sqlite:///{(repo / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(repo)
    client = mlflow.MlflowClient(tracking_uri=uri)
    experiment = client.create_experiment("retrain-test")
    laptop_run = client.create_run(experiment).info.run_id
    record = repo / "reports" / "metrics" / "b0.json"
    record.write_text(json.dumps({"mlflow_run_id": laptop_run}), encoding="utf-8")
    write_bundle(repo, "b0", laptop_run)
    git(repo, "commit", "-qam", "b0 trained on a laptop")
    git(repo, "checkout", "-q", "-b", "retrain/run-2")
    # The retrain fits the candidate only, so b0's record still names the laptop's run.
    # The bundle names the run, so labelling works before `evaluate` has run at all.
    runner_run = client.create_run(experiment).info.run_id
    write_bundle(repo, CANDIDATE, runner_run)

    runs = tag_runs(repo / "runs.json", "retrain/run-1", root=repo)

    tags = client.get_run(runner_run).data.tags
    assert tags[MACHINE_TAG].startswith("GitHub-hosted runner")
    assert "not NFR-10's laptop reference" in tags[ENERGY_TAG]
    assert MACHINE_TAG not in client.get_run(laptop_run).data.tags
    for name in VARIANTS:
        assert runs[name]["url"].endswith(
            f"/#/experiments/{experiment}/runs/{runs[name]['run_id']}"
        )
    assert json.loads((repo / "runs.json").read_text("utf-8")) == runs
