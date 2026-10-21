"""The CI check that a promotion pull request carries a passing gate verdict (FR-15).

The gate computes its verdict into the committed `metrics.json`, and a pull
request that moves `dvc.lock` promotes the model it records (EDN-08). Without
this check, a rejected model could be merged, and from M5 on deployed.

The tests build a throwaway git repository with a `main` commit and a branch,
so what is exercised is the real question the job asks: did `dvc.lock` change
between the base and the head, and does the head's verdict allow it.
"""

import hashlib
import json
from pathlib import Path
import subprocess

import pytest
from tools import promotion_gate
from tools.promotion_gate import check, main, problems, promoted_variant
import yaml

CANDIDATE = "lgbm-basic"

PASSING = {"gate_passed": True, "deployable_variant": CANDIDATE, "n_variants_passing": 2}
OTHER_WINS = {
    "gate_passed": True,
    "deployable_variant": "lgbm-extended",
    "variants": {CANDIDATE: {"gate_passed": True}, "lgbm-extended": {"gate_passed": True}},
}
FAILING = {"gate_passed": False, "deployable_variant": None, "n_variants_passing": 0}


def git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
        + ["-c", "commit.gpgsign=false", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def lock_for(metrics_text: str, *, run: int = 1) -> str:
    """A minimal `dvc.lock` whose `evaluate` stage records `metrics.json` by its MD5."""
    md5 = hashlib.md5(metrics_text.encode(), usedforsecurity=False).hexdigest()
    return (
        "schema: '2.0'\n"
        "stages:\n"
        "  evaluate:\n"
        f"    cmd: run {run}\n"
        "    outs:\n"
        "    - path: metrics.json\n"
        "      hash: md5\n"
        f"      md5: {md5}\n"
    )


def write_promotion(
    root: Path, metrics: dict | str, *, candidate: str = CANDIDATE, run: int = 2
) -> None:
    text = metrics if isinstance(metrics, str) else json.dumps(metrics, indent=2) + "\n"
    (root / "metrics.json").write_text(text, encoding="utf-8", newline="\n")
    (root / "dvc.lock").write_text(lock_for(text, run=run), encoding="utf-8", newline="\n")
    (root / "params.yaml").write_text(
        f"report:\n  candidate_variant: {candidate}\n", encoding="utf-8", newline="\n"
    )


@pytest.fixture
def repo(tmp_path):
    """A repository whose `main` holds the previous, failing run; the base ref is `main`."""
    git(tmp_path, "init", "-q", "-b", "main")
    write_promotion(tmp_path, FAILING, run=1)
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "previous run")
    git(tmp_path, "checkout", "-q", "-b", "change")
    return tmp_path


def commit(root: Path) -> None:
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "change")


@pytest.mark.req("FR-15", "NFR-01")
def test_a_promotion_with_a_passing_gate_for_the_promoted_model_may_merge(repo):
    write_promotion(repo, PASSING)
    commit(repo)
    assert check("main", repo) == []


@pytest.mark.req("FR-15", "NFR-01")
def test_a_promotion_whose_gate_failed_is_blocked(repo):
    write_promotion(repo, FAILING)
    commit(repo)
    found = check("main", repo)
    assert any("gate_passed is false" in each for each in found)
    assert any("deployable_variant is null" in each for each in found)


@pytest.mark.req("FR-15", "NFR-01")
def test_a_passing_gate_for_another_variant_than_the_promoted_one_is_blocked(repo):
    write_promotion(repo, {**PASSING, "deployable_variant": "lgbm-extended"})
    commit(repo)
    found = check("main", repo)
    assert len(found) == 1
    assert found[0].startswith(
        'deployable_variant is "lgbm-extended", not "lgbm-basic", '
        "the model report.candidate_variant in params.yaml puts forward"
    )


@pytest.mark.req("FR-15", "NFR-01")
def test_a_failing_candidate_beside_another_passing_variant_is_said_to_fail(repo):
    other = {**OTHER_WINS, "variants": {CANDIDATE: {"gate_passed": False}, "lgbm-extended": {}}}
    write_promotion(repo, other)
    commit(repo)
    (message,) = check("main", repo)
    assert message.endswith("and the candidate itself fails the gate")


@pytest.mark.req("FR-15", "NFR-01")
def test_a_passing_candidate_beaten_by_another_variant_is_told_what_to_decide(repo):
    write_promotion(repo, OTHER_WINS)
    commit(repo)
    (message,) = check("main", repo)
    assert (
        'The candidate passes the gate, but "lgbm-extended" passes it with the lower MdAPE'
        in message
    )
    assert (
        "change report.candidate_variant (an EDN decision); the gate cannot pass while "
        "another variant is deployable" in message
    )


@pytest.mark.req("FR-15", "NFR-01")
def test_the_promoted_model_is_the_one_params_names_not_a_fixed_one(repo):
    extended = {**PASSING, "deployable_variant": "lgbm-extended"}
    write_promotion(repo, extended, candidate="lgbm-extended")
    commit(repo)
    assert check("main", repo) == []


@pytest.mark.req("FR-15", "NFR-01")
@pytest.mark.parametrize("missing", ["gate_passed", "deployable_variant"])
def test_a_verdict_with_a_missing_field_is_blocked(repo, missing):
    write_promotion(repo, {key: value for key, value in PASSING.items() if key != missing})
    commit(repo)
    assert [each for each in check("main", repo) if missing in each] == [
        f"metrics.json has no {missing} field"
    ]


@pytest.mark.req("FR-15", "NFR-01")
@pytest.mark.parametrize("not_true", ["true", 1, None])
def test_only_a_boolean_true_passes_the_gate(repo, not_true):
    write_promotion(repo, {**PASSING, "gate_passed": not_true})
    commit(repo)
    assert any("gate_passed is" in each for each in check("main", repo))


@pytest.mark.req("FR-15", "NFR-01")
@pytest.mark.parametrize("broken", ["{not json", "[]", ""])
def test_a_metrics_file_that_is_not_a_json_object_is_blocked(repo, broken):
    write_promotion(repo, broken)
    commit(repo)
    assert check("main", repo)


@pytest.mark.req("FR-15", "NFR-01")
def test_metrics_that_are_not_the_ones_the_lock_records_are_blocked(repo):
    # A passing verdict pasted over the failing run, or a lock from a new run
    # committed beside the old metrics: the verdict is not the promoted model's.
    write_promotion(repo, FAILING)
    (repo / "metrics.json").write_text(json.dumps(PASSING) + "\n", encoding="utf-8", newline="\n")
    commit(repo)
    found = check("main", repo)
    assert len(found) == 1
    assert "is not the file dvc.lock records" in found[0]


@pytest.mark.req("FR-15", "NFR-01")
def test_a_lock_without_the_evaluate_metrics_is_blocked(repo):
    write_promotion(repo, PASSING)
    (repo / "dvc.lock").write_text("schema: '2.0'\nstages: {}\n", encoding="utf-8", newline="\n")
    commit(repo)
    assert check("main", repo) == ["dvc.lock records no metrics.json under the evaluate stage"]


@pytest.mark.req("FR-15", "NFR-01")
def test_a_params_file_that_names_no_candidate_blocks_a_promotion(repo):
    write_promotion(repo, PASSING)
    (repo / "params.yaml").write_text("report: {}\n", encoding="utf-8", newline="\n")
    commit(repo)
    assert check("main", repo) == [
        "params.yaml has no report.candidate_variant, so no model is named"
    ]


@pytest.mark.req("FR-15", "NFR-01")
def test_a_change_that_leaves_the_lock_alone_passes_whatever_the_verdict_says(repo):
    # `main` carries a failing verdict, and this change touches neither the lock
    # nor the metrics: the check has to report success, because it is a required
    # check on every pull request.
    (repo / "README.md").write_text("docs only\n", encoding="utf-8")
    commit(repo)
    assert check("main", repo) == []


@pytest.mark.req("FR-15", "NFR-01")
def test_a_change_to_the_metrics_alone_is_not_a_promotion(repo):
    # The promotion is the lock; the gate of a model is read when the lock moves.
    (repo / "metrics.json").write_text(json.dumps(FAILING) + "\n", encoding="utf-8", newline="\n")
    commit(repo)
    assert check("main", repo) == []


def test_an_unknown_base_ref_fails_instead_of_passing(repo):
    # A shallow checkout that lacks the base must not read as "no change".
    with pytest.raises(RuntimeError, match="could not compare dvc.lock"):
        check("no-such-ref", repo)


@pytest.mark.req("FR-15", "NFR-01")
def test_main_exits_with_the_verdict(repo, monkeypatch, capsys):
    monkeypatch.setattr(promotion_gate, "check", lambda base: check(base, repo))
    write_promotion(repo, FAILING)
    commit(repo)
    assert main(["--base", "main"]) == 1
    assert "::error::" in capsys.readouterr().out
    git(repo, "checkout", "-q", "main")
    git(repo, "checkout", "-q", "-b", "docs")
    assert main(["--base", "main"]) == 0
    assert main(["--base", "no-such-ref"]) == 2


def test_problems_lists_every_reason_at_once():
    found = problems(b'{"gate_passed": false}', {"stages": {}}, CANDIDATE)
    assert len(found) == 3


def test_promoted_variant_reads_the_candidate():
    assert promoted_variant({"report": {"candidate_variant": "b1"}}) == "b1"


def test_the_repository_params_name_a_candidate_variant_the_pipeline_trains():
    # The check reads this key of the real params.yaml, so a rename of it must
    # fail here and not silently block every promotion.
    params = yaml.safe_load((promotion_gate.PROJ_ROOT / "params.yaml").read_text("utf-8"))
    assert promoted_variant(params) in params["train"]["variants"]
