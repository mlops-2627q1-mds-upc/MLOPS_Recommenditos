"""What a run's provenance tags identify, checked against a pipeline DVC really runs.

The claim NFR-14 rests on is a comparison between two records written by two
different programs at two different times: the tags a stage computes when it
starts, and the `dvc.lock` DVC writes once it has finished. A test that only
checked the tags exist, or that they change when something changes, passed
while that comparison failed for every stage after the first of a `dvc repro`
(issue #79). So the central test here runs `dvc repro` for real, on a pipeline
small enough to take seconds, and compares the two records the way the report's
reader would: the committed lock against what each stage reported.
"""

from collections.abc import Iterator
from dataclasses import dataclass
import json
from pathlib import Path

import pytest
from tests.conftest import build_dvc_pipeline, commit_all, dvc_repro, git_in, isolate_dvc
import yaml

from recommenditos.provenance import (
    DVC_STAGE_ENV_VAR,
    dvc_written_paths,
    git_state,
    resumed_run_tags,
    run_tags,
)

#: The stages of the `dvc_pipeline` fixture, in the order DVC runs them.
STAGES = ("prepare", "train@small", "train@large")


def _lock_tags(root: Path, stage: str) -> dict[str, str]:
    """The input tags a stage's run should carry, read off the lock DVC wrote."""
    prefix = stage.partition("@")[0]
    lock = yaml.safe_load((root / "dvc.lock").read_text(encoding="utf-8"))
    return {f"{prefix}.deps.{dep['path']}": dep["md5"] for dep in lock["stages"][stage]["deps"]}


def _records(directory: Path) -> dict[str, dict]:
    return {
        stage: json.loads((directory / f"{stage}.json").read_text(encoding="utf-8"))
        for stage in STAGES
    }


@dataclass(frozen=True)
class Reproduced:
    """The issue #79 scenario, run once for the tests that read it."""

    root: Path
    #: The commit the second `dvc repro` started from.
    head: str
    #: What each stage reported, by stage, in the first and the second run.
    first: dict[str, dict]
    second: dict[str, dict]


@pytest.fixture(scope="module")
def reproduced(tmp_path_factory) -> Iterator[Reproduced]:
    """A clean commit with an out-of-date lock, reproduced by DVC.

    The second commit changes a listing and nothing else, as any change to the
    data or the code leaves the committed lock out of date. So every stage
    reruns, and DVC rewrites `dvc.lock` and the git-tracked `summary.json`
    between them: every stage after the first starts in a tree git reports as
    changed. Module-scoped, because each `dvc repro` costs seconds and the tests
    only read the outcome.
    """
    base = tmp_path_factory.mktemp("reproduced")
    with pytest.MonkeyPatch.context() as monkeypatch:
        isolate_dvc(monkeypatch, base / "dvc-site-cache")
        root = build_dvc_pipeline(base / "pipeline")
        dvc_repro(root, base / "first")
        commit_all(root, "the pipeline's first run")
        (root / "raw" / "b.csv").write_text("bmw,320d,2019\n", encoding="utf-8")
        head = commit_all(root, "a corrected listing")
        dvc_repro(root, base / "second")
        yield Reproduced(root, head, _records(base / "first"), _records(base / "second"))


@pytest.mark.req("NFR-14")
def test_every_stage_of_a_committed_dvc_repro_names_the_inputs_its_lock_records(
    reproduced: Reproduced,
):
    """The run and the committed lock compare line for line, and the tree is clean.

    Before the fix, every stage after the first was tagged dirty and carried the
    digest of a lock that only existed mid-run, which no commit holds.
    """
    # The scenario has to have happened, or the assertions below prove nothing:
    # DVC rewrote files git tracks while the stages ran.
    changed = git_in(reproduced.root, "status", "--porcelain")
    assert "dvc.lock" in changed
    assert "summary.json" in changed

    for stage, record in reproduced.second.items():
        prefix = stage.partition("@")[0]
        inputs = _lock_tags(reproduced.root, stage)
        assert record["run"] == {
            "git_commit": reproduced.head,
            "git_dirty": "false",
            **inputs,
        }, stage
        assert record["resumed"] == {
            f"{prefix}.git_commit": reproduced.head,
            f"{prefix}.git_dirty": "false",
            **inputs,
        }, stage


@pytest.mark.req("NFR-14")
def test_an_input_hash_moves_with_its_input_and_with_nothing_else(reproduced: Reproduced):
    """A hash that did not move when the data moved would record nothing.

    The other half matters as much: a hash that moved with anything else would
    make two runs on the same input look like runs on different ones.
    """
    before = reproduced.first["prepare"]["run"]
    after = reproduced.second["prepare"]["run"]

    assert before["prepare.deps.raw"] != after["prepare.deps.raw"]
    assert before["prepare.deps.raw"].endswith(".dir"), "a directory is hashed as one"
    assert before["prepare.deps.stage.py"] == after["prepare.deps.stage.py"]


@pytest.mark.req("NFR-14")
def test_only_what_dvc_writes_is_ignored(dvc_pipeline: Path):
    """The lock and the uncached outputs are DVC's; everything else is a change.

    Read off the pipeline rather than listed: `summary.json` is ignored because
    `dvc.yaml` says it is not cached, so an uncached output added later is
    covered the day it is declared.
    """
    root = dvc_pipeline
    assert dvc_written_paths(root) == ("dvc.lock", "summary.json")

    (root / "dvc.lock").write_text("schema: '2.0'\n", encoding="utf-8")
    (root / "summary.json").write_text("{}\n", encoding="utf-8")
    commit_all(root, "a lock and a summary, as a run would leave them")
    (root / "dvc.lock").write_text("schema: '2.0'\nstages: {}\n", encoding="utf-8")
    (root / "summary.json").write_text('{"listings": 2}\n', encoding="utf-8")

    assert git_state(root)["git_dirty"] == "false"


@pytest.mark.req("NFR-14")
@pytest.mark.parametrize(
    "change",
    ["an edited tracked file", "an untracked file", "an edited parameter file"],
)
def test_a_change_dvc_did_not_write_still_makes_the_tree_dirty(dvc_pipeline: Path, change: str):
    """Ignoring DVC's writes must not turn into ignoring the developer's.

    An untracked file counts: an untracked module the code imports makes a run
    as unreproducible from its commit as an edited tracked file does.
    """
    root = dvc_pipeline
    if change == "an edited tracked file":
        (root / "stage.py").write_text("# edited\n", encoding="utf-8")
    elif change == "an untracked file":
        (root / "helpers.py").write_text("SCALE = 2\n", encoding="utf-8")
    else:
        (root / "params.yaml").write_text("seed: 1\n", encoding="utf-8")
        commit_all(root, "a parameter file")
        (root / "params.yaml").write_text("seed: 2\n", encoding="utf-8")

    assert git_state(root)["git_dirty"] == "true"


@pytest.mark.req("NFR-14")
def test_a_run_outside_dvc_records_the_commit_and_no_input_hashes(dvc_pipeline: Path):
    """A test, or a stage called by hand, may read other files than `dvc.yaml` says.

    Every test of this project points the stages at a temporary directory, and no
    lock records what such a run read, so input hashes would describe files the
    run may never have opened.
    """
    head = git_in(dvc_pipeline, "rev-parse", "HEAD")

    assert run_tags(dvc_pipeline) == {"git_commit": head, "git_dirty": "false"}
    assert resumed_run_tags(dvc_pipeline) == {}


@pytest.mark.req("NFR-14")
def test_a_run_without_git_records_that_rather_than_failing(monkeypatch):
    """An image built without the `.git` directory still has to be able to train."""

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("recommenditos.provenance.subprocess.run", no_git)
    monkeypatch.delenv(DVC_STAGE_ENV_VAR, raising=False)

    assert run_tags() == {"git_commit": "unknown", "git_dirty": "false"}


@pytest.mark.req("NFR-14")
def test_a_checkout_without_dvc_counts_every_change(tmp_path: Path):
    """No pipeline to read DVC's writes off, so nothing is excused.

    A tarball or an image may carry `.git` without `.dvc`; that is no reason to
    fail the run, and no reason to call a changed lock file clean either.
    """
    git_in(tmp_path, "init", "--quiet")
    (tmp_path / "dvc.lock").write_text("schema: '2.0'\n", encoding="utf-8")
    commit_all(tmp_path, "a lock file and no DVC repository")
    (tmp_path / "dvc.lock").write_text("schema: '2.0'\nstages: {}\n", encoding="utf-8")

    assert dvc_written_paths(tmp_path) == ()
    assert git_state(tmp_path)["git_dirty"] == "true"
