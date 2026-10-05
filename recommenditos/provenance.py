"""Which committed state a run was produced from, as MLflow tags (NFR-14).

A run is evidence only if a reader can go from it back to what produced it, and
only a committed state can be gone back to. So the tags name two things, both of
which a commit and the DVC remote can reproduce: git restores the code, and
`dvc pull` the data.

**The code: `git_commit` and `git_dirty`.** The commit the working tree was at,
and whether the tree had moved on from it. What does not count as moving on is
what DVC itself writes while `dvc repro` runs: each pipeline's lock file, which
DVC rewrites after every stage, and the outputs DVC does not cache, which git
tracks instead (`metrics.json`, `reports/metrics/`, the validation summary).
Counting them made every stage after the first of a clean `dvc repro` report a
dirty tree, for a change that was the run itself: three of the four `train`
runs of one ladder were tagged dirty only because `dvc.lock` had changed under
them (issue #79, EDN-74). Every other change still counts, an untracked file
included, because an untracked module the code imports makes a run as
unreproducible from its commit as an edited tracked file does; the noise that is
not that belongs in `.gitignore`.

Ignoring those files hides nothing a run depends on today: no stage that opens or
appends to a run declares an uncached output as a dependency, so a hand edit of
one cannot change what such a run read. A stage that did declare one would have
it hashed among its inputs below like any other dependency, so the edit would
show there instead of in `git_dirty`.

**The inputs: `<stage>.deps.<path>`.** One tag per dependency `dvc.yaml`
declares for the stage that is running, holding the hash DVC computes for it: an
MD5 for a file, an MD5 with a `.dir` suffix for a directory. It is computed by
DVC's own code, the code that writes `dvc.lock`, when the stage starts; DVC
records the same value under `stages.<stage>.deps` in `dvc.lock` once the stage
has finished, so a run and the committed lock compare line for line. For a data
dependency the hash is also the address of the object in the DVC cache and on
the remote, so `dvc pull` at a commit whose lock records it restores the input
itself; a code file is not in the DVC cache, and git restores it at the commit,
where a clean tree guarantees it has that hash.

The tags are prefixed with the stage's name rather than its full address,
`train` for `train@b0`, because MLflow does not allow an `@` in a tag key; the
`dvc_stage` tag the `train` stage sets carries the full address.

Input hashes are only recorded when DVC runs the stage. DVC sets `DVC_STAGE` in
the environment of every command it runs, so the stage learns which stage it is
from DVC rather than from its own arguments. The stage is looked up by its name
alone, because the rest of the address is relative to wherever `dvc repro` was
started: `../dvc.yaml:train@b0` from a subdirectory. Outside DVC - a test, or a
stage module called by hand - those arguments can point the stage at other files
than the ones `dvc.yaml` declares, as every test of this project does, and no
lock records what it read. Such a run carries the commit and no input hashes, rather
than hashes of files it may not have read.

This module talks to git and to DVC and to nothing else, so the tags can be
computed, and tested, without an MLflow server; `recommenditos/tracking.py` puts
them on the runs.
"""

import os
from pathlib import Path
import subprocess

from dvc.dependency.param import ParamsDependency
from dvc.exceptions import NotDvcRepoError
from dvc.repo import Repo
from dvc.stage import PipelineStage

from recommenditos.config import PROJ_ROOT

#: Set by DVC in the environment of every command it runs, to the address of the
#: stage the command belongs to: `train@b0`, `evaluate` (`dvc/stage/run.py`).
DVC_STAGE_ENV_VAR = "DVC_STAGE"


def running_stage() -> str | None:
    """The address of the DVC stage this process runs as, or `None` outside DVC."""
    return os.environ.get(DVC_STAGE_ENV_VAR) or None


def run_tags(root: Path | None = None) -> dict[str, str]:
    """The provenance of a run this process opens.

    `git_commit` and `git_dirty` for every run, a credential check or a test
    included, and the input hashes of the running stage when DVC runs it.
    """
    root = root or PROJ_ROOT
    tags = git_state(root)
    stage = running_stage()
    if stage is not None:
        tags |= input_tags(stage, root)
    return tags


def resumed_run_tags(root: Path | None = None) -> dict[str, str]:
    """The provenance of what a stage appends to a run another stage opened.

    `evaluate` appends to the run `train` opened (EDN-55), possibly at a later
    commit: a change to `evaluate.py` alone reruns only `evaluate`, against the
    runs of an earlier `train`. The run's unprefixed `git_commit` and `git_dirty`
    belong to the stage that opened it and keep saying where the fit came from,
    so the appending stage records its own state under its own name:
    `evaluate.git_commit`, `evaluate.git_dirty` and `evaluate.deps.<path>`.

    Outside DVC there is no stage to name the tags after and no lock to compare
    them with, so nothing is added.
    """
    root = root or PROJ_ROOT
    stage = running_stage()
    if stage is None:
        return {}
    prefix = tag_prefix(stage)
    return {f"{prefix}.{name}": value for name, value in git_state(root).items()} | input_tags(
        stage, root
    )


def tag_prefix(stage: str) -> str:
    """The prefix of a stage's tags: its name, without the `foreach` key."""
    return stage_name(stage).partition("@")[0]


def stage_name(stage: str) -> str:
    """A stage's name, from its address: `train@b0` from `../dvc.yaml:train@b0`.

    DVC addresses a stage of the top-level `dvc.yaml` by its name alone and any
    other as `<path>:<name>`, where `<path>` is relative to the directory
    `dvc repro` was started in. The stage cannot know that directory, and the
    path is therefore of no use to it.
    """
    return stage.rpartition(":")[2]


def input_tags(stage: str, root: Path | None = None) -> dict[str, str]:
    """`stage_inputs` as tags, `<stage>.deps.<path>`."""
    return {input_tag_name(stage, path): md5 for path, md5 in stage_inputs(stage, root).items()}


def input_tag_name(stage: str, path: str) -> str:
    """The tag that holds the hash of the dependency `path` of `stage`."""
    return f"{tag_prefix(stage)}.deps.{path}"


def declared_inputs(stage: str, root: Path | None = None) -> tuple[str, ...]:
    """The paths of the dependencies `dvc.yaml` declares for `stage`, as `dvc.lock` keys them.

    Nothing is hashed, so this answers on a clone that has not pulled the data.
    """
    with Repo(str(root or PROJ_ROOT)) as repo:
        target = _find_stage(repo, stage)
        return tuple(_lock_path(dep, target) for dep in _path_dependencies(target))


def stage_inputs(stage: str, root: Path | None = None) -> dict[str, str]:
    """The hash of every dependency `dvc.yaml` declares for `stage`, by path.

    Keyed and valued exactly as `dvc.lock` writes the stage's `deps`: the path
    relative to the stage's working directory, and the `md5` field. Computed
    through DVC rather than with `hashlib`, because a directory's hash is DVC's
    own construction - a digest over the sorted listing of its files' hashes,
    after `.dvcignore` - and a reimplementation would be one more thing that can
    disagree with the lock. `tests/test_provenance.py` holds the two equal on a
    pipeline DVC really runs.

    The parameters a stage declares are not among them. They are in
    `params.yaml` at the commit, which a clean tree guarantees is the file the
    stage read, and `train` also logs its own as the run's parameters.
    """
    with Repo(str(root or PROJ_ROOT)) as repo:
        target = _find_stage(repo, stage)
        return {
            _lock_path(dep, target): dep.get_hash().value for dep in _path_dependencies(target)
        }


def _find_stage(repo: Repo, stage: str) -> PipelineStage:
    """The pipeline stage named in the address `stage`, which must be the only one.

    By name, among every stage of the repository's pipelines, because the path
    part of the address is relative to a directory the stage does not know (see
    `stage_name`). Two stages of one name in different `dvc.yaml` files would
    make that ambiguous, so it raises rather than picking one; this repository has
    a single `dvc.yaml`, and the git-ignored worktrees under `.claude/` are not
    part of its index.
    """
    name = stage_name(stage)
    found = [
        candidate
        for candidate in repo.index.stages
        if isinstance(candidate, PipelineStage) and candidate.name == name
    ]
    if len(found) != 1:
        raise LookupError(
            f"DVC is running stage {stage!r}, and {len(found)} pipeline stages of the "
            f"repository at {repo.root_dir} are named {name!r}, where one was expected."
        )
    return found[0]


def _path_dependencies(target: PipelineStage) -> list:
    """The stage's `deps`, without the parameters, which `dvc.lock` records apart."""
    return [dep for dep in target.deps if not isinstance(dep, ParamsDependency)]


def _lock_path(dep, target: PipelineStage) -> str:
    """A dependency's path as `dvc.lock` writes it.

    The expression `Output.dumpd` writes it with, so a path declared as
    `./models` is keyed `models`, as in the lock.
    """
    return dep.fs.as_posix(os.path.relpath(dep.fs_path, target.wdir))


def git_state(root: Path | None = None) -> dict[str, str]:
    """`git_commit`, and `git_dirty` for any change DVC did not write itself.

    A checkout without git - a tarball, an image built without `.git`, a machine
    without git - records `unknown` and `false` rather than failing the run: none
    of those is a reason not to train. A repository whose commit git can name but
    whose status it cannot read - a corrupt index, say - records `git_dirty:
    unknown`, because an empty answer there would read as a clean tree.
    """
    root = root or PROJ_ROOT
    commit = _git(root, "rev-parse", "HEAD")
    if not commit:
        return {"git_commit": "unknown", "git_dirty": "false"}
    # `--untracked-files=all` lists the files of an untracked directory one by
    # one, so a directory holding a DVC output and a stray file is not judged as
    # a whole. `literal`, so a path is a path and never a glob.
    excluded = [f":(exclude,literal){path}" for path in dvc_written_paths(root)]
    changes = _git(root, "status", "--porcelain", "--untracked-files=all", "--", ".", *excluded)
    if changes is None:
        return {"git_commit": commit, "git_dirty": "unknown"}
    return {"git_commit": commit, "git_dirty": "true" if changes else "false"}


def dvc_written_paths(root: Path | None = None) -> tuple[str, ...]:
    """The git-tracked paths `dvc repro` writes, relative to `root`.

    Each pipeline's lock file, and every output a stage declares with
    `cache: false`, which git tracks because DVC does not. Read from the
    pipeline rather than listed here, so an uncached output a later stage
    declares is covered the day it is added.

    Not included: the `.gitignore` DVC appends an entry to when a cached output
    is new. DVC writes it only for an output whose entry is missing, which is on
    the first run of a new stage, while the file it would have to ignore is also
    the hand-written `.gitignore` of the repository root (`gx/` is an output
    there). So the first run of a new stage is reported dirty, until its entry is
    committed, rather than every edit to the root `.gitignore` going unreported.
    """
    root = root or PROJ_ROOT
    try:
        repo = Repo(str(root))
    except NotDvcRepoError:
        return ()
    paths: set[str] = set()
    with repo:
        for stage in repo.index.stages:
            if isinstance(stage, PipelineStage):
                paths.add(os.path.splitext(stage.dvcfile.path)[0] + ".lock")
            paths.update(out.fs_path for out in stage.outs if out.is_in_repo and not out.use_cache)
        return tuple(
            sorted(Path(os.path.relpath(path, repo.root_dir)).as_posix() for path in paths)
        )


def _git(root: Path, *args: str) -> str | None:
    """`git`'s output in `root`, or `None` where git cannot answer.

    `None` rather than an empty string, because an empty output is an answer:
    for `git status` it is the one that says the tree is clean.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()
