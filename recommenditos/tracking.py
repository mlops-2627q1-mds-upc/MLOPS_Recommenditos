"""MLflow against the shared DagsHub tracking server, in one place.

MLflow already reads `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and
`MLFLOW_TRACKING_PASSWORD` from the environment by itself, so this module is not
here to pass them along. It is here so that a missing one fails immediately,
with a message naming `.env.template`, instead of surfacing as an HTTP 401 from
inside a half-finished training run - the cheapest failure is the one that
happens before the fit.

`tracked_run` is the seam a stage uses: it configures the server, opens the run
and tags it with the git commit and the DVC data version that NFR-06 requires.
Those tags live here rather than in each caller because a caller that has to
remember a second call eventually forgets one, and a run without them cannot be
reproduced from the report.

`python -m recommenditos.tracking` goes through the same seam with a throwaway
experiment, which is how a new contributor proves their credentials work before
touching the pipeline. The `train` stage takes the same route with
`params.yaml`'s `train.mlflow_experiment`, so the setup a newcomer verifies is
the setup the pipeline runs on and there is one place to change if DagsHub's
authentication ever does.

A pipeline stage cannot use `tracked_run` directly, though, because the suite
and a fresh clone have no credentials and must still be able to train.
`optional_run` is the stage's entry point: same run, same tags, but it yields a
`DisabledRun` instead of failing when nothing is configured or the server cannot
be reached. `RECOMMENDITOS_REQUIRE_TRACKING=1` turns that back into a failure,
which is how whoever produces the report's numbers asserts that the runs
actually landed rather than discovering a silent skip afterwards.

`resume_run` is the other half of the grouping: `train` writes its run id into
`models/<variant>/model.json` and `evaluate` appends the test metrics and the
gate verdict to that same run, so the experiment holds exactly one run per
variant carrying its hyperparameters, its artefact, its energy figures and its
metrics. The alternative - one run per DVC stage - is the course demo's wart and
would scatter four variants over eight runs nothing joins.
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

# Set before `import mlflow`, because mlflow reads this one at import time and
# logs an unsolicited pointer to a bundled authoring skill whenever it thinks a
# coding agent is driving the process. On this project that is most runs, and it
# would be the first line a contributor sees in step 6 of the getting-started
# page and in every train-stage log. `setdefault`, so anyone who wants the hint
# can still ask for it. `.env` cannot do this job: isort puts `import mlflow`
# above `import recommenditos.config`, so python-dotenv has not run yet.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

# MLflow retries a failed HTTP request with exponential backoff, and its default
# budget is far larger than a pipeline stage can afford. Measured against a
# closed port: 5 retries (the default) take 246 s to give up, 3 take 13.7 s, 2
# take 4.3 s and 1 takes 0.12 s. A `train` stage that degrades to no tracking
# after 246 s would spend 16 minutes over four variants doing nothing, which is
# NFR-10's entire training budget; 3 retries keeps the ladder under a minute and
# still rides out a connection that is merely flaky. `setdefault`, so a caller
# who wants MLflow's own budget - a large artefact upload over a bad line - can
# ask for it.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "3")

from loguru import logger
import mlflow
import typer

# Importing config for its side effect: it is the single place that calls
# `load_dotenv()`, so the credentials from the gitignored .env are in the
# environment by the time this module reads them. Every DVC stage imports it
# too, which is why the loading is not repeated here.
from recommenditos.config import PROJ_ROOT

app = typer.Typer()

# All three are needed: DagsHub's tracking server rejects anonymous requests, so
# a URI without credentials fails as surely as no URI at all.
REQUIRED_ENV_VARS = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
)

# The experiment the setup check writes to. Deliberately not the pipeline's
# experiment from params.yaml: a credential check is not an experiment result
# and has no business sitting next to the runs the report cites.
SETUP_CHECK_EXPERIMENT = "setup-check"

# `dvc.lock` pins the hash of every artefact the pipeline produced, so one digest
# over it identifies the data a run saw without the run enumerating its inputs.
# That is the "DVC data version" NFR-06 asks for. MD5 because DVC's own hashes
# are MD5, so the two are read side by side.
DVC_LOCK_FILE = PROJ_ROOT / "dvc.lock"

# Where MLflow is pointed when nothing configured it. Since 3.16 an unset
# tracking URI does not mean "tracking off": it resolves to
# `sqlite:///<cwd>/mlflow.db`, and the first logging call creates that file in
# whatever directory the process started in - the repository root, for a
# notebook or a stray script. This scheme has no store behind it, so such a call
# fails instead, and the URI itself is the message.
UNCONFIGURED_TRACKING_URI = "unconfigured://use-recommenditos.tracking.tracked_run"

if not os.environ.get("MLFLOW_TRACKING_URI"):
    # `set_tracking_uri` also exports MLFLOW_TRACKING_URI, so that subprocesses
    # inherit it. Here that is welcome - a stage DVC spawns gets the same
    # refusal - but it means the variable no longer distinguishes "configured"
    # from "guarded", which is why `_is_configured` below knows the sentinel.
    mlflow.set_tracking_uri(UNCONFIGURED_TRACKING_URI)


def configure_tracking(experiment: str) -> str:
    """Point MLflow at the DagsHub tracking server and select `experiment`.

    Returns the tracking URI, so a caller can log where its runs went.
    Raises `RuntimeError` naming the variables that are missing.
    """
    missing = [name for name in REQUIRED_ENV_VARS if not _is_configured(name)]
    if missing:
        raise RuntimeError(
            f"MLflow is not configured: {', '.join(missing)} "
            "not set. Copy .env.template to .env and fill in your DagsHub "
            "credentials (see docs/docs/getting-started.md)."
        )

    uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    logger.info(f"MLflow tracking to {uri}, experiment {experiment!r}.")
    return uri


def _is_configured(name: str) -> bool:
    """Whether `name` carries a value somebody actually chose.

    A variable left blank in `.env` is the likeliest way to get here, and the
    sentinel above is the other: it means "nothing configured this", so treating
    it as a value would make a missing URI undetectable.
    """
    value = os.environ.get(name, "")
    return bool(value) and value != UNCONFIGURED_TRACKING_URI


def run_provenance() -> dict[str, str]:
    """The code and data version of this checkout, as MLflow tags (NFR-06).

    `git_dirty` is separate from `git_commit` on purpose: a commit alone says
    nothing about a working tree that has moved on from it, and a run nobody can
    map back to a state of the code is not evidence.

    It counts untracked files, because an untracked module the code imports makes
    a run as unreproducible from its commit as an edited tracked file does. The
    cost of that reading is that anything left lying in the tree marks every run
    dirty, so whatever is genuinely noise has to be in `.gitignore` - which is
    where the agent tooling's `/.claude/worktrees/` entry comes from.
    """
    tags = {"git_commit": _git("rev-parse", "HEAD") or "unknown", "git_dirty": "false"}
    if _git("status", "--porcelain"):
        tags["git_dirty"] = "true"
    if DVC_LOCK_FILE.exists():
        tags["dvc_lock_md5"] = hashlib.md5(DVC_LOCK_FILE.read_bytes()).hexdigest()
    return tags


@contextmanager
def tracked_run(experiment: str, run_name: str | None = None) -> Iterator[mlflow.ActiveRun]:
    """Open an MLflow run on the shared server, tagged with its provenance.

    The parameters a stage read stay with the stage: only it knows which keys of
    `params.yaml` its `dvc.yaml` entry declares, so it logs those itself. The
    commit and the data version are the same question for every run, which is
    why they are answered here instead.
    """
    configure_tracking(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(run_provenance())
        yield run


class Run:
    """An open MLflow run, as much of it as a pipeline stage needs.

    A handle rather than the module functions, so that every call site in a stage
    is unconditional and the "is tracking on?" branch exists once, in
    `optional_run`. That is what lets issue #38 add `run.log_metrics(...)` for the
    energy figures without touching a conditional.
    """

    #: What `model.json` records about how the run was tracked, so a bundle says
    #: whether its numbers reached the server or only the disk.
    mode = "enabled"

    def __init__(self, run_id: str | None) -> None:
        #: `None` only on a `DisabledRun`, which is what `model.json` records so
        #: that `evaluate` knows there is no run to append to.
        self.run_id = run_id

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict) -> None:
        # A metric a stage could not measure is `None`, and there is no number to
        # record for it. Dropped here rather than at each call site, so that a
        # stage can hand over the record it built without first learning which of
        # its own keys are optional - which is what `evaluate` does with the
        # criteria that are still null.
        mlflow.log_metrics({name: value for name, value in metrics.items() if value is not None})

    def set_tags(self, tags: dict) -> None:
        mlflow.set_tags(tags)

    def log_artifacts(self, directory: Path, artifact_path: str) -> None:
        mlflow.log_artifacts(str(directory), artifact_path=artifact_path)


class DisabledRun(Run):
    """The same handle when there is no server to talk to: every call does nothing.

    A null object rather than `None`, because the alternative is an `if run` in
    front of every log call in two stages, and the one that gets forgotten is the
    one that raises on the machine that *has* credentials.
    """

    mode = "disabled"

    def __init__(self) -> None:
        super().__init__(run_id=None)

    def log_params(self, params: dict) -> None:
        pass

    def log_metrics(self, metrics: dict) -> None:
        pass

    def set_tags(self, tags: dict) -> None:
        pass

    def log_artifacts(self, directory: Path, artifact_path: str) -> None:
        pass


#: Set to `1` to make a tracking failure fail the stage. Off by default because
#: CI and a fresh clone have no credentials and still have to be able to train;
#: on for the runs whose numbers are cited, where a silent skip would be found
#: only once the report was being written.
REQUIRE_TRACKING_ENV_VAR = "RECOMMENDITOS_REQUIRE_TRACKING"


@contextmanager
def optional_run(experiment: str, run_name: str) -> Iterator[Run]:
    """`tracked_run`, degrading to a run handle that logs nothing.

    Three ways tracking can be off, all of them handled the same way: nothing is
    configured, the credentials are refused, or the server cannot be reached.
    Only the first is free to detect; the other two cost the retry budget bounded
    at the top of this module, so a stage pointed at a dead server pauses for
    about fourteen seconds and then trains anyway.

    What is never done is falling back to a local store. Since 3.16 an unset
    tracking URI resolves to `sqlite:///$PWD/mlflow.db`, so "local fallback"
    means a database in whatever directory the process started in, which for a
    DVC stage is the repository root.
    """
    if _require_tracking():
        with tracked_run(experiment, run_name) as run:
            yield Run(run.info.run_id)
        return
    with _degrading(
        lambda: tracked_run(experiment, run_name),
        on_failure=(
            "MLflow tracking is off for this run ({cause}). The model is written normally; set "
            f"{', '.join(REQUIRED_ENV_VARS)} to record it, or {REQUIRE_TRACKING_ENV_VAR}=1 to "
            "make this a failure."
        ),
    ) as run:
        yield run


@contextmanager
def resume_run(run_id: str | None) -> Iterator[Run]:
    """Reopen the run `train` created, so one variant is one run.

    Ends the run it opened, which matters because `evaluate` is one stage over
    four variants in one process: without it, variant two would append to variant
    one's run.
    """
    missing = run_id is None or not _tracking_is_configured()
    if missing and not _require_tracking():
        # Nothing to resume, and no server to resume it on. Reported at info
        # level rather than as a warning: on a clone with no credentials this is
        # the expected state, and a warning per variant would train people to
        # ignore the one that matters.
        logger.info(f"no MLflow run to resume (run_id={run_id!r}); tracking is off.")
        yield DisabledRun()
        return
    if missing:
        raise RuntimeError(
            f"{REQUIRE_TRACKING_ENV_VAR} is set, and there is no run to resume: "
            f"run_id={run_id!r}, tracking configured={_tracking_is_configured()}. Retrain so "
            f"that the model records the run its metrics belong to."
        )
    # The URI explicitly, because importing this module points MLflow at the
    # sentinel when nothing was configured, and that would outlive a `.env`
    # loaded afterwards.
    if _require_tracking():
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        with mlflow.start_run(run_id=run_id):
            yield Run(run_id)
        return

    def reopen() -> AbstractContextManager[mlflow.ActiveRun]:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        return mlflow.start_run(run_id=run_id)

    with _degrading(
        reopen, on_failure=f"could not resume MLflow run {run_id} " + "({cause})."
    ) as run:
        yield run


@contextmanager
def _degrading(
    open_run: Callable[[], AbstractContextManager[mlflow.ActiveRun]], *, on_failure: str
) -> Iterator[Run]:
    """The run `open_run` opens as a handle, or a `DisabledRun` if it cannot open.

    The `yield` below sits outside every `except` in this function, and that is
    the whole point of the function existing. A context manager that catches
    around its own `yield` catches the *caller's* exception too: contextlib
    throws it in at the yield, so the handler reads a failing fit as a failing
    server, logs that tracking is off when it was on, and yields a second
    time - which contextlib reports to the caller as `RuntimeError: generator
    didn't stop after throw()`, leaving the real exception in `__context__`
    where no stage log shows it. Opening the run is the only failure this
    degrades on.

    Ending the run is a third case and neither of the two: a server that dies
    between the last metric and the terminal status must not turn a model that
    was written into a failed stage, so that failure is a warning and the run is
    left unfinished on the server.
    """
    try:
        opened = open_run()
        active = opened.__enter__()
    except Exception as error:  # noqa: BLE001 - any failure to reach the server
        logger.warning(on_failure.format(cause=f"{type(error).__name__}: {error}"))
        yield DisabledRun()
        return
    try:
        yield Run(active.info.run_id)
    except BaseException:
        _end_run(opened, sys.exc_info())
        raise
    _end_run(opened, (None, None, None))


def _end_run(opened: AbstractContextManager[mlflow.ActiveRun], failure: tuple) -> None:
    """Close `opened`, recording `failure` as the run's status where there is one."""
    try:
        opened.__exit__(*failure)
    except Exception as error:  # noqa: BLE001 - see `_degrading`
        logger.warning(
            f"the MLflow run was not closed cleanly ({type(error).__name__}: {error}); it stays "
            f"unfinished on the server. Everything logged before this point landed."
        )


def _tracking_is_configured() -> bool:
    return all(_is_configured(name) for name in REQUIRED_ENV_VARS)


def _require_tracking() -> bool:
    return os.environ.get(REQUIRE_TRACKING_ENV_VAR, "") not in {"", "0"}


def _git(*args: str) -> str:
    """`git` in the project root, or an empty string where git cannot answer.

    A tarball, a container build without the `.git` directory or a machine
    without git installed are all reasons a run may not know its commit. None of
    them is a reason to fail the run, so the tag says `unknown` instead.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(PROJ_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


@app.command()
def main(experiment: str = SETUP_CHECK_EXPERIMENT):
    """Log one trivial run, so a new contributor can see it in the DagsHub UI."""
    try:
        log_setup_check(experiment)
    except RuntimeError as error:
        # Logged rather than raised: typer renders an uncaught exception as a
        # pretty panel with a dozen lines of this module's source and the actual
        # message at the very bottom. A missing variable is something to read,
        # not a crash to debug, and this command is the first thing a new
        # contributor runs. The `train` stage would print the same panel into its
        # stage log, which is the other half of why this is caught here.
        logger.error(str(error))
        raise typer.Exit(1) from error


def log_setup_check(experiment: str) -> str:
    """Log the throwaway run and say where it went. Returns its run id."""
    with tracked_run(experiment, f"setup-check-{time.strftime('%Y%m%d-%H%M%S')}") as run:
        mlflow.log_param("checked", "credentials")
        mlflow.log_metric("ok", 1)

    uri = mlflow.get_tracking_uri()
    logger.success(
        f"Logged run {run.info.run_id} to {experiment!r}. "
        f"Open {uri.removesuffix('.mlflow')}/experiments to see it."
    )
    return run.info.run_id


if __name__ == "__main__":
    app()
