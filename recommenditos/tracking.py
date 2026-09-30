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
authentication ever does. Wiring that up is issue #37's work, along with adding
this module to the stage's `deps` in dvc.yaml.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
import os
import subprocess
import time

# Set before `import mlflow`, because mlflow reads this one at import time and
# logs an unsolicited pointer to a bundled authoring skill whenever it thinks a
# coding agent is driving the process. On this project that is most runs, and it
# would be the first line a contributor sees in step 6 of the getting-started
# page and in every train-stage log. `setdefault`, so anyone who wants the hint
# can still ask for it. `.env` cannot do this job: isort puts `import mlflow`
# above `import recommenditos.config`, so python-dotenv has not run yet.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

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
