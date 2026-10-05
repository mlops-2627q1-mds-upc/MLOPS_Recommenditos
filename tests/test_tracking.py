"""The MLflow configuration helper, and the template that documents it.

No test here reaches DagsHub: the point of the helper is what it does *before*
the network is touched, and the tests that let MLflow run point it at a SQLite
database under `tmp_path`. The suite therefore still passes on a clean clone
with no credentials (see `tests/conftest.py`).

One database is shared by the whole module rather than created per test. Each
fresh MLflow store pays for creating and migrating a schema, which was most of
this module's runtime; the tests stay independent by using a different
experiment name each, which is the only state they care about.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys

from loguru import logger
import mlflow
import pytest
from typer.testing import CliRunner

from recommenditos.config import PROJ_ROOT
from recommenditos.tracking import (
    DVC_LOCK_FILE,
    REQUIRED_ENV_VARS,
    UNCONFIGURED_TRACKING_URI,
    app,
    configure_tracking,
    log_setup_check,
    run_provenance,
    tracked_run,
)

ENV_TEMPLATE = PROJ_ROOT / ".env.template"


@pytest.fixture(autouse=True)
def _isolated_mlflow_environment(monkeypatch):
    """Neither the developer's `.env` nor one test may leak into the next.

    `recommenditos.config` loads `.env` at import time, so on a machine that has
    one, the variables are already set when the suite starts. Each test declares
    the environment it wants instead of inheriting that one.

    The URI is captured *before* the variables go, because `get_tracking_uri()`
    invents `sqlite:///$PWD/mlflow.db` once there is nothing to read - restoring
    that would leave every later test pointed at a database in the repository
    root, and the only reason nothing writes there today is collection order.
    """
    before = mlflow.get_tracking_uri()
    for name in REQUIRED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    yield
    mlflow.set_tracking_uri(before)


@pytest.fixture(scope="module")
def _local_store(tmp_path_factory) -> str:
    """A SQLite tracking store, built and migrated once for this module."""
    # SQLite rather than a `file:` directory: MLflow 3 put the filesystem
    # tracking backend into maintenance mode and raises on it unless
    # `MLFLOW_ALLOW_FILE_STORE` is set. A local database is the supported
    # equivalent and needs no server.
    return f"sqlite:///{tmp_path_factory.mktemp('mlflow') / 'mlflow.db'}"


@pytest.fixture
def local_tracking(monkeypatch, _local_store: str) -> str:
    """A complete, credential-shaped environment pointed at the local store."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", _local_store)
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "someone")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "a-token")
    return _local_store


def _template_assignments() -> dict[str, str]:
    assignments = {}
    for raw in ENV_TEMPLATE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            assignments[name.strip()] = value.strip()
    return assignments


# --------------------------------------------------------------------------
# Failing before the network
# --------------------------------------------------------------------------


def test_missing_variables_are_all_named_at_once():
    """One message, every missing variable - not one failure per round trip."""
    with pytest.raises(RuntimeError) as failure:
        configure_tracking("whatever")

    message = str(failure.value)
    for name in REQUIRED_ENV_VARS:
        assert name in message
    # The message has to say what to do about it, not only what is wrong.
    assert ".env.template" in message


def test_an_empty_value_counts_as_missing(monkeypatch):
    """A variable left blank in `.env` is the likeliest way to get here."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.invalid")
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "someone")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "")

    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_PASSWORD"):
        configure_tracking("whatever")


def test_the_local_store_guard_does_not_pass_for_a_configured_uri(monkeypatch):
    """The guard must not make the thing it guards look configured.

    `mlflow.set_tracking_uri` exports `MLFLOW_TRACKING_URI` so subprocesses
    inherit it, so pointing MLflow at the sentinel at import time writes it into
    the environment. Read naively, that is a tracking URI, and a clone with no
    credentials at all would be told only that its username is missing.
    """
    monkeypatch.setenv("MLFLOW_TRACKING_URI", UNCONFIGURED_TRACKING_URI)

    with pytest.raises(RuntimeError, match="MLFLOW_TRACKING_URI") as failure:
        configure_tracking("whatever")

    assert UNCONFIGURED_TRACKING_URI not in str(failure.value)


def test_the_cli_reports_a_missing_variable_without_a_traceback():
    """The documented newcomer command has to print its message, not a panel.

    Typer renders an uncaught exception as a Rich panel holding a dozen lines of
    `tracking.py`'s own source, with the message at the very bottom. The `train`
    stage would put the same panel in its stage log.
    """
    reported: list[str] = []
    sink = logger.add(reported.append, format="{message}")
    try:
        result = CliRunner().invoke(app, [])
    finally:
        logger.remove(sink)

    assert result.exit_code == 1
    # A clean `typer.Exit` arrives as SystemExit; anything else was rendered as a
    # traceback on its way out.
    assert isinstance(result.exception, SystemExit), (
        f"a {type(result.exception).__name__} reached typer, which prints it as a panel"
    )
    assert any("MLFLOW_TRACKING_URI" in message for message in reported), (
        f"the command exited without naming what is missing: {reported}"
    )


def test_an_unconfigured_process_cannot_log_into_the_repository(monkeypatch, tmp_path):
    """Importing the module must not leave `mlflow.db` reachable as a default.

    Since 3.16 an unset `MLFLOW_TRACKING_URI` means `sqlite:///$PWD/mlflow.db`,
    not "tracking off", and the first logging call creates it. So a notebook or a
    stray script that logs before going through `tracked_run` would drop a
    database into whatever directory it started in.
    """
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(UNCONFIGURED_TRACKING_URI)

    with pytest.raises(Exception, match="unconfigured"), mlflow.start_run():
        mlflow.log_metric("ok", 1)

    assert not (tmp_path / "mlflow.db").exists()
    assert not (tmp_path / "mlruns").exists()


def test_the_helper_does_not_read_the_environment_at_import_time():
    """Importing the module must not fail on a clone without credentials.

    The autouse fixture has just unset the variables, so if the import had done
    the checking, this module could not have been imported at all.
    """
    assert not any(os.environ.get(name) for name in REQUIRED_ENV_VARS)


# --------------------------------------------------------------------------
# Against a local store
# --------------------------------------------------------------------------


def test_a_complete_environment_selects_the_experiment(local_tracking: str):
    assert configure_tracking("an-experiment") == local_tracking
    assert mlflow.get_tracking_uri() == local_tracking
    assert mlflow.get_experiment_by_name("an-experiment") is not None


def test_the_setup_check_logs_a_run_that_can_be_read_back(local_tracking: str):
    """The check is only worth running if it leaves something to look at."""
    log_setup_check("a-setup-check")

    experiment = mlflow.get_experiment_by_name("a-setup-check")
    runs = mlflow.MlflowClient().search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FINISHED"
    assert runs[0].data.metrics == {"ok": 1.0}


@pytest.mark.req("NFR-14")
def test_every_run_records_the_commit_and_the_data_version(local_tracking: str):
    """NFR-14 asks for this on every run, so the seam does it, not the caller."""
    with tracked_run("a-provenance-check") as run:
        pass

    tags = mlflow.MlflowClient().get_run(run.info.run_id).data.tags
    assert (
        tags["git_commit"]
        == subprocess.run(
            ["git", "-C", str(PROJ_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    assert tags["git_dirty"] in {"true", "false"}
    assert tags["dvc_lock_md5"], "dvc.lock is the data version, and this repo has one"


@pytest.mark.req("NFR-14")
def test_a_run_without_git_records_that_rather_than_failing(monkeypatch):
    """An image built without the `.git` directory still has to be able to train."""

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("recommenditos.tracking.subprocess.run", no_git)

    provenance = run_provenance()

    assert provenance["git_commit"] == "unknown"
    assert provenance["git_dirty"] == "false"


@pytest.mark.req("NFR-14")
def test_the_data_version_changes_with_the_lock_file(monkeypatch, tmp_path):
    """A tag that does not move when the data moves records nothing."""
    assert DVC_LOCK_FILE == PROJ_ROOT / "dvc.lock", "the data version reads the real lock file"

    lock = tmp_path / "dvc.lock"
    monkeypatch.setattr("recommenditos.tracking.DVC_LOCK_FILE", lock)

    lock.write_text("outs:\n- md5: one\n", encoding="utf-8")
    first = run_provenance()["dvc_lock_md5"]
    lock.write_text("outs:\n- md5: two\n", encoding="utf-8")

    assert run_provenance()["dvc_lock_md5"] != first


# --------------------------------------------------------------------------
# The template, and where `.env` is read from
# --------------------------------------------------------------------------


def test_the_template_documents_every_variable_the_project_reads():
    """Rename a variable in the code and this fails until the template follows.

    Exactly, not merely: a template that lists a variable nothing reads is how a
    contributor comes to believe that filling it in configures something.
    """
    assert set(_template_assignments()) == set(REQUIRED_ENV_VARS)


def test_the_template_carries_no_credential():
    """A template committed with somebody's token filled in is the failure mode."""
    documented = _template_assignments()

    filled = {name for name, value in documented.items() if value}
    # The tracking URI is the one value that is the same for the whole team, so
    # it is the one value the template may carry.
    assert filled == {"MLFLOW_TRACKING_URI"}
    assert documented["MLFLOW_TRACKING_URI"].startswith("https://dagshub.com/")


@pytest.fixture(scope="module")
def nested_clone(tmp_path_factory) -> Path:
    """A checkout nested under a directory that holds somebody else's `.env`.

    Only `config.py` and the package's `__init__` are copied, because that is
    everything the loading involves: `PROJ_ROOT` comes from `config.py`'s own
    `__file__`, so a copy is a faithful stand-in for a second clone and a symlink
    is not - `resolve()` would follow it back to this repository.
    """
    outer = tmp_path_factory.mktemp("outer")
    (outer / ".env").write_text(
        "MLFLOW_TRACKING_URI=https://the-parent-checkout.invalid\n", encoding="utf-8"
    )
    package = outer / "clone" / "recommenditos"
    package.mkdir(parents=True)
    for name in ("__init__.py", "config.py"):
        shutil.copyfile(PROJ_ROOT / "recommenditos" / name, package / name)
    return outer / "clone"


#: Printed by the subprocess below: where it thinks the repository root is, and
#: which tracking URI importing `config` put in its environment.
_REPORT = (
    "import os;from recommenditos.config import PROJ_ROOT;"
    "print(PROJ_ROOT);print(os.environ.get('MLFLOW_TRACKING_URI', ''))"
)


def _report_env_from(clone: Path) -> dict[str, str]:
    """Import `recommenditos.config` inside `clone` and report what it loaded."""
    environment = {k: v for k, v in os.environ.items() if not k.startswith("MLFLOW_")}
    completed = subprocess.run(
        [sys.executable, "-c", _REPORT],
        cwd=clone,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    # Not `.strip()`: an unset URI prints an empty line, and stripping eats it.
    root, uri = completed.stdout.splitlines()[-2:]
    return {"root": root, "uri": uri}


def test_a_nested_clone_does_not_inherit_the_parent_env(nested_clone: Path):
    """`.env` is scoped to its own repository, or the tracking rule is a fiction.

    A bare `load_dotenv()` searches upwards until it finds a file, so a clone
    inside another checkout picks up that one's credentials. The clone then looks
    configured when it is not, "an absent tracking URI means tracking is off"
    stops being falsifiable anywhere, and a stale token two directories up sends
    a run to the wrong server.
    """
    reported = _report_env_from(nested_clone)

    assert reported["root"] == str(nested_clone), "the subprocess imported the wrong copy"
    assert reported["uri"] == "", f"the parent checkout's .env was loaded: {reported['uri']}"


def test_the_clone_does_read_its_own_env(nested_clone: Path):
    """The other half: scoping it to `PROJ_ROOT` must not stop it working."""
    (nested_clone / ".env").write_text(
        "MLFLOW_TRACKING_URI=https://this-clone.invalid\n", encoding="utf-8"
    )
    try:
        reported = _report_env_from(nested_clone)
    finally:
        (nested_clone / ".env").unlink()

    assert reported["root"] == str(nested_clone)
    assert reported["uri"] == "https://this-clone.invalid"
