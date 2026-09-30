"""The MLflow configuration helper, and the template that documents it.

No test here reaches DagsHub: the point of the helper is what it does *before*
the network is touched, and the one test that lets MLflow run points it at a
directory under `tmp_path`. The suite therefore still passes on a clean clone
with no credentials (see `tests/conftest.py`).
"""

import os

import mlflow
import pytest

from recommenditos.config import PROJ_ROOT
from recommenditos.tracking import REQUIRED_ENV_VARS, configure_tracking, main

ENV_TEMPLATE = PROJ_ROOT / ".env.template"

#: The variables the DVC remote needs. They are not in `REQUIRED_ENV_VARS`,
#: because DVC reads its credentials from `.dvc/config.local` rather than from
#: the environment, but `.env` is where a contributor keeps them, so the
#: template has to carry them too.
DVC_ENV_VARS = ("DAGSHUB_USERNAME", "DAGSHUB_USER_TOKEN")


@pytest.fixture(autouse=True)
def _isolated_mlflow_environment(monkeypatch, tmp_path):
    """Neither the developer's `.env` nor one test may leak into the next.

    `recommenditos.config` loads `.env` at import time, so on a machine that has
    one, the variables are already set when the suite starts. Each test declares
    the environment it wants instead of inheriting that one.
    """
    for name in REQUIRED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    before = mlflow.get_tracking_uri()
    yield
    mlflow.set_tracking_uri(before)


def _template_assignments() -> dict[str, str]:
    assignments = {}
    for raw in ENV_TEMPLATE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            name, value = line.split("=", 1)
            assignments[name.strip()] = value.strip()
    return assignments


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


def test_a_complete_environment_selects_the_experiment(monkeypatch, tmp_path):
    # SQLite rather than a `file:` directory: MLflow 3 put the filesystem
    # tracking backend into maintenance mode and raises on it unless
    # `MLFLOW_ALLOW_FILE_STORE` is set. A local database is the supported
    # equivalent and needs no server.
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "someone")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "a-token")

    assert configure_tracking("an-experiment") == uri
    assert mlflow.get_tracking_uri() == uri
    assert mlflow.get_experiment_by_name("an-experiment") is not None


def test_the_setup_check_logs_a_run_that_can_be_read_back(monkeypatch, tmp_path):
    """The check is only worth running if it leaves something to look at."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_TRACKING_USERNAME", "someone")
    monkeypatch.setenv("MLFLOW_TRACKING_PASSWORD", "a-token")

    main(experiment="a-setup-check")

    experiment = mlflow.get_experiment_by_name("a-setup-check")
    runs = mlflow.MlflowClient().search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].info.status == "FINISHED"
    assert runs[0].data.metrics == {"ok": 1.0}


def test_the_template_documents_every_variable_the_project_reads():
    """Rename a variable in the code and this fails until the template follows."""
    documented = _template_assignments()

    for name in (*REQUIRED_ENV_VARS, *DVC_ENV_VARS):
        assert name in documented, f"{name} is missing from .env.template"


def test_the_template_carries_no_credential():
    """A template committed with somebody's token filled in is the failure mode."""
    documented = _template_assignments()

    filled = {name for name, value in documented.items() if value}
    # The tracking URI is the one value that is the same for the whole team, so
    # it is the one value the template may carry.
    assert filled == {"MLFLOW_TRACKING_URI"}
    assert documented["MLFLOW_TRACKING_URI"].startswith("https://dagshub.com/")


def test_the_helper_does_not_read_the_environment_at_import_time():
    """Importing the module must not fail on a clone without credentials.

    The autouse fixture has just unset the variables, so if the import had done
    the checking, this module could not have been imported at all.
    """
    assert not any(os.environ.get(name) for name in REQUIRED_ENV_VARS)
