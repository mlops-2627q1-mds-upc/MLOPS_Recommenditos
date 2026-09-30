"""MLflow against the shared DagsHub tracking server, in one place.

MLflow already reads `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and
`MLFLOW_TRACKING_PASSWORD` from the environment by itself, so this module is not
here to pass them along. It is here so that a missing one fails immediately,
with a message naming `.env.template`, instead of surfacing as an HTTP 401 from
inside a half-finished training run - the cheapest failure is the one that
happens before the fit.

`python -m recommenditos.tracking` calls `configure_tracking` with a throwaway
experiment, which is how a new contributor proves their credentials work before
touching the pipeline. The `train` stage takes the same route with
`params.yaml`'s `train.mlflow_experiment`, so the setup a newcomer verifies is
the setup the pipeline runs on and there is one place to change if DagsHub's
authentication ever does. Wiring that up is issue #37's work, along with adding
this module to the stage's `deps` in dvc.yaml.
"""

import os
import time

from loguru import logger
import mlflow
import typer

# Importing config for its side effect: it is the single place that calls
# `load_dotenv()`, so the credentials from the gitignored .env are in the
# environment by the time this module reads them. Every DVC stage imports it
# too, which is why the loading is not repeated here.
import recommenditos.config  # noqa: F401

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


def configure_tracking(experiment: str) -> str:
    """Point MLflow at the DagsHub tracking server and select `experiment`.

    Returns the tracking URI, so a caller can log where its runs went.
    Raises `RuntimeError` naming the variables that are missing.
    """
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
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


@app.command()
def main(experiment: str = SETUP_CHECK_EXPERIMENT):
    """Log one trivial run, so a new contributor can see it in the DagsHub UI."""
    uri = configure_tracking(experiment)
    with mlflow.start_run(run_name=f"setup-check-{time.strftime('%Y%m%d-%H%M%S')}") as run:
        mlflow.log_param("checked", "credentials")
        mlflow.log_metric("ok", 1)
    logger.success(
        f"Logged run {run.info.run_id} to {experiment!r}. "
        f"Open {uri.removesuffix('.mlflow')}/experiments to see it."
    )


if __name__ == "__main__":
    app()
