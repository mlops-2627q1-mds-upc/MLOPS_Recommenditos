"""Run the API: `python -m recommenditos.api [uvicorn options]`.

Empties the metrics' multiprocess directory once, before uvicorn starts its
workers (`recommenditos.api.metrics`), then hands every option to uvicorn's own
command line with the app factory and the logging configuration filled in, so
`--host`, `--port`, `--workers` (or `WEB_CONCURRENCY`) and `--log-level` work as
they do for uvicorn. The logging configuration sends uvicorn's lines to the
service log and drops its access log (`recommenditos.api.logs`).

The app is not imported here: with several workers this process only
supervises them, and the model's libraries would cost it memory for nothing.
"""

from pathlib import Path
import sys

import uvicorn

from recommenditos.api.metrics import prepare_multiprocess_dir

#: The app factory uvicorn calls in every worker.
APP_FACTORY = "recommenditos.api.app:create_app"

#: The `logging.config.dictConfig` uvicorn applies in every process it starts.
LOG_CONFIG = Path(__file__).with_name("log_config.json")


def main(args: list[str] | None = None) -> None:
    prepare_multiprocess_dir()
    options = sys.argv[1:] if args is None else args
    uvicorn.main.main(
        args=[APP_FACTORY, "--factory", "--log-config", str(LOG_CONFIG), *options],
        prog_name="python -m recommenditos.api",
    )


if __name__ == "__main__":
    main()
