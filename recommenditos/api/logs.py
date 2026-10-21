"""The service log: loguru, one JSON object per line in the container, text for `make api`.

`RECOMMENDITOS_LOG_FORMAT` picks the format: `json` (the image sets it) or
`text` (the default). Either way a line written while a request is answered
carries that request's server-generated ID, which `RequestIdMiddleware` puts
into loguru's context; the JSON form has it as `request_id`, beside the
fields the line was logged with. A line holds no client address and no
request body (NFR-08): the middleware writes one line per request with its
method, route template, status and duration, in place of uvicorn's access log, which
names the client.

uvicorn configures the standard library's logging in every process it starts,
the supervisor and each worker, from `log_config.json` (`python -m
recommenditos.api` passes it). That file builds its one handler with
`uvicorn_handler`, which configures loguru in that process and forwards
uvicorn's own lines, and every other library's, to it.
"""

from collections.abc import Callable
from datetime import UTC
import inspect
import json
import logging
import os
import sys
import traceback
from typing import TYPE_CHECKING, Any, TextIO

from loguru import logger

if TYPE_CHECKING:
    from loguru import Message, Record

#: The environment variable that picks the log format.
LOG_FORMAT_ENV_VAR = "RECOMMENDITOS_LOG_FORMAT"
LOG_FORMATS = ("text", "json")

#: The logger whose level uvicorn's `--log-level` sets; the service log follows it.
_LEVEL_SOURCE = "uvicorn.error"

_TEXT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>"
)


def log_format() -> str:
    """The format `RECOMMENDITOS_LOG_FORMAT` names; anything else stops the service."""
    value = os.environ.get(LOG_FORMAT_ENV_VAR, "").strip().lower() or "text"
    if value not in LOG_FORMATS:
        raise ValueError(f"{LOG_FORMAT_ENV_VAR} is {value!r}; it must be one of {LOG_FORMATS}")
    return value


def configure(fmt: str | None = None, stream: TextIO | None = None) -> None:
    """Send loguru's lines to `stream` (stderr) in `fmt`, at the level uvicorn runs at."""
    fmt = fmt or log_format()
    stream = stream or sys.stderr
    logger.remove()
    if fmt == "json":
        logger.add(json_sink(stream), level=0, filter=_at_uvicorn_level, format="{message}")
    else:
        # `diagnose=False`: a traceback names no variable's value, which could be
        # a request's content.
        logger.add(stream, level=0, filter=_at_uvicorn_level, format=text_format, diagnose=False)


def json_sink(stream: TextIO) -> Callable[["Message"], None]:
    """A loguru sink that writes each record to `stream` as one JSON object per line."""

    def write(message: "Message") -> None:
        stream.write(json_line(message.record) + "\n")
        stream.flush()

    return write


def json_line(record: "Record") -> str:
    """A record as one line of JSON: the context and the logged fields, then the basics."""
    line: dict[str, Any] = {
        "time": record["time"].astimezone(UTC).isoformat(timespec="milliseconds"),
        "level": record["level"].name,
        "logger": record["name"],
        "message": record["message"],
    }
    line |= {key: value for key, value in record["extra"].items() if key not in line}
    exception = record["exception"]
    if exception is not None:
        line["exception"] = "".join(
            traceback.format_exception(exception.type, exception.value, exception.traceback)
        ).rstrip()
    return json.dumps(line, default=str)


def text_format(record: "Record") -> str:
    """loguru's default line, with the request ID after the source when there is one."""
    request_id = " | {extra[request_id]}" if "request_id" in record["extra"] else ""
    return _TEXT_FORMAT + request_id + " - <level>{message}</level>\n{exception}"


def _at_uvicorn_level(record: "Record") -> bool:
    return bool(record["level"].no >= logging.getLogger(_LEVEL_SOURCE).getEffectiveLevel())


class InterceptHandler(logging.Handler):
    """Forward a standard-library record to loguru, from the frame that logged it.

    The recipe of loguru's documentation: the depth makes loguru name the module
    that called `logging`, not this handler.
    """

    def emit(self, record: logging.LogRecord) -> None:
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = inspect.currentframe(), 0
        while frame is not None and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def uvicorn_handler() -> logging.Handler:
    """The handler `log_config.json` gives uvicorn's loggers and the root logger.

    `logging.config.dictConfig` calls it once in every process uvicorn starts,
    before that process logs anything, so it configures loguru for the process
    too. Python's warnings, such as scikit-learn's, are logged rather than
    printed, so that they reach the same sink.
    """
    configure()
    logging.captureWarnings(True)
    return InterceptHandler()
