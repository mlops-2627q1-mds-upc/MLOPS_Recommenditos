"""The monitoring jobs' command line: `python -m recommenditos_monitoring <command>`.

    replay    send held-out listings to the API and their prices to /v1/feedback (#227)
    run-once  test every window of 1,000 requests the prediction log completed (#228)
    report    the finding and the evidence of one window of monitor.db (#228)

docs/docs/monitoring.md has the commands for a local stack and for the VM.
"""

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import sys

import httpx

from recommenditos_monitoring.job import (
    WINDOW_SIZE,
    JobError,
    Paths,
    chosen_window,
    evidence,
    finding,
    run_once,
)
from recommenditos_monitoring.replay import (
    THEN_SAMPLES,
    Replay,
    ReplayError,
    Summary,
    format_summary,
    phases_of,
    reference_date_of,
)
from recommenditos_monitoring.store import Store, Window

#: Where the image holds the samples; run from the repository, `data/monitoring`.
DATA_DIR_VARIABLE = "RECOMMENDITOS_MONITORING_DATA_DIR"
API_KEY_VARIABLE = "RECOMMENDITOS_API_KEY"
#: The API's own variable for its log, which the drift job reads.
LOG_VARIABLE = "RECOMMENDITOS_PREDICTION_LOG"
DB_VARIABLE = "RECOMMENDITOS_MONITOR_DB"
TEXTFILE_VARIABLE = "RECOMMENDITOS_MONITOR_TEXTFILE_DIR"


def positive_rate(text: str) -> float:
    rate = float(text)
    if not 0 < rate <= 100:  # noqa: PLR2004 - the bounds are the message
        raise argparse.ArgumentTypeError("the rate is above 0 and at most 100 listings/s")
    return rate


#: The longest wait for an answer: below nginx's default proxy_read_timeout of
#: 60 s, after which the proxy answers 503 (deploy/nginx.conf) for a valuation the
#: API may still answer and log. The replay retries a 503, so with a longer wait
#: that listing would be logged twice; with this one its own read timeout comes
#: first, and a read timeout is never sent again.
MAX_TIMEOUT_SECONDS = 50.0


def answer_timeout(text: str) -> float:
    seconds = float(text)
    if not 0 < seconds <= MAX_TIMEOUT_SECONDS:
        raise argparse.ArgumentTypeError(
            f"the timeout is above 0 and at most {MAX_TIMEOUT_SECONDS:g} s, "
            "below the proxy's 60 s, after which it answers 503"
        )
    return seconds


def parser() -> argparse.ArgumentParser:
    main = argparse.ArgumentParser(prog="python -m recommenditos_monitoring")
    commands = main.add_subparsers(dest="command", required=True)
    replay = commands.add_parser(
        "replay",
        help="replay held-out listings against the API",
        description="Send control listings, then the ES market, to the API at a fixed rate, "
        "and report each listing's observed price to /v1/feedback.",
    )
    replay.add_argument(
        "--control", type=int, default=0, metavar="N", help="control listings to send first"
    )
    replay.add_argument(
        "--then", choices=THEN_SAMPLES, help="the sample to send after the control listings"
    )
    replay.add_argument(
        "--then-count",
        type=int,
        metavar="N",
        help="how many listings of --then to send (default: all of them)",
    )
    replay.add_argument(
        "--rate", type=positive_rate, default=5.0, help="listings per second (default: 5)"
    )
    replay.add_argument(
        "--base-url",
        default="http://proxy:8080",
        help="where the valuations go, the reverse proxy as a client reaches it",
    )
    replay.add_argument(
        "--feedback-url",
        default="http://api:8000",
        help="where the feedback goes, the API inside the Compose network",
    )
    replay.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get(DATA_DIR_VARIABLE, "data/monitoring")),
        help=f"the samples' directory (default: ${DATA_DIR_VARIABLE} or data/monitoring)",
    )
    replay.add_argument(
        "--retries", type=int, default=3, help="further attempts of a failed request"
    )
    replay.add_argument(
        "--timeout",
        type=answer_timeout,
        default=10.0,
        help=f"seconds to wait for an answer (default: 10, at most {MAX_TIMEOUT_SECONDS:g})",
    )

    run = commands.add_parser(
        "run-once",
        help="test the windows the prediction log completed since the last pass",
        description="Read the prediction log from the last closed window on, test every "
        "full window of requests for input drift against the reference, store the "
        "results in monitor.db, and write them for node-exporter's textfile collector.",
    )
    _store_argument(run)
    run.add_argument(
        "--log",
        type=Path,
        default=Path(os.environ.get(LOG_VARIABLE, "prediction-log.sqlite3")),
        help=f"the API's prediction log (default: ${LOG_VARIABLE})",
    )
    run.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get(DATA_DIR_VARIABLE, "data/monitoring")),
        help=f"the samples' directory with the reference (default: ${DATA_DIR_VARIABLE})",
    )
    run.add_argument(
        "--textfile-dir",
        type=Path,
        default=_optional_path(os.environ.get(TEXTFILE_VARIABLE)),
        help=f"where monitor.prom goes (default: ${TEXTFILE_VARIABLE}; unset writes none)",
    )
    run.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help=f"requests per window (default: {WINDOW_SIZE}, NFR-11's)",
    )
    report = commands.add_parser(
        "report",
        help="the finding of one window, for retrain.yml's reason, or its evidence as JSON",
        description="Print the finding of a window of monitor.db: the latest tested one, "
        "or the one --window names; with --json, the window and every feature's result.",
    )
    _store_argument(report)
    report.add_argument("--window", type=int, metavar="N", help="the window's number")
    report.add_argument(
        "--json", action="store_true", help="print the evidence as JSON instead of the finding"
    )
    return main


def _optional_path(text: str | None) -> Path | None:
    return Path(text) if text else None


def _store_argument(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get(DB_VARIABLE, "monitor.db")),
        help=f"the job's results (default: ${DB_VARIABLE} or monitor.db)",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "run-once":
        return run_once_command(args)
    if args.command == "report":
        return report_command(args)
    return replay_command(args)


def run_once_command(args: argparse.Namespace) -> int:
    paths = Paths(log=args.log, db=args.db, data_dir=args.data_dir, textfile_dir=args.textfile_dir)
    try:
        outcome = run_once(paths, window_size=args.window_size)
    except Exception as error:  # noqa: BLE001 - every failure ends the pass with its cause
        print(f"the drift pass failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    if outcome.skipped:
        print("another pass is running; this one leaves it to finish")
        return 0
    for window in outcome.closed:
        print(f"{window.status} window {window.window_id}: " + _short(window))
    print(f"open window: {outcome.open_requests} of {args.window_size} requests")
    return 0


def _short(window: Window) -> str:
    if window.flagged is None:
        return f"seq {window.first_seq}-{window.last_seq}, {window.requests} requests, untested"
    verdict = "flagged" if window.flagged else "not flagged"
    return (
        f"seq {window.first_seq}-{window.last_seq}, {verdict}, "
        f"{window.drifted_features} drifted features, {window.es_share:.0%} ES"
    )


def report_command(args: argparse.Namespace) -> int:
    if not args.db.is_file():
        print(f"there is no monitor.db at {args.db}; has a pass run?", file=sys.stderr)
        return 1
    store = Store.open(args.db)
    try:
        window = chosen_window(store, args.window)
        features = store.features(window.window_id)
    except JobError as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        store.close()
    print(evidence(window, features) if args.json else finding(window, features))
    return 0


def replay_command(args: argparse.Namespace) -> int:
    if args.control < 0 or (args.then_count is not None and args.then_count < 0):
        print("counts are 0 or more", file=sys.stderr)
        return 2
    if args.control == 0 and args.then is None:
        print("nothing to send: give --control N, --then es, or both", file=sys.stderr)
        return 2
    api_key = os.environ.get(API_KEY_VARIABLE, "")
    if not api_key:
        print(f"{API_KEY_VARIABLE} is not set, so no feedback could be stored", file=sys.stderr)
        return 2
    summary = Summary(rate=args.rate)
    with httpx.Client(timeout=args.timeout) as client:
        replay = Replay(
            client=client,
            base_url=args.base_url.rstrip("/"),
            feedback_url=args.feedback_url.rstrip("/"),
            api_key=api_key,
            reference_date=reference_date_of(args.data_dir),
            rate=args.rate,
            retries=args.retries,
        )
        try:
            replay.run(phases_of(args.data_dir, args.control, args.then, args.then_count), summary)
        except ReplayError as error:
            print(f"stopped: {error}", file=sys.stderr)
            print(format_summary(summary))
            return 1
        except KeyboardInterrupt:
            print("stopped by the user", file=sys.stderr)
            print(format_summary(summary))
            return 130
    print(format_summary(summary))
    return 0 if summary.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
