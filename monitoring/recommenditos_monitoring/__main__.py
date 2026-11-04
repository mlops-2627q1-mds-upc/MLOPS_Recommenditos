"""The monitoring jobs' command line: `python -m recommenditos_monitoring <command>`.

    replay    send held-out listings to the API and their prices to /v1/feedback (#227)

docs/docs/monitoring.md has the commands for a local stack and for the VM.
"""

import argparse
from collections.abc import Sequence
import os
from pathlib import Path
import sys

import httpx

from recommenditos_monitoring.replay import (
    THEN_SAMPLES,
    Replay,
    ReplayError,
    Summary,
    format_summary,
    phases_of,
    reference_date_of,
)

#: Where the image holds the samples; run from the repository, `data/monitoring`.
DATA_DIR_VARIABLE = "RECOMMENDITOS_MONITORING_DATA_DIR"
API_KEY_VARIABLE = "RECOMMENDITOS_API_KEY"


def positive_rate(text: str) -> float:
    rate = float(text)
    if not 0 < rate <= 100:  # noqa: PLR2004 - the bounds are the message
        raise argparse.ArgumentTypeError("the rate is above 0 and at most 100 listings/s")
    return rate


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
        "--timeout", type=float, default=10.0, help="seconds to wait for an answer"
    )
    return main


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
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
