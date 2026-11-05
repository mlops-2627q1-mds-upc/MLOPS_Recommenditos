"""Pooled MdAPE and coverage over several drift windows, as the drift job computes them.

Uses the job's own label join and error formula, so each window's figures equal
`report --window N`. Run in the monitor container on the VM:

    docker exec -i recommenditos-monitor-1 python - 13 14 < pooled.py > pooled.json
"""

import json
import os
from pathlib import Path
import statistics
import sys

from recommenditos_monitoring.log import Label, connect_read_only, labels
from recommenditos_monitoring.performance import absolute_percentage_error
from recommenditos_monitoring.store import Store


def in_range(label: Label) -> bool | None:
    if label.lower_eur is None or label.upper_eur is None:
        return None
    return label.lower_eur <= label.observed_price_eur <= label.upper_eur


def figures(found: list[Label]) -> dict[str, float | int]:
    ranged = [hit for hit in map(in_range, found) if hit is not None]
    return {
        "labelled": len(found),
        "mdape": statistics.median(absolute_percentage_error(label) for label in found),
        "ranges": len(ranged),
        "coverage": sum(ranged) / len(ranged),
    }


def main(window_ids: list[int]) -> None:
    store = Store.open(Path(os.environ["RECOMMENDITOS_MONITOR_DB"]))
    log = connect_read_only(Path(os.environ["RECOMMENDITOS_PREDICTION_LOG"]))
    windows = {window.window_id: window for window in store.windows()}
    per_window, pooled, rows = {}, [], []
    for window_id in window_ids:
        window = windows[window_id]
        held = store.requests(window_id)
        found = list(labels(log, [r.request_id for r in held], window.first_seq, window.last_seq))
        per_window[window_id] = figures(found) | {"model": window.model_version}
        pooled += found
        rows += [
            {
                "window": window_id,
                "observed": label.observed_price_eur,
                "ape": absolute_percentage_error(label),
                "in_range": in_range(label),
            }
            for label in found
        ]
    print(json.dumps({"windows": per_window, "pooled": figures(pooled), "rows": rows}))


if __name__ == "__main__":
    main([int(argument) for argument in sys.argv[1:]])
