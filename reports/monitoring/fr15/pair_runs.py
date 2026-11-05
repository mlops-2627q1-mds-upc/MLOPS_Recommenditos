"""Pair the before and after `ES` windows of the FR-15 rehearsal from pooled.py's outputs.

Checks that both runs sent the same `ES` rows (the observed prices match in order),
then writes the pooled figures of both runs, the paired per-request errors, and the
criterion's pooled MdAPE and coverage:

    python pair_runs.py before-pooled-13-14.json after-pooled-17-18.json

writes pooled-es-windows.json, paired-es-requests.csv and criterion-calculation.txt
next to this script.
"""

import csv
import json
from pathlib import Path
import statistics
import sys
from typing import Any

HERE = Path(__file__).parent


def in_range(row: dict[str, Any]) -> bool | None:
    if row["lower"] is None:
        return None
    return bool(row["lower"] <= row["observed"] <= row["upper"])


def cell(hit: bool | None) -> str:
    return "" if hit is None else str(int(hit))


def main(before_path: Path, after_path: Path) -> None:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    b, a = before["rows"], after["rows"]
    same = [r["observed"] for r in b] == [r["observed"] for r in a]
    if not same:
        raise SystemExit("the two runs did not send the same ES rows")
    summary = {
        "before": {"windows": before["windows"], "pooled": before["pooled"]},
        "after": {"windows": after["windows"], "pooled": after["pooled"]},
    }
    (HERE / "pooled-es-windows.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (HERE / "paired-es-requests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "es_index",
                "observed_price_eur",
                "ape_before",
                "ape_after",
                "in_range_before",
                "in_range_after",
            ]
        )
        for i, (x, y) in enumerate(zip(b, a, strict=True)):
            writer.writerow(
                [
                    i,
                    x["observed"],
                    f"{x['ape']:.6f}",
                    f"{y['ape']:.6f}",
                    cell(in_range(x)),
                    cell(in_range(y)),
                ]
            )
    mdape_before = statistics.median(r["ape"] for r in b)
    mdape_after = statistics.median(r["ape"] for r in a)
    ranged_before = [hit for hit in map(in_range, b) if hit is not None]
    ranged_after = [hit for hit in map(in_range, a) if hit is not None]
    lower = sum(x["ape"] > y["ape"] for x, y in zip(b, a, strict=True))
    lines = [
        f"rows {len(b)} {len(a)} observed prices identical in order: {same}",
        (
            f"pooled MdAPE before {mdape_before:.4%} after {mdape_after:.4%} "
            f"delta {(mdape_after - mdape_before) * 100:+.2f} pp"
        ),
        (
            f"pooled coverage before {sum(ranged_before)}/{len(ranged_before)} = "
            f"{sum(ranged_before) / len(ranged_before):.1%}, after {sum(ranged_after)}/"
            f"{len(ranged_after)} = {sum(ranged_after) / len(ranged_after):.1%}"
        ),
        f"per-request APE lower after: {lower} of {len(a)}",
    ]
    (HERE / "criterion-calculation.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
