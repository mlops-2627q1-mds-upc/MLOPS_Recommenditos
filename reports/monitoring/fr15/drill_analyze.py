"""The time line of the rollback drill's /health probe (drill-probe.txt).

Each line: UTC time, HTTP status or curl error, model version. Prints every run of
answers other than 200, every change of the served version, and any 200 whose
version is neither the old nor the new model.

    python drill_analyze.py drill-probe.txt
"""

from datetime import datetime
from pathlib import Path
import sys

OLD, NEW = "afc3335eefa80dafe3d9ae6ab85c1ae3", "2a60912bd232083e6950f26fe72fe6c4"


def clock(moment: datetime) -> str:
    return moment.strftime("%H:%M:%S.%f")[:-3]


def main(path: Path) -> None:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        moment, code, version = line.split(maxsplit=2)
        rows.append((datetime.fromisoformat(moment), code, version))
    print(f"{len(rows)} probes {clock(rows[0][0])} - {clock(rows[-1][0])}")
    start, version = None, None
    for i, (moment, code, served) in enumerate(rows):
        if code != "200":
            start = i if start is None else start
            continue
        if start is not None:
            first, last = rows[start][0], rows[i - 1][0]
            codes = sorted({row[1] for row in rows[start:i]})
            seconds = (moment - first).total_seconds()
            print(
                f"non-200 {clock(first)} .. {clock(last)}: {i - start} probes, "
                f"{seconds:.1f} s until the next 200; codes {codes}"
            )
            start = None
        if version and served != version:
            print(f"version switch {clock(moment)}: {version[:8]} -> {served[:8]}")
        version = served
    if start is not None:
        print(f"non-200 from {clock(rows[start][0])} to the end")
    odd = [row for row in rows if row[1] == "200" and row[2] not in (OLD, NEW)]
    print("200 answers with a version other than old/new:", len(odd), odd[:3])


if __name__ == "__main__":
    main(Path(sys.argv[1]))
