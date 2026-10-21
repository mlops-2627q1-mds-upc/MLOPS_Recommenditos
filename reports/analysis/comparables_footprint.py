"""What the comparables index costs the API: disk, memory and latency (#138, NFR-02, NFR-04).

NFR-04 caps the API image and the API container at 1 GB each, and NFR-02 asks
for a p95 of at most 300 ms for `/comparables` on the target VM. This measures
the committed `models/comparables.parquet` on the laptop:

- its size on disk, which is what it adds to the image;
- the resident memory of a process that has built the app, once around
  `models/lgbm-basic` with the index beside it and once around a copy of the
  bundle without it, each in a fresh interpreter, `RUNS` times; the difference
  is what the index costs every uvicorn worker;
- the in-process latency of `/v1/comparables` with FastAPI's test client, so it
  holds the validation, the model's checks and the search, and no network,
  nginx or uvicorn. Requests are built from test-split listings of the served
  makes, which the index does not hold, in four shapes: make only, make and
  model, P1 (make, model, registration date and mileage) and P1 with `k` 20.
  `ROUNDS` rounds of `PER_SHAPE` listings each, the shapes interleaved, after
  `WARM_UP` unmeasured requests. The load average is printed before and after,
  because the laptop is shared.

Linux only, because the resident memory is read from `/proc`. Run from the
repository root after `dvc pull models/lgbm-basic models/comparables.parquet
data/processed/test.parquet`:

    uv run python reports/analysis/comparables_footprint.py

which, with the log lines removed, gives `comparables_footprint_results.txt`.
"""

from http import HTTPStatus
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from recommenditos.api.app import create_app
from recommenditos.config import MODELS_DIR, PROCESSED_DATA_DIR
from recommenditos.data.build_comparables import INDEX_FILE

PER_SHAPE = 100
ROUNDS = 3
WARM_UP = 30
RUNS = 3
SEED = 0

P1 = ("make", "model", "registration_date", "mileage_km_raw")

#: Builds the app in a fresh interpreter and prints its resident memory in KiB.
_RSS = """
import re, sys
from pathlib import Path
from recommenditos.api.app import create_app
app = create_app(Path(sys.argv[1]))
print(re.search(r"VmRSS:\\s+(\\d+)", Path("/proc/self/status").read_text()).group(1))
"""


def rss_mib(bundle: Path) -> float:
    run = subprocess.run(
        [sys.executable, "-c", _RSS, str(bundle)], capture_output=True, text=True, check=True
    )
    return int(run.stdout.split()[-1]) / 1024


def body(row: pd.Series, fields: tuple[str, ...]) -> dict:
    """`row` as a request with `fields`, the ones the listing leaves empty left out."""
    out = {}
    for name in fields:
        value = row[name]
        if pd.isna(value):
            continue
        if name == "registration_date":
            value = value.strftime("%Y-%m")
        elif isinstance(value, np.floating):
            value = float(value)
        out[name] = value
    return out


def memory() -> None:
    bundle = MODELS_DIR / "lgbm-basic"
    with tempfile.TemporaryDirectory() as scratch:
        alone = Path(scratch) / "lgbm-basic"
        shutil.copytree(bundle, alone)
        with_index = [rss_mib(bundle) for _ in range(RUNS)]
        without = [rss_mib(alone) for _ in range(RUNS)]
    size = (MODELS_DIR / INDEX_FILE).stat().st_size
    print(f"index on disk: {size:,} bytes ({size / 2**20:.1f} MiB)")
    print(f"app RSS with the index:    {', '.join(f'{v:.1f}' for v in with_index)} MiB")
    print(f"app RSS without the index: {', '.join(f'{v:.1f}' for v in without)} MiB")
    print(f"the index costs a worker:  {np.median(with_index) - np.median(without):.1f} MiB")


def latency() -> None:
    client = TestClient(create_app())
    served = client.app.state.served
    index = client.app.state.comparables
    frame = pd.read_parquet(PROCESSED_DATA_DIR / "test.parquet")
    frame = frame[frame["make"].isin(served.allowed["make"])]
    rows = frame.sample(PER_SHAPE * ROUNDS + WARM_UP, random_state=SEED)
    shapes = {
        "make only": (("make",), {}),
        "make+model": (("make", "model"), {}),
        "P1": (P1, {}),
        "P1, k 20": (P1, {"k": 20}),
    }

    for _, row in rows.head(WARM_UP).iterrows():
        client.post("/v1/comparables", json=body(row, P1))

    load_before = os.getloadavg()
    seconds: dict[str, list[float]] = {name: [] for name in shapes}
    found: dict[str, list[int]] = {name: [] for name in shapes}
    for _, row in rows.iloc[WARM_UP:].iterrows():
        for name, (fields, extra) in shapes.items():
            request = body(row, fields) | extra
            start = time.perf_counter()
            response = client.post("/v1/comparables", json=request)
            seconds[name].append(time.perf_counter() - start)
            assert response.status_code == HTTPStatus.OK, (request, response.text)
            found[name].append(len(response.json()["listings"]))
    load_after = os.getloadavg()

    print(f"index {index.version}, {len(index):,} listings; lgbm-basic {served.info.version}")
    print(f"in process, {os.cpu_count()} logical CPUs")
    print(f"load average before: {', '.join(f'{v:.2f}' for v in load_before)}")
    print(f"load average after:  {', '.join(f'{v:.2f}' for v in load_after)}")
    print()
    print(
        f"{'shape':<11} {'requests':>9} {'p50 ms':>8} {'p95 ms':>8} {'max ms':>8}"
        f" {'empty':>6} {'mean listings':>14}"
    )
    every = [value for values in seconds.values() for value in values]
    every_found = [value for values in found.values() for value in values]
    for name, values, listed in [
        *((name, seconds[name], found[name]) for name in shapes),
        ("all", every, every_found),
    ]:
        ms = np.asarray(values) * 1000
        counts = np.asarray(listed)
        print(
            f"{name:<11} {len(ms):>9} {np.percentile(ms, 50):>8.1f} {np.percentile(ms, 95):>8.1f}"
            f" {ms.max():>8.1f} {np.mean(counts == 0):>6.1%} {counts.mean():>14.1f}"
        )
    print()
    print("NFR-02: p95 at most 300 ms for /comparables on the target VM.")


if __name__ == "__main__":
    memory()
    print()
    latency()
