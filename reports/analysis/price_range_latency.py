"""The in-process latency of `/v1/price-range` on the committed `lgbm-basic` bundle (#139, NFR-02).

NFR-02 asks for a p95 of at most 300 ms on the target VM. This is the laptop figure, measured in
process with FastAPI's test client, so it holds the validation, the matrix, the point model and the
three quantile boosters, and no network, nginx or uvicorn. It is an early check, not the NFR-03 load
test.

Requests are built from test-split listings of the served makes, as a client would send them, in
three shapes: make only, P1 (make, model, registration date and mileage) and full (every field of
the basic feature set the listing has). `ROUNDS` rounds of `PER_SHAPE` listings each, the shapes
interleaved, after `WARM_UP` unmeasured requests. The load average is printed before and after,
because the laptop is shared.

Run: `uv run python reports/analysis/price_range_latency.py` gives
`price_range_latency_results.txt` (2026-10-20).
"""

import os
import time

from fastapi.testclient import TestClient
from loguru import logger
import numpy as np
import pandas as pd

from recommenditos.api.app import create_app
from recommenditos.config import PROCESSED_DATA_DIR

PER_SHAPE = 100
ROUNDS = 3
WARM_UP = 30
SEED = 0

P1 = ("make", "model", "registration_date", "mileage_km_raw")
FULL = (
    *P1,
    "power_kw",
    "fuel_category",
    "transmission",
    "country_code",
    "body_type",
    "seller_type",
    "nr_prev_owners",
    "drive_train",
    "gears",
    "cylinders_volume_cc",
    "nr_seats",
    "nr_doors",
)
_INTEGERS = {"nr_prev_owners", "gears", "nr_seats", "nr_doors"}


def body(row: pd.Series, fields: tuple[str, ...]) -> dict:
    """`row` as a request with `fields`, the ones the listing leaves empty left out."""
    out = {}
    for name in fields:
        value = row[name]
        if pd.isna(value):
            continue
        if name == "registration_date":
            value = value.strftime("%Y-%m")
        elif name in _INTEGERS:
            value = int(value)
        elif isinstance(value, np.floating):
            value = float(value)
        out[name] = value
    return out


def main():
    # The stages' log lines go to stderr, and none of them belongs in the results file.
    logger.remove()
    client = TestClient(create_app())
    served = client.app.state.served
    frame = pd.read_parquet(PROCESSED_DATA_DIR / "test.parquet")
    frame = frame[frame["make"].isin(served.allowed["make"])]
    rows = frame.sample(PER_SHAPE * ROUNDS + WARM_UP, random_state=SEED)
    shapes = {"make only": ("make",), "P1": P1, "full": FULL}

    for _, row in rows.head(WARM_UP).iterrows():
        client.post("/v1/price-range", json=body(row, FULL))

    load_before = os.getloadavg()
    seconds: dict[str, list[float]] = {name: [] for name in shapes}
    for _, row in rows.iloc[WARM_UP:].iterrows():
        for name, fields in shapes.items():
            request = body(row, fields)
            start = time.perf_counter()
            response = client.post("/v1/price-range", json=request)
            seconds[name].append(time.perf_counter() - start)
            assert response.status_code == 200, (request, response.text)
    load_after = os.getloadavg()

    print(f"lgbm-basic, {served.info.version}, in process, {os.cpu_count()} logical CPUs")
    print(f"load average before: {', '.join(f'{v:.2f}' for v in load_before)}")
    print(f"load average after:  {', '.join(f'{v:.2f}' for v in load_after)}")
    print()
    print(f"{'shape':<10} {'requests':>9} {'p50 ms':>8} {'p95 ms':>8} {'max ms':>8}")
    every = [value for values in seconds.values() for value in values]
    for name, values in [*seconds.items(), ("all", every)]:
        ms = np.asarray(values) * 1000
        print(
            f"{name:<10} {len(ms):>9} {np.percentile(ms, 50):>8.1f} {np.percentile(ms, 95):>8.1f}"
            f" {ms.max():>8.1f}"
        )
    print()
    print("NFR-02: p95 at most 300 ms for /price-range on the target VM.")


if __name__ == "__main__":
    main()
