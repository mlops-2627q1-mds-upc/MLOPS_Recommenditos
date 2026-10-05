"""What Great Expectations' batch fingerprint costs `validate-data` (EDN-68).

Great Expectations hashes every in-memory batch into a fingerprint for the stored
result's markers. `validate_data._without_batch_fingerprint` switches that off.
This runs the real stage on the real snapshot, with the switch as committed
(`off`) or bypassed (`on`), against a context built in a temporary directory, so
the pipeline's own outputs are not touched. Run each mode in its own process
under `/usr/bin/time -v`, which reports the wall time and the peak RSS:

    /usr/bin/time -v uv run python reports/analysis/gx_fingerprint_cost.py off
    /usr/bin/time -v uv run python reports/analysis/gx_fingerprint_cost.py on

`gx_fingerprint_cost_results.txt` holds both runs' figures (2026-10-05).
"""

from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import time

from recommenditos.config import INTERIM_DATA_DIR, PARAMS_FILE, RAW_DATA_DIR
from recommenditos.data import gx_context_configuration, validate_data

mode = sys.argv[1]
if mode == "on":
    validate_data._without_batch_fingerprint = nullcontext
elif mode != "off":
    raise SystemExit("usage: gx_fingerprint_cost.py on|off")

with tempfile.TemporaryDirectory() as scratch:
    root = Path(scratch)
    gx_context_configuration.main(root / "gx", root / "results", root / "docs", PARAMS_FILE)
    started = time.perf_counter()
    validate_data.main(
        RAW_DATA_DIR / "listings.parquet",
        INTERIM_DATA_DIR / "listings.parquet",
        root / "summary.json",
        root / "gx",
    )
    print(f"fingerprint {mode}: validate_data.main took {time.perf_counter() - started:.1f} s")
