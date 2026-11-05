# ruff: noqa
# fmt: off
# Kept as it ran on the VM on 2026-11-05 (these five header lines are the only addition): its
# output is pooled-es-windows.json via pair_runs.py, so lint and format fixes would make the
# committed script differ from the one that produced the evidence.
# Pooled MdAPE and coverage over several windows, with the drift job's own label join and formulas.
# Run inside recommenditos-monitor-1: python - <ids...> < pooled.py ; prints JSON with per-request rows.
import json, os, statistics, sys
from pathlib import Path
from recommenditos_monitoring.store import Store
from recommenditos_monitoring.log import connect_read_only, labels
from recommenditos_monitoring.performance import absolute_percentage_error
ids = [int(a) for a in sys.argv[1:]]
store = Store.open(Path(os.environ["RECOMMENDITOS_MONITOR_DB"]))
log = connect_read_only(Path(os.environ["RECOMMENDITOS_PREDICTION_LOG"]))
windows = {w.window_id: w for w in store.windows()}
rows, per = [], {}
for i in ids:
    w = windows[i]; held = store.requests(i)
    found = labels(log, [r.request_id for r in held], w.first_seq, w.last_seq)
    ape = [absolute_percentage_error(l) for l in found]
    rng = [l.lower_eur <= l.observed_price_eur <= l.upper_eur for l in found if l.lower_eur is not None and l.upper_eur is not None]
    per[i] = {"requests": len(held), "labelled": len(found), "mdape": statistics.median(ape), "ranges": len(rng), "coverage": sum(rng) / len(rng), "model": w.model_version}
    rows += [{"window": i, "observed": l.observed_price_eur, "price": l.price_eur, "lower": l.lower_eur, "upper": l.upper_eur, "ape": a} for l, a in zip(found, ape)]
ape = [r["ape"] for r in rows]; rng = [r["lower"] <= r["observed"] <= r["upper"] for r in rows if r["lower"] is not None]
print(json.dumps({"windows": per, "pooled": {"labelled": len(ape), "mdape": statistics.median(ape), "ranges": len(rng), "coverage": sum(rng) / len(rng)}, "rows": rows}))
