"""Where the CPU time of a fit comes from although `train.num_threads` is 1 (issue #113).

The energy comparison of the committed runs shows more CPU time than wall time for
the measured fit of `b1`, `lgbm-basic` and `lgbm-extended`, and `train.num_threads`
is 1. This script attributes the CPU time inside the span `train` measures to the
threads that spent it and to the steps of the fit that ran while they did.

Each fit runs in a process of its own, because the thread pools of OpenBLAS and
OpenMP are sized when their libraries load, under five configurations:

- `stage`: the environment as `dvc repro` has it, CodeCarbon's tracker around the
  span exactly as `train` sets it up (`energy.measure`).
- `bare`: the same span without the tracker.
- `omp1`: `OMP_NUM_THREADS=1`, with the tracker.
- `blas1`: `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=1`, with the tracker.
- `all1`: both variables at 1, without the tracker.

Inside a process, the CPU time of every thread is read from
`/proc/self/task/<tid>/stat` before and after the span and around each step, so
this script runs on Linux only. A thread is named by what started it: the main
thread, a Python thread (CodeCarbon's sampler is one), or a native thread that
existed before the span (OpenBLAS starts its pool when numpy is imported) or that
started inside it (OpenMP starts its pool at the first parallel region). The
steps are `fit_variant`'s and `fit_with_intervals`'s: the blanked copies, the
estimator's fit (and for `b1`, inside it, the solver alone), the training and validation L1 the record carries (two
predictions over the training and validation rows), and for a variant with a
price range the quantile models and their calibration.

Nothing is written into the repository: the matrices are read from
`data/processed`, and CodeCarbon writes its CSV into a temporary directory.
Run from the repository root, with the matrices pulled:

    uv run python reports/analysis/fit_cpu_threads.py | tee reports/analysis/fit_cpu_threads_results.txt

It takes about 20 minutes, mostly `lgbm-basic`, whose span includes its quantile
models.
"""

from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time

# `b1` is fitted on OpenBLAS's Haswell kernel (`check_openblas_kernel`); numpy has to see this first.
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")

VARIANTS = ("b1", "lgbm-basic", "lgbm-extended")
BLAS_VARIABLES = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
CONFIGS = {
    "stage": ({}, True),
    "bare": ({}, False),
    "omp1": ({"OMP_NUM_THREADS": "1"}, True),
    "blas1": ({name: "1" for name in BLAS_VARIABLES}, True),
    "all1": ({"OMP_NUM_THREADS": "1", **{name: "1" for name in BLAS_VARIABLES}}, False),
}
TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def thread_cpu() -> dict[int, float]:
    """CPU seconds (user plus system) of every live thread of this process, by thread id."""
    seconds = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            stat = Path(f"/proc/self/task/{tid}/stat").read_text()
        except FileNotFoundError:  # the thread ended between listdir and the read
            continue
        fields = stat[stat.rindex(")") + 2 :].split()
        seconds[int(tid)] = (int(fields[11]) + int(fields[12])) / TICKS
    return seconds


def split_main(before: dict[int, float], after: dict[int, float], main: int) -> tuple[float, float]:
    """CPU seconds of the main thread and of every other live thread between two snapshots."""
    deltas = {tid: after[tid] - before.get(tid, 0.0) for tid in after}
    main_s = deltas.pop(main, 0.0)
    return main_s, sum(deltas.values())


def child(variant: str, config: str) -> None:
    from recommenditos.config import PARAMS_FILE, PROCESSED_DATA_DIR
    from recommenditos.data.build_features import read_supported_makes
    from recommenditos.modeling import intervals as intervals_module
    from recommenditos.modeling import model as model_module
    from recommenditos.modeling.energy import EnergySettings, measure
    from recommenditos.modeling.intervals import fit_with_intervals
    from recommenditos.modeling.model import ESTIMATORS, RequestContract, interval_settings
    from recommenditos.modeling.train import read_calibration, read_matrices
    from recommenditos.pipeline import load_params
    from sklearn.linear_model import Ridge
    from threadpoolctl import threadpool_info

    params = load_params(PARAMS_FILE)
    settings = params["train"]["variants"][variant]
    input_dir = PROCESSED_DATA_DIR / "features"
    supported = read_supported_makes(input_dir.parent)
    data = read_matrices(
        input_dir, settings["feature_set"], supported, request=RequestContract.from_params(params)
    )
    if interval_settings(variant, settings) is not None:
        data = replace(
            data, calibration=read_calibration(input_dir, settings["feature_set"], supported)
        )
    pools_before = threadpool_info()
    main = threading.main_thread().native_id

    # Each step's wall time and the CPU time of the main thread and the others while
    # it ran. Only the outermost step counts, so a step called inside another one is
    # not counted twice, except an `inner` one, which is reported as part of its step.
    steps: dict[str, list[float]] = {}
    depth = [0]

    def timed(name, function, *, inner=False):
        def wrapper(*args, **kwargs):
            if depth[0] and not inner:
                return function(*args, **kwargs)
            depth[0] += not inner
            wall, before = time.perf_counter(), thread_cpu()
            try:
                return function(*args, **kwargs)
            finally:
                main_s, others_s = split_main(before, thread_cpu(), main)
                row = steps.setdefault(name, [0.0, 0.0, 0.0])
                row[0] += time.perf_counter() - wall
                row[1] += main_s
                row[2] += others_s
                depth[0] -= not inner

        return wrapper

    model_module._with_blanked_copies = timed("blanked copies", model_module._with_blanked_copies)
    model_module._l1_log_price = timed("train and validation L1", model_module._l1_log_price)
    estimator = ESTIMATORS[settings["estimator"]]
    estimator.fit = classmethod(
        lambda cls, *args, _fit=timed("estimator fit", estimator.fit), **kwargs: _fit(*args, **kwargs)
    )
    intervals_module.fit_quantile_estimators = timed(
        "quantile models", intervals_module.fit_quantile_estimators
    )
    intervals_module.conformalize = timed("calibration", intervals_module.conformalize)
    # Inside `estimator fit` for `b1`: the solver alone, so the encoding and the
    # solver can be told apart.
    Ridge.fit = timed("  of which Ridge.fit (LSQR)", Ridge.fit, inner=True)

    span_threads: dict[int, str] = {}
    before_span: dict[int, float] = {}
    after_span: dict[int, float] = {}

    def fit():
        before_span.update(thread_cpu())
        result = fit_with_intervals(
            variant, settings, data, seed=params["seed"], num_threads=params["train"]["num_threads"]
        )
        after_span.update(thread_cpu())
        # Named while they are alive: CodeCarbon's sampler stops with the tracker.
        span_threads.update({t.native_id: t.name for t in threading.enumerate()})
        return result

    _, with_tracker = CONFIGS[config]
    if with_tracker:
        with tempfile.TemporaryDirectory() as directory:
            model, cost = measure(
                fit,
                name=variant,
                settings=EnergySettings.from_params(params["train"]["energy"]),
                output_file=Path(directory) / f"{variant}.csv",
            )
        wall_s, cpu_s = cost.fit_seconds, cost.fit_cpu_seconds
    else:
        cpu_started, wall_started = time.process_time(), time.perf_counter()
        model = fit()
        wall_s = time.perf_counter() - wall_started
        cpu_s = time.process_time() - cpu_started

    by_kind = {"main thread": 0.0}
    counts = {"main thread": 1}
    for tid, seconds in after_span.items():
        delta = seconds - before_span.get(tid, 0.0)
        if tid == main:
            kind = "main thread"
        elif tid in span_threads:
            kind = f"Python thread {span_threads[tid]!r}"
        elif tid in before_span:
            kind = "native threads started before the span"
        else:
            kind = "native threads started in the span"
        by_kind[kind] = by_kind.get(kind, 0.0) + delta
        counts[kind] = counts.get(kind, 0) + (kind != "main thread")
    # A thread that ended inside the span is in the process total and in no snapshot.
    by_kind["threads that ended in the span"] = cpu_s - sum(by_kind.values())

    print(
        json.dumps(
            {
                "variant": variant,
                "config": config,
                "wall_s": wall_s,
                "cpu_s": cpu_s,
                "threads": {k: [v, counts.get(k, 0)] for k, v in by_kind.items()},
                "steps": steps,
                # The two L1 the record carries, to show the threads change no number.
                "l1": [model.metadata["training"][f"{split}_l1_log_price"] for split in ("train", "validation")],
                "pools": [
                    {k: pool[k] for k in ("user_api", "internal_api", "prefix", "num_threads")}
                    for pool in threadpool_info()
                ],
                "pools_before_span": len(pools_before),
            }
        )
    )


def report(rows: list[dict]) -> None:
    print("\n## CPU over wall time of the measured span\n")
    print(f"{'variant':<14}" + "".join(f"{c:>18}" for c in CONFIGS))
    for variant in VARIANTS:
        cells = []
        for config in CONFIGS:
            row = next(r for r in rows if r["variant"] == variant and r["config"] == config)
            cells.append(f"{row['cpu_s']:6.2f}/{row['wall_s']:6.2f}={row['cpu_s'] / row['wall_s']:4.2f}")
        print(f"{variant:<14}" + "".join(f"{c:>18}" for c in cells))
    print("(each cell: CPU seconds / wall seconds = ratio)")

    for row in rows:
        print(f"\n## {row['variant']}, {row['config']}: {row['cpu_s']:.2f} s CPU in {row['wall_s']:.2f} s")
        print(f"train and validation L1 in the record: {row['l1'][0]!r}, {row['l1'][1]!r}")
        print("CPU seconds by thread:")
        for kind, (seconds, count) in row["threads"].items():
            number = f" ({count})" if count and kind != "main thread" else ""
            print(f"  {kind + number:<52} {seconds:8.2f}")
        print("CPU seconds by step (wall, main thread, other threads):")
        for name, (wall, main_s, others_s) in row["steps"].items():
            print(f"  {name:<28} wall {wall:8.2f}   main {main_s:8.2f}   others {others_s:8.2f}")
        if row["config"] == "stage":
            print("Thread pools after the span (threadpoolctl):")
            for pool in row["pools"]:
                print(
                    f"  {pool['internal_api']:<9} {pool['user_api']:<7} {pool['num_threads']:>2} threads"
                    f"  {pool['prefix']}"
                )


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
        return
    import psutil

    print(
        f"{datetime.now():%Y-%m-%d %H:%M}, {platform.processor() or platform.machine()}, "
        f"{psutil.cpu_count()} logical CPUs, Python {platform.python_version()}, "
        f"load {os.getloadavg()[0]:.2f}"
    )
    base = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", *BLAS_VARIABLES)}
    rows = []
    for variant in VARIANTS:
        for config, (extra, _) in CONFIGS.items():
            started = time.perf_counter()
            out = subprocess.run(
                [sys.executable, __file__, "--child", variant, config],
                env={**base, **extra},
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            rows.append(json.loads(out.strip().splitlines()[-1]))
            print(f"{variant} {config}: done in {time.perf_counter() - started:.0f} s", flush=True)
    report(rows)


if __name__ == "__main__":
    main()
