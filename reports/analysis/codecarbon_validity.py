"""CodeCarbon on our machines: what its figure is, and what configuration measures a fit.

Issue #38 assumes CodeCarbon *measures* the energy of a fit. This script tests that
assumption on the machine the ladder is trained on before any number is quoted,
and it is the evidence behind EDN-69 and the settings in `params.yaml`'s
`train.energy`. Six questions:

1. Which CPU power method and TDP does CodeCarbon pick when RAPL is root-only, as it
   is on every Linux kernel since 5.10 (Platypus, CVE-2020-8694), and what does it
   say about it?
2. Does the figure follow the work, or is it wall time times a constant? Four
   workloads of different CPU load - an idle sleep, a single-threaded busy loop,
   a LightGBM fit at 1 thread and the same fit at 4 - in `process` and in
   `machine` mode, compared against CodeCarbon's own model:
   CPU = TDP / logical CPUs x process CPU time, RAM = 10 W x duration.
3. Does `OfflineEmissionsTracker` make any network call? Every socket connection
   and name lookup in this process is made to fail and recorded.
4. Does `measure_power_secs` change how completely a fit is integrated? 1 s, as
   the course demo uses, against CodeCarbon's default of 15 s, three repeats each.
5. What does the tracker cost on the 744-row fixture the test suite trains on?
   Each variant fitted bare and tracked, five repeats, plus the one-off cold start
   of the first tracker in a process.
6. What does `on_csv_write="append"` do to a file that is a DVC stage output, given
   that DVC deletes a stage's outputs before running it?

Run from the repository root, so that `recommenditos` is importable:

    uv run python reports/analysis/codecarbon_validity.py

It takes about ten minutes. The committed output was produced on a laptop that
other jobs were sharing - the 1-minute load average is printed beside every row -
and that is deliberate for question 2, whose `machine` rows are about exactly that.
"""

import logging
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time

ATTEMPTS: list[tuple[str, object]] = []


def _refuse_connect(self, address):
    ATTEMPTS.append(("connect", address))
    raise OSError("network blocked by codecarbon_validity.py")


def _refuse_lookup(host, *args, **kwargs):
    ATTEMPTS.append(("getaddrinfo", host))
    raise OSError("network blocked by codecarbon_validity.py")


# Before anything that could open a connection is imported.
socket.socket.connect = _refuse_connect
socket.getaddrinfo = _refuse_lookup

from codecarbon import OfflineEmissionsTracker, OutputMethod  # noqa: E402
import codecarbon  # noqa: E402
import lightgbm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from recommenditos.config import PARAMS_FILE  # noqa: E402
from recommenditos.data import build_features, preprocess, split_data  # noqa: E402
from recommenditos.data.build_features import read_supported_makes  # noqa: E402
from recommenditos.data.synthetic import generate_raw_listings  # noqa: E402
from recommenditos.modeling import train  # noqa: E402
from recommenditos.modeling.model import fit_variant  # noqa: E402
from recommenditos.pipeline import load_params  # noqa: E402

# The stages this calls log to stdout, which is where the results go.
from loguru import logger  # noqa: E402

logger.remove()

COUNTRY = "ESP"
SECONDS = 8.0
RAM_FLOOR_W = 10.0

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

rng = np.random.default_rng(0)
X = rng.normal(size=(60_000, 16))
y = X @ rng.normal(size=16) + rng.normal(size=60_000)


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        message = record.getMessage().strip().splitlines()[0]
        if message not in self.messages:
            self.messages.append(message)


def sleep():
    time.sleep(SECONDS)


def spin():
    end = time.perf_counter() + SECONDS
    while time.perf_counter() < end:
        pass


def lightgbm_fit(threads):
    def fit():
        lightgbm.LGBMRegressor(
            n_estimators=400, num_leaves=63, n_jobs=threads, verbose=-1, random_state=0
        ).fit(X, y)

    return fit


WORKLOADS = {
    "sleep": sleep,
    "busy loop, 1 thread": spin,
    "LightGBM, 1 thread": lightgbm_fit(1),
    "LightGBM, 4 threads": lightgbm_fit(4),
}


def tracked(fn, out: Path, *, mode="process", interval=15.0, name="probe"):
    tracker = OfflineEmissionsTracker(
        project_name=name,
        country_iso_code=COUNTRY,
        tracking_mode=mode,
        measure_power_secs=interval,
        output_dir=str(out),
        output_file="probe.csv",
        output_methods=[OutputMethod.CSV],
        on_csv_write="append",
        log_level="warning",
    )
    cpu0, wall0 = time.process_time(), time.perf_counter()
    with tracker:
        fn()
    wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
    frame = pd.read_csv(out / "probe.csv", dtype={"run_id": "str"})
    row = frame[frame["run_id"] == str(tracker.run_id)].iloc[0]
    hardware = next(h for h in tracker._hardware if type(h).__name__ == "CPU")
    return tracker, row, hardware, wall, cpu


def question_1_and_2(out: Path):
    print("== 1. What CodeCarbon picks without RAPL, and 2. whether it tracks the work ==\n")
    collect = _Collect()
    logging.getLogger("codecarbon").addHandler(collect)
    rows = []
    for mode in ("process", "machine"):
        for name, fn in WORKLOADS.items():
            load = os.getloadavg()[0]
            _, row, cpu_hw, wall, cpu = tracked(fn, out, mode=mode, name=name)
            tdp, n_cpus = cpu_hw._tdp, int(row["cpu_count"])
            rows.append(
                {
                    "mode": mode,
                    "workload": name,
                    "wall_s": round(wall, 2),
                    "cpu_s": round(cpu, 2),
                    "cpu_W": round(row["cpu_energy"] * 3.6e6 / row["duration"], 2),
                    "ram_W": round(row["ram_energy"] * 3.6e6 / row["duration"], 2),
                    "total_W": round(row["energy_consumed"] * 3.6e6 / row["duration"], 2),
                    "cpu_vs_model": round(
                        row["cpu_energy"] / (tdp / n_cpus * cpu / 3.6e6), 2
                    )
                    if cpu > 0.1
                    else None,
                    "g_per_kWh": round(row["emissions"] / row["energy_consumed"] * 1000, 2),
                    "load_1min": round(load, 1),
                }
            )
    logging.getLogger("codecarbon").removeHandler(collect)
    print(f"CodeCarbon {codecarbon.__version__}, CPU model {row['cpu_model']!r}, "
          f"{row['cpu_count']} logical CPUs, RAM {row['ram_total_size']:.1f} GB")
    print(f"CPU power method: {cpu_hw._mode}, TDP {cpu_hw._tdp} W "
          f"(generic fallback: {cpu_hw._is_generic_tdp})")
    print(f"RAPL energy_uj readable: {os.access('/sys/class/powercap/intel-rapl:0/energy_uj', os.R_OK)}")
    print("\nWhat CodeCarbon logs at WARNING and above:")
    for message in collect.messages:
        print(f"  - {message}")
    print("\nAverage power over each workload (energy / duration). `cpu_vs_model` is the CPU "
          "energy over TDP / logical CPUs x process CPU time:")
    print(pd.DataFrame(rows).to_string(index=False))
    print()


def question_3():
    print("== 3. Network calls made by the offline tracker ==\n")
    print(f"{len(ATTEMPTS)} attempt(s): {ATTEMPTS}\n")


def question_4(out: Path):
    print("== 4. measure_power_secs: how completely a fit is integrated ==\n")
    rows = []
    for interval in (1.0, 15.0):
        for name in ("busy loop, 1 thread", "LightGBM, 1 thread", "LightGBM, 4 threads"):
            for repeat in range(3):
                _, row, cpu_hw, _, cpu = tracked(
                    WORKLOADS[name], out, interval=interval, name=f"{name}-{interval}"
                )
                rows.append(
                    {
                        "measure_power_secs": interval,
                        "workload": name,
                        "repeat": repeat,
                        "duration_s": round(row["duration"], 2),
                        "cpu_vs_model": round(
                            row["cpu_energy"] / (cpu_hw._tdp / row["cpu_count"] * cpu / 3.6e6), 3
                        ),
                        "ram_vs_model": round(
                            row["ram_energy"] / (RAM_FLOOR_W * row["duration"] / 3.6e6), 3
                        ),
                        "total_W": round(row["energy_consumed"] * 3.6e6 / row["duration"], 2),
                    }
                )
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    print("\nRange per setting (min to max over the three repeats):")
    summary = frame.groupby(["measure_power_secs", "workload"]).agg(
        cpu_min=("cpu_vs_model", "min"),
        cpu_max=("cpu_vs_model", "max"),
        ram_min=("ram_vs_model", "min"),
        ram_max=("ram_vs_model", "max"),
        total_W_min=("total_W", "min"),
        total_W_max=("total_W", "max"),
    )
    print(summary.to_string())
    print()


def question_5(root: Path):
    print("== 5. Tracker overhead on the test fixture ==\n")
    raw = root / "raw" / "listings.parquet"
    raw.parent.mkdir(parents=True)
    generate_raw_listings(2000).to_parquet(raw, index=False)
    processed = root / "processed"
    preprocess.main(raw, root / "interim" / "listings.parquet", PARAMS_FILE)
    split_data.main(root / "interim" / "listings.parquet", processed, PARAMS_FILE)
    for feature_set in ("basic", "extended"):
        build_features.main(feature_set, processed, processed / "features", PARAMS_FILE)
    params = load_params(PARAMS_FILE)
    makes = read_supported_makes(processed)
    out = root / "emissions"
    out.mkdir()

    # A fresh interpreter, because the first tracker of a process pays for the
    # TDP registry and the hardware probe, and this one has already paid.
    cold = subprocess.run(
        [
            sys.executable,
            "-c",
            "import time, tempfile; t = time.perf_counter(); "
            "from codecarbon import OfflineEmissionsTracker; "
            "d = tempfile.mkdtemp(); "
            "tr = OfflineEmissionsTracker(country_iso_code='ESP', tracking_mode='process', "
            "output_dir=d, log_level='error'); tr.start(); tr.stop(); "
            "print(round(time.perf_counter() - t, 2))",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    print(f"First tracker in a fresh process, import included: {cold} s\n")

    rows = []
    for variant, settings in params["train"]["variants"].items():
        data = train.read_matrices(processed / "features", settings["feature_set"], makes)

        def fit(variant=variant, settings=settings, data=data):
            fit_variant(variant, settings, data, seed=params["seed"], num_threads=1)

        fit()
        for how, run in (("bare", fit), ("tracked", lambda: tracked(fit, out, name="fixture"))):
            times = []
            for _ in range(5):
                started = time.perf_counter()
                run()
                times.append(time.perf_counter() - started)
            rows.append(
                {
                    "variant": variant,
                    "n_train_rows": len(data.train),
                    "how": how,
                    "median_s": round(statistics.median(times), 3),
                    "min_s": round(min(times), 3),
                    "max_s": round(max(times), 3),
                }
            )
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    medians = frame.pivot(index="variant", columns="how", values="median_s")
    print("\nMedian cost of tracking one fit (tracked - bare), seconds:")
    print((medians["tracked"] - medians["bare"]).round(3).to_string())
    print()


def question_6(root: Path):
    print("== 6. on_csv_write='append' on a DVC stage output ==\n")
    dvc = shutil.which("dvc") or str(Path(sys.executable).parent / "dvc")
    repo = root / "dvc-append"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run([dvc, "init", "-q"], cwd=repo, check=True)
    (repo / "dvc.yaml").write_text(
        "stages:\n"
        "  declared:\n"
        "    cmd: python3 -c \"open('declared.csv', 'a').write('row\\\\n')\"\n"
        "    outs:\n"
        "      - declared.csv:\n"
        "          cache: false\n"
        "  persisted:\n"
        "    cmd: python3 -c \"open('persisted.csv', 'a').write('row\\\\n')\"\n"
        "    outs:\n"
        "      - persisted.csv:\n"
        "          cache: false\n"
        "          persist: true\n",
        encoding="utf-8",
    )
    for _ in range(3):
        subprocess.run([dvc, "repro", "--force", "-q"], cwd=repo, check=True, capture_output=True)
    for name in ("declared", "persisted"):
        lines = (repo / f"{name}.csv").read_text(encoding="utf-8").splitlines()
        print(f"after three `dvc repro --force`: {name}.csv holds {len(lines)} row(s)")
    print()


def main():
    print(f"python {sys.version.split()[0]}, codecarbon {codecarbon.__version__}, "
          f"lightgbm {lightgbm.__version__}, {time.strftime('%Y-%m-%d %H:%M')}\n")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        question_1_and_2(root)
        question_4(root)
        question_5(root / "fixture")
        question_6(root)
    question_3()


if __name__ == "__main__":
    main()
