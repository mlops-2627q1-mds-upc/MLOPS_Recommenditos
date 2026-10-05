"""One-off: CodeCarbon's estimate of a fit against the RAPL energy counter (EDN-69).

The pipeline reports CodeCarbon's `cpu_load` estimate, because RAPL - the energy
counter in every Intel CPU since Sandy Bridge - is readable by root only since
Linux 5.10 (Platypus, CVE-2020-8694). `codecarbon_validity.py` showed what that
estimate is made of: 10 W of RAM times wall time, plus the CPU's TDP divided by its
logical CPUs times the process's CPU time. This script asks the question that
leaves open: how far is that from what the hardware counts, and does the ladder's
ranking - the thing the report concludes from - survive the difference?

It measures, around the same fit and in the same run:

- the RAPL counters themselves, read directly from sysfs: the package and its
  core, uncore and DRAM subzones, the platform (`psys`) zone where it exists, and
  the MMIO view of the package. Sampled twice a second so that a counter that
  wraps (at `max_energy_range_uj`, about 262 kJ here) is caught, and with an idle
  baseline before and after the fits so that a fit's *marginal* energy can be told
  from what the machine draws anyway;
- CodeCarbon's estimate exactly as the pipeline gets it without RAPL, by forcing
  its `cpu_load` mode (`force_mode_cpu_load=True`), with the same settings
  `params.yaml` gives the `train` stage;
- what CodeCarbon itself reports once RAPL is readable (`cc_default`), which is
  what the pipeline would switch to on a machine that grants the access
  permanently (EDN-69's rejected option B).

The four ladder variants are fitted three times each, round robin, on the feature
matrices in `data/processed/features`, with `train.num_threads` from params.yaml.
MLflow is not involved: the script calls `fit_variant` directly.

Because RAPL counts the whole package, the script refuses to run on a busy
machine: the 1-minute load average must be at most `MAX_LOAD_1MIN` and the whole
machine at most `MAX_BUSY_SHARE` busy over a 10-second sample, and it says which
processes were busy when it refuses. Every fit also records how many CPU-seconds
other processes used while it ran, so a row disturbed after the check is visible.

Run it from the repository root on an otherwise idle laptop, after granting read
access to the counters for the duration of the run (a reboot also revokes it):

    sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj \\
        /sys/class/powercap/intel-rapl-mmio:*/energy_uj
    MLFLOW_TRACKING_URI= uv run python reports/analysis/rapl_validation.py \\
        > reports/analysis/rapl_validation_results.txt
    sudo chmod 0400 /sys/class/powercap/intel-rapl:*/energy_uj \\
        /sys/class/powercap/intel-rapl-mmio:*/energy_uj

It takes about four minutes. `--dry-run` skips both checks and replaces the sysfs
tree with a synthetic counter that draws 5 W and wraps every 10 J, which exercises
the whole script, wrap handling included, without any permission; its output is
labelled as such and is not evidence.
"""

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

import codecarbon
from codecarbon import OfflineEmissionsTracker, OutputMethod

# The stages this calls log to stdout, which is where the results go.
from loguru import logger
import pandas as pd
import psutil

from recommenditos.config import PARAMS_FILE, PROCESSED_DATA_DIR
from recommenditos.data.build_features import read_supported_makes
from recommenditos.modeling import train
from recommenditos.modeling.energy import EnergySettings, read_record
from recommenditos.modeling.model import fit_variant
from recommenditos.pipeline import load_params

logger.remove()

POWERCAP = Path("/sys/class/powercap")
ZONE_PATTERNS = ("intel-rapl:*", "intel-rapl-mmio:*")

#: How idle the machine has to be. A laptop with a desktop session and nothing
#: else sits well below both; a single busy thread on 8 logical CPUs is 12.5 %.
MAX_LOAD_1MIN = 1.5
MAX_BUSY_SHARE = 0.08
IDLE_SAMPLE_S = 10.0

BASELINE_S = 30.0
REPEATS = 3
SAMPLE_EVERY_S = 0.5
J_PER_KWH = 3.6e6
UJ_PER_J = 1e6

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


# --------------------------------------------------------------------------
# The counters
# --------------------------------------------------------------------------


def zones(root: Path) -> dict[str, Path]:
    """Every RAPL zone under `root`, keyed `<directory>:<name>`, e.g. `intel-rapl:0:package-0`."""
    found = {}
    for pattern in ZONE_PATTERNS:
        for directory in sorted(root.glob(pattern)):
            name = (directory / "name").read_text().strip()
            found[f"{directory.name}:{name}"] = directory
    return found


def readable(directory: Path) -> bool:
    try:
        int((directory / "energy_uj").read_text())
    except (PermissionError, OSError, ValueError):
        return False
    return True


class Counters:
    """The energy of every readable zone over a window, wrap-safe.

    Each zone's counter is read every `SAMPLE_EVERY_S` and the positive deltas
    are summed; a reading below the previous one is a wrap, and the delta is
    taken through `max_energy_range_uj`. Sampling, rather than reading only at
    the two ends, is what makes a window longer than one wrap period safe.
    """

    def __init__(self, directories: dict[str, Path]):
        self.directories = directories
        self.ranges = {
            key: int((path / "max_energy_range_uj").read_text())
            for key, path in directories.items()
        }
        self.wraps = dict.fromkeys(directories, 0)
        self._stop = threading.Event()

    def _read(self) -> dict[str, int]:
        return {
            key: int((path / "energy_uj").read_text()) for key, path in self.directories.items()
        }

    def _accumulate(self, now: dict[str, int]) -> None:
        for key, value in now.items():
            delta = value - self._last[key]
            if delta < 0:
                delta += self.ranges[key]
                self.wraps[key] += 1
            self.total_uj[key] += delta
        self._last = now

    def _run(self) -> None:
        while not self._stop.wait(SAMPLE_EVERY_S):
            self._accumulate(self._read())

    def __enter__(self) -> "Counters":
        self.total_uj = dict.fromkeys(self.directories, 0)
        self._last = self._read()
        self.started = time.perf_counter()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self._accumulate(self._read())
        self.seconds = time.perf_counter() - self.started

    def joules(self) -> dict[str, float]:
        return {key: value / UJ_PER_J for key, value in self.total_uj.items()}


@contextmanager
def synthetic_powercap(watts: float = 5.0, wrap_j: float = 10.0):
    """A fake sysfs tree whose package counter rises at `watts` and wraps at `wrap_j`."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        zone = root / "intel-rapl:0"
        zone.mkdir()
        (zone / "name").write_text("package-0\n")
        (zone / "max_energy_range_uj").write_text(f"{int(wrap_j * UJ_PER_J)}\n")
        (zone / "energy_uj").write_text("0\n")
        stop = threading.Event()

        def tick():
            started = time.perf_counter()
            while not stop.wait(0.05):
                value = int((time.perf_counter() - started) * watts * UJ_PER_J) % int(
                    wrap_j * UJ_PER_J
                )
                # Atomically, so a reader never sees a half-written file.
                (zone / "energy_uj.tmp").write_text(f"{value}\n")
                os.replace(zone / "energy_uj.tmp", zone / "energy_uj")

        thread = threading.Thread(target=tick, daemon=True)
        thread.start()
        try:
            yield root
        finally:
            stop.set()
            thread.join()


# --------------------------------------------------------------------------
# The two checks
# --------------------------------------------------------------------------


def check_idle() -> list[str]:
    """Why the machine is too busy to measure on, or an empty list."""
    problems = []
    load = os.getloadavg()[0]
    if load > MAX_LOAD_1MIN:
        problems.append(f"the 1-minute load average is {load:.2f}, above {MAX_LOAD_1MIN}")
    processes = list(psutil.process_iter(["pid", "name", "cmdline"]))
    for process in processes:
        try:
            process.cpu_percent(None)
        except psutil.Error:
            pass
    busy = psutil.cpu_percent(interval=IDLE_SAMPLE_S) / 100
    if busy > MAX_BUSY_SHARE:
        problems.append(
            f"the machine was {busy:.0%} busy over {IDLE_SAMPLE_S:.0f} s, above {MAX_BUSY_SHARE:.0%}"
        )
    if problems:
        heavy = []
        for process in processes:
            try:
                share = process.cpu_percent(None)
            except psutil.Error:
                continue
            if share >= 5 and process.pid != os.getpid():
                command = " ".join(process.info["cmdline"] or [process.info["name"] or "?"])
                heavy.append((share, process.pid, command[:120]))
        for share, pid, command in sorted(heavy, reverse=True)[:10]:
            problems.append(f"  busy: {share:5.1f} % of one CPU, pid {pid}: {command}")
    return problems


def chmod_hint() -> str:
    return (
        "grant read access for this run with\n"
        "    sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj "
        "/sys/class/powercap/intel-rapl-mmio:*/energy_uj\n"
        "and revoke it afterwards with the same command and `0400` instead of `a+r` "
        "(a reboot also resets it)"
    )


# --------------------------------------------------------------------------
# One measured fit
# --------------------------------------------------------------------------


def tracker(settings: EnergySettings, out: Path, name: str, *, estimate: bool):
    """The pipeline's tracker, with CodeCarbon's `cpu_load` estimate forced or not.

    Not forced, CodeCarbon picks what it would pick in the pipeline on this
    machine: RAPL while the counters are readable, `cpu_load` otherwise.
    """
    return OfflineEmissionsTracker(
        project_name=name,
        country_iso_code=settings.country_iso_code,
        tracking_mode=settings.tracking_mode,
        measure_power_secs=settings.measure_power_secs,
        output_dir=str(out),
        output_file=f"{'estimate' if estimate else 'default'}.csv",
        output_methods=[OutputMethod.CSV],
        on_csv_write="append",
        force_mode_cpu_load=estimate,
        force_cpu_power=None,
        force_ram_power=None,
        pue=1.0,
        allow_multiple_runs=True,
        log_level="error",
    )


def cpu_method(tracker_: OfflineEmissionsTracker) -> str:
    for hardware in tracker_._hardware:
        if type(hardware).__name__ == "CPU":
            return hardware._mode
    return "unknown"


def machine_cpu_seconds() -> float:
    """CPU-seconds the whole machine has spent, so a window's share of it can be taken."""
    times = psutil.cpu_times()
    return times.user + times.nice + times.system + times.irq + times.softirq + times.steal


def measured_fit(fit, directories, settings, out: Path, name: str) -> dict:
    estimate = tracker(settings, out, name, estimate=True)
    default = tracker(settings, out, name, estimate=False)
    with estimate, default, Counters(directories) as rapl:
        machine0, cpu0 = machine_cpu_seconds(), time.process_time()
        fit()
        cpu = time.process_time() - cpu0
        machine = machine_cpu_seconds() - machine0
    row = {"fit_s": rapl.seconds, "cpu_s": cpu, "others_cpu_s": max(machine - cpu, 0.0)}
    for label, used, file in (
        ("cc_estimate", estimate, "estimate.csv"),
        ("cc_default", default, "default.csv"),
    ):
        record = read_record(out / file, str(used.run_id))
        row[f"{label}_method"] = cpu_method(used)
        row[f"{label}_cpu_J"] = record["cpu_energy"] * J_PER_KWH
        row[f"{label}_ram_J"] = record["ram_energy"] * J_PER_KWH
        row[f"{label}_total_J"] = record["energy_consumed"] * J_PER_KWH
    for key, value in rapl.joules().items():
        row[f"rapl {key} J"] = value
    row["wraps"] = sum(rapl.wraps.values())
    return row


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def baseline(directories) -> dict[str, float]:
    with Counters(directories) as rapl:
        time.sleep(BASELINE_S)
    return {key: value / rapl.seconds for key, value in rapl.joules().items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--variants", nargs="*")
    args = parser.parse_args()
    global BASELINE_S, REPEATS
    if args.dry_run:
        BASELINE_S, REPEATS = 3.0, 1
        print("DRY RUN: synthetic counter, no idle check. Not evidence.\n")
    else:
        problems = check_idle()
        if problems:
            print(
                "Refusing to measure: RAPL counts the whole package, so other work would be "
                "charged to the fits.",
                file=sys.stderr,
            )
            print("\n".join(problems), file=sys.stderr)
            sys.exit(3)

    with synthetic_powercap() if args.dry_run else _real_powercap() as root:
        found = zones(root)
        usable = {key: path for key, path in found.items() if readable(path)}
        if not any(key.endswith("package-0") for key in usable):
            print(
                f"No RAPL package counter is readable under {root}; {chmod_hint()}.",
                file=sys.stderr,
            )
            sys.exit(2)
        run(usable, sorted(set(found) - set(usable)), args)


@contextmanager
def _real_powercap():
    yield POWERCAP


def run(directories: dict[str, Path], unreadable: list[str], args) -> None:
    params = load_params(PARAMS_FILE)
    settings = EnergySettings.from_params(params["train"]["energy"])
    variants = {
        name: spec
        for name, spec in params["train"]["variants"].items()
        if not args.variants or name in args.variants
    }
    features = PROCESSED_DATA_DIR / "features"
    makes = read_supported_makes(PROCESSED_DATA_DIR)
    data = {
        name: train.read_matrices(features, spec["feature_set"], makes)
        for name, spec in variants.items()
    }

    print(
        f"{time.strftime('%Y-%m-%d %H:%M')}, codecarbon {codecarbon.__version__}, "
        f"{psutil.cpu_count()} logical CPUs, train.num_threads {params['train']['num_threads']}, "
        f"load {os.getloadavg()[0]:.2f}"
    )
    print(f"Readable zones: {', '.join(directories)}")
    if unreadable:
        print(f"Unreadable zones, left out: {', '.join(unreadable)}")
    print()

    idle_before = baseline(directories)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        # CodeCarbon's first tracker in a process pays for its hardware probe;
        # paid here, outside any measured window.
        with tracker(settings, out, "warm-up", estimate=True):
            pass
        for repeat in range(REPEATS):
            for name, spec in variants.items():

                def fit(name=name, spec=spec):
                    fit_variant(
                        name,
                        spec,
                        data[name],
                        seed=params["seed"],
                        num_threads=params["train"]["num_threads"],
                    )

                row = {
                    "variant": name,
                    "repeat": repeat,
                    **measured_fit(fit, directories, settings, out, name),
                }
                rows.append(row)
                print(f"  {name} #{repeat}: {row['fit_s']:.2f} s", file=sys.stderr, flush=True)
    idle_after = baseline(directories)

    report(pd.DataFrame(rows), idle_before, idle_after)


def report(frame: pd.DataFrame, idle_before: dict, idle_after: dict) -> None:
    package = next(
        column
        for column in frame.columns
        if column.endswith("package-0 J") and column.startswith("rapl intel-rapl:")
    )
    package_key = package.removeprefix("rapl ").removesuffix(" J")
    print("== Idle draw, W (30 s with nothing running, before and after the fits) ==\n")
    print(pd.DataFrame({"before": idle_before, "after": idle_after}).round(3).to_string())
    idle_w = (idle_before[package_key] + idle_after[package_key]) / 2
    print(f"\nPackage idle draw used for the marginal energy: {idle_w:.3f} W\n")

    frame["rapl_package_J"] = frame[package]
    frame["rapl_package_marginal_J"] = frame[package] - idle_w * frame["fit_s"]
    frame["estimate_cpu_vs_marginal"] = (
        frame["cc_estimate_cpu_J"] / frame["rapl_package_marginal_J"]
    )
    frame["estimate_total_vs_package"] = frame["cc_estimate_total_J"] / frame["rapl_package_J"]

    print("== Every fit ==\n")
    print(frame.round(3).to_string(index=False))

    columns = [
        "fit_s",
        "cpu_s",
        "others_cpu_s",
        "rapl_package_J",
        "rapl_package_marginal_J",
        "cc_estimate_cpu_J",
        "cc_estimate_total_J",
        "cc_default_cpu_J",
        "cc_default_total_J",
        "estimate_cpu_vs_marginal",
        "estimate_total_vs_package",
    ]
    medians = frame.groupby("variant", sort=False)[columns].median()
    print("\n== Median per variant ==\n")
    print(medians.round(3).to_string())

    reference = "lgbm-basic" if "lgbm-basic" in medians.index else medians.index[0]
    ratios = medians[
        [
            "fit_s",
            "rapl_package_J",
            "rapl_package_marginal_J",
            "cc_estimate_total_J",
            "cc_default_total_J",
        ]
    ].div(medians.loc[reference])
    print(f"\n== Each variant relative to {reference}: does the ranking survive? ==\n")
    print(ratios.round(3).to_string())
    print(
        f"\nMethods: estimate {sorted(frame['cc_estimate_method'].unique())}, "
        f"CodeCarbon's own choice {sorted(frame['cc_default_method'].unique())}; "
        f"counter wraps seen: {int(frame['wraps'].sum())}; "
        f"largest CPU time of other processes during a fit: {frame['others_cpu_s'].max():.2f} s"
    )
    print(
        "Spread over repeats (max/min of the package energy per variant): "
        + ", ".join(
            f"{name} {group['rapl_package_J'].max() / group['rapl_package_J'].min():.3f}"
            for name, group in frame.groupby("variant", sort=False)
        )
    )


if __name__ == "__main__":
    main()
