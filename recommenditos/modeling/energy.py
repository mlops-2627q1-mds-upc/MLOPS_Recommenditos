"""The energy and emissions of one fit, measured with CodeCarbon (NFR-10, issue #38).

`train` calls `measure` around `fit_variant` and around nothing else, so the
figure covers the fit and not the reading of the matrices, the MLflow round trips
or the writing of the bundle. The course demo does the same; what this module
adds is everything the demo leaves to chance, each of which was measured on our
machines before it was written down (EDN-69, `reports/analysis/codecarbon_validity.py`).

**What the number is.** CodeCarbon prefers a hardware energy counter, and on
Linux that is RAPL. Since kernel 5.10 RAPL is readable by root only, because
unprivileged power readings are a side channel (Platypus, CVE-2020-8694), so on a
contributor's laptop and on a GitHub runner CodeCarbon falls back to its
`cpu_load` model: the CPU draws the chip's TDP, looked up in CodeCarbon's own
table, times the share of the machine's logical CPUs this process kept busy, and
the RAM draws a constant 10 W, which is its floor of two DIMMs at 5 W for any x86
machine. The energy of a fit is therefore a function of two times, not a reading
of a meter:

    energy = 10 W x wall time + TDP / logical CPUs x process CPU time

On the i5-10210U this was measured on (TDP 15 W, 8 logical CPUs), a LightGBM fit
at `train.num_threads: 1` comes out at 11.75 to 11.80 W for every second it runs
over repeated fits, and a process that only sleeps at 10.15 W. On a GitHub runner
the same kind of fit is charged 79.8 W per second, because CodeCarbon divides the
280 W TDP of a 64-core AMD EPYC 7763 by the four vCPUs the VM sees. So the energy
of a fit is its duration times a machine-specific constant, and a figure is only
comparable with another from the same machine under the same load.

That is why `fit_cpu_seconds` is recorded beside the energy: it is a
measurement, and it is the quantity the CPU half is proportional to. Which method
produced a figure is logged with it (`cpu_power_method`), so a run on a machine
that does expose RAPL cannot be compared with one that does not without anyone
noticing.

**Settings, all from `params.yaml` (`train.energy`).**

- `tracking_mode: process` rather than `machine`. In `machine` mode the CPU half
  is the load of the whole machine, so a fit is charged for whatever else runs:
  measured while other work shared the laptop, a `time.sleep` was charged 4.5 W of
  CPU in `machine` mode and 0.3 W in `process` mode.
- `OfflineEmissionsTracker` with a pinned `country_iso_code`, never the online
  tracker. Per its source, the online one geolocates the machine through an HTTP
  call to geojs, writes its latitude and longitude into the CSV, and queries the
  AWS, Azure and GCP metadata endpoint, switching to the provider's regional
  factor when one answers - which on a GitHub runner, an Azure VM, one would. So
  the emissions of an identical fit would depend on where it happened to run, a
  git-tracked file would carry a contributor's location, and the test suite would
  make network calls. The offline tracker made none, measured by failing every
  socket connection in the process.
- `measure_power_secs: 15`, CodeCarbon's default, rather than the demo's 1. The
  power is sampled every second by a separate thread either way; this is how
  often the samples are integrated into energy, and at 1 s that integration races
  the thread that stops the tracker. Measured over three repeats each: at 15 s a
  single-threaded LightGBM fit integrates to 0.987x to 0.993x of CodeCarbon's own
  model, at 1 s the same fit ranged from 0.86x to 1.07x of it and a busy loop's
  RAM half from 0.71x to 4.31x, an interval counted several times over.

Every other argument is passed explicitly. CodeCarbon reads `~/.codecarbon.config`,
`./.codecarbon.config` and any `CODECARBON_*` environment variable for every
argument it is not given, so a contributor's own configuration could otherwise
switch on its API output, which is a network call, or force a CPU power, which
silently changes what the figure means.

**Failures are loud here, because they are silent there.** CodeCarbon wraps its
constructor, `start` and `stop` in a handler that logs a warning and carries on:
an unknown keyword leaves a half-built tracker, and an unknown country is
measured against the world average. So `EnergySettings` checks the country
against CodeCarbon's own table, and `measure` reads its row back by run id and
raises if it is not there. NFR-10 asks that every training run records its
emissions, and a stage that trained without recording them has not done that.

**The CSV.** `reports/emissions/<variant>.csv`, CodeCarbon's own format, appended
to as the demo does (`on_csv_write="append"`). Under `dvc repro` the stage's
outputs are deleted before it runs, so the file holds exactly the one row of the
fit that produced the model; a run outside DVC - a test, a variant retrained by
hand - appends, which is why the row is found by run id rather than by position.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import time

from codecarbon import OfflineEmissionsTracker, OutputMethod
from codecarbon.input import DataSource
from loguru import logger
import pandas as pd

from recommenditos.config import REPORTS_DIR

#: Where `train` writes CodeCarbon's record of each fit, one CSV per variant.
#: Defined here rather than in `config.py`, which every stage depends on, so that
#: only the two stages that read the record rerun if it ever moves.
EMISSIONS_DIR = REPORTS_DIR / "emissions"

#: The keys of `train.energy` in params.yaml, which are exactly the ones read.
SETTINGS_KEYS: tuple[str, ...] = ("tracking_mode", "country_iso_code", "measure_power_secs")

#: The two modes CodeCarbon implements. `process` is the one params.yaml sets.
TRACKING_MODES: frozenset[str] = frozenset({"process", "machine"})

#: The CSV columns this module reads back, by name. CodeCarbon writes about 40;
#: these are the ones a run is logged with, and a release that renames one fails
#: the read-back naming it rather than logging a number from the wrong column -
#: which is what the demo's positional `iloc[-1, 4:13]` would do.
RECORD_COLUMNS: tuple[str, ...] = (
    "run_id",
    "project_name",
    "duration",
    "emissions",
    "energy_consumed",
    "cpu_energy",
    "ram_energy",
    "cpu_power",
    "ram_power",
    "country_iso_code",
    "tracking_mode",
    "codecarbon_version",
    "cpu_model",
    "cpu_count",
    "ram_total_size",
    "cpu_utilization_percent",
)

#: The CPU power methods that estimate from the chip's TDP rather than read a
#: counter. On Linux without root, `cpu_load` is what CodeCarbon falls back to.
_ESTIMATED_FROM_TDP: frozenset[str] = frozenset({"cpu_load", "constant"})

_WH_PER_KWH = 1000.0


class EnergyError(RuntimeError):
    """The fit was not measured as configured, so there is no figure to record."""


@dataclass(frozen=True)
class EnergySettings:
    """How a fit is measured: the `train.energy` block of params.yaml."""

    tracking_mode: str
    country_iso_code: str
    measure_power_secs: float

    @classmethod
    def from_params(cls, block: dict) -> "EnergySettings":
        """The block, checked in both directions and against CodeCarbon's own data.

        The country in particular, because CodeCarbon measures an unknown one
        against the world average with nothing but a log line, which would put a
        plausible-looking emissions figure on every run.
        """
        given = dict(block)
        unknown = sorted(set(given) - set(SETTINGS_KEYS))
        missing = sorted(set(SETTINGS_KEYS) - set(given))
        if unknown or missing:
            raise EnergyError(
                f"train.energy has to set exactly {', '.join(SETTINGS_KEYS)}"
                + (f"; it does not set {', '.join(missing)}" if missing else "")
                + (f"; it sets {', '.join(unknown)}, which nothing reads" if unknown else "")
                + "."
            )
        if given["tracking_mode"] not in TRACKING_MODES:
            raise EnergyError(
                f"train.energy.tracking_mode is {given['tracking_mode']!r}; CodeCarbon "
                f"implements {', '.join(sorted(TRACKING_MODES))}."
            )
        country = str(given["country_iso_code"])
        if country not in DataSource().get_global_energy_mix_data():
            raise EnergyError(
                f"train.energy.country_iso_code is {country!r}, which CodeCarbon has no "
                f"carbon intensity for. It would measure the fit against the world average "
                f"and say so only in a log line; use a three-letter ISO code it knows."
            )
        seconds = float(given["measure_power_secs"])
        if seconds <= 0:
            raise EnergyError(f"train.energy.measure_power_secs must be positive, not {seconds}.")
        return cls(
            tracking_mode=given["tracking_mode"],
            country_iso_code=country,
            measure_power_secs=seconds,
        )


@dataclass(frozen=True)
class FitMeasurement:
    """What one fit cost: the two times taken here, and CodeCarbon's row for it.

    `fit_seconds` and `fit_cpu_seconds` bracket the fit and nothing else, so
    CodeCarbon's own `duration`, which also covers its start and stop, is a few
    hundredths of a second longer. `record` is the row exactly as CodeCarbon
    wrote it, so what MLflow receives and what the CSV holds cannot disagree.
    """

    fit_seconds: float
    fit_cpu_seconds: float
    record: dict
    cpu_power_method: str
    #: The TDP the estimate assumed, for the two methods that estimate from one,
    #: and `None` for a reading, where a TDP plays no part in the number.
    cpu_tdp_w: float | None

    @property
    def run_id(self) -> str:
        return str(self.record["run_id"])

    def metrics(self) -> dict[str, float]:
        """The figures of the fit, in CodeCarbon's units, for `Run.log_metrics`."""
        return {
            "fit_seconds": self.fit_seconds,
            "fit_cpu_seconds": self.fit_cpu_seconds,
            "energy_kwh": float(self.record["energy_consumed"]),
            "cpu_energy_kwh": float(self.record["cpu_energy"]),
            "ram_energy_kwh": float(self.record["ram_energy"]),
            "emissions_kg_co2eq": float(self.record["emissions"]),
            "cpu_power_w": float(self.record["cpu_power"]),
            "ram_power_w": float(self.record["ram_power"]),
            # The whole machine's CPU load while the fit ran, as CodeCarbon sampled
            # it. Not part of the figure in `process` mode, and recorded because it
            # is what the figure depends on anyway: the RAM half is 10 W times wall
            # time, and a fit sharing the machine takes longer for the same work.
            "machine_cpu_utilization_pct": float(self.record["cpu_utilization_percent"]),
        }

    def params(self) -> dict[str, str | float | int]:
        """How the figures were produced, beyond the settings, for `Run.log_params`.

        Parameters rather than tags because they are inputs to the number, the way
        a hyperparameter is: the same fit measured by `intel_rapl` and by
        `cpu_load` gives two figures that are not comparable, and MLflow's compare
        view shows parameters beside metrics. The settings themselves are
        params.yaml keys and `train` logs them with the others it declares, so
        this is only what no key decides: the library and the machine.
        """
        return {
            "codecarbon.version": str(self.record["codecarbon_version"]),
            "codecarbon.cpu_power_method": self.cpu_power_method,
            "codecarbon.cpu_tdp_w": "none" if self.cpu_tdp_w is None else self.cpu_tdp_w,
            "codecarbon.cpu_model": str(self.record["cpu_model"]),
            "codecarbon.cpu_count": int(self.record["cpu_count"]),
            "codecarbon.ram_total_size_gb": round(float(self.record["ram_total_size"]), 1),
        }

    def summary(self) -> str:
        """One log line saying what was measured and how."""
        return (
            f"{float(self.record['energy_consumed']) * _WH_PER_KWH:.4f} Wh, "
            f"{float(self.record['emissions']) * 1e6:.3f} mg CO2eq over a "
            f"{self.fit_seconds:.2f} s fit ({self.fit_cpu_seconds:.2f} CPU-s), estimated by "
            f"CodeCarbon {self.record['codecarbon_version']} as {self.cpu_power_method}"
            + ("" if self.cpu_tdp_w is None else f" at a TDP of {self.cpu_tdp_w:g} W")
            + f" for {self.record['cpu_model']} in {self.record['tracking_mode']} mode, "
            f"against the {self.record['country_iso_code']} grid"
        )


def measure[T](
    fit: Callable[[], T], *, name: str, settings: EnergySettings, output_file: Path
) -> tuple[T, FitMeasurement]:
    """Call `fit`, measuring its energy, and return its result with the measurement.

    A callable rather than a context manager, so that what is measured is a
    single expression at the call site and nothing can drift into the measured
    section by being written on the line below it.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tracker = _tracker(name, settings, output_file)
    with tracker:
        cpu_started, wall_started = time.process_time(), time.perf_counter()
        result = fit()
        fit_seconds = time.perf_counter() - wall_started
        fit_cpu_seconds = time.process_time() - cpu_started
    method, tdp = _cpu_power_method(tracker)
    record = read_record(output_file, str(tracker.run_id))
    _check_the_record_obeyed(record, settings)
    measurement = FitMeasurement(
        fit_seconds=fit_seconds,
        fit_cpu_seconds=fit_cpu_seconds,
        record=record,
        cpu_power_method=method,
        cpu_tdp_w=tdp,
    )
    logger.info(f"{name}: {measurement.summary()}.")
    return result, measurement


def _check_the_record_obeyed(record: dict, settings: EnergySettings) -> None:
    """Refuse a row that was measured some other way than params.yaml says.

    Every argument is passed explicitly, so this cannot happen through a
    contributor's CodeCarbon configuration; it is the check that it does not,
    and the reason the run logs the settings once, from params.yaml, rather than
    a second copy from CodeCarbon that could disagree with them.
    """
    stated = {
        "tracking_mode": settings.tracking_mode,
        "country_iso_code": settings.country_iso_code,
    }
    recorded = {key: record[key] for key in stated}
    if recorded != stated:
        raise EnergyError(
            f"CodeCarbon recorded {recorded} for a fit params.yaml configured as {stated}, so "
            f"the figure is not the measurement train.energy describes."
        )


def read_record(path: Path, run_id: str) -> dict:
    """The row CodeCarbon wrote for `run_id`, by column name, as plain Python values.

    The energy comparison stage reads its records through this too, so there is
    one definition of which columns a record has.
    """
    if not path.exists():
        raise EnergyError(
            f"CodeCarbon wrote no {path}. It swallows its own exceptions, so the reason is "
            f"in a `codecarbon WARNING` line above, not in a traceback."
        )
    frame = pd.read_csv(path, dtype={"run_id": "str"})
    absent = [column for column in RECORD_COLUMNS if column not in frame.columns]
    if absent:
        raise EnergyError(
            f"{path} has no column(s) {', '.join(absent)}, so it was written by a CodeCarbon "
            f"whose format this module does not know. Re-verify the measurement before "
            f"upgrading (pyproject.toml says why the version is bounded)."
        )
    rows = frame[frame["run_id"] == run_id]
    if len(rows) != 1:
        raise EnergyError(
            f"{path} holds {len(rows)} row(s) for run {run_id}, so the fit was not recorded "
            f"once. CodeCarbon swallows its own exceptions; the reason is in a "
            f"`codecarbon WARNING` line above."
        )
    row = rows.iloc[0]
    # `.item()` turns numpy scalars into Python ones, so the record survives
    # `json.dumps` and MLflow's parameter serialisation unchanged.
    return {
        column: row[column].item() if hasattr(row[column], "item") else row[column]
        for column in RECORD_COLUMNS
    }


def _tracker(name: str, settings: EnergySettings, output_file: Path) -> OfflineEmissionsTracker:
    """An offline tracker with every argument that changes the figure stated.

    The ones params.yaml does not carry are not knobs: they are pinned to what the
    measurement means - one CSV output and no other sink, no forced power, no
    datacentre overhead, no external carbon-intensity service - so that a
    `.codecarbon.config` or a `CODECARBON_*` variable on someone's machine
    cannot change it.
    """
    tracker = OfflineEmissionsTracker(
        project_name=name,
        country_iso_code=settings.country_iso_code,
        region=None,
        cloud_provider=None,
        cloud_region=None,
        tracking_mode=settings.tracking_mode,
        measure_power_secs=settings.measure_power_secs,
        output_dir=str(output_file.parent),
        output_file=output_file.name,
        output_methods=[OutputMethod.CSV],
        on_csv_write="append",
        emissions_endpoint=None,
        electricitymaps_api_token=None,
        force_cpu_power=None,
        force_ram_power=None,
        force_mode_cpu_load=False,
        force_carbon_intensity_g_co2e_kwh=None,
        pue=1.0,
        wue=0.0,
        rapl_include_dram=False,
        rapl_prefer_psys=False,
        allow_multiple_runs=True,
        log_level="warning",
    )
    # A constructor that failed half way has no run id, because CodeCarbon
    # assigns it last; it would otherwise "measure" by doing nothing.
    if not hasattr(tracker, "run_id"):
        raise EnergyError(
            "CodeCarbon could not build its tracker; the reason is in the `codecarbon "
            "WARNING` lines above."
        )
    return tracker


def _cpu_power_method(tracker: OfflineEmissionsTracker) -> tuple[str, float | None]:
    """Which method produced the CPU half of the figure, and the TDP it assumed.

    CodeCarbon logs this and does not write it into its CSV, so it is read off
    the tracker's hardware list. That is a private attribute, which is why the
    version is bounded in pyproject.toml and why `tests/test_energy.py` asserts the
    method is a known one: a release that moves the attribute fails there rather
    than silently logging `unknown`.

    `cpu_load` and `constant` are estimates from a TDP; `intel_rapl` and the other
    counters are readings, and for those the TDP CodeCarbon carries is a
    placeholder that plays no part in the number, so none is reported. A CPU that
    is not in CodeCarbon's table is estimated at a generic 85 W, which is said in
    the method itself, because the figure then describes no particular machine.
    """
    for hardware in getattr(tracker, "_hardware", []):
        if type(hardware).__name__ != "CPU":
            continue
        method = str(getattr(hardware, "_mode", "unknown"))
        if method not in _ESTIMATED_FROM_TDP:
            return method, None
        tdp = getattr(hardware, "_tdp", None)
        if getattr(hardware, "_is_generic_tdp", False):
            logger.warning(
                f"CodeCarbon does not know the TDP of {hardware.get_model()!r} and assumes "
                f"{tdp} W, so this machine's energy figure describes no particular CPU."
            )
            method = f"{method} (generic TDP)"
        return method, None if tdp is None else float(tdp)
    return "unknown", None
