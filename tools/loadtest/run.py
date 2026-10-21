"""Run the load test and take the measurements around it (issue #143).

    uv run python -m tools.loadtest.run --host http://localhost:8080 --rate 20 \
        --duration 5m --out reports/loadtest/<name>                  (make loadtest)

Runs `locustfile.py` against `--host` and, on the Docker host of the stack it
targets, measures what the run did to the API container:

- error rate and p50, p95, p99 latency per endpoint, from Locust (NFR-02, NFR-03);
- the share of answers within 200 ms and 300 ms per endpoint, from two scrapes of
  `/metrics` inside the API container, before and after (NFR-02);
- the API container's memory and CPU, sampled with `docker stats` (NFR-04);
- energy per request, estimated by CodeCarbon on this machine over the run, and
  over an idle period before it as a baseline (NFR-10);
- with a prediction log, that it gained exactly one record per successful
  valuation and none besides (NFR-03).

It must run on the machine whose Docker daemon runs the stack (or with
`DOCKER_HOST` pointing at it); `--no-docker` measures from Locust alone. It
writes `<out>.json`, `<out>.md` and Locust's `<out>-locust_stats.csv` and
`<out>-locust_failures.csv`. Standard library, Locust and CodeCarbon only, so a
checkout with `uv sync --only-group loadtest` runs it; docs/docs/load-testing.md
says how to run it on the VM.
"""

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from codecarbon import OfflineEmissionsTracker
from codecarbon.output import OutputMethod

from tools.loadtest import measure

ROOT = Path(__file__).resolve().parents[2]
LOCUSTFILE = Path(__file__).with_name("locustfile.py")

#: Targets the report states beside each figure (docs/docs/specification.md).
P95_TARGET_MS = {"/v1/predict": 200.0, "/v1/price-range": 300.0, "/v1/comparables": 300.0}
#: NFR-03: less than 1 % errors.
ERROR_RATE = 0.01
MEMORY_TARGET_BYTES = 1e9
VALUATION_ENDPOINTS = tuple(P95_TARGET_MS)

#: The environment variable of the prediction log's file in the API container (#140).
PREDICTION_LOG_ENV_VAR = "RECOMMENDITOS_PREDICTION_LOG"

_SCRAPE = (
    "import urllib.request; "
    "print(urllib.request.urlopen('http://127.0.0.1:8000/metrics', timeout=10).read().decode())"
)
#: Prints the log's highest sequence number, or the request IDs of the records
#: above the one given; `-` when the container has no log. A plain string,
#: not an f-string, so it is visibly no query built from input; the variable it
#: reads is `PREDICTION_LOG_ENV_VAR`.
_READ_LOG = """
import os, sqlite3, sys
path = os.environ.get("RECOMMENDITOS_PREDICTION_LOG", "")
if not path or not os.path.exists(path):
    print("-")
    raise SystemExit
log = sqlite3.connect(path)
if len(sys.argv) == 1:
    print(log.execute("SELECT COALESCE(MAX(seq), 0) FROM predictions").fetchone()[0])
else:
    query = "SELECT request_id FROM predictions WHERE seq > ? ORDER BY seq"
    for (request_id,) in log.execute(query, (int(sys.argv[1]),)):
        print(request_id)
"""


def seconds(duration: str) -> int:
    """`'90'`, `'90s'`, `'5m'` or `'1h30m'`, Locust's run-time format, in seconds."""
    if duration.isdigit():
        return int(duration)
    parts = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", duration)
    if not parts or not any(parts.groups()):
        raise argparse.ArgumentTypeError(f"not a duration: {duration!r}")
    hours, minutes, secs = (int(part or 0) for part in parts.groups())
    return hours * 3600 + minutes * 60 + secs


def docker(*args: str) -> str:
    # Arguments this module builds itself; `docker` from PATH, as a user runs it.
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    ).stdout


def find_container(project: str, service: str) -> str:
    ids = docker(
        "ps",
        "-q",
        "--filter",
        f"label=com.docker.compose.project={project}",
        "--filter",
        f"label=com.docker.compose.service={service}",
    ).split()
    if len(ids) != 1:
        raise SystemExit(f"expected one running {project}/{service} container, found {len(ids)}")
    return ids[0]


class StatsSampler(threading.Thread):
    """Samples the container's memory and CPU with `docker stats` until stopped."""

    def __init__(self, container: str) -> None:
        super().__init__(daemon=True)
        self.container = container
        self.samples: list[tuple[float, float]] = []
        self.stopping = threading.Event()

    def run(self) -> None:
        while not self.stopping.is_set():
            line = docker(
                "stats", "--no-stream", "--format", "{{.MemUsage}}|{{.CPUPerc}}", self.container
            )
            self.samples.append(measure.parse_docker_stats(line))
            self.stopping.wait(1.0)

    def stop(self) -> dict[str, float]:
        self.stopping.set()
        self.join()
        memory = [m for m, _ in self.samples]
        cpu = [c for _, c in self.samples]
        return {
            "samples": len(self.samples),
            "memory_max_bytes": max(memory, default=math.nan),
            "memory_mean_bytes": sum(memory) / len(memory) if memory else math.nan,
            "cpu_mean_percent": sum(cpu) / len(cpu) if cpu else math.nan,
            "cpu_max_percent": max(cpu, default=math.nan),
        }


def tracker(output_dir: Path, name: str) -> OfflineEmissionsTracker:
    """CodeCarbon's offline tracker, set up as `recommenditos/modeling/energy.py` sets it up.

    Every argument that changes the figure is passed, so no `.codecarbon.config`
    or `CODECARBON_*` variable changes it, with one difference: `machine` mode,
    because the API runs in a container and CodeCarbon's `process` mode measures
    only the process that runs it and its children. The figure is the whole
    machine's, which on the VM is mostly the stack and on a shared laptop is
    whatever else runs too.
    """
    return OfflineEmissionsTracker(
        project_name=name,
        country_iso_code="ESP",
        region=None,
        cloud_provider=None,
        cloud_region=None,
        tracking_mode="machine",
        measure_power_secs=15,
        output_dir=str(output_dir),
        output_file="emissions.csv",
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
        log_level="error",
    )


def cpu_power_method(tracked: Any) -> str:
    """How CodeCarbon got the CPU's power: a counter, or an estimate from a TDP."""
    for hardware in getattr(tracked, "_hardware", []):
        if type(hardware).__name__ == "CPU":
            mode = str(getattr(hardware, "_mode", "unknown"))
            tdp = getattr(hardware, "_tdp", None)
            generic = " (generic TDP)" if getattr(hardware, "_is_generic_tdp", False) else ""
            model = hardware.get_model() if hasattr(hardware, "get_model") else "?"
            return f"{mode}{generic}, {model}, TDP {tdp} W"
    return "unknown"


def measure_energy(output_dir: Path, name: str, during: Any) -> dict[str, Any]:
    """CodeCarbon's energy while `during()` runs, in joules, and its seconds."""
    tracked = tracker(output_dir, name)
    tracked.start()
    started = time.monotonic()
    during()
    elapsed = time.monotonic() - started
    tracked.stop()
    data = tracked.final_emissions_data
    return {
        "seconds": elapsed,
        "energy_j": data.energy_consumed * 3.6e6,
        "cpu_energy_j": data.cpu_energy * 3.6e6,
        "ram_energy_j": data.ram_energy * 3.6e6,
        "cpu_power_method": cpu_power_method(tracked),
    }


def load_average() -> list[float] | None:
    return [round(load, 2) for load in os.getloadavg()] if hasattr(os, "getloadavg") else None


class ContainerProbe:
    """What the run measures inside and around the API container."""

    def __init__(self, container: str) -> None:
        self.container = container
        self.sampler = StatsSampler(container)
        self.before: measure.Buckets = {}
        self.log_seq = "-"

    def exec_python(self, script: str, *args: str) -> str:
        return docker("exec", self.container, "python", "-c", script, *args)

    def describe(self) -> dict[str, Any]:
        inspected = json.loads(docker("inspect", self.container))[0]
        env = dict(pair.split("=", 1) for pair in inspected["Config"]["Env"])
        return {
            "image": inspected["Config"]["Image"],
            "workers": int(env.get("WEB_CONCURRENCY", "1")),
            "prediction_log": env.get(PREDICTION_LOG_ENV_VAR, ""),
        }

    def begin(self) -> None:
        self.before = measure.latency_buckets(self.exec_python(_SCRAPE))
        self.log_seq = self.exec_python(_READ_LOG).strip()
        self.sampler.start()

    def finish(self, answered: list[str]) -> dict[str, Any]:
        result: dict[str, Any] = {"docker_stats": self.sampler.stop()}
        after = measure.latency_buckets(self.exec_python(_SCRAPE))
        result["metrics"] = measure.latency_shares(self.before, after)
        result["prediction_log"] = None
        if self.log_seq != "-":
            logged = self.exec_python(_READ_LOG, self.log_seq).split()
            check = measure.check_log(answered, logged)
            result["prediction_log"] = {**asdict(check), "exactly_once": check.exactly_once}
        return result


def energy_per_request(result: dict[str, Any], requests: int) -> dict[str, float]:
    """The run's energy per request, in total and above the idle baseline."""
    if "energy_run" not in result or not requests:
        return {}
    run_energy = result["energy_run"]
    figures = {"energy_per_request_j": run_energy["energy_j"] / requests}
    if "energy_idle" in result:
        idle_w = result["energy_idle"]["energy_j"] / result["energy_idle"]["seconds"]
        above_idle = run_energy["energy_j"] - idle_w * run_energy["seconds"]
        figures["energy_per_request_above_idle_j"] = above_idle / requests
    return figures


def locust_command(args: argparse.Namespace, users: int, ids_file: Path) -> list[str]:
    return [
        *(sys.executable, "-m", "locust", "-f", str(LOCUSTFILE), "--headless"),
        *("--host", args.host, "-u", str(users), "-r", str(users)),
        *("-t", f"{seconds(args.duration)}s", "--rate", str(args.rate)),
        *("--bodies", str(args.bodies), "--request-ids", str(ids_file)),
        *("--csv", f"{args.out}-locust", "--only-summary", "--exit-code-on-error", "0"),
        # At the end of the run, a user finishes the request it is waiting for
        # instead of abandoning it, so that every answer the API logged reached
        # the client and the prediction-log check compares like with like.
        *("--stop-timeout", "30"),
    ]


def run(args: argparse.Namespace, work: Path) -> dict[str, Any]:
    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    users = args.users or max(1, math.ceil(args.rate))
    result: dict[str, Any] = {
        "label": out.name,
        "note": args.note,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": args.host,
        "rate_per_s": args.rate,
        "duration_s": seconds(args.duration),
        "users": users,
        "machine": {
            "hostname": platform.node(),
            "cpus": os.cpu_count(),
            "load_average_before": load_average(),
        },
    }
    probe = None if args.no_docker else ContainerProbe(find_container(args.project, args.service))
    if probe:
        result["container"] = probe.describe()
    if args.idle and not args.no_energy:
        print(f"loadtest: idle baseline, {args.idle} s")
        result["energy_idle"] = measure_energy(work, "idle", lambda: time.sleep(args.idle))
    if probe:
        probe.begin()

    ids_file = work / "request-ids.txt"
    command = locust_command(args, users, ids_file)
    environment = {**os.environ, "PYTHONPATH": str(ROOT)}

    def load() -> None:
        # This interpreter running Locust with arguments built above.
        subprocess.run(command, cwd=ROOT, env=environment, check=True)

    if args.no_energy:
        load()
    else:
        result["energy_run"] = measure_energy(work, "run", load)
    result["machine"]["load_average_after"] = load_average()
    for suffix in ("_stats_history.csv", "_exceptions.csv"):
        Path(f"{out}-locust{suffix}").unlink(missing_ok=True)

    stats = measure.read_locust_stats(Path(f"{out}-locust_stats.csv"))
    result["locust"] = {n: {**asdict(s), "error_rate": s.error_rate} for n, s in stats.items()}
    result |= energy_per_request(result, stats["Aggregated"].requests)
    if probe:
        result |= probe.finish(ids_file.read_text(encoding="utf-8").split())
    return result


def met(value: float, target: float) -> str:
    return "yes" if value <= target else "no"


def _overview(result: dict[str, Any]) -> list[str]:
    machine = result["machine"]
    lines = [
        (
            f"- Started {result['started_at']} against `{result['host']}`"
            f" from `{machine['hostname']}` ({machine['cpus']} logical CPUs)."
        ),
        (
            f"- Target rate {result['rate_per_s']:g} requests/s for {result['duration_s']} s,"
            f" {result['users']} Locust users."
        ),
        (
            f"- Load average (1, 5, 15 min) before: {machine['load_average_before']},"
            f" after: {machine['load_average_after']}."
        ),
    ]
    if "container" in result:
        container = result["container"]
        lines.append(f"- API image `{container['image']}`, {container['workers']} worker(s).")
    return lines


def _latency(result: dict[str, Any]) -> list[str]:
    rows = []
    for name, s in result["locust"].items():
        target = P95_TARGET_MS.get(name)
        p95 = f"{s['p95_ms']:.0f}"
        if target:
            p95 += f" ({met(s['p95_ms'], target)}, target {target:.0f})"
        errors = f"{measure.percent(s['error_rate'])} ({met(s['error_rate'], ERROR_RATE)})"
        row = [name, s["requests"], f"{s['rate_per_s']:.2f}", errors, f"{s['p50_ms']:.0f}", p95]
        rows.append([*row, f"{s['p99_ms']:.0f}"])
    header = ["Endpoint", "Requests", "Req/s", "Errors (< 1 %)", "p50 ms", "p95 ms", "p99 ms"]
    lines = [
        "## Latency and errors (Locust, client side)",
        "",
        measure.markdown_table(header, rows),
    ]
    if "metrics" in result:
        valuation = {p: f for p, f in result["metrics"].items() if p in VALUATION_ENDPOINTS}
        header = ["Endpoint", "Answers", "Within 200 ms", "Within 300 ms"]
        lines += [
            "",
            "## Share of answers within NFR-02's targets (`/metrics`, server side)",
            "",
            measure.markdown_table(header, measure.mapping_rows(valuation)),
        ]
    return lines


def _resources(result: dict[str, Any]) -> list[str]:
    lines = []
    if "docker_stats" in result:
        d = result["docker_stats"]
        mib = 2**20
        memory = (
            f"max {d['memory_max_bytes'] / mib:.0f} MiB, mean {d['memory_mean_bytes'] / mib:.0f}"
        )
        below = met(d["memory_max_bytes"], MEMORY_TARGET_BYTES)
        cpu = f"mean {d['cpu_mean_percent']:.0f} %, max {d['cpu_max_percent']:.0f} %"
        lines += [
            "## API container (`docker stats`, NFR-04)",
            "",
            f"- Memory: {memory} MiB ({below}, below 1 GB), {d['samples']} samples.",
            f"- CPU: {cpu} (100 % is one core).",
            "",
        ]
    if "energy_run" in result:
        e = result["energy_run"]
        per_request = result.get("energy_per_request_j", math.nan)
        lines += [
            "## Energy (CodeCarbon, whole machine, NFR-10)",
            "",
            f"- Method: {e['cpu_power_method']}; `machine` tracking mode.",
            (
                f"- Run: {e['energy_j']:.0f} J over {e['seconds']:.0f} s"
                f" ({e['energy_j'] / e['seconds']:.1f} W), {per_request:.3f} J per request."
            ),
        ]
        if "energy_idle" in result:
            i = result["energy_idle"]
            above = result.get("energy_per_request_above_idle_j", math.nan)
            # Below zero when the rest of the machine was busier during the
            # baseline than during the run, as on a shared laptop.
            above_text = f"{above:.3f} J" if above >= 0 else "not measurable"
            lines.append(
                f"- Idle baseline: {i['energy_j'] / i['seconds']:.1f} W over {i['seconds']:.0f} s;"
                f" above it, {above_text} per request."
            )
        lines.append("")
    return lines


def _log(result: dict[str, Any]) -> list[str]:
    log = result.get("prediction_log")
    if not log:
        return ["## Prediction log (NFR-03)", "", "- Not checked: the API has no prediction log."]
    verdict = "yes" if log["exactly_once"] else "no"
    return [
        "## Prediction log (NFR-03)",
        "",
        f"- {log['answered']} successful valuations; the log gained {log['logged']} records.",
        (
            f"- Missing {log['missing']}, unexpected {log['unexpected']},"
            f" duplicated {log['duplicated']}."
        ),
        f"- Exactly one record per successful request: **{verdict}**.",
    ]


def report(result: dict[str, Any]) -> str:
    """The run as Markdown, each figure beside its target."""
    lines = [f"# Load test: {result['label']}", ""]
    if result["note"]:
        lines += [result["note"], ""]
    lines += [*_overview(result), "", *_latency(result), "", *_resources(result), *_log(result)]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", required=True, help="Base URL, e.g. http://localhost:8080")
    parser.add_argument("--rate", type=float, default=20.0, help="Total requests per second")
    parser.add_argument("--duration", default="5m", help="Locust run time: 90s, 5m, 1h")
    parser.add_argument("--users", type=int, default=0, help="Locust users (default: the rate)")
    parser.add_argument("--out", type=Path, required=True, help="Output path without extension")
    parser.add_argument("--bodies", type=Path, default=ROOT / "data/interim/loadtest/bodies.jsonl")
    parser.add_argument("--idle", type=int, default=60, help="Idle energy baseline in s (0: none)")
    parser.add_argument("--note", default="", help="Text for the report: machine, conditions")
    parser.add_argument("--project", default="recommenditos", help="Compose project name")
    parser.add_argument("--service", default="api", help="Compose service of the API")
    parser.add_argument("--no-docker", action="store_true", help="Measure from Locust alone")
    parser.add_argument("--no-energy", action="store_true", help="Skip CodeCarbon")
    args = parser.parse_args()
    seconds(args.duration)
    with tempfile.TemporaryDirectory(prefix="loadtest-") as work:
        result = run(args, Path(work))
    Path(f"{args.out}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    text = report(result)
    Path(f"{args.out}.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
