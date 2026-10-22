Load testing
============

The load test sends the traffic of NFR-03 to a running stack and measures what the [specification](specification.md) asks of it: latency (NFR-02), throughput and errors, and exactly-once logging with two workers (NFR-03), the API container's memory (NFR-04) and the energy per request (NFR-10).
Its results on the VM settle the targets of NFR-02 to NFR-04, which are proposed estimates until then ([decision 1](specification.md#decisions-pending-confirmation), issue #143).

The tool is [Locust](https://locust.io/): it is Python, it is on the course's tools list, and M6 can reuse it to replay the `ES` listings.
k6 was considered and would add a JavaScript toolchain.

## What it does

| File | What it is |
|------|------------|
| `tools/loadtest/bodies.py` | Writes request bodies from real held-out listings (`make loadtest-bodies`) |
| `tools/loadtest/locustfile.py` | The Locust users: the valuation requests, at a fixed total rate |
| `tools/loadtest/arrivals.py` | When each user sends: exponential gaps at its share of the rate |
| `tools/loadtest/plan.py` | Which endpoints a run calls, read from the target's `/openapi.json` |
| `tools/loadtest/run.py` | Runs Locust and takes the measurements around it (`make loadtest`) |
| `tools/loadtest/measure.py` | Turns what a run collected into its figures |

**The requests.**
`make loadtest-bodies` samples 2,000 listings of the processed test split, which the model never saw, and writes each as a `/v1/predict` body into `data/interim/loadtest/bodies.jsonl`.
A listing that is no valid request is left out: one without a field FR-01 requires, with a value the schema refuses, or with a category the served bundle does not accept, so that every error the run counts is the API's.
The test split holds no `ES` listings; M6's replay of them is a separate run with `--split data/processed/holdout_es.parquet`.

**The mix.**
At start, the Locust file reads the target's OpenAPI document and calls `/v1/predict`, `/v1/price-range` and `/v1/comparables` in the ratio 3 : 1 : 1, the main use case UC1 three times as often as each of the others.
An endpoint the target does not serve is left out and named in the output, and each body is cut down to the fields the endpoint's request schema defines.

**The rate.**
`--rate` requests per second are shared evenly by four Locust users per request per second (80 users at 0.25 requests per second each for NFR-03's 20 requests/s).
Each user waits an exponential gap before every request, its first included, 4 s on average at 0.25 requests per second, so the users together send like a Poisson process at the total rate, as independent clients do.
Locust starts all users within the first second, and a fixed gap per user made them send together, all 80 within a second and again every 4 s: the API saw the rate as bursts, and the requests of a burst queued behind each other.
On one stack with one worker, at 10 requests/s for 40 s and two runs each, the fixed gap measured `/v1/predict` at p50 850 to 920 ms and p95 1,500 to 1,600 ms, the exponential gaps at p50 89 to 91 ms and p95 160 to 190 ms; the bursts inflated the latency about tenfold.
The [first laptop dry run](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/2bcebf0/reports/loadtest/laptop-dry-run-2026-10-21.md), with the fixed gap, measured p50 3,100 ms and p95 5,700 ms for `/v1/predict`.
A user waits for its answer before it sends the next request, so a slow target gets fewer requests per second rather than a growing queue.
The time between a user's sends is then the longer of its gap and the answer, and the share of the rate a run achieves is `m / (R + m * exp(-R / m))` on average, with `m` the mean gap of 4 s and `R` the answer time: 99.9 % at 200 ms, 97 % at 1 s, 95 % at about 1.4 s, 90 % at 2 s and 73 % at 4 s.
A simulation of 80 users at 20 requests/s over 600 s with `tools/loadtest/arrivals.py` agrees within its noise.
The 95 % verdict below thus fails from answers of about 1.4 s, long after NFR-02's 200 and 300 ms.
The report therefore states the achieved rate, the answers divided by the duration, against the target, and counts the target as sustained when at least 95 % of it was achieved.
The number of requests a run sends is random, with a standard deviation of 1.3 % of the target over NFR-03's 5 minutes at 20 requests/s, which puts the 95 % threshold four standard deviations below the target; over a 60 s dry run it is 2.9 %.
At the end of the run, each user finishes the request it is waiting for.
The counts and percentiles come from Locust's `--json-file` statistics, which Locust writes once at the end; its CSV statistics are a snapshot taken every second and miss the last answers.

**The measurements.**
`tools/loadtest/run.py` takes them on the Docker host of the stack, around the Locust run:

| Figure | Source | Requirement |
|--------|--------|-------------|
| Requests, achieved rate against the target, error rate, p50, p95 and p99 latency per endpoint | Locust, at the client | NFR-02, NFR-03 |
| Share of answers within 200 ms and 300 ms per endpoint | Two scrapes of `/metrics` inside the API container, before and after the run | NFR-02 |
| API container memory (max and mean) and CPU | `docker stats`, sampled about every 2 s | NFR-04 |
| Locust's own CPU, which competes with the API's when both run on one machine | psutil, every 2 s | - |
| Energy per request, in total and above an idle baseline | CodeCarbon over the run and over an idle period before it | NFR-10 |
| One prediction-log record per successful valuation, none missing, none extra | The request IDs Locust received against the records the log gained during the run | NFR-03 |

Locust's latency includes nginx and the network between Locust and the VM, and Locust rounds times above 100 ms to two significant digits.
The `/metrics` shares are measured inside the API, from the request reaching it to its last byte, so they exclude both.
Every 4xx or 5xx answer is an error.

**What CodeCarbon measures.**
CodeCarbon runs in `machine` mode, because its `process` mode measures only the process that runs it and its children, and the API runs in a container.
No machine we use lets a normal user read the RAPL energy counter (NFR-10), so the figure is CodeCarbon's estimate from the CPU's TDP and the load of the whole machine, plus a constant for the RAM.
On a laptop that is everything the laptop does, the IDE and Locust included, so a laptop figure says nothing about the API.
On the VM it is the stack and Locust; the report gives the API container's CPU and Locust's beside it.
Most of the total is the RAM term, a constant power whatever the load (87 % of the latest dry run's energy), so the energy per request in total mostly measures the run's duration.
The CPU term follows CodeCarbon's load model, the TDP times 0.1 + 0.9 x load³ with load the whole machine's CPU share, which stays near a tenth of the TDP at moderate load: the latest dry run's API used 0.93 cores on average, which are 12 % of the laptop's 8 logical CPUs, and the model gave 1.5 W for the CPU.
CodeCarbon integrates the power every 15 s, so a 60 s run has about four samples; the report states the method, the TDP and the interval, and a 5-minute run on the VM has twenty.
The idle baseline is measured just before the run, and the energy above it, divided by the requests, leaves the constant RAM term out and is the informative figure; it is reported as not measurable when the machine was busier during the baseline than during the run.
On a VM CodeCarbon may not find the virtual CPU in its TDP table and fall back to a generic TDP; the report names the method and the TDP it used.

## Run it locally

```bash
uv run dvc pull data/processed/test.parquet models/lgbm-basic
make loadtest-bodies
WEB_CONCURRENCY=2 make stack
make loadtest LOADTEST_DURATION=60s LOADTEST_OUT=reports/loadtest/<name> \
    LOADTEST_ARGS='--note "Dry run on <machine>, not the VM."'
make stack-down
```

`make loadtest` takes `LOADTEST_URL` (default `http://localhost:8080`), `LOADTEST_RATE` (20), `LOADTEST_DURATION` (`5m`) and `LOADTEST_OUT`, and passes `LOADTEST_ARGS` on: `--idle 0` skips the 60 s idle baseline, `--no-energy` skips CodeCarbon, and `--no-docker` measures from Locust alone, for a target whose Docker daemon this machine cannot reach.
It writes `<out>.md`, the figures beside their targets, `<out>.json`, every figure, and Locust's statistics, `<out>-locust.json`.

A laptop run is a dry run of the procedure and is labelled as one: [`reports/loadtest/laptop-dry-run-2026-10-22.md`](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/loadtest/laptop-dry-run-2026-10-22.md) is the latest, with two workers at 20 requests/s for 60 s.

## Run it on the VM

After the lab session of 2026-10-27, once the stack is deployed ([Deployment](deployment.md)).

**Where Locust runs.**
On the VM itself, against `http://localhost:<HTTP_PORT>`.
`docker stats`, the `/metrics` scrape and the prediction log need the VM's Docker daemon, and CodeCarbon measures the machine it runs on.
Locust then shares the VM's CPU with the stack; the report gives its CPU beside the API container's.
Its latency then contains no network, which is what NFR-02's "on the target VM" asks for.

**Setup on the VM**, as the deploy user, once: a checkout of the public repository, without any DagsHub token, and an environment with Locust and CodeCarbon alone.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos.git loadtest
cd loadtest && uv sync --locked --only-group loadtest
```

The bodies come from the laptop, which has the data: `make loadtest-bodies`, then `scp data/interim/loadtest/bodies.jsonl recommenditos@<vm>:loadtest/data/interim/loadtest/` (create the directory first).
The VM needs neither DVC nor the data.

**The runs.**
Two runs of 20 requests/s for 5 minutes, both against the image the VM serves:

1. With the worker count the VM is meant to serve with (`WEB_CONCURRENCY` in the stack's `.env`, 1 by default): NFR-02, NFR-03's throughput and error rate, NFR-04 and NFR-10.
2. With `WEB_CONCURRENCY=2` in `.env` and `docker compose up -d --wait`: NFR-03's exactly-once check, and whether a second worker changes latency and memory.

Each from the checkout, with `uv run --no-sync` so that uv does not install the other groups:

```bash
uv run --no-sync python -m tools.loadtest.run --host http://localhost:80 --rate 20 --duration 5m \
    --out reports/loadtest/vm-<date>-workers-<n> \
    --note "VM run: <nproc> vCPUs, <free -h total> RAM, Locust on the VM, <anything else running>."
```

Afterwards, set `WEB_CONCURRENCY` back to the value the VM serves with.

**What to record.**
Copy the three files of each run back into `reports/loadtest/` and commit them in a pull request, with `nproc`, `free -h` and the image tag in the note.
The prediction log keeps the run's records; they are 2 x 6,000 requests of about 750 bytes, far below its 2 GB cap; the report's start time and duration mark them, so M6's drift job can leave them out.

**What follows.**
With both runs committed, the team confirms or adjusts the targets of NFR-02 to NFR-04 against the measurements, the specification is updated, and the decision is recorded in the EDN as EDN-88, with the choice of Locust (issue #143).
