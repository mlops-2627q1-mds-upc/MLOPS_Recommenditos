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
`--rate` requests per second are shared evenly by as many Locust users as the rate (20 users at 1 request per second each for NFR-03's 20 requests/s).
A user waits for its answer before it sends the next request, so a target that cannot keep up gets fewer requests per second rather than a growing queue: the achieved rate is part of the result.
At the end of the run, each user finishes the request it is waiting for.

**The measurements.**
`tools/loadtest/run.py` takes them on the Docker host of the stack, around the Locust run:

| Figure | Source | Requirement |
|--------|--------|-------------|
| Requests, achieved rate, error rate, p50, p95 and p99 latency per endpoint | Locust, at the client | NFR-02, NFR-03 |
| Share of answers within 200 ms and 300 ms per endpoint | Two scrapes of `/metrics` inside the API container, before and after the run | NFR-02 |
| API container memory (max and mean) and CPU | `docker stats`, sampled about every 2 s | NFR-04 |
| Energy per request, in total and above an idle baseline | CodeCarbon over the run and over an idle period before it | NFR-10 |
| One prediction-log record per successful valuation, none missing, none extra | The request IDs Locust received against the records the log gained during the run | NFR-03 |

Locust's latency includes nginx and the network between Locust and the VM, and Locust rounds times above 100 ms to two significant digits.
The `/metrics` shares are measured inside the API, from the request reaching it to its last byte, so they exclude both.
Every 4xx or 5xx answer is an error.

**What CodeCarbon measures.**
CodeCarbon runs in `machine` mode, because its `process` mode measures only the process that runs it and its children, and the API runs in a container.
No machine we use lets a normal user read the RAPL energy counter (NFR-10), so the figure is CodeCarbon's estimate from the CPU's TDP and the load of the whole machine, plus a constant for the RAM.
On a laptop that is everything the laptop does, the IDE and Locust included, so a laptop figure says nothing about the API.
On the VM it is the stack and Locust; the report gives the API container's CPU from `docker stats` beside it.
The idle baseline is measured just before the run, and the energy above it, divided by the requests, is the closest figure to the API's own cost; it is reported as not measurable when the machine was busier during the baseline than during the run.
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
It writes `<out>.md`, the figures beside their targets, `<out>.json`, every figure, and Locust's `<out>-locust_stats.csv` and `<out>-locust_failures.csv`.

A laptop run is a dry run of the procedure and is labelled as one: [`reports/loadtest/laptop-dry-run-2026-10-21.md`](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/loadtest/laptop-dry-run-2026-10-21.md) is the first.

## Run it on the VM

After the lab session of 2026-10-27, once the stack is deployed ([Deployment](deployment.md)).

**Where Locust runs.**
On the VM itself, against `http://localhost:<HTTP_PORT>`.
`docker stats`, the `/metrics` scrape and the prediction log need the VM's Docker daemon, and CodeCarbon measures the machine it runs on.
Locust then shares the VM's CPU with the stack; at 20 requests/s it needs a few percent of one core, and its share is visible in the report beside the API container's.
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
Copy the four files of each run back into `reports/loadtest/` and commit them in a pull request, with `nproc`, `free -h` and the image tag in the note.
The prediction log keeps the run's records; they are 2 x 6,000 requests of about 750 bytes, far below its 2 GB cap; the report's start time and duration mark them, so M6's drift job can leave them out.

**What follows.**
With both runs committed, the team confirms or adjusts the targets of NFR-02 to NFR-04 against the measurements, the specification is updated, and the decision is recorded in the EDN as EDN-88, with the choice of Locust (issue #143).
