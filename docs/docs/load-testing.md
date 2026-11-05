Load testing
============

The load test sends the traffic of NFR-03 to a running stack and measures what the [specification](specification.md) asks of it: latency (NFR-02), throughput and errors, and exactly-once logging with two workers (NFR-03), the API container's memory (NFR-04) and the energy per request (NFR-10).
Its results on the VM settled the targets of NFR-02 to NFR-04 on 2026-10-31 ([decision 1](specification.md#decisions-pending-confirmation), [EDN-88](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md), issue #143).

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
The 95 % verdict below thus fails from answers of about 1.4 s, long after NFR-02's 250 and 300 ms.
The report therefore states the achieved rate, the answers divided by the duration, against the target, and counts the target as sustained when at least 95 % of it was achieved.
The number of requests a run sends is random, with a standard deviation of 1.3 % of the target over NFR-03's 5 minutes at 20 requests/s, which puts the 95 % threshold four standard deviations below the target; over a 60 s dry run it is 2.9 %.
At the end of the run, each user finishes the request it is waiting for.
The counts and percentiles come from Locust's `--json-file` statistics, which Locust writes once at the end; its CSV statistics are a snapshot taken every second and miss the last answers.

**The measurements.**
`tools/loadtest/run.py` takes them on the Docker host of the stack, around the Locust run:

| Figure | Source | Requirement |
|--------|--------|-------------|
| Requests, achieved rate against the target, error rate, p50, p95 and p99 latency per endpoint | Locust, at the client | NFR-02, NFR-03 |
| Share of answers within 250 ms and 300 ms per endpoint | Two scrapes of `/metrics` inside the API container, before and after the run | NFR-02 |
| API container memory (max and mean) and CPU | `docker stats`, sampled about every 2 s | NFR-04 |
| Locust's own CPU, which competes with the API's when both run on one machine | psutil, every 2 s | - |
| Energy per request, in total and above an idle baseline | CodeCarbon over the run and over an idle period before it | NFR-10 |
| One prediction-log record per successful valuation, none missing, none extra | The request IDs Locust received against the records the log gained during the run | NFR-03 |

Locust's latency includes nginx and the network between Locust and the VM, and Locust rounds times above 100 ms to two significant digits.
The `/metrics` shares are measured inside the API, from the request reaching it to its last byte, so they exclude both.
Every 4xx or 5xx answer is an error.
`run.py` does not record restarts: on the VM, compare the API container's `RestartCount` (`docker inspect -f '{{.RestartCount}}' recommenditos-api-1`) before and after the run.
It has to stay the same, because a restart during the run means the healthcheck took the loaded API for a hung one ([EDN-93](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md), [Deployment](deployment.md)).
Re-measure the hang drill on the VM too, with the SIGSTOP that [Deployment](deployment.md) describes.
Its worst case of 21 s assumes the laptop's restart of about 4 s from the kill to the next answer; a cold restart of 13 s or more on the VM would break NFR-05's 30 s, and the probe's timings would have to shrink.

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

`make loadtest` takes `LOADTEST_URL` (default `http://localhost:8080`, or the port `HTTP_PORT` names), `LOADTEST_RATE` (20), `LOADTEST_DURATION` (`5m`) and `LOADTEST_OUT`, and passes `LOADTEST_ARGS` on: `--idle 0` skips the 60 s idle baseline, `--no-energy` skips CodeCarbon, and `--no-docker` measures from Locust alone, for a target whose Docker daemon this machine cannot reach.
It writes `<out>.md`, the figures beside their targets, `<out>.json`, every figure, and Locust's statistics, `<out>-locust.json`.

A laptop run is a dry run of the procedure and is labelled as one: [`reports/loadtest/laptop-dry-run-2026-10-22.md`](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/loadtest/laptop-dry-run-2026-10-22.md) is the latest, with two workers at 20 requests/s for 60 s.

## Run it on the VM

Once the stack is deployed ([Deployment](deployment.md)); it has run on the VM since 2026-10-30.

**Where Locust runs.**
On the VM itself, against the proxy at `http://localhost:8080` (the stack's `HTTP_PORT`).
`docker stats`, the `/metrics` scrape and the prediction log need the VM's Docker daemon, and CodeCarbon measures the machine it runs on.
It could not run elsewhere anyway: the public URL does not answer from the UPC network or UPClink, and the SSH port does not answer from the public internet ([The VM](deployment.md#the-vm)).
Locust then shares the VM's one vCPU with the stack; the report gives its CPU beside the API container's.
Its latency then contains no network, which is what NFR-02's "on the target VM" asks for.

Every command below runs from a laptop with UPClink connected, on campus too (from the university's Wi-Fi, eduroam, the SSH port timed out on 2026-11-04, [The VM](deployment.md#the-vm)), over SSH as the deploy user, whose home is the stack's directory `/opt/recommenditos`.

**Setup on the VM**, once: uv in the deploy user's home, a checkout of the public repository at the commit the VM runs, without any DagsHub token, and an environment with Locust and CodeCarbon alone (about 320 MB).
The installer puts uv into `~/.local/bin`; `UV_NO_MODIFY_PATH=1` keeps it from editing the shell profiles, and the commands call uv by its path.

```bash
ssh -p 22052 recommenditos@nattech.fib.upc.edu 'curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh'
ssh -p 22052 recommenditos@nattech.fib.upc.edu './deploy.sh status'    # the tag that runs: its commit SHA
ssh -p 22052 recommenditos@nattech.fib.upc.edu 'git clone https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos.git loadtest &&
    cd loadtest && git checkout <SHA> && ~/.local/bin/uv sync --locked --only-group loadtest &&
    mkdir -p data/interim/loadtest'
```

The bodies come from the laptop, which has the data; the VM needs neither DVC nor the data:

```bash
make loadtest-bodies
scp -P 22052 data/interim/loadtest/bodies.jsonl recommenditos@nattech.fib.upc.edu:loadtest/data/interim/loadtest/
```

**The runs.**
Two runs of 20 requests/s for 5 minutes, both against the image the VM serves; check with `./deploy.sh status` before and after each run that the tag did not change, since a deploy in between would mix two images:

1. With the worker count the VM is meant to serve with (`WEB_CONCURRENCY` in `/opt/recommenditos/.env`, 1 by default): NFR-02, NFR-03's throughput and error rate, NFR-04 and NFR-10.
2. With `WEB_CONCURRENCY=2` in `.env` and `./deploy.sh deploy <the tag that runs>`, which starts the stack again with the changed `.env`: NFR-03's exactly-once check, and whether a second worker changes latency and memory.

Each from the checkout, with `uv run --no-sync` so that uv does not install the other groups.
A run takes about 6.5 minutes with the idle baseline; started with `nohup`, it survives a dropped SSH connection or VPN:

```bash
ssh -p 22052 recommenditos@nattech.fib.upc.edu 'cd loadtest && nohup ~/.local/bin/uv run --no-sync python -m tools.loadtest.run \
    --host http://localhost:8080 --rate 20 --duration 5m --out reports/loadtest/vm-<date>-workers-<n> \
    --note "VM run: <nproc> vCPUs, <free -h total> RAM, Locust on the VM, <anything else running>." \
    > ~/loadtest-<n>.log 2>&1 &'
```

Before and after each run, also take the API container's `RestartCount` and start time (`docker inspect -f '{{.RestartCount}} {{.State.StartedAt}}' recommenditos-api-1`).
Compose creates a new container when `.env` changes, so compare within one run.

Afterwards, set `WEB_CONCURRENCY` back to the value the VM serves with, run `./deploy.sh deploy <the tag that runs>` again and check `./deploy.sh status` says healthy.

**What to record.**
Copy the three files of each run back into `reports/loadtest/` and commit them in a pull request, with `nproc`, `free -h` and the image tag in the note:

```bash
scp -P 22052 'recommenditos@nattech.fib.upc.edu:loadtest/reports/loadtest/vm-<date>-*' reports/loadtest/
```

The prediction log keeps the run's records; they are 2 x 6,000 requests of about 750 bytes, far below its 2 GB cap. M6's drift job leaves them out: the runs of 2026-10-31 predate #191, so their `inputs_sent_only` is NULL, and a later run's requests carry `X-Client: loadtest` ([Reading the log](api.md#reading-the-log)).

**Results of 2026-10-31.**
Both runs were made against image 902de35, with Locust on the VM: [one worker](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/loadtest/vm-2026-10-31-workers-1.md) and [two workers](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/loadtest/vm-2026-10-31-workers-2.md).

| Figure | 1 worker | 2 workers | Target |
|--------|----------|-----------|--------|
| Achieved rate, errors | 19.61 of 20 requests/s, 0 of 5,883 | 19.87 of 20 requests/s, 0 of 5,960 | 95 % of 20/s, below 1 % (NFR-03) |
| `/v1/predict` p50 / p95 / p99 (Locust) | 65 / 210 / 340 ms | 78 / 240 / 440 ms | p95 at most 250 ms, 200 ms when measured (NFR-02) |
| `/v1/predict` answers within 200 ms (`/metrics`) | 94.7 % | 93.3 % | 95 %, NFR-02's target when measured |
| `/v1/price-range` p95 | 170 ms | 230 ms | at most 300 ms (NFR-02) |
| `/v1/comparables` p95 | 16 ms | 20 ms | at most 300 ms (NFR-02) |
| API container memory, max (`docker stats`) | 295 MiB | 609 MiB | below 1 GB (NFR-04); the limit is 768 MiB |
| API container CPU, mean / max | 61 % / 96 % of the one vCPU | 60 % / 96 % | - |
| Locust's CPU, mean | 2 % | 2 % | - |
| Energy per request, total / above idle | 0.59 / 0.058 J | 0.59 / 0.060 J | reported, no target (NFR-10) |
| Prediction log | 5,883 records, exactly once | 5,960 records, exactly once | exactly once (NFR-03) |
| API `RestartCount` before / after | 0 / 0 | 0 / 0 | unchanged |

`/v1/predict` missed NFR-02's target of the time, 200 ms at the 95th percentile, with either worker count, by 10 ms with one worker; the other two endpoints and NFR-03 and NFR-04 met their targets.
The team raised the `/predict` target to 250 ms and kept one worker ([EDN-88](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)); the two reports state the verdicts against 200 ms, as `run.py` judged them then.
A second worker adds no CPU on one vCPU: it doubles the memory and makes the tail slower.
Locust's 2 % is confirmed by `ps`: 6 s of CPU time over the 299 s the process ran.
CodeCarbon took a TDP of 4 W from its table for the VM's CPU ("Intel Core Processor (Broadwell, no TSX, IBRS)"), so its CPU term is low; 86 to 87 % of each run's energy is the constant 10 W it assumes for the RAM.

**Results of 2026-11-05, with the full M6 stack.**
One run of 20 requests/s for 5 minutes, one worker, with the eight services of the stack up (api, proxy, prometheus, cadvisor, node-exporter, alertmanager, grafana, monitor): [the report](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/loadtest/vm-2026-11-05-full-stack.md).
It differs from the runs of 2026-10-31 in where Locust ran: on the laptop, against the public URL `http://nattech.fib.upc.edu:40520` with UPClink disconnected, over the internet, through nginx; the round trip was not recorded in a result file, and the median of `/v1/comparables` below (31 ms against 5 ms on the VM) gives its order.
So its latencies contain the network, which the runs on the VM did not, and are not directly comparable with them.
The run used `--no-docker --no-energy --idle 0`: the laptop reaches neither the VM's Docker daemon nor `/metrics` (the proxy answers 403), so `run.py` measured neither memory, energy, nor the exactly-once check, and the server-side figures below were read afterwards from the stack's own Prometheus.
The window was 2026-11-04 23:08:01 to 23:13:03 UTC, 00:08:01 to 00:13:03 CET on 2026-11-05, for reading the dashboards of that time.

| Figure | 2026-10-31, 1 worker, Locust on the VM | 2026-11-05, full stack, Locust on the laptop | Target |
|--------|----------|-----------|--------|
| Achieved rate, errors | 19.61 of 20 requests/s, 0 of 5,883 | 20.04 of 20 requests/s, 0 of 6,013 | 95 % of 20/s, below 1 % (NFR-03) |
| `/v1/predict` p50 / p95 / p99 | 65 / 210 / 340 ms | 97 / 280 / 1,100 ms | p95 at most 250 ms (NFR-02) |
| `/v1/price-range` p50 / p95 / p99 | 52 / 170 / 320 ms | 82 / 270 / 1,200 ms | p95 at most 300 ms (NFR-02) |
| `/v1/comparables` p50 / p95 / p99 | 5 / 16 / 23 ms | 31 / 120 / 950 ms | p95 at most 300 ms (NFR-02) |

Server side, from the API's histogram `http_request_duration_seconds` in the stack's Prometheus (its time from receiving a request to the last byte of the answer, so without the network and nginx):

| Figure | 2026-10-31, 1 worker | 2026-11-05, full stack | Target |
|--------|----------|-----------|--------|
| `/v1/predict` answers within 250 ms | not read (94.7 % within 200 ms, the target then) | 97.1 % (93.8 % within 200 ms) | 95 % (NFR-02) |
| `/v1/predict` p50 / p95 / p99 | - | 69 / 218 / 376 ms | p95 at most 250 ms (NFR-02) |
| `/v1/price-range` within 300 ms, p50 / p95 / p99 | - | 98.6 %, 55 / 190 / 347 ms | p95 at most 300 ms (NFR-02) |
| `/v1/comparables` within 300 ms, p50 / p95 / p99 | - | 100 % (all within 50 ms), 6 / 22 / 39 ms | p95 at most 300 ms (NFR-02) |
| Answers counted by the API, status | - | 3,607, 1,184, and 1,222, all 200 | the requests Locust sent |
| API container CPU, mean (cAdvisor) | 61 % of the one vCPU | 64 % | - |
| The seven other containers' CPU, together | - | below 2 % | - |
| VM CPU idle, steal, CPU pressure (node-exporter) | - | 32 %, below 0.01 %, 14.3 % | - |
| API container memory, max | 295 MiB (`docker stats`) | 303 MiB (cAdvisor's working set) | below 1 GB (NFR-04) |

NFR-03's rate and error targets hold with the full stack: the rate was reached, and there were no errors.
Its latency part is NFR-02's check, which is still owed, and its exactly-once part was not re-checked.
NFR-02 was not re-run with the full stack by its decided method, a Locust load test on the VM ([EDN-88](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)), because this run's client was on the laptop.
The API's own histogram supports it: 97.1 % of the 3,607 `/v1/predict` answers took at most 250 ms in the API (p95 218 ms), and `/v1/price-range` and `/v1/comparables` are well within 300 ms.
The histogram leaves out the time before the API's timer starts and the network, so it is a lower bound of what a client on the VM sees; the run of 2026-10-31 on the VM, whose client-side p95 of 210 ms includes the server's time, bounds the server's time on the bare stack from above.
The full stack costs `/v1/predict` about one point: 93.8 % of its answers within 200 ms against 94.7 % on the bare stack with one worker.
The one vCPU was busy but not saturated: the API used 64 % of it, 32 % stayed idle, the hypervisor took none, and some task waited for the CPU 14.3 % of the time (node-exporter's CPU pressure).
The monitor ran no pass during the load: its passes ended at 23:06:57 and 23:21:58 UTC.
The API container was 71 s old when the load began, recreated by the deploy of e9def2e at 23:06:50 UTC, and its tail shows no warm-up: the answers above 250 ms are spread over the five minutes, except for one burst in a single scrape interval, about 23:11:30 to 23:11:45 UTC, that holds all 14 `/v1/predict` answers above 750 ms, 4 of them above 1 s, and the 3 `/v1/price-range` answers above 750 ms.
The client-side p95 of 280 ms for `/v1/predict` is the server's time plus everything between the laptop and the API's timer, so it is no measurement of NFR-02 either way.

**The client's p99 of about 1 s.**
Most of it is not the API's measured time; where it comes from is an inference, not a measurement.
Locust saw 85 of its 6,013 answers take 950 to 1,600 ms (55 for `/v1/predict`, 17 for `/v1/price-range`, and 13 for `/v1/comparables`), while the API took more than 750 ms for only 17 answers of the whole run, and at most 50 ms for every `/v1/comparables` answer.
So at least 68 of the 85 spent more than 200 ms outside the API's timer, and the 13 of `/v1/comparables` more than 900 ms.
Locust's distribution has a gap: few answers between 500 and 950 ms (29 in all), then a cluster from 950 ms to 1.6 s, each endpoint's usual time plus about one second.
That is the shape of a fixed delay of about one second added to some requests, not of a slow server.
None of the earlier runs had a single answer of 950 ms or more: neither the two on the VM of 2026-10-31, which went through the same proxy to the API with one worker at a similar CPU (61 %), nor the laptop's dry run of 2026-10-22.
What those runs did not have: the internet and the university's NAT between the laptop and the VM, Locust on the laptop (load average about 3 on 8 logical CPUs, the Locust process at 7 % of one core on average and 74 % at most), the seven other services of the full stack, and a newer API image.
Two causes on the VM are not excluded either.
The API's timer starts only when its ASGI app receives the request (`MetricsMiddleware` in `recommenditos/api/metrics.py`), so time spent in the socket backlog or waiting for uvicorn's single event loop is outside the histogram.
And the API itself stalled once, at about 23:11:30 to 23:11:45 UTC, which holds all 17 answers above 750 ms.
A delay of one second is what TCP adds when it has to resend the first packet of a new connection (its initial retransmission timeout), but that is a hypothesis nothing in this run proves: the proxy's access log, which records each request's time in nginx (`$request_time`), and the API's logs of the window were lost when the deploy of e378e90 recreated their containers at 23:34 UTC, and no packets were captured.
Settling it takes per-request timestamps on both sides: Locust's full request history (its CSV history, not only the summary) and the proxy's access log with `$request_time`, saved before the next deploy recreates the proxy; matched by time, they show for each slow request whether the second was spent before nginx, in nginx and the API, or in the API's timer.
A run with Locust on the VM and the full stack up, NFR-02's decided method, is still owed.

**Reading the server side again.**
Prometheus keeps 60 days (and at most 2 GB), so these figures can be read again until early January 2027, in Grafana's Explore or on the VM with `docker exec recommenditos-prometheus-1 wget -qO- 'http://127.0.0.1:9090/api/v1/query?time=<time>&query=<URL-encoded query>'`.
The queries and their verbatim answers are in [vm-2026-11-05-full-stack-prometheus.json](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/loadtest/vm-2026-11-05-full-stack-prometheus.json).
The API's counters are cumulative, so the answers of the window are the difference between the scrape after it and the scrape before it, 5 min 15 s earlier, evaluated at `2026-11-04T23:13:15Z`:

```promql
# The answers of the window, by endpoint and bucket; histogram_quantile(0.95, ...) of it gives the p95.
sum by (path, le) (http_request_duration_seconds_bucket{job="api", path=~"/v1/(predict|price-range|comparables)"})
  - sum by (path, le) (http_request_duration_seconds_bucket{job="api", path=~"/v1/(predict|price-range|comparables)"} offset 5m15s)

# The answers of the window, by endpoint and status.
sum by (path, status) (http_requests_total{job="api", path=~"/v1/.+"})
  - sum by (path, status) (http_requests_total{job="api", path=~"/v1/.+"} offset 5m15s)
```

The answers above 750 ms per scrape, as a range query from `2026-11-04T23:06:30Z` to `2026-11-04T23:20:00Z` with a step of 15 s (the difference between two steps is the answers of that interval):

```promql
sum by (path) (http_request_duration_seconds_count{job="api"}) - sum by (path) (http_request_duration_seconds_bucket{job="api", le="0.75"})
```

The CPU, as a mean over the five minutes, evaluated at `2026-11-04T23:13:00Z`:

```promql
sum by (container_label_com_docker_compose_service) (rate(container_cpu_usage_seconds_total{container_label_com_docker_compose_project="recommenditos"}[5m]))
sum by (mode) (rate(node_cpu_seconds_total[5m]))
```

The histogram resolves both targets exactly: 0.2, 0.25, and 0.3 s are bucket bounds (`LATENCY_BUCKETS` in `recommenditos/api/metrics.py`), so the share within 250 or 300 ms is a count, and only the percentiles are interpolated within their bucket (`/v1/predict`'s p95 lies between 200 and 250 ms, its p99 between 300 and 400 ms).
Prometheus scrapes every 15 s, which sets only when an answer is counted, not how its time is measured.

**The decision.**
On 2026-10-31 the team raised NFR-02's `/predict` target to 250 ms, confirmed NFR-03, NFR-04 and NFR-10 as measured, and kept one API worker in production ([EDN-88](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md), issue #143).
