Monitoring
==========

Milestone 6 watches the running system from three sides.
Better Stack checks the public `/health` from outside, so it notices when the VM itself is gone ([External uptime check](deployment.md#external-uptime-check)).
Prometheus, inside the stack, keeps the figures of the API and of the machine it runs on ([Resource monitoring](#resource-monitoring)).
The drift job, a later part of M6, compares what the API is asked with what the model was trained on ([#228](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/228)).
With no real users, its traffic is the [replay](#replay) of held-out listings.
Grafana's dashboards and the alerts follow in [#225](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/225), on top of the Prometheus described here.

## Resource monitoring

Three services of `deploy/compose.yaml` run beside the API and nginx, on a laptop, in CI and on the VM alike ([EDN-99](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md), [#224](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/224)):

| Service | What it reports | Memory limit |
|---------|-----------------|--------------|
| `prometheus` | Scrapes the four targets below every 15 s and stores the samples; its own figures too (TSDB size, failed scrapes) | 384 MiB |
| `cadvisor` | Per container: CPU, memory against the container's limit, the start time and block I/O | 192 MiB |
| `node-exporter` | The host: memory, CPU, load, and the free bytes of each file system, the root's among them | 64 MiB |

Prometheus scrapes, by Compose service name and inside the Compose network:

| Job | Target | Why |
|-----|--------|-----|
| `api` | `api:8000/metrics` | Requests, latency and errors per route, the predicted prices, the warnings and the served model (FR-11, [API docs](api.md#metrics)); one target whatever the number of workers, since in multiprocess mode any worker answers with the sum of all |
| `cadvisor` | `cadvisor:8080` | What the API cannot see of itself: its container's memory and CPU, and its restarts |
| `node` | `node-exporter:9100` | Whether the VM runs out of memory or disk (NFR-04) |
| `prometheus` | `localhost:9090` | Whether Prometheus itself keeps up and how much disk its TSDB takes |

The configuration is [`deploy/prometheus/prometheus.yml`](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/deploy/prometheus/prometheus.yml), mounted read-only, and it travels with every release like the other stack files ([The stack files on the VM](deployment.md#the-stack-files-on-the-vm)).
`make prometheus-check` runs `promtool check config` on it with the stack's own Prometheus image, and CI's integration job runs it before it starts the stack.

### The questions it answers

Each figure answers a question someone on call would ask, and the integration suite checks that each one reaches Prometheus with a value (`tests/integration/test_deployed_prometheus.py`):

| Question | PromQL |
|----------|--------|
| How close is the API to its memory limit? | `container_memory_working_set_bytes{container_label_com_docker_compose_service="api"} / container_spec_memory_limit_bytes{container_label_com_docker_compose_service="api"}` |
| Did a container restart, after a crash, an out-of-memory kill of its main process or a hang (EDN-93)? | `changes(container_start_time_seconds{container_label_com_docker_compose_service="api"}[1h])` |
| How busy is the one vCPU, per container? | `rate(container_cpu_usage_seconds_total{container_label_com_docker_compose_service!=""}[5m])` |
| How much memory is left on the host? | `node_memory_MemAvailable_bytes` |
| How far is the disk from NFR-04's 4 GB floor? | `node_filesystem_avail_bytes{mountpoint="/"}` |
| How much disk does Prometheus take? | `prometheus_tsdb_storage_blocks_bytes + prometheus_tsdb_wal_storage_size_bytes` |

cAdvisor names a container's series by its Compose service and project, the labels `container_label_com_docker_compose_service` and `container_label_com_docker_compose_project`, and by no other label of the image, so that the series stay few.
It is started with `--docker_only`, so it reports Docker's containers and the host's root cgroup and none of the host's other cgroups.
It reads the cgroups every 30 s (`--housekeeping_interval`), which keeps its own CPU low on the VM's one vCPU, and up to 60 s when a container's figures do not change (dynamic housekeeping), so a container's figures are up to 60 s old, and right after a restart they can still show the container as it started.

What the stack does not show:

- **Out-of-memory kills of a worker.** cAdvisor's OOM events read the kernel log from `/dev/kmsg`, which it is not given; a kill of the API's main process shows as a restart, and a kill of one of several workers only as memory at the limit and in the API's log.
- **Network traffic, per container or of the host.** The request traffic comes from the API's own metrics (`http_requests_total` and its latency histogram, [API docs](api.md#metrics)).
  Per-container network bytes are left out on purpose: cAdvisor reads them from the host's `/proc`, which it would need under `/rootfs` with `/var/lib/docker`, widening its root exception for no requirement.
  node-exporter does not use the host's network, so its network figures are those of its own container.
- **The CI runner.** The self-hosted runner is a process on the host, not a container: it shows in the host's memory and CPU, not per container.

### Retention

Prometheus keeps at most 2 GB or 60 days of samples, whichever it reaches first, and drops the oldest block first (`--storage.tsdb.retention.size=2GB`, `--storage.tsdb.retention.time=60d`, NFR-04).
Prometheus reads the units of a size as powers of two, so the cap is 2 GiB, 2.15 GB of the VM's 20 GB disk; the size counts the TSDB's blocks and its write-ahead log.
Sixty days are the rest of the term with room to spare.
The TSDB lives in the named volume `prometheus-data`, so it survives a new image, a changed configuration and a rollback; `docker compose down` keeps it, and only `down -v` removes it.
The integration suite checks both flags and the volume, as NFR-04's "a deployment check that the retention caps are configured, not just intended" asks.

### The textfile collector

node-exporter also exports every `*.prom` file in `/var/lib/node-exporter/textfile` of its container, which is the named volume `node-textfile`, mounted read-only there.
That is how a job that is not a server publishes figures: M6's drift job ([#228](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/228)) will mount the same volume writable and write its results there, without a port or a push gateway of its own.
Nothing writes there yet.
A writer has to write each file under a temporary name and rename it, so that a scrape never reads half a file, and leave it readable to all, because node-exporter runs as `nobody`.

### Reachable only inside the stack

None of the three publishes a port: the VM's ports 8080 to 8089 are public through the NAT ([The VM](deployment.md#the-vm)), and nothing but nginx is meant to be (NFR-09).
The integration suite checks that no service but the proxy publishes one and that none uses the host's network, which would open its ports on the host without a `ports` entry.
Grafana ([#225](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/225)) will be the way in, through an SSH tunnel to the VM.
Until then, Prometheus' HTTP API is reached from inside its container, whose busybox `wget` asks it on `127.0.0.1:9090`:

```bash
# Locally, from the repository root:
docker compose -f deploy/compose.yaml exec prometheus \
  wget -qO- 'http://127.0.0.1:9090/api/v1/targets?state=active'
docker compose -f deploy/compose.yaml exec prometheus \
  wget -qO- 'http://127.0.0.1:9090/api/v1/query?query=node_filesystem_avail_bytes'

# On the VM, from /opt/recommenditos, with the paths deploy.sh uses:
docker compose -f "$(readlink -f stack)/compose.yaml" --env-file .env exec prometheus \
  wget -qO- 'http://127.0.0.1:9090/api/v1/targets?state=active'
```

For Prometheus' own web page during debugging, a temporary tunnel reaches the container's address on the VM's Docker network without publishing anything, from the UPC network or UPClink:

```bash
ip=$(ssh -p 22052 recommenditos@nattech.fib.upc.edu \
  docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' recommenditos-prometheus-1)
ssh -p 22052 -N -L 9090:"$ip":9090 recommenditos@nattech.fib.upc.edu   # then open http://localhost:9090
```

The address changes when the container is recreated, so look it up again after a deploy.

### Hardening, and cAdvisor's exception

All three run with the shared hardening of the stack: a read-only root file system, no Linux capability, `no-new-privileges`, and limits on memory and processes ([The stack](deployment.md#the-stack)).
Prometheus and node-exporter run as `nobody`.
cAdvisor runs as root, because it reads the Docker socket and `/var/lib/docker`, which only root can.
The Docker socket is root-equivalent on the host, however read-only its mount is: whoever can talk to it can start a privileged container.
That is the price of per-container figures, and it is kept small: cAdvisor drops every capability, mounts the socket, `/sys` and `/var/lib/docker` read-only and nothing else of the host, publishes no port, and serves only its metrics and its read-only web page to the Compose network.
The integration suite names cAdvisor as the one service that runs as root and checks that every host path it mounts is read-only.
node-exporter reads the host's `/` under `/host`, read-only too, and needs neither the host's network nor its PID namespace for the figures above.

cAdvisor reads the containers' cgroups, which are cgroup v2 on CI's runners and current laptops, and cgroup v1 on the VM (Ubuntu 20.04).
The figures above have the same names under both.

### When a monitoring service does not start

The three services are part of the release, so a deploy waits for them as for the API: `deploy.sh` runs `docker compose up --wait`, and a monitoring service that does not become healthy fails the deploy, which then puts the previous release back, the API included ([Every deploy](deployment.md#every-deploy)).
That is wanted for a broken configuration, which `make prometheus-check` and CI's integration job should catch first, and unwanted for a registry that cannot be reached: the images come from quay.io and ghcr.io, and a VM that cannot pull one keeps the release that runs.
Their healthchecks therefore probe only that each serves (`/-/ready`, `/healthz`, the landing page), not what it scrapes, so a failing scrape target never fails a deploy.
Each also restarts on its own after a crash and after a reboot (`restart: unless-stopped`, NFR-05).

## Replay

M6 has no real users ([EDN-13](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)), so the traffic that makes drift, error and coverage visible is a replay of held-out listings ([#227](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/227), [EDN-100](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
It sends listings to the running API as a client would, and reports each listing's real price to `/v1/feedback` (FR-10), so the prediction log holds the requests the drift job compares with the training data and the labels it computes the error and the coverage from (NFR-11, FR-14).

It is the service `replay` of `deploy/compose.yaml`, behind the profile `replay`, in the monitoring image (`monitoring/Dockerfile`), and it runs by hand, to its end.
`up`, and with it every deploy, neither starts it nor removes a running one.
The monitoring image is its own uv project, `monitoring/`, with numpy 1.26 and pandas 2, because the drift job's Alibi Detect needs them; the API image stays the serving runtime (NFR-04).

### What it sends

The listings come from the samples of the `monitoring_samples` stage, which CD bakes into the image ([The monitoring samples](pipeline.md#the-monitoring-samples)):

| Sample | Listings | What it is |
|--------|----------|------------|
| `control` | 20,000 | Listings of `train` outside the drift reference, drawn at random from the same rows, so a control window is i.i.d. with the reference and the drift job must not flag it |
| `es` | 5,779 | Every listing of the `ES` holdout the API accepts (EDN-03), the market the model never saw, which the drift job must flag |

`--control N` sends the first N control listings, and `--then es` sends the ES listings after them, so a dashboard shows a clean before and after; `--then-count N` stops after N of them.
The listings are in a random order, so any run of consecutive ones is a random sample of its set.
The model was fitted on the control listings, so their error and coverage are in-sample and optimistic: they show that the labels flow, not how good the model is.
The drift job compares the error and coverage of a window with the test values in `metrics.json` instead ([#229](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/229)).

Each listing is sent with the body its sample holds: the fields the listing has, and nothing for a value it lacks, so the log records what such a client sends (#191).
The listings alternate between `/v1/predict` and `/v1/price-range`, so both endpoints' panels move and a window gets about as many ranges as point estimates.
They go through nginx, `http://proxy:8080`, as a client's requests do, with `X-Client: replay`, which the log records as `client`.

`registration_date` is moved forward by the whole months from the samples' `reference_date`, 2025-11-08, to the send time.
The model computes ages on 2025-11-08 and the API on the day it answers, so a listing of the 2025 snapshot replayed a year later would otherwise look a year older, and every control window would be flagged on `age_years`.
A month is complete at 00:00 UTC on the 8th; the drift job computes the age of a logged request with the same rule, so the two shifts cancel to within three days (`drift_features.py`).

After each priced answer, the listing's observed price goes to `http://api:8000/v1/feedback` with the answer's `request_id` and `RECOMMENDITOS_API_KEY` from `.env` in `X-API-Key`, inside the Compose network, because nginx refuses the path from outside.
A feedback that fails is sent again, up to three times with a growing pause, because the same price again is answered 200 and changes nothing.
A valuation is sent again only when it surely reached no route: no connection, or a 502 or 503.
After a read timeout the API may have logged the answer, so it is counted and not sent again, and no listing is logged twice.
A 401 from `/v1/feedback` stops the run, since no price could be stored with that key.

The listings go one at a time at `--rate` per second (5 by default), so 5,000 control listings take about 17 minutes and all 25,779 about 86 minutes; when the API answers more slowly, the rate drops, and the summary says so.

### Run it locally

From the repository root, with the samples, the model and the index:

```bash
uv run dvc pull monitoring_samples models/lgbm-basic data/processed/comparables.parquet
export RECOMMENDITOS_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
docker compose -f deploy/compose.yaml up -d --build --wait
docker compose -f deploy/compose.yaml --profile replay build replay
docker compose -f deploy/compose.yaml --profile replay run --rm replay \
    --control 300 --then es --then-count 300 --rate 5
```

`run` starts the API and nginx first if they are not running, and waits for a healthy API.
The replay streams the samples a listing at a time and keeps only its counts: in the run in [The summary](#the-summary) its container peaked at 76 MB (`memory.peak`, page cache included) of its 256 MiB limit.

### Run it on the VM

Over SSH, with UPClink off campus, as `recommenditos`, from `/opt/recommenditos`, with the paths `deploy.sh` uses ([The stack files on the VM](deployment.md#the-stack-files-on-the-vm)).
`deploy.sh` pulls the monitoring image of every release beside the API image, so the image of the running tag is there; `.env` names its repository in `MONITORING_IMAGE`, as `.env.template` does.
A long run goes in the background, so a dropped SSH connection does not end it:

```bash
ssh -p 22052 recommenditos@nattech.fib.upc.edu
cd /opt/recommenditos
docker compose -f "$(readlink -f stack)/compose.yaml" --env-file .env --profile replay \
    run -d --name replay replay --control 5000 --then es --rate 5
docker logs -f replay          # the progress, and the summary at the end
docker rm replay               # once it has ended
```

The drift job's windows of 1,000 requests fill at the rate the replay is given; NFR-11's twenty control windows take the whole control sample.

### The summary

The run ends with one line per sample and a total: the listings sent, the valuations by answer and the feedback by answer, then the duration and the rate it reached.
It exits 0 when every listing was priced (200) and its price stored (201, or 200 for a repeat), and 1 otherwise; a 409 or 404 from `/v1/feedback` is counted there.
For example, the run of 300 and 300 listings above on a laptop on 2026-11-04:

```text
phase    sent  answered  feedback
control  300   200: 300  201: 300
es       300   200: 300  201: 300
total    600   200: 600  201: 600
duration 119.9 s, 5.01 listings/s (target 5)
result: ok
```

### What it leaves in the log

Every replayed answer is a row with `client = 'replay'` and `inputs_sent_only = 1`, which the drift job reads ([Reading the log](api.md#reading-the-log)), and each one with a price has its feedback, which `labelled_predictions` joins to it:

```sql
SELECT client, inputs_sent_only, endpoint, COUNT(*) FROM predictions GROUP BY 1, 2, 3;
SELECT json_extract(inputs, '$.country_code') AS country, COUNT(*)
FROM labelled_predictions GROUP BY 1;
```
