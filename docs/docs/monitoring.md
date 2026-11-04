Monitoring
==========

Milestone 6 watches the running system from three sides.
Better Stack checks the public `/health` from outside, so it notices when the VM itself is gone ([External uptime check](deployment.md#external-uptime-check)).
Prometheus, inside the stack, keeps the figures of the API and of the machine it runs on ([Resource monitoring](#resource-monitoring)).
The [drift job](#drift-job) compares what the API is asked with what the model was trained on.
With no real users, its traffic is the [replay](#replay) of held-out listings.
[Grafana](#grafana) shows Prometheus' figures against the limits the requirements set, and [alerts](#alerts) post to the team's Discord before a limit is reached, each with what to do ([#225](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/225)).

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

node-exporter also exports every `*.prom` file in `/var/lib/node-exporter/textfile` of its container, which is the named volume `monitor-textfile`, mounted read-only there.
That is how a job that is not a server publishes figures: the [drift job](#drift-job) mounts the same volume writable and writes `monitor.prom` there after every pass, without a port or a push gateway of its own.
It writes the file under a temporary name and renames it, so that a scrape never reads half a file, and leaves it readable to all, because node-exporter runs as `nobody`.
The volume is a tmpfs that Docker mounts with the job's user, 10001, as its owner: a volume on disk would be root's when node-exporter, which runs as `nobody` from an image without that directory, mounted it first, and the job could not write it.
A reboot empties it until the next pass, and the volume `node-textfile` of the first stacks is no longer used.

### Reachable only inside the stack

None of the three publishes a port: the VM's ports 8080 to 8089 are public through the NAT ([The VM](deployment.md#the-vm)), and nothing but nginx is meant to be (NFR-09).
The integration suite checks that no service but the proxy publishes one beyond the host's loopback, and that none uses the host's network, which would open its ports on the host without a `ports` entry.
[Grafana](#grafana) is the way in, through an SSH tunnel to the VM.
For a figure the dashboard does not show, Prometheus' HTTP API is reached from inside its container, whose busybox `wget` asks it on `127.0.0.1:9090`:

```bash
# Locally, after make stack:
docker exec recommenditos-prometheus-1 \
  wget -qO- 'http://127.0.0.1:9090/api/v1/targets?state=active'
docker exec recommenditos-prometheus-1 \
  wget -qO- 'http://127.0.0.1:9090/api/v1/query?query=node_filesystem_avail_bytes'

# On the VM, from /opt/recommenditos, with the paths deploy.sh uses:
docker compose -f "$(readlink -f stack)/compose.yaml" --env-file .env exec prometheus \
  wget -qO- 'http://127.0.0.1:9090/api/v1/targets?state=active'
```

For Prometheus' own web page during debugging, a temporary tunnel reaches the container's address on the VM's Docker network without publishing anything, with UPClink connected ([Open it](#open-it)):

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

The monitoring services, Grafana and Alertmanager below included, are part of the release, so a deploy waits for them as for the API: `deploy.sh` runs `docker compose up --wait`, and a monitoring service that does not become healthy fails the deploy, which then puts the previous release back, the API included ([Every deploy](deployment.md#every-deploy)).
That is wanted for a broken configuration, which `make prometheus-check` and CI's integration job should catch first, and unwanted for a registry that cannot be reached: the images come from quay.io, ghcr.io and, for Grafana, Docker Hub, whose anonymous pull limit counts per address and the FIB's NAT is shared; a VM that cannot pull one keeps the release that runs.
A deploy pulls an image only when its digest in `compose.yaml` changed.
Their healthchecks therefore probe only that each serves (`/-/ready`, `/healthz`, the landing page, Grafana's `/api/health`), not what it scrapes, so a failing scrape target never fails a deploy.
Each also restarts on its own after a crash and after a reboot (`restart: unless-stopped`, NFR-05).

## Grafana

Grafana runs in the stack and shows one provisioned dashboard, **Service health**, which is also its home page: the figures of [Resource monitoring](#resource-monitoring) against the limits the requirements set ([EDN-99](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md), [#225](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/225)).

| Panel | Against | Alert |
|-------|---------|-------|
| API up, scrape target health | NFR-05: back within 30 s | [ApiDown](#apidown), [TargetDown](#targetdown) |
| API uptime in the time range | NFR-05: monitored and reported | - |
| Firing alerts | - | all of [Alerts](#alerts) |
| Request rate per route | FR-11 | - |
| p95 latency of `/v1/predict` | NFR-02: 250 ms ([EDN-88](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)) | [PredictLatencyHigh](#predictlatencyhigh-and-rangelatencyhigh) |
| p95 latency of `/v1/price-range` and `/v1/comparables` | NFR-02: 300 ms | [RangeLatencyHigh](#predictlatencyhigh-and-rangelatencyhigh) |
| Server error rate | NFR-03: 1 % | [ErrorRateHigh](#errorratehigh), [ServerErrors](#servererrors) |
| API memory | its 768 MiB limit (NFR-04 caps it at 1 GB) | [ContainerMemoryNearLimit](#containermemorynearlimit) |
| Memory of each container, share of its limit | 90 % | [ContainerMemoryNearLimit](#containermemorynearlimit) |
| CPU of each container | the VM's one vCPU | - |
| Host memory available | 512 MiB | [HostMemoryLow](#hostmemorylow) |
| Disk free | NFR-04: 4 GB | [DiskFreeLow](#diskfreelow) |
| Container restarts in the last 30 minutes | 3 | [ContainerRestarting](#containerrestarting) |
| Predicted prices, prediction warnings by field | FR-11, FR-05 | drift and error follow in [#230](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/230) |

Every target and limit is drawn in red, as a threshold line or, for the API's memory limit and the host's memory and disk floors, as a dashed series, so that none is mistaken for data.
The uptime panel is the share of Prometheus' scrapes of the API that succeeded in the dashboard's time range, seen from inside the stack.
It cannot see the VM or its network being gone; the uptime NFR-05 reports is Better Stack's, from outside ([External uptime check](deployment.md#external-uptime-check), EDN-10).

![The Service health dashboard on a laptop's stack under a light load](https://raw.githubusercontent.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/main/reports/figures/grafana-service-health-local.png)

The screenshot is a local stand-in, taken on a laptop's stack under 5 requests per second for two and a half minutes; the one for the report is taken on the VM under NFR-03's load test.

### Provisioned, not clicked

Everything Grafana shows comes from `deploy/grafana/`, which travels with every release like the other stack files:

| File | What it provisions |
|------|--------------------|
| `provisioning/datasources/prometheus.yml` | The data source `Prometheus`, uid `prometheus`, at `http://prometheus:9090`, not editable |
| `provisioning/dashboards/dashboards.yml` | The dashboards in `dashboards/`, which the UI can neither save over (`allowUiUpdates: false`) nor delete |
| `dashboards/service-health.json` | The dashboard, uid `service-health` |

A change to a panel is a change to the JSON file, reviewed in a pull request: change it in a local Grafana, copy the JSON (**Export**, **Export as JSON**), and remove the ids Grafana adds, the dashboard's `id`, `version` and `iteration`.
`tests/test_monitoring_config.py` fails on such ids, on repeated panel ids and on a panel that queries another data source, and checks that every alert links a panel that exists.
Panel ids stay what they are, since the alerts link panels by id.
Grafana reads the folder again every minute, so a new release's dashboard shows without a restart.
The integration suite checks that a fresh stack has the data source, reaches Prometheus through it, and shows the dashboard to an anonymous visitor (`tests/integration/test_deployed_grafana.py`).

### Open it

Grafana listens on the host's `127.0.0.1:3000` only, on a laptop and on the VM alike, so no other machine reaches it; on the VM it is opened through an SSH tunnel:

```bash
ssh -p 22052 -N -L 3000:127.0.0.1:3000 recommenditos@nattech.fib.upc.edu
# then open http://localhost:3000 in the browser
```

The tunnel needs UPClink connected, on campus too: from the university's Wi-Fi (eduroam) the SSH port timed out on 2026-11-04 ([The VM](deployment.md#the-vm)).
The links in the alerts' messages point at `http://localhost:3000`, so they open once the tunnel is up.

Inside the tunnel, a visitor is an anonymous Viewer: every dashboard can be seen, nothing can be changed.
The admin logs in as `admin` with `GRAFANA_ADMIN_PASSWORD` from the stack's `.env`; compose.yaml has no default for it and refuses to start without one.
Grafana sets the password when it creates its database, in the named volume `grafana-data`, on its first start; a later change to `.env` does not change it, `docker exec recommenditos-grafana-1 grafana cli admin reset-admin-password <new>` does.
The rejected alternatives, Grafana read-only behind nginx's public port and screenshots only, are in [EDN-99](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md).

Grafana runs with the shared hardening (read-only root, no capability, `no-new-privileges`, its own user 472), with its database in the volume and a tmpfs on `/tmp`.
It sends nothing out: usage reports, update checks, the news feed, avatars and plugin downloads at start-up are off, and so is its own alerting, since the alerts are Prometheus' ([Alerts](#alerts)).
The data sources the image bundles besides Prometheus are turned off: each would run as a process of its own, and together they exhausted the container's 128 processes.

## Alerts

Prometheus evaluates the alerting rules in `deploy/prometheus/rules/` every 15 s and hands what fires to Alertmanager, which posts it to the team's Discord channel through a webhook ([EDN-99](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
Each message carries the alert's summary, what to do, a link to its section below and a link to its dashboard panel through the [tunnel](#open-it).
Alertmanager groups the alerts by name, waits 30 s so that alerts firing together arrive in one message, posts again every 4 hours while a problem lasts, and posts when it is resolved.
While the API is down, its latency and error alerts are held back, and while ErrorRateHigh fires, so is ServerErrors.

| Alert | Fires when | For | Severity |
|-------|------------|-----|----------|
| [ApiDown](#apidown) | Prometheus cannot scrape `api:8000` | 1 minute | critical |
| [PredictLatencyHigh](#predictlatencyhigh-and-rangelatencyhigh) | more than 5 % of the `/v1/predict` answers of the last minute took over 250 ms, that is, the p95 is above NFR-02's target, with at least 20 answers in that minute | 5 minutes | warning |
| [RangeLatencyHigh](#predictlatencyhigh-and-rangelatencyhigh) | the same for `/v1/price-range` or `/v1/comparables` and 300 ms, per route | 5 minutes | warning |
| [ErrorRateHigh](#errorratehigh) | more than 1 % of the `/v1/` answers of the last minute are server errors (5xx), with at least 20 answers in that minute | 5 minutes | critical |
| [ServerErrors](#servererrors) | the API answered at least 5 `/v1/` requests with a server error in the last 15 minutes, at any traffic | at once | warning |
| [DiskFreeLow](#diskfreelow) | the host's root file system has less than 4 GB free | 5 minutes | critical |
| [HostMemoryLow](#hostmemorylow) | the host has less than 512 MiB available | 5 minutes | warning |
| [ContainerMemoryNearLimit](#containermemorynearlimit) | a container uses more than 90 % of its memory limit | 5 minutes | warning |
| [ContainerRestarting](#containerrestarting) | a container restarted 3 times in 30 minutes | at once | critical |
| [TargetDown](#targetdown) | a monitoring target other than the API cannot be scraped | 5 minutes | warning |

Why these thresholds:

- **Latency** is read from the histogram's 0.25 s and 0.3 s buckets, which are NFR-02's targets, so "more than 5 % slower than the target" is exact where an interpolated p95 is not; five minutes are as long as NFR-03's load test.
- **Errors** count server errors only: a 4xx is the API's correct answer to a bad request (FR-16), and the public internet sends many.
  The `/v1/` routes only, so that the healthcheck's `/health` every 5 s does not dilute the share; the dashboard shows the 4xx share beside it.
- **A one-minute share, for 5 minutes:** the share is taken over the last minute and must be above the target on every evaluation, every 15 s, for 5 minutes, so a breach fires only after it has lasted about 5 minutes; the rules' tests show a 6-minute breach at the replay's 5 requests per second firing 5 minutes 15 s after it began.
  A share over 5 minutes would stay above the target for 5 minutes after a spike of 30 s, so with `for: 5m` it would fire on that spike: at the replay's rate, any half minute of a busy vCPU (the drift job runs every 15 minutes) would post a firing and a resolved message to Discord.
  The tests show a one-minute spike of slow answers, and one of server errors, staying quiet.
- **At least 20 answers** in that minute, the rate of 100 in 5 minutes, so that one slow answer or one server error on a quiet day does not fire.
- **ServerErrors** has no such floor: outside the replay and the load tests the API gets almost no traffic ([EDN-13](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)), and ErrorRateHigh would not see an API that fails every one of its 10 requests a minute.
  A healthy API answers no server errors at all, so five in 15 minutes are worth a look at any traffic; Prometheus' `increase` does not count the first answer of a status code the API has not given since it started, so it takes up to six.
- **Disk:** NFR-04's 4 GB floor is headroom, not the end of the disk, so the alert fires with room left to act.
- **Host memory:** the VM has no swap, so memory that runs out ends with the kernel killing a process; 512 MiB is the drift job's peak of up to 400 MiB per run with room to spare.
- **Restarts:** one restart is NFR-05 working (a crash, or a hang the healthcheck ends, EDN-93); three in half an hour are a loop.
  A deploy recreates a container, which counts as a new one, not as a restart.

The rules have unit tests, `deploy/prometheus/tests/service-health.test.yml`: for each alert, series that make it fire after its `for` duration with its labels and annotations, and series that must not.
`make prometheus-check` runs them with `promtool test rules`, after `promtool check config`, and checks Alertmanager's configuration with `amtool check-config`; CI's integration job runs it before it starts the stack.
The integration suite checks that Prometheus has loaded every rule and found Alertmanager, and that Alertmanager runs the Discord receiver.

### The Discord webhook

The webhook URL is a secret, since whoever has it can post to the channel: it is `DISCORD_WEBHOOK_URL` in the stack's `.env` ([`.env.template`](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/deploy/.env.template)), and compose.yaml has no default for it, so a stack without it does not start rather than drop every alert.
Alertmanager does not expand environment variables in its configuration, and reads the URL from a file (`webhook_url_file`): the service's entrypoint writes it to a tmpfs that only Alertmanager's user can read, and starts Alertmanager without it in its environment.
A Compose secret would keep it out of `docker inspect` too, but Compose does not mount one from the environment into a read-only container.

A local stack (`make stack`) and CI's get a webhook that goes nowhere, so a laptop's or a runner's alerts never reach the team; `make stack LOCAL_DISCORD_WEBHOOK_URL=<url>` sends them somewhere on purpose.
The receiver was tested on 2026-11-04: a local Alertmanager posted the test alert `DiscordReceiverTest` (about 18:15 UTC) to the team's channel, where it arrived.

### What the alerts do not cover

- **Prometheus or Alertmanager down.** Nothing then evaluates or sends the alerts, and nothing says so; the dashboard's target health shows it, and Better Stack still notices the API being gone ([External uptime check](deployment.md#external-uptime-check)).
- **The VM or its network down.** The same: Better Stack's check from outside sends its own email.
- **Slowness on a quiet API.** The latency alerts need 20 answers a minute, so outside the replay and the load tests a slow API stays unnoticed; ServerErrors covers only its errors.
- **Fewer than five server errors in 15 minutes**, and errors that nginx answers itself (a 502 or 503 when the API does not answer in time): the API's metrics count only the answers it gives, and ApiDown and Better Stack cover an API that gives none.
- **Drift, error and coverage** of the model follow in [#230](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/230), as more rule files in `deploy/prometheus/rules/` and more panels.

### Runbook

On the VM, the commands below run as `recommenditos` over SSH, from `/opt/recommenditos` ([Deploy to the VM](deployment.md#deploy-to-the-vm)).

#### ApiDown

The API has not answered Prometheus' scrape for a minute, so it did not recover within NFR-05's 30 s on its own.

1. `./deploy.sh status` and `docker ps -a`: is the container running, restarting or gone?
2. `docker logs --tail 100 recommenditos-api-1`: a traceback at start-up is a broken image or bundle; a kill is memory (the API memory panel).
3. If the last deploy brought it, `./deploy.sh rollback` starts the previous release.
4. If the container is up but does not answer, the healthcheck ends it within about 21 s (EDN-93); if that does not happen, `docker restart recommenditos-api-1`.

#### PredictLatencyHigh and RangeLatencyHigh

More than 5 % of a route's answers took longer than NFR-02's target for five minutes.

1. On the dashboard, the CPU panel: does another container (the drift job, cAdvisor) or the self-hosted runner take the one vCPU at the same time? A load test or the replay running is expected to cause it.
2. The request rate: a traffic peak beyond NFR-03's 20 requests per second is outside the promise.
3. If neither, compare with the last deploy: a slower model or image is rolled back with `./deploy.sh rollback`, and the regression gets an issue with the panel.

#### ErrorRateHigh

More than 1 % of the `/v1/` answers were server errors for five minutes.

1. `docker logs --since 10m recommenditos-api-1`: the failing requests, their status codes and request IDs, and the tracebacks.
2. A full disk or a locked prediction log shows in the disk panel and the log; free space ([DiskFreeLow](#diskfreelow)).
3. If the errors began with a deploy, `./deploy.sh rollback`.

#### ServerErrors

The API answered at least five `/v1/` requests with a server error (5xx) in the last 15 minutes, which a healthy API never does, at any traffic.
On a busy API that stays above NFR-03's 1 %, [ErrorRateHigh](#errorratehigh) follows and holds this warning back.

1. `docker logs --since 20m recommenditos-api-1`: the failing requests, their status codes, request IDs and tracebacks.
2. One client's requests, or every request? A single malformed body that the API fails on instead of answering 422 is a bug: open an issue with the request ID.
3. If the errors began with a deploy, `./deploy.sh rollback`.

#### DiskFreeLow

Less than NFR-04's 4 GB are free on the host's root file system.

1. `docker system df` and `sudo du -xh -d1 /var/lib/docker`: images, volumes or container logs?
2. Old image tags: `docker image prune -a` removes the images no container uses, the previous release's too, so keep the tag `.deploy-previous` names (`docker image ls`).
3. Prometheus' TSDB is capped at 2 GB and the prediction log is pruned to its own cap; if one of them is larger, that cap is broken: open an issue.

#### HostMemoryLow

The host has less than 512 MiB available, and the VM has no swap.

1. The panel of each container's memory: which one grew?
2. Is a load test, the replay or a CI job on the self-hosted runner running? Those end on their own.
3. A container that keeps growing without load leaks: `docker restart recommenditos-<service>-1`, and an issue with the panel.

#### ContainerMemoryNearLimit

A container uses more than 90 % of its memory limit, and the kernel will soon kill its largest process.

1. Steady growth without more traffic is a leak: restart the container (`docker restart recommenditos-<service>-1`) and open an issue with the panel.
2. Growth with the traffic means the limit in `deploy/compose.yaml` is too small for that load: raise it in a pull request, with the measurement.

#### ContainerRestarting

A container restarted three times in 30 minutes.

1. `docker logs --since 30m recommenditos-<service>-1`: what it logged before each restart.
2. The memory panel: at its limit, it is killed for memory ([ContainerMemoryNearLimit](#containermemorynearlimit)); for the API, a hang ends with the healthcheck's kill (EDN-93).
3. If the restarts began with a deploy, `./deploy.sh rollback`.

#### TargetDown

Prometheus could not scrape a monitoring target for five minutes, so its panels and the alerts built on it see nothing.

1. `docker ps -a` and `docker logs --tail 100 recommenditos-<service>-1` for the target's service (cAdvisor, node-exporter, Alertmanager, Grafana, Prometheus itself).
2. Each restarts on its own after a crash, so one that stays down needs its log read; a configuration it refuses fails `make prometheus-check` in CI first.

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
A month is complete at 00:00 UTC on the 8th; the drift job computes the age of a logged request with the same rule, so the two shifts cancel to within four days, three in the year after 2025-11-08 (`drift_features.py`).

After each priced answer, the listing's observed price goes to `http://api:8000/v1/feedback` with the answer's `request_id` and `RECOMMENDITOS_API_KEY` from `.env` in `X-API-Key`, inside the Compose network, because nginx refuses the path from outside.
A feedback that fails is sent again, up to three times with a growing pause, because the same price again is answered 200 and changes nothing.
A valuation is sent again only when it surely reached no route: no connection, or a 502 or 503.
After a read timeout the API may have logged the answer, so it is counted and not sent again, and no listing is logged twice.
That is why `--timeout` is at most 50 s: after 60 s without the API's answer nginx answers 503 itself (its default `proxy_read_timeout`, mapped to 503 in `deploy/nginx.conf`), the API may still log that listing, and a 503 is sent again, so the replay's own read timeout has to come first.
A 401 from `/v1/feedback` stops the run, since no price could be stored with that key.

The listings go one at a time at `--rate` per second (5 by default), so 5,000 control listings take about 17 minutes and all 25,779 about 86 minutes; when the API answers more slowly, the rate drops, and the summary says so.

### Run it locally

From the repository root, with the samples, the model and the index:

```bash
uv run dvc pull monitoring_samples models/lgbm-basic data/processed/comparables.parquet
export RECOMMENDITOS_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
# The stack refuses to start without these two; locally, alerts go nowhere.
export GRAFANA_ADMIN_PASSWORD=$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')
export DISCORD_WEBHOOK_URL=http://127.0.0.1:9/discarded
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
A long run goes in the background, so a dropped SSH connection does not end it.
Its output goes into a file beside `.env` as it is written, because the next deploy's `up --remove-orphans` removes a replay container that has ended, and its logs with it, so the summary must not wait in `docker logs`:

```bash
ssh -p 22052 recommenditos@nattech.fib.upc.edu
cd /opt/recommenditos
docker compose -f "$(readlink -f stack)/compose.yaml" --env-file .env --profile replay \
    run -d --name replay replay --control 5000 --then es --rate 5
nohup docker logs -f replay > "replay-$(date -u +%Y-%m-%dT%H%M%SZ).log" 2>&1 &
tail -f replay-*.log           # the progress, and the summary at the end
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

## Drift job

The drift job ([#228](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/228), [EDN-101](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)) compares each window of 1,000 logged requests with the 10,000 training listings of the drift reference, and says which input features drifted (FR-14, NFR-11).
It is the `monitor` service of the stack, in the monitoring image of the replay.

### What it compares

A window is 1,000 consecutive rows of the prediction log by `seq` that are answers of `/v1/predict` or `/v1/price-range`, with `inputs_sent_only = 1`, from no marked client or from the replay ([Reading the log](api.md#reading-the-log)), and with all ten fields FR-01 requires.
The load test, the smoke checks, the web page and the rows from before #191 are left out by `client` and `inputs_sent_only`; a range request with only some fields is left out by the last rule, because the reference has every required field.
A change of `model_version` closes the window being filled as `partial`, untested, so that every window belongs to one model.

Each request becomes the 23 features of `drift_features.py`, its age computed on `as_of_for(answered_at)`, the rule that cancels the replay's shift of `registration_date` ([What it sends](#what-it-sends)).
Each feature has an Alibi Detect detector of its own, fitted on `data/monitoring/reference.parquet`:

- the eight numeric features, `KSDrift` on their non-missing values, of the reference and of the window alike;
- the seven categorical features and the eight missing indicators, `ChiSquareDrift`, over the categories of the reference and of the window together, so a category the reference never saw counts.

A feature has drifted when its p-value is below 0.005 / 23, about 0.000217, and a window is flagged when any feature has (NFR-11, EDN-14).
`model` is not compared: its 349 levels leave well under one expected count per cell of a chi-square on 1,000 requests.
`TabularDrift`, which runs the same tests over a whole table, cannot leave out the missing values of one column alone, so the job runs one detector per feature and applies the Bonferroni correction itself, as the validated scipy check `reports/analysis/nfr11_check.py` did.
The indicators `age_years__missing`, `mileage_km_raw__missing` and `power_kw__missing` are `present` in every accepted request; `ChiSquareDrift` accepts such a one-category feature and gives it the p-value 1, so they stay in the 23 and the threshold is the validated one.

### How it runs

The service is a shell loop, `monitoring/monitor-loop.sh`, that starts one pass, `python -m recommenditos_monitoring run-once`, and sleeps 15 minutes after it ends (`RECOMMENDITOS_MONITOR_INTERVAL_S`, 900 s).
An interval that is not a positive whole number of seconds would make `sleep` fail at once and run the passes back to back, so the loop prints a warning to the service log and sleeps 900 s instead.
The Python process ends after every pass, so between passes only the shell, its `sleep` and Docker's init run, 4 MiB of resident memory in `docker top` (10.5 MiB in `docker stats`, which counts the page cache).
A pass runs at the container's start too, so every deploy runs one.
A pass reads the log from the end of the last closed window, tests every window that has filled since, stores each with its results in one transaction, and records the progress of the window being filled; a rerun never stores a window twice, and a second pass while one runs leaves it to finish.
A pass that fails prints `the drift pass failed: ...` to the service log, records the failure in `monitor.db`, leaves the last metrics in place, and the loop goes on.

The service has no healthcheck at all: with none, `up --wait` waits only for the container to run, so a failing pass can never fail a deploy, and Compose 2.35 on CI and the VM fails `--wait` on `disable: true`.
It is limited to 512 MiB and half a vCPU, which leaves the API's latency (NFR-02) the other half of the VM's one vCPU; passes that tested three windows peaked at 197 MiB in the container's cgroup (`memory.peak`) on a laptop, most of it Alibi Detect's import (it raised a Python process's peak resident memory from 110 MB with pandas and pyarrow alone to 291 MB), which a pass with no full window skips.
It runs as the API's user, 10001, not as the image's: SQLite's WAL needs the log's directory writable for its shared-memory file even to read it, and that directory on the `prediction-log` volume is the API user's.
The job opens the log read-only all the same.

Alibi Detect 0.13 pins Pillow below 11 and transformers below 5, both with fixed HIGH findings that the image scan of NFR-09 fails on.
The image leaves Pillow out, because only Alibi Detect's image detectors and Matplotlib use it and the job imports neither; the integration suite checks that it is gone and that a pass runs.
transformers stays, since Alibi Detect imports it on import; its three findings are about loading and saving models, which the job never does, and are accepted in `.trivyignore` until 2026-12-04.
The image is 1.26 GB, most of it Alibi Detect's dependencies.

### Where the results go

`monitor.db`, on the volume `monitor-results`, keeps everything; it survives a new image and a rollback, and only `down -v` removes it.

| Table | One row per |
|-------|-------------|
| `windows` | Closed window: `status` (`tested` or `partial`), its first and last `seq`, its time range (`first_answered_at`, `last_answered_at`), the model, `es_share`, and when tested `flagged`, `drifted_features` and the threshold |
| `window_features` | Feature of a tested window: the test (`ks` or `chi2`), the p-value, the statistic, the values compared, and `drifted` |
| `window_requests` | Request of a window that the log still holds, with its `request_id`, so a window's labels are the feedback of exactly its requests ([#229](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/229)) |
| `open_window` | The window being filled: n of 1,000 |
| `runs` | Pass, with its outcome, the last 10,000 |

After each pass the job writes `monitor.prom` into the volume `monitor-textfile`, which node-exporter serves ([The textfile collector](#the-textfile-collector)): the latest tested window's flag, its number of drifted features, each feature's p-value (`recommenditos_drift_feature_p_value{feature, test}`), the threshold, its number, last `seq`, end time and ES share, the open window's progress, and `recommenditos_monitor_last_success_timestamp_seconds`, the end of the last pass that succeeded.
That last one, and node-exporter's `node_textfile_mtime_seconds`, show a job that stopped.

### How large it grows

`monitor.db` is bounded, as NFR-04 asks of storage that grows: it never holds more than the log's cap allows, and its evidence grows by a few kilobytes a window.

- `window_requests` takes 121 bytes a request, its two indexes included.
  The log's rotation deletes the oldest answers with their feedback ([Rotation](api.md#rotation)), after which no label can be joined to them, so every pass drops the rows of requests below the log's lowest `seq`.
  The table therefore holds at most the requests the log holds.
  The log's 2 GB hold 820,000 answers of the replay's full listings (2,445 bytes each) or 3.1 million of the ten required fields alone (637 bytes each), so the table stays at about 100 MB with the replay's traffic and at about 380 MB in the worst case.
- `windows` and `window_features` take about 2.7 KB a window of 1,000 requests, and stay: they are the evidence for the report and for retraining.
  At 5 requests a second around the clock, which only a replay sends, that is 1.2 MB a day and 0.4 GB a year, against the 20 GB disk with at least 8 GB free.
- `runs` keeps the last 10,000 passes, 104 days of one every 15 minutes: 0.8 MB with one failed pass in ten.

The sizes were measured with SQLite's `dbstat` on a log of 20,000 answers written by the API's own store, with 20 windows tested on it.
SQLite reuses the pages a deletion frees, so the file does not grow past its largest content.

### How to read a finding

`report` prints a finding for a window, the latest tested one by default:

```text
Input drift in window 3 (log seq 2001 to 3000, 1000 requests answered ... by lgbm-basic ..., 100% ES): 14 of 23 features are below p 0.000217 (0.005 / 23, NFR-11): country_code (p 0), nr_prev_owners__missing (p 0), body_type (p 2.7e-34), ...
```

A flagged window says the requests of that window are unlikely to come from the training distribution; the features are sorted by p-value, the strongest evidence first.
A flag on `country_code` alone is a new market mix; several features besides it, as the `ES` replay gives, are a population the model has not seen, which is the case for retraining (FR-15): the finding is the text for `retrain.yml`'s `reason`.
A control window of the training distribution is flagged in at most 1 of 20 windows (NFR-11), so a single flag between unflagged windows is a false alarm until the next window confirms it.
A `partial` window was cut short by a new model and is not tested; an open window shows only its progress.

### Run it by hand

Locally, on the stack of `make stack`, a pass and the reports; Compose refuses to read `compose.yaml` without the two variables of [Run it locally](#run-it-locally), and any value does for `exec`:

```bash
export GRAFANA_ADMIN_PASSWORD=unused DISCORD_WEBHOOK_URL=http://127.0.0.1:9/discarded
docker compose -f deploy/compose.yaml exec monitor python -m recommenditos_monitoring run-once
docker compose -f deploy/compose.yaml exec monitor python -m recommenditos_monitoring report
docker compose -f deploy/compose.yaml exec monitor python -m recommenditos_monitoring report --window 3 --json
```

On the VM, from `/opt/recommenditos`, with the paths deploy.sh uses; the JSON evidence goes straight into a file on the laptop:

```bash
ssh -p 22052 recommenditos@nattech.fib.upc.edu \
  'cd /opt/recommenditos && docker compose -f stack/compose.yaml --env-file .env exec -T monitor python -m recommenditos_monitoring report --window 3 --json' \
  > reports/monitoring/window-3.json
```

`docker compose ... cp monitor:/var/lib/recommenditos-monitor/monitor.db .` copies the whole file off the container instead.

### NFR-11, tested on the real samples

`monitoring/tests/test_nfr11.py` runs the job's own pass with the real detectors over a log of 20 control windows of 1,000 and the first 1,000 `ES` listings of the versioned samples, sent as the replay sends them, across a month boundary of the age rule.
It asserts at most 1 flagged control window, and the `ES` window flagged with at least three features besides `country_code`.
On the samples of 2026-11-04, no control window was flagged, the lowest p-value of any control feature was 0.0035 against the threshold of 0.000217, and the `ES` window was flagged in 14 features.
The test needs `dvc pull data/monitoring`, which only main can do (the DagsHub credentials are main's, EDN-91), so it runs in CD's publish job, before any image is built or pushed, where a missing sample fails it (`RECOMMENDITOS_REQUIRE_MONITORING_SAMPLES=1`).
A pull request's Tests job skips it and lists the skip in its job summary; run it locally with `make test-monitoring` after `dvc pull data/monitoring`.
The requirement matrix collects the monitoring tests in their own environment, so NFR-11 and FR-14 count as verified by a test.

### The licence

Alibi Detect is licensed under the Business Source License 1.1, not an open-source licence: it permits non-production use, and its Additional Use Grant lets non-profit educational institutions use it in production as long as nothing built on it is sold.
A course project of the UPC is such a use.
Each version becomes Apache 2.0 four years after its release, 0.13.0 on 2029-12-11.
