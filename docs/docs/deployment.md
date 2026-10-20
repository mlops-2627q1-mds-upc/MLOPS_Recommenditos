Deployment
==========

The API runs as a Docker Compose stack of two containers: the API image, which holds the serving runtime and one model bundle, and nginx in front of it.
The same stack runs on a laptop and on the FIB Virtech VM ([EDN-87](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
This page says what the stack guarantees, how to run it locally, how to set up a fresh VM and deploy to it, and what to check at the lab session where we get the VM.
Milestone 5 adds CI/CD, the image scan and the hardening on top of this stack ([What M5 adds](#what-m5-adds)).

> **Unverified: everything about the VM.**
> We get access to the VM at the lab on 2026-10-27.
> Its operating system, size, sudo rights, installed software and which ports are reachable from outside are all unknown until then.
> Everything below that depends on them is written for a fresh Ubuntu or Debian VM with sudo, and the [lab checklist](#lab-checklist) is how we find out.

## The stack

```
client --HTTP--> proxy (nginx, host port 8080 locally, 80 on the VM)
                   |-- /feedback, /metrics: 403, never forwarded
                   '-- everything else --> api:8000 (uvicorn, one model bundle)
                                          only reachable inside the Compose network
```

| File | What it is |
|------|------------|
| `Dockerfile`, `.dockerignore` | The API image |
| `deploy/compose.yaml` | The stack: the API and the proxy |
| `deploy/nginx.conf` | The proxy's configuration |
| `deploy/.env.template` | The template of the stack's `.env` on the VM |
| `deploy/bootstrap.sh` | Prepares a fresh VM |
| `tests/integration/` | The integration suite: tests a running stack through its proxy (`make test-integration`, `make smoke`) |

What each part guarantees:

- **The image** installs `[project] dependencies` from `uv.lock` and nothing else ([EDN-47](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)), following the multi-stage pattern of [uv's Docker guide](https://docs.astral.sh/uv/guides/integration/docker/).
  It copies `models/lgbm-basic` unchanged, so `/health` reports the hash `dvc.lock` records for that directory (FR-12, EDN-08); `--build-arg MODEL_VARIANT=lgbm-extended` builds it with another bundle.
  The API runs as the unprivileged user 10001, and the image's `HEALTHCHECK` asks `/health` with Python's `urllib`, so the image needs no curl.
  It starts with `python -m recommenditos.api`, which empties `PROMETHEUS_MULTIPROC_DIR` (`/tmp/prometheus`, a tmpfs in Compose) and starts uvicorn; `WEB_CONCURRENCY` sets the number of workers, and `/metrics` adds up their figures ([API docs](api.md#metrics)).
- **Compose** publishes only the proxy's port.
  The proxy starts once the API's healthcheck passes, and both containers restart on their own after a crash and after a reboot of the VM (`restart: unless-stopped`, NFR-05).
  A hung API, one whose process lives on but stops answering, is only marked unhealthy: Docker's restart policy acts when the process exits, so in M4 nothing restarts it automatically; doing so is an M5 item ([What M5 adds](#what-m5-adds)).
- **nginx** refuses `/feedback` and `/metrics`, versioned or not, with a 403 (FR-10, EDN-13).
  The replay job and Prometheus reach those endpoints at `api:8000` inside the Compose network, never through the proxy.
  nginx matches the normalised path, so `//feedback`, `/v1/./feedback` and percent-encoded spellings are refused too.
  It sends `X-Content-Type-Options: nosniff` and `X-Frame-Options: DENY`, hides its version (`server_tokens off`), and sends no HSTS header, because there is no TLS (NFR-09).
  It refuses bodies above 16 KB, as defence in depth behind the API's own 10 KB limit, and answers its own errors (403, 413, 503) as RFC 9457 problems like the API does (FR-16).
  While the API is down, for example in the seconds after a crash, it answers 503 within 2 s.
  Its access log carries the API's `X-Request-ID`, and it resolves the API's address again every 10 s, so a recreated API container is found without restarting nginx.
  nginx runs as a non-root user too (`nginxinc/nginx-unprivileged`), which is why it listens on 8080 inside its container.

## Run it locally

You need Docker with the compose plugin and the pulled model:

```bash
uv run dvc pull models/lgbm-basic
make stack          # builds the image, starts both containers, waits until healthy
make test-integration   # the integration suite against http://localhost:8080
make stack-down
```

`make stack` is `docker compose -f deploy/compose.yaml up -d --build --wait`, so it runs the same way on Windows without `make`.
Docker copies a symbolic link into the image as a link, so `models/lgbm-basic` has to hold the files themselves: `find models/lgbm-basic -type l` must print nothing.
That holds for DVC's default cache type, `copy`, and for `hardlink`; with `symlink`, set `cache.type` to one of those and run `uv run dvc checkout --relink models/lgbm-basic`.

Measured on 2026-10-20 on a 4-core laptop (i5-10210U, 15 GB RAM) with Docker 29.9:

| What | Measured | Target |
|------|----------|--------|
| Image size (`docker image inspect`, uncompressed) | 665 MB | at most 1 GB (NFR-04) |
| API container memory, idle after the smoke check (`docker stats`) | 157 MiB (nginx: 10 MiB) | below 1 GB under load (NFR-04, load test in #143) |
| `docker compose up -d` to the first 200 from `/health` through nginx | 5.2 s, image already built | - |
| SIGKILL of the API process to the next 200 from `/health` through nginx, 3 runs | 3.1 to 3.4 s | within 30 s (NFR-05) |

The crash test kills the uvicorn process from the host, as the kernel's OOM killer or a segfault would: `docker run --rm --pid=host alpine kill -9 $(docker inspect -f '{{.State.Pid}}' recommenditos-api-1)`.
`docker kill` is no crash test, because Docker counts it as a manual stop and `unless-stopped` then leaves the container down.
The integration suite runs the same crash test ([Integration tests](#integration-tests)).

## Integration tests

`tests/integration/` tests the stack as a client reaches it, over HTTP through nginx, and the containers as they run.
It covers what the in-process API tests cannot see:

- through the proxy: `/health` answers 200 and reports the bundle in the image, `GET /`, `/docs` and `/openapi.json` answer, a valid `/v1/predict` returns its documented shape, an invalid request and an unknown route return RFC 9457 problems, the security headers are set and the `Server` header names no version;
- the limits and refusals: a body over the API's 10 KB is refused by the API, one over the proxy's 16 KB by nginx, and `/feedback` and `/metrics`, versioned or not, get the proxy's 403 (FR-10, NFR-09);
- on the Docker host: both containers run as a non-root user, the image holds no package outside the serving runtime of `uv.lock`, the API's port is not published, and after a SIGKILL of the API process `/health` is 200 again within 30 s (NFR-05).

```bash
make test-integration                                # against http://localhost:8080
make test-integration STACK_URL=http://<host>:<port> # against another stack
```

The base URL is `RECOMMENDITOS_STACK_URL`, which the make targets set.
The tests that need the Docker host run when that URL is a loopback address and are skipped otherwise, so the same suite runs in CI, on a laptop and from outside the VM.
Against a local URL they include the crash test, which takes the API down for a few seconds: run the suite from outside the VM, never on it.
`/health` is compared with the hash `dvc.lock` records for the bundle (FR-12), unless `RECOMMENDITOS_EXPECTED_MODEL_VERSION` names another version.
`make test` leaves the suite out (`-m "not integration"` in `pyproject.toml`).

CI's `Integration (Compose stack)` job runs the suite on every pull request that touches the package, the `Dockerfile`, `deploy/`, the dependencies or the suite.
CI cannot `dvc pull` the real bundle, so the job fits `lgbm-basic` on the synthetic fixture into `models/` (`uv run python -m tests.fixtures.synthetic_bundle lgbm-basic models`, which prints the bundle's version), builds the image around it, starts the stack, runs the suite with that version as the expected one, prints both containers' logs if anything failed and always tears the stack down.

## Deploy to the VM

### How the image gets there

Until CD exists, the image is built on a laptop and streamed to the VM with `docker save | ssh docker load`.
The VM then needs Docker and three files, and none of the repository, uv, DVC or a DagsHub token, which would otherwise sit on a shared university machine.
It also keeps the DVC cache and the build cache off the VM's 20 GB disk (NFR-04).
In M5, CD builds the same image in GitHub Actions and the VM pulls it from GHCR instead.

### First time: bootstrap a fresh VM

From the repository root on your laptop, with `<admin>@<vm>` the account the lecturer gives us (it needs sudo):

```bash
ssh-copy-id <admin>@<vm>                                   # once, if you log in with a password
scp -r deploy <admin>@<vm>:
ssh -t <admin>@<vm> sudo bash deploy/bootstrap.sh
```

Run the same two commands to bring a changed `bootstrap.sh`, `compose.yaml` or `nginx.conf` to the VM: `scp -r deploy <vm>:` replaces the files of `~/deploy`, whereas a named target directory that already exists, such as `<vm>:some-dir`, would receive a nested `some-dir/deploy/` and leave the old files in place.

`bootstrap.sh` is idempotent and does four things:

1. installs Docker Engine and the compose plugin from Docker's apt repository, following Docker's install page (it first removes the distribution's conflicting packages, such as `docker.io`), and enables Docker at boot;
2. creates the deploy user `recommenditos`, a member of the `docker` group, whose home `/opt/recommenditos` holds the stack;
3. lets every SSH key of the account that ran it log in as `recommenditos`, so that every later step is `ssh recommenditos@<vm>` without sudo;
4. copies `compose.yaml`, `nginx.conf` and `.env.template` into `/opt/recommenditos`, and creates `.env` from the template, readable by the deploy user alone, unless it exists.

Then set the port the [lab checklist](#lab-checklist) found reachable:

```bash
ssh recommenditos@<vm> 'nano .env'      # HTTP_PORT=80, or the forwarded port
```

The `docker` group is root-equivalent on that machine; only the deploy user and the admins belong in it.
FR-10's API key goes into the same `.env` once `/feedback` exists (#140); it never enters the repository (NFR-15).

### Every deploy

From the repository root, at the commit to deploy, after `uv run dvc pull models/lgbm-basic`:

```bash
make image                                   # recommenditos-api:<short commit SHA>
TAG=$(git rev-parse --short HEAD)
docker save recommenditos-api:$TAG | gzip | ssh recommenditos@<vm> docker load
ssh recommenditos@<vm> "sed -i 's/^API_IMAGE_TAG=.*/API_IMAGE_TAG=$TAG/' .env && docker compose up -d --wait"
make smoke SMOKE_URL=http://<vm-host>:<port>
```

`--wait` returns once both containers are healthy, and `make smoke` is what says the deployment worked: it runs the [integration suite](#integration-tests) against the VM and exits non-zero on any failure.
Run it from outside the VM, because the proxy rules it checks are about requests from outside; the tests that need the Docker host are skipped from there.
On Windows, run these in Git Bash or WSL: Windows PowerShell 5 corrupts binary data in a pipe.
Without either, save to a file instead: `docker save -o api.tar recommenditos-api:$TAG`, `scp api.tar recommenditos@<vm>:`, `ssh recommenditos@<vm> docker load -i api.tar`.

A change to `compose.yaml` or `nginx.conf` reaches the VM by copying it and starting the stack again:

```bash
scp deploy/compose.yaml deploy/nginx.conf recommenditos@<vm>:
ssh recommenditos@<vm> docker compose up -d --wait
```

### Roll back, inspect, clean up

On the VM, as `recommenditos`:

```bash
docker image ls recommenditos-api            # the tags the VM holds
sed -i 's/^API_IMAGE_TAG=.*/API_IMAGE_TAG=<previous>/' .env && docker compose up -d --wait
docker compose ps                            # state and health of both containers
docker compose logs -f --tail 100            # both logs; the proxy's lines carry request_id
docker image rm recommenditos-api:<old>      # each tag costs disk, of which there is 20 GB
```

## Lab checklist

At the lab session on 2026-10-27, whoever gets access records the answers in #134 and, for the three questions, in the [specification](specification.md#decisions-pending-confirmation).

**1. What the VM is.** On the VM:

| What | Command | What we assume until then |
|------|---------|---------------------------|
| Operating system | `cat /etc/os-release` | Ubuntu or Debian |
| CPU architecture | `uname -m` | `x86_64`, the architecture a laptop builds for |
| CPU cores | `nproc` | - |
| RAM | `free -h` | 4 GB (EDN-17) |
| Disk | `df -h /` | 20 GB (EDN-17) |
| sudo | `sudo -v && echo yes` | yes |
| Docker | `docker --version; docker compose version` | not installed |
| Outbound HTTPS, which the bootstrap needs | `curl -sI https://download.docker.com \| head -1` | allowed |
| Private or public address | `ip -4 addr show scope global` and `curl -s https://ifconfig.me` | behind NAT |

**2. The three questions of #134.**

- *Is the VM reachable from the public internet, and on which ports?*
  Ask the lecturer which ports are forwarded.
  For each port, start a listener on the VM, `sudo python3 -m http.server <port>`, and request it from a machine outside the UPC network, such as a laptop on a phone's hotspot: `curl -m 5 -sI http://<vm-host>:<port>/`.
  An answer means the port is reachable; a timeout means it is not.
  Afterwards, the same `curl` against `/health` once the stack runs.
- *Is there a host name we control enough to get a certificate for?*
  `hostname -f` on the VM, and `host <public address>` from outside, show the names that point to it.
  A certificate from Let's Encrypt needs port 80 of that name to reach the VM, so this is only yes if the first answer includes port 80 and the name is ours or the lecturer's to use.
- *Can a GitHub-hosted runner reach the VM over SSH (NFR-13)?*
  A GitHub-hosted runner is one more machine on the public internet, so test from outside the UPC network as above: `ssh -o ConnectTimeout=5 -p <ssh-port> <admin>@<vm-host> true`.
  If that only works from inside the UPC network or through its VPN, the answer is no, and CD falls back to a self-hosted runner on the VM.

**3. Deploy and check.**
Run the [bootstrap](#first-time-bootstrap-a-fresh-vm) and the [first deploy](#every-deploy), then the smoke check from outside the UPC network.
Finally check NFR-05's reboot case: `sudo reboot`, and time until `curl http://<vm-host>:<port>/health` answers 200 again; the 30 s count from when the VM is up, which `uptime -s` tells afterwards.

## Without sudo or without Docker

`bootstrap.sh` needs root for all four of its steps.
If Docker is already installed and our account is in the `docker` group, none of them is needed: copy `compose.yaml` and `nginx.conf` into a directory of our own, copy `.env.template` there as `.env` and fill it in, and deploy as above, under our account.
If Docker is installed and we are not in the group, we ask the lecturer to add us; rootless Docker would also work, but it needs the `uidmap` package and subordinate IDs that only an admin can provide, and it cannot publish a port below 1024, so `HTTP_PORT` would move to 8080 or above.
If there is no Docker and no sudo, the fallback is the course demo's recipe in our home directory: uv installs Python and `uv sync --locked --no-default-groups` the runtime, the bundle is copied with `scp`, uvicorn runs on an unprivileged port under `systemd --user` or a `@reboot` cron entry, and nginx, without root, runs on an unprivileged port as well.
That loses the image as the unit we test and deploy, so we would ask for Docker before building it.

## What M5 adds

- CI builds the image on every merge to `main`, tags it with the commit SHA, pushes it to GHCR, and CD deploys it over SSH with `docker compose pull && docker compose up -d --wait`, then runs `make smoke` and puts the previous tag back if it fails (NFR-13).
  The steps of [Every deploy](#every-deploy) are what CD automates; `API_IMAGE_TAG` in the VM's `.env` is already the variable it sets.
- `trivy` scans the image before it is deployed (NFR-09).
- Hardening of the containers: a read-only root file system, dropped capabilities, `no-new-privileges` and memory limits.
- Recovery from a hung API: a container that stays unhealthy is restarted, for example by a watchdog that restarts unhealthy containers, since Docker's restart policy covers only a process that exits.
