Deployment
==========

The API runs as a Docker Compose stack of two containers: the API image, which holds the serving runtime and one model bundle, and nginx in front of it.
The same stack runs on a laptop and on the FIB Virtech VM ([EDN-87](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
This page says what the stack guarantees, how to run it locally, how CI builds and publishes the image, how to set up a fresh VM and deploy to it, and what to check at the lab session where we get the VM.
Milestone 5 adds CI/CD, the image scan and the hardening on top of this stack ([Continuous delivery](#continuous-delivery), [What M5 adds](#what-m5-adds)).

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
| `deploy/deploy.sh` | Runs one image tag on the VM and puts the previous one back if it fails |
| `tests/integration/` | The integration suite: tests a running stack through its proxy (`make test-integration`, `make smoke`) |

What each part guarantees:

- **The image** installs `[project] dependencies` from `uv.lock` and nothing else ([EDN-47](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)), following the multi-stage pattern of [uv's Docker guide](https://docs.astral.sh/uv/guides/integration/docker/).
  It copies `models/lgbm-basic` unchanged, so `/health` reports the hash `dvc.lock` records for that directory (FR-12, EDN-08); `--build-arg MODEL_VARIANT=lgbm-extended` builds it with another bundle.
  It also copies FR-09's comparables index, `data/processed/comparables.parquet`, to `/app/models/comparables.parquet` beside the bundle, where `RECOMMENDITOS_COMPARABLES_FILE` points the API, and `/health` reports the md5 `dvc.lock` records for that file as the index's version.
  The API runs as the unprivileged user 10001, and the image's `HEALTHCHECK` runs `recommenditos/api/healthcheck.py`, copied to `/app/healthcheck.py`, which asks `/health` with Python's `urllib`, so the image needs no curl.
  It runs as a script with `python -I -S`, so a probe imports neither the package's `__init__` nor anything from site-packages: that halves its CPU time, which matters at one probe every 5 s.
  It starts with `python -m recommenditos.api`, which empties `PROMETHEUS_MULTIPROC_DIR` (`/tmp/prometheus`, a tmpfs in Compose) and starts uvicorn; `WEB_CONCURRENCY` sets the number of workers (Compose passes it from `.env`, 1 by default), and `/metrics` adds up their figures ([API docs](api.md#metrics)).
  With more than one worker, a single worker whose event loop blocks goes unnoticed: the others still answer `/health`, so the healthcheck passes while the requests that worker accepted hang, and uvicorn's supervisor replaces a worker only when it stops answering its ping, which runs in a thread of its own.
  The API therefore runs one worker until the load test on the VM ([#143](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/143)) settles the count ([EDN-93](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
- **Compose** publishes only the proxy's port.
  Both containers run with a read-only root file system, without any Linux capability and with `no-new-privileges` (NFR-09), so they write only to their tmpfs mounts and the API also to its volume.
  Neither needed a capability back or any other exception.
  The API needed one writable path more than its metric files: a tmpfs on `/tmp`, where SQLite writes a sort too large for its cache and Python's `tempfile` writes; without it a large query of the prediction log fails with "disk I/O error".
  nginx's unprivileged image writes only its pid file and buffered bodies, all under `/tmp`, which is a tmpfs too.
  Each container has a memory limit without swap and a process limit (NFR-04): 768 MiB and 256 processes for the API, 64 MiB and 64 for nginx, at least 1.2 times the peak of the laptop's load tests with two workers; `deploy/compose.yaml` gives the measurements, and the VM's [load test](load-testing.md#run-it-on-the-vm) re-checks them.
  In a container that exceeds its memory limit, the kernel kills the largest process: usually a worker, which uvicorn's supervisor or nginx's master replaces, and when it is the main process, Docker restarts the container as after a crash.
  The proxy starts once the API's healthcheck passes, and both containers restart on their own after a crash and after a reboot of the VM (`restart: unless-stopped`, NFR-05).
  The API writes its prediction log and the feedback to the named volume `prediction-log`, mounted at `/var/lib/recommenditos`, so they survive a new image; M6's drift job mounts the same volume ([EDN-86](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md), [API docs](api.md#the-prediction-log-and-the-feedback)).
  A hung API, one whose process lives on but stops answering, is restarted too, although Docker's restart policy acts only when the process exits ([EDN-93](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
  Once the API has answered a probe since it started, or at the latest once the healthcheck's 60 s start period is over, a probe that gets no answer from `/health` in two attempts of 6 s kills the server.
  The first answer leaves a marker on the API's `/tmp` tmpfs with the start time of PID 1, so a marker from before a restart counts for nothing.
  Docker's init, which `init: true` puts in front of the server as PID 1, then exits, and `restart: unless-stopped` starts the container again.
  The init is needed because the kernel ignores a kill sent from inside the container to its PID 1, and uvicorn's own signal handlers only ask the blocked event loop to stop.
  The probe kills only the children of Docker's init, and nothing when PID 1 is another process: a container of the image that shares the host's PID namespace inherits the healthcheck too.
  The 6 s also bound overload: `/health` waits once for a thread of a worker's threadpool, behind the valuations (its dependencies are async), so an API that is busy but not hung is killed only when about 128 requests, at about 47 ms each, wait ahead of the probe in one worker, more than the load test's 80 users can send at once ([Load testing](load-testing.md)).
  An overload beyond that, which the load test does not reach, would be taken for a hang and restart the API.
- **nginx** refuses `/feedback` and `/metrics`, versioned or not, with a 403 (FR-10, EDN-13).
  The replay job and Prometheus reach those endpoints at `api:8000` inside the Compose network, never through the proxy.
  nginx matches the normalised path, so `//feedback`, `/v1/./feedback` and percent-encoded spellings are refused too.
  It sends `X-Content-Type-Options: nosniff` and `X-Frame-Options: DENY`, hides its version (`server_tokens off`), and sends no HSTS header, because there is no TLS (NFR-09).
  It refuses bodies above 16 KB, as defence in depth behind the API's own 10 KB limit, and answers its own errors (403, 413, 503) as RFC 9457 problems like the API does (FR-16).
  While the API is down, for example in the seconds after a crash, it answers 503 within 2 s.
  Its access log carries the request ID of each answer: the API's `X-Request-ID`, or nginx's own for a 403, 413 or 503 it answers itself, which it also sends as `X-Request-ID`.
  It resolves the API's address again every 10 s, so a recreated API container is found without restarting nginx.
  nginx runs as a non-root user too (`nginxinc/nginx-unprivileged`), which is why it listens on 8080 inside its container.
- **Docker** keeps each container's log in at most five files of 10 MB, so the two logs take at most 100 MB of the VM's disk (NFR-04); the API's 50 MB hold about 120,000 requests of history.

## Run it locally

You need Docker with the compose plugin and the pulled model and comparables index:

```bash
uv run dvc pull models/lgbm-basic data/processed/comparables.parquet
RECOMMENDITOS_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))') make stack
                    # builds the image, starts both containers, waits until healthy
make test-integration   # the integration suite against http://localhost:8080
make stack-down
```

`make stack` is `docker compose -f deploy/compose.yaml up -d --build --wait`, so it runs the same way on Windows without `make`.
The `Dockerfile` installs third-party wheels from uv's build cache, but builds the `recommenditos` wheel from the build context every time (`--reinstall-package`): the cache is shared by every build on the machine, and uv would otherwise install a wheel it built from another worktree or branch ([#180](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/180)); the integration suite checks that the image holds the checkout's package.
Docker copies a symbolic link into the image as a link, so `models/lgbm-basic` and `data/processed/comparables.parquet` have to be the files themselves: `find models data/processed/comparables.parquet -type l` must print nothing.
That holds for DVC's default cache type, `copy`, and for `hardlink`; with `symlink`, set `cache.type` to one of those and run `uv run dvc checkout --relink models/lgbm-basic data/processed/comparables.parquet`.

Measured on 2026-10-20 on a 4-core laptop (i5-10210U, 15 GB RAM) with Docker 29.9:

| What | Measured | Target |
|------|----------|--------|
| Image size (`docker image inspect`, uncompressed) | 702 MB, measured on 2026-10-21 with the three interval boosters of [#158](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/158) | at most 1 GB (NFR-04) |
| API container memory, idle after the smoke check (`docker stats`), measured on 2026-10-22 | 248 MiB with one worker; 539 MiB with two, a 52 MiB supervisor and two workers (nginx: 7 MiB) | below 1 GB under load (NFR-04, [load test](load-testing.md)) |
| `docker compose up -d` to the first 200 from `/health` through nginx | 5.2 s, image already built | - |
| SIGKILL of the API process to the next 200 from `/health` through nginx, 3 runs | 3.1 to 3.4 s | within 30 s (NFR-05) |
| SIGSTOP of the API's processes (a hang) to the next 200 from `/health` through nginx, measured on 2026-10-23 with the hardening of #200 | 19.4 to 20.8 s in 4 runs, the hang placed right after the API turned healthy, the latest point for a probe to catch it; at most about 21 s (5 s interval, two attempts of 6 s, about 4 s to restart) | within 30 s (NFR-05) |

The crash test kills the uvicorn process from the host, as the kernel's OOM killer or a segfault would: `docker run --rm --pid=host alpine kill -9 $(docker top recommenditos-api-1 -eo pid,args | awk '$4 == "recommenditos.api" {print $1}')`.
The process is found with `docker top`, because `docker inspect`'s `State.Pid` is now Docker's init in front of it.
`docker kill` is no crash test, because Docker counts it as a manual stop and `unless-stopped` then leaves the container down.
The integration suite runs the same crash test ([Integration tests](#integration-tests)).
Its hang test stops the server and its workers with SIGSTOP instead, as soon as Docker sees the API healthy: they hold the port, so nginx's connection is accepted and never answered, as with a blocked event loop.

## Integration tests

`tests/integration/` tests the stack as a client reaches it, over HTTP through nginx, and the containers as they run.
It covers what the in-process API tests cannot see:

- through the proxy: `/health` answers 200 and reports the bundle in the image, `GET /`, `/docs` and `/openapi.json` answer, a valid `/v1/predict` returns its documented shape, an invalid request and an unknown route return RFC 9457 problems, the security headers are set and the `Server` header names no version;
- the limits and refusals: a body over the API's 10 KB is refused by the API, one over the proxy's 16 KB by nginx, and `/feedback` and `/metrics`, versioned or not, get the proxy's 403 (FR-10, NFR-09);
- on the Docker host: both containers run as a non-root user with a read-only root file system, no capability, `no-new-privileges` and limits on memory and processes, the image holds no package outside the serving runtime of `uv.lock`, its `recommenditos` package matches the checkout module by module, the API's port is not published, `/metrics` answers inside the Compose network, `/v1/feedback` with the key stores a price that the prediction log on the volume joins to its answer (skipped when the stack runs without a key), and after a SIGKILL of the API process, and after the API hangs, `/health` is 200 again within 30 s (NFR-05);
- the rollback drill: `deploy.sh` puts the previous tag back when a new one does not start, as a second Compose project beside the stack ([Every deploy](#every-deploy)).

```bash
make test-integration                                # against http://localhost:8080
make test-integration STACK_URL=http://<host>:<port> # against another stack
```

The base URL is `RECOMMENDITOS_STACK_URL`, which the make targets set.
The tests that need the Docker host, the crash test among them, run when that URL is a loopback address and are skipped otherwise.
`make smoke` never touches containers: it deselects those tests whatever the URL, so a smoke run on the VM, from a runner on it, or through an SSH tunnel cannot kill the production API or inspect the machine's own containers.
`make test-integration` is the full run, for CI and the local stack.
`/health` is compared with the hashes `dvc.lock` records for the bundle and the index (FR-12), unless `RECOMMENDITOS_EXPECTED_MODEL_VERSION` and `RECOMMENDITOS_EXPECTED_COMPARABLES_VERSION` name other versions.
`make test` leaves the suite out (`-m "not integration"` in `pyproject.toml`).

CI's `Integration (Compose stack)` job runs the suite on every pull request that touches the package, the `Dockerfile`, `deploy/`, the dependencies or the suite.
CI cannot `dvc pull` the real bundle, so the job fits `lgbm-basic` on the synthetic fixture into `models/` (`uv run python -m tests.fixtures.synthetic_bundle lgbm-basic models`, which prints the bundle's version and writes a comparables index of the same fixture to `data/processed/comparables.parquet`), builds the image around both, starts the stack, runs the suite with their versions as the expected ones, prints both containers' logs if anything failed and always tears the stack down.

## Continuous delivery

The `CD` workflow ([`.github/workflows/cd.yml`](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/.github/workflows/cd.yml)) builds the API image of every merge to `main` and pushes it to GHCR (NFR-13, [EDN-91](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)):

```text
ghcr.io/mlops-2627q1-mds-upc/mlops_recommenditos/api:<full commit SHA>
ghcr.io/mlops-2627q1-mds-upc/mlops_recommenditos/api:latest
```

It starts when a CI run on a push to `main` finishes, and only if that run passed, so a commit whose CI failed never publishes an image.
It works on the commit CI tested (`github.event.workflow_run.head_sha`), not on the tip of `main`, which a later merge may have moved.
A commit that changes nothing but `docs/`, `reports/` and the Markdown files at the top level builds nothing; the run is green and its summary says so.
A manual run (**Actions**, **CD**, **Run workflow** on `main`) builds the tip of `main` again if CI passed on it, for a run that failed on something outside the commit, such as a registry outage.
A failed step that depends on a GitHub setting, the DagsHub secrets or the permission to push the package, says in the job summary which setting to check.

For every other commit, one job:

1. pulls `models/lgbm-basic` and `data/processed/comparables.parquet` at the versions `dvc.lock` points to (FR-12, EDN-08);
2. builds the image with the OCI labels `source`, `revision` (the commit), `version` and `created`, reusing unchanged layers from the GitHub Actions cache;
3. fails if the image is larger than 1 GB (NFR-04);
4. starts the Compose stack with it and runs the [integration suite](#integration-tests) with no expected-version override, so `/health` has to report the hashes `dvc.lock` records;
5. scans the API image and the nginx image with trivy and fails on a HIGH or CRITICAL vulnerability that has a fix available ([Vulnerabilities and dependency updates](#vulnerabilities-and-dependency-updates));
6. pushes `:<commit SHA>` and moves `:latest` to it, unless `:latest` already names a newer commit, which happens when the CI runs of two merges finish in the opposite order;
7. writes the pushed tags, the digest, the model and index versions `/health` reported, the image size and the job's runtime into the job summary.

Only the `dvc pull` step sees the DagsHub token.
It is a secret of the GitHub Environment `dagshub`, which only `main` may use, and the step writes it into `.dvc/config.local`, as on a laptop, and deletes that file when it ends.
The job's token may read the repository and write packages, nothing more.

The build stages' layers come from the GitHub Actions cache (`type=gha`, `mode=max`), so a commit that leaves `uv.lock` and `pyproject.toml` unchanged skips the `uv sync` of the third-party dependencies.
The runtime stage copies the environment in two layers, with the bundle and the index between them, ordered from the least to the most often changed (`docker history`, measured on 2026-10-23):

1. the third-party dependencies, 535 MB, which only a change to `uv.lock` or `pyproject.toml` rebuilds;
2. the model bundle, 45 MB, and the comparables index, 2 MB, which a model promotion changes;
3. the `recommenditos` package, 1.5 MB, laid over the dependencies' environment.

Docker rebuilds every layer above a changed one, so a code change rebuilds and pushes only the package's layer, 0.6 MB compressed, and the VM pulls only that.
While the environment was a single layer with the package in it, a code change pushed that layer and the two above it, 183 MB compressed.
A model promotion, which changes only `dvc.lock`, rebuilds the three upper layers, 48 MB.
The job summary states each run's runtime.
Built locally with the job's steps on 2026-10-22, the image was 704 MB and passed the 27 tests of the integration suite against the hashes `dvc.lock` records.

The package is public, so the VM and anyone else pull it without a credential; the image holds no secret and no personal data, as the API key is passed at run time.
GHCR creates a new package as private, so after the first push an organisation owner makes it public once: the organisation's **Packages** tab, `mlops_recommenditos/api`, **Package settings**, **Change visibility**, **Public**.
After that, any commit's image can be checked from anywhere:

```bash
docker pull ghcr.io/mlops-2627q1-mds-upc/mlops_recommenditos/api:<SHA>
docker inspect --format '{{json .Config.Labels}}' ghcr.io/mlops-2627q1-mds-upc/mlops_recommenditos/api:<SHA>
```

## Deploy to the VM

### How the image gets there

[Continuous delivery](#continuous-delivery) pushes the image of every merge to GHCR, and the VM pulls it from there: `deploy/deploy.sh` on the VM pulls the commit's tag and starts it.
The VM therefore needs Docker and four files, and none of the repository, uv, DVC or a DagsHub token, which would otherwise sit on a shared university machine.
It also keeps the DVC cache and the build cache off the VM's 20 GB disk (NFR-04).
When GHCR cannot be reached, a laptop builds the image and streams it to the VM with `docker save | ssh docker load` instead ([Every deploy](#every-deploy)).

### First time: bootstrap a fresh VM

From the repository root on your laptop, with `<admin>@<vm>` the account the lecturer gives us (it needs sudo):

```bash
ssh-copy-id <admin>@<vm>                                   # once, if you log in with a password
scp -r deploy <admin>@<vm>:
ssh -t <admin>@<vm> sudo bash deploy/bootstrap.sh
```

Run the same two commands to bring a changed `bootstrap.sh`, `compose.yaml`, `nginx.conf` or `deploy.sh` to the VM: `scp -r deploy <vm>:` replaces the files of `~/deploy`, whereas a named target directory that already exists, such as `<vm>:some-dir`, would receive a nested `some-dir/deploy/` and leave the old files in place.

`bootstrap.sh` is idempotent and does four things:

1. installs Docker Engine and the compose plugin from Docker's apt repository, following Docker's install page (it first removes the distribution's conflicting packages, such as `docker.io`), and enables Docker at boot;
2. creates the deploy user `recommenditos`, a member of the `docker` group, whose home `/opt/recommenditos` holds the stack;
3. lets every SSH key of the account that ran it log in as `recommenditos`, so that every later step is `ssh recommenditos@<vm>` without sudo;
4. copies `compose.yaml`, `nginx.conf`, `.env.template` and `deploy.sh` into `/opt/recommenditos`, and creates `.env` from the template, readable by the deploy user alone, unless it exists.

Then set the port the [lab checklist](#lab-checklist) found reachable:

```bash
ssh recommenditos@<vm> 'nano .env'      # HTTP_PORT=80, or the forwarded port
```

The `docker` group is root-equivalent on that machine; only the deploy user and the admins belong in it.
Set FR-10's API key in the same `.env` (`RECOMMENDITOS_API_KEY`, generated on the VM as the template says); it never enters the repository (NFR-15).

### Every deploy

The `Deploy` workflow ([`.github/workflows/deploy.yml`](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/.github/workflows/deploy.yml)) deploys every image that [continuous delivery](#continuous-delivery) pushes, once it is [turned on](#turn-continuous-deployment-on) (NFR-13).
It starts when a CD run on `main` finishes, and only if that run passed and pushed an image: a commit that changes only `docs/`, `reports/` or top-level Markdown files builds none and deploys nothing.
It deploys the image's commit SHA tag, never `:latest`, in one job of the GitHub Environment `production`:

1. reads the tag the VM runs, and skips the commit if the VM already runs a newer one that contains it, which happens when the CD runs of two merges finish in the opposite order (the rule that keeps `:latest` moving forward);
2. runs `deploy.sh deploy <SHA>` on the VM, which pulls the image, sets `API_IMAGE_TAG` in `.env`, runs `docker compose up -d --wait`, and asks the proxy for `/health`; if the stack does not come up healthy, `deploy.sh` starts the previous tag again and the run fails;
3. runs `make smoke` from the runner against the VM's public address, the outside view the proxy rules are about; if it fails, `deploy.sh rollback` starts the previous tag again and the run fails;
4. writes the deployed commit, the outcome and the tag that runs now into the job summary.

Deploys never overlap and are never cancelled: the workflow's runs queue in the order they arrive (`concurrency` with `queue: max`), and `deploy.sh` holds a lock on the VM, so a deploy by hand waits for a running one.
The job checks out the deployed commit, so the smoke suite holds `/health` to that commit's `dvc.lock` (FR-12).

A manual run (**Actions**, **Deploy**, **Run workflow** on `main`) deploys the full commit SHA it is given, an older one too: that is how an earlier version is redeployed on purpose, for example after a promoted model regressed (FR-15).
Any tag CD pushed can be deployed this way, since GHCR keeps them all.

`deploy.sh` is also how a deploy runs by hand, on the VM as `recommenditos` or over SSH:

```bash
ssh recommenditos@<vm> ./deploy.sh deploy <full commit SHA>
make smoke SMOKE_URL=http://<vm-host>:<port>
```

`--wait` returns once both containers are healthy, and `make smoke` is what says the deployment worked: it runs the [integration suite](#integration-tests) against the VM, without the tests that need the Docker host, and exits non-zero on any failure.
Run it from outside the VM, because the proxy rules it checks are about requests from outside.

The rollback is rehearsed against the local stack by `tests/integration/test_deploy_rollback.py`, part of `make test-integration` and CI's integration job.
It copies the VM's stack directory (`compose.yaml`, `nginx.conf`, `.env` and `deploy.sh`) into a temporary directory and runs it as a second Compose project on a port of its own.
It deploys the image under test, then the same image with a server that exits as soon as it starts, and checks that the first tag is back in `.env`, runs and answers through the proxy; a second test checks `deploy.sh rollback`, the step that follows a failed smoke check.
A third test closes `deploy.sh`'s output as soon as it starts the broken tag, as a cancelled run or a lost SSH connection would, and checks that the rollback still finishes; a fourth checks that a deploy removes the image of the tag two deploys back and keeps the rest.
The drill's tags are full commit SHAs, the only tags `deploy.sh` accepts.
On a laptop on 2026-10-22, `deploy.sh deploy` of the broken tag returned 7 s after it started, with the previous tag serving again, and the four tests took 62 s.
CI's integration job runs beside the `Tests` job, which took 5 minutes on the pull request that added the drill, so the drill does not lengthen a CI run.

**Fallback without GHCR.** From the repository root, at the commit to deploy, after `uv run dvc pull models/lgbm-basic data/processed/comparables.parquet`, the image is built locally, given the name the VM's `.env` expects and streamed to the VM:

```bash
TAG=$(git rev-parse HEAD)
IMAGE=ghcr.io/mlops-2627q1-mds-upc/mlops_recommenditos/api
make image IMAGE_TAG=$TAG                    # recommenditos-api:<full commit SHA>
docker tag recommenditos-api:$TAG $IMAGE:$TAG
docker save $IMAGE:$TAG | gzip | ssh recommenditos@<vm> docker load
ssh recommenditos@<vm> ./deploy.sh deploy $TAG
make smoke SMOKE_URL=http://<vm-host>:<port>
```

`deploy.sh` finds the image on the VM and does not pull it.
On Windows, run these in Git Bash or WSL: Windows PowerShell 5 corrupts binary data in a pipe.
Without either, save to a file instead: `docker save -o api.tar $IMAGE:$TAG`, `scp api.tar recommenditos@<vm>:`, `ssh recommenditos@<vm> docker load -i api.tar`.

A change to `compose.yaml`, `nginx.conf` or `deploy.sh` reaches the VM by running the [bootstrap](#first-time-bootstrap-a-fresh-vm) again, then starting the stack again:

```bash
scp -r deploy <admin>@<vm>:
ssh -t <admin>@<vm> sudo bash deploy/bootstrap.sh
ssh recommenditos@<vm> docker compose up -d --wait
```

### Roll back, inspect, clean up

On the VM, as `recommenditos`:

```bash
./deploy.sh status                           # the tag that runs, its health and the previous tag
./deploy.sh rollback                         # start the previous tag again
./deploy.sh deploy <full commit SHA>         # start any tag CD pushed
cat deploy-history.log                       # every deploy and rollback, with its time and outcome
cat deploy-last-run.log                      # the whole output of the last deploy or rollback
docker compose ps                            # state and health of both containers
docker compose logs -f --tail 100            # both logs, joined by request_id (api.md#following-one-request)
```

`rollback` goes back one step, to the tag that ran before the last deploy; to go further back, deploy an older tag.
After a successful deploy, `deploy.sh` removes the images of the tags it deployed before, except the one that runs and the previous one, since each costs disk, of which there is 20 GB; any other tag can be pulled again.

### Turn continuous deployment on

The `Deploy` workflow is off until the repository variable `DEPLOY_VIA` is set: until then each run is green and its summary says that deploying is off.
Its deploying job, the one in the Environment `production`, runs only once it is on.
GitHub's documentation says that running a workflow that references a missing environment creates it, without any protection; whether a skipped job does that too is not documented, so `production` may already exist, empty, after the first run.
Once `DEPLOY_VIA` is set, a missing variable or secret fails the run with a line naming it.
The workflow reaches the VM over SSH from a GitHub-hosted runner, option A of [#198](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/198); if the [lab checklist](#lab-checklist) finds the VM's SSH port unreachable from outside, a self-hosted runner on the VM replaces the step `Reach the VM` and the job's `runs-on`, and nothing else.

After the [bootstrap](#first-time-bootstrap-a-fresh-vm) and a first deploy by hand, and once the GHCR package `mlops_recommenditos/api` is public (see [Continuous delivery](#continuous-delivery)), since the VM pulls without a credential; `docker manifest inspect ghcr.io/mlops-2627q1-mds-upc/mlops_recommenditos/api:latest` answers without logging in once it is:

1. **The deploy key.** Generate a key pair for CD alone, on a laptop: `ssh-keygen -t ed25519 -N '' -C recommenditos-cd -f cd_key`.
   Append its public half to `/opt/recommenditos/.ssh/authorized_keys` on the VM, restricted to `deploy.sh`, so it can do nothing but deploy, roll back and report the tag (NFR-15):

    ```text
    command="/opt/recommenditos/deploy.sh",restrict ssh-ed25519 AAAA... recommenditos-cd
    ```

    sshd passes the command the workflow asks for to `deploy.sh` in `SSH_ORIGINAL_COMMAND`, and `deploy.sh` refuses anything but `deploy <tag>`, `rollback` and `status`; `restrict` turns off port, agent and X11 forwarding and the terminal.
2. **The host key**, pinned so the runner connects to the VM and nothing else: `ssh-keyscan -p <ssh-port> <vm-host>` from a trusted network, compared with `ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub` on the VM.
3. **The Environment `production`:** create it (**Settings**, **Environments**, **New environment**), or open it if a run has already created it, and limit **Deployment branches and tags** to `main`; `gh api repos/mlops-2627q1-mds-upc/MLOPS_Recommenditos/environments --jq '.environments[] | [.name, (.deployment_branch_policy | tostring)]'` shows both. It holds:

    | Name | Kind | Value |
    |------|------|-------|
    | `VM_SSH_KEY` | secret | the private half, `cd_key`; then delete the local copy |
    | `VM_HOST` | variable | the VM's public host name or address |
    | `VM_SSH_PORT` | variable | the forwarded SSH port, if it is not 22 |
    | `VM_USER` | variable | `recommenditos` is the default; set it only if it differs |
    | `VM_KNOWN_HOSTS` | variable | the lines of step 2 |
    | `SMOKE_URL` | variable | `http://<vm-host>:<port>`, the address `make smoke` checks |

4. **The switch:** the repository variable `DEPLOY_VIA` = `ssh` (**Settings**, **Secrets and variables**, **Actions**, **Variables**).
5. **The first run:** run the workflow by hand with the SHA of the newest image, then check its summary and the [rollback drill](#every-deploy) on the VM: deploy a tag whose image does not start, or one whose smoke check fails, and link the run that put the previous tag back in [#198](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/198).

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

**4. Load test.**
Run the [load test on the VM](load-testing.md#run-it-on-the-vm): its results settle the targets of NFR-02 to NFR-04.

## Without sudo or without Docker

`bootstrap.sh` needs root for all four of its steps.
If Docker is already installed and our account is in the `docker` group, none of them is needed: copy `compose.yaml` and `nginx.conf` into a directory of our own, copy `.env.template` there as `.env` and fill it in, and deploy as above, under our account.
If Docker is installed and we are not in the group, we ask the lecturer to add us; rootless Docker would also work, but it needs the `uidmap` package and subordinate IDs that only an admin can provide, and it cannot publish a port below 1024, so `HTTP_PORT` would move to 8080 or above.
If there is no Docker and no sudo, the fallback is the course demo's recipe in our home directory: uv installs Python and `uv sync --locked --no-default-groups` the runtime, the bundle is copied with `scp`, uvicorn runs on an unprivileged port under `systemd --user` or a `@reboot` cron entry, and nginx, without root, runs on an unprivileged port as well.
That loses the image as the unit we test and deploy, so we would ask for Docker before building it.

## Vulnerabilities and dependency updates

NFR-09 requires that a dependency with a known and fixable vulnerability does not reach the running system.
Three checks keep it that way.

**The image scan.**
`tools/image_scan.py` scans an image with [trivy](https://trivy.dev) for HIGH and CRITICAL vulnerabilities, in the Debian packages and in the Python packages of the virtual environment.
A finding with a fix available fails the job; one without a fix is counted in the summary and does not, so that a CVE of the base image that Debian has not patched yet does not block every pull request and every deploy.
CI's `Integration (Compose stack)` job scans both images of the stack, the API image it built and the nginx image, on every pull request that changes the image, and CD scans them again before it pushes.
The result, a table per image, goes into the job summary.
Before the scan, a canary scans `python:3.12.0-slim-bookworm` as published in October 2023, pinned by digest and pulled from Amazon ECR Public, or from Docker Hub when ECR answers with its pull limit, and fails the job unless the same scan finds fixable findings there (93 on 2026-10-22), so a scan that passes is evidence, as the gitleaks job's planted token is for NFR-15.
trivy is the release binary, pinned by version and SHA-256 in `.github/actions/setup-trivy`, not `aquasecurity/trivy-action`, whose tags an attacker moved to a credential stealer in March 2026 ([GHSA-69fq-xp46-6x23](https://github.com/aquasecurity/trivy/security/advisories/GHSA-69fq-xp46-6x23)).
Its vulnerability database is downloaded on each run, in about 5 s in CI, rather than kept in the Actions cache: it is rebuilt every few hours, so a cached copy would be stale by the next run.
When one registry answers with a rate limit or a server error, trivy takes the database from the next one: `mirror.gcr.io`, then Amazon ECR Public, then `ghcr.io`.

To scan locally, with trivy installed:

```bash
python3 tools/image_scan.py scan $(docker compose -f deploy/compose.yaml config --images)
```

When the scan fails, update the base image or the package; Dependabot's pull request may already do so.
If neither is possible yet, accept the finding in `.trivyignore`: the vulnerability ID with an expiry date (`CVE-2026-12345 exp:2026-12-31`), after which trivy reports it again, and a comment line above it that says why the risk is accepted.
The scan refuses an entry without either.
On 2026-10-22 the API image had no fixable HIGH or CRITICAL finding and 44 without a fix, all in Debian packages; the nginx image had none.

**The Dockerfile lint.**
[hadolint](https://github.com/hadolint/hadolint) checks the `Dockerfile` in the pre-commit hook and in CI's required `Lint & format` job.
Its one exception, DL3008 for the unpinned `libgomp1`, carries its reason in the `Dockerfile`: Debian removes a package version from its archive with every security update, so a pinned version would break the build instead of patching it.

**Dependabot.**
`.github/dependabot.yml` proposes updates every Monday for the GitHub Actions, the images of the `Dockerfile`, of `deploy/compose.yaml` and of the report's TeX Live image (`reports/latex/Dockerfile`), and `uv.lock`, one grouped pull request per ecosystem, with titles such as `chore(deps): bump the docker group with 2 updates`.
Every image is pinned by tag and digest (`python:3.12-slim-trixie@sha256:...`), so a rebuild starts from the bytes the last scan saw, and Dependabot moves both; a base image rebuilt under the same tag with Debian's security fixes is a new digest and so a pull request too, whose integration job scans the new image.
Python's minor version and nginx's stable branch are not moved: those are changes of their own.
The actions are pinned by commit SHA with the release in a comment, which Dependabot moves the same way.

## What M5 adds

- Continuous deployment to the VM: the `Deploy` workflow and `deploy.sh` with its rollback are built and rehearsed against the local stack ([Every deploy](#every-deploy)); turning them on needs the VM ([Turn continuous deployment on](#turn-continuous-deployment-on)), and so does the rollback drill on the VM itself and NFR-05's reboot check (NFR-13, FR-15).
- The containers are hardened ([#200](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/200)): a read-only root file system, no capabilities, `no-new-privileges`, and limits on memory and processes, as [the Compose bullet above](#the-stack) describes and the integration suite checks.
