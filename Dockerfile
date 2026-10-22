# The API image: the serving runtime from uv.lock (EDN-47), one model bundle and
# the comparables index (EDN-08, FR-12), built after `dvc pull` so that
# `models/<variant>` and `data/processed/comparables.parquet` are present.
# docs/docs/deployment.md says how it is built, run and deployed.
#
# Two stages, after uv's Docker guide: the builder installs the project into a
# virtual environment with uv, and the runtime stage copies that environment
# and the bundle onto the same slim Python image, without uv, the build cache or
# the source tree.
#
# Every image is pinned by digest as well as by tag, so a rebuild starts from
# the bytes the last scan saw (NFR-09). Dependabot moves tag and digest together
# (.github/dependabot.yml); it reads only `FROM` lines, written out in full, so
# the base image is not an ARG and uv comes in through a stage of its own.

# A pinned uv, so that a rebuild resolves nothing differently from CI.
FROM ghcr.io/astral-sh/uv:0.11.11@sha256:798712e57f879c5393777cbda2bb309b29fcdeb0532129d4b1c3125c5385975a AS uv

FROM python:3.12-slim-trixie@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2 AS builder

COPY --from=uv /uv /bin/uv

# Bytecode compiled at build time, so the first request does not pay for it;
# files copied out of the cache mount rather than hard-linked into it; the
# interpreter of the base image, so the environment runs in the runtime stage.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# The dependencies first, in a layer of their own that only a change to the lock
# invalidates. `--no-default-groups` leaves out every dependency group: the
# image holds `[project] dependencies`, the serving runtime, and nothing else.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-default-groups --no-install-project

# Then the package itself, installed non-editable, so the environment holds it
# and the source tree stays in this stage. flit reads README.md and LICENSE.
# `--reinstall-package` (which implies `--refresh-package`) builds the wheel from
# this context every time. The cache mount is shared by every build on the
# machine and every checkout's project sits at /app, so when pyproject.toml, the
# cache key uv checks for a local project, matches another checkout's, uv would
# install the wheel it built from that checkout (#180).
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY recommenditos ./recommenditos
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable --reinstall-package recommenditos


FROM python:3.12-slim-trixie@sha256:cb33f7f771ecec2ef1225cf39b159591a040391bd73e4c01f00fdfc4c47611d2

# The bundle to serve. `/health` reports the DVC hash of this directory, so it is
# copied whole and unchanged, and FR-12's check compares it with dvc.lock.
ARG MODEL_VARIANT=lgbm-basic

# LightGBM's native library links against GNU OpenMP, which the slim image lacks.
# Not pinned to a version: Debian replaces a package version in its archive with
# every security update, so a pin would break the build instead of patching it.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system --gid 10001 api \
    && useradd --system --uid 10001 --gid api --no-create-home --shell /usr/sbin/nologin api

# The one directory the API writes: the prediction log and the feedback (EDN-86).
# Compose mounts a named volume here, which takes this owner when it is created,
# so the service user can write it and the drift job can mount it too.
RUN mkdir -p /var/lib/recommenditos && chown 10001:10001 /var/lib/recommenditos

# Owned by root and only readable by the service user: the API writes nothing
# into its own code or model.
COPY --from=builder /app/.venv /app/.venv
COPY models/${MODEL_VARIANT} /app/models/${MODEL_VARIANT}
# FR-09's index, beside the bundle, where the API looks for it; `/health`
# reports its md5, which is what dvc.lock records for it.
COPY data/processed/comparables.parquet /app/models/comparables.parquet

# PROMETHEUS_MULTIPROC_DIR: where every uvicorn worker writes its metrics, so
# that /metrics adds them up whatever the worker count (docs/docs/api.md#metrics).
# Under /tmp, the one place the service user may write; Compose mounts a tmpfs
# there, and the launcher creates the directory if nothing is mounted.
# RECOMMENDITOS_LOG_FORMAT: one JSON object per log line, each line of a request
# with its request ID (docs/docs/api.md#logs).
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    RECOMMENDITOS_LOG_FORMAT=json \
    RECOMMENDITOS_MODEL_DIR=/app/models/${MODEL_VARIANT} \
    RECOMMENDITOS_COMPARABLES_FILE=/app/models/comparables.parquet \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus \
    RECOMMENDITOS_PREDICTION_LOG=/var/lib/recommenditos/prediction-log.sqlite3

WORKDIR /app
# Numeric, so that a runtime can check it is not root without reading /etc/passwd.
USER 10001:10001
EXPOSE 8000

# A probe from the standard library, so the image needs no curl. Every second
# while the API starts, then every 10 s; 3 failures in a row mark it unhealthy.
# After the start period, a probe whose two attempts of 2 s both go unanswered
# also ends the hung server, so that the restart policy brings it back; that
# needs an init in front of the server, which Compose's `init: true` adds
# (recommenditos/api/healthcheck.py, NFR-05, EDN-93).
HEALTHCHECK --interval=10s --timeout=6s --start-period=60s --start-interval=1s --retries=3 \
    CMD ["python", "-m", "recommenditos.api.healthcheck"]

# The launcher empties PROMETHEUS_MULTIPROC_DIR, then starts uvicorn with the app
# factory; WEB_CONCURRENCY sets the number of workers (one by default).
CMD ["python", "-m", "recommenditos.api", "--host", "0.0.0.0", "--port", "8000"]
