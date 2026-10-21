# The API image: the serving runtime from uv.lock (EDN-47) and one model bundle
# (EDN-08, FR-12), built after `dvc pull` so that `models/<variant>` is present.
# docs/docs/deployment.md says how it is built, run and deployed.
#
# Two stages, after uv's Docker guide: the builder installs the project into a
# virtual environment with uv, and the runtime stage copies that environment
# and the bundle onto the same slim Python image, without uv, the build cache or
# the source tree.

ARG PYTHON_IMAGE=python:3.12-slim-trixie

FROM ${PYTHON_IMAGE} AS builder

# A pinned uv, so that a rebuild resolves nothing differently from CI.
COPY --from=ghcr.io/astral-sh/uv:0.11.11 /uv /bin/uv

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
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY recommenditos ./recommenditos
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable


FROM ${PYTHON_IMAGE}

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

# PROMETHEUS_MULTIPROC_DIR: where every uvicorn worker writes its metrics, so
# that /metrics adds them up whatever the worker count (docs/docs/api.md#metrics).
# Under /tmp, the one place the service user may write; Compose mounts a tmpfs
# there, and the launcher creates the directory if nothing is mounted.
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    RECOMMENDITOS_MODEL_DIR=/app/models/${MODEL_VARIANT} \
    PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus \
    RECOMMENDITOS_PREDICTION_LOG=/var/lib/recommenditos/prediction-log.sqlite3

WORKDIR /app
# Numeric, so that a runtime can check it is not root without reading /etc/passwd.
USER 10001:10001
EXPOSE 8000

# urllib from the standard library, so the image needs no curl. Every second
# while the API starts, then every 10 s; 3 failures in a row mark it unhealthy.
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --start-interval=1s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

# The launcher empties PROMETHEUS_MULTIPROC_DIR, then starts uvicorn with the app
# factory; WEB_CONCURRENCY sets the number of workers (one by default).
CMD ["python", "-m", "recommenditos.api", "--host", "0.0.0.0", "--port", "8000"]
