# The API image: the serving runtime from uv.lock (EDN-47), one model bundle and
# the comparables index (EDN-08, FR-12), built after `dvc pull` so that
# `models/<variant>` and `data/processed/comparables.parquet` are present.
# docs/docs/deployment.md says how it is built, run and deployed.
#
# After uv's Docker guide: the `dependencies` stage installs the third-party
# dependencies into a virtual environment with uv, the `builder` stage adds the
# project, and the runtime stage copies both and the bundle onto the same slim
# Python image, without uv, the build cache or the source tree. The runtime stage
# copies the dependencies and the project in separate layers, so after a code
# change the VM pulls a layer of about 0.6 MB compressed, not the whole
# environment.
#
# Every image is pinned by digest as well as by tag, so a rebuild starts from
# the bytes the last scan saw (NFR-09). Dependabot moves tag and digest together
# (.github/dependabot.yml); it reads only `FROM` lines, written out in full, so
# the base image is not an ARG and uv comes in through a stage of its own.

# A pinned uv, so that the image builds the same way every time; Dependabot moves
# the pin.
FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

FROM python:3.12-slim-trixie@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159 AS dependencies

COPY --from=uv /uv /bin/uv

# Bytecode compiled at build time, so the first request does not pay for it;
# files copied out of the cache mount rather than hard-linked into it; the
# interpreter of the base image, so the environment runs in the runtime stage.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# The dependencies, in a stage of its own that only a change to the lock or to
# pyproject.toml invalidates. `--no-default-groups` leaves out every dependency
# group: the image holds `[project] dependencies`, the serving runtime, and
# nothing else.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-default-groups --no-install-project


FROM dependencies AS builder

# Then the package itself, installed non-editable, so the environment holds it
# and the source tree stays in this stage. flit reads README.md and LICENSE.
# `--reinstall-package` (which implies `--refresh-package`) builds the wheel from
# this context every time. The cache mount is shared by every build on the
# machine and every checkout's project sits at /app, so when pyproject.toml, the
# cache key uv checks for a local project, matches another checkout's, uv would
# install the wheel it built from that checkout (#180).
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY recommenditos ./recommenditos
# The wheel's files are then copied, with their bytecode and their parent
# directories, to /project, under the same paths, for the runtime stage to lay
# over the dependencies; the healthcheck's module also to /app/healthcheck.py,
# for the HEALTHCHECK to run as a script. The wheel installs nothing else, no
# script and no other top-level package; the RECORD check fails the build when
# that changes, and when it cannot read RECORD (grep exits with 1 only when it
# read the file and printed no line).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --no-editable --reinstall-package recommenditos \
    && site="$(/app/.venv/bin/python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')" \
    && { grep -v -e '^recommenditos/' -e '^recommenditos-[^/]*\.dist-info/' "$site"/recommenditos-*.dist-info/RECORD; test $? -eq 1; } \
    && mkdir /project \
    && cp -a --parents "$site"/recommenditos "$site"/recommenditos-*.dist-info /project/ \
    && cp -a "$site"/recommenditos/api/healthcheck.py /project/app/healthcheck.py


FROM python:3.12-slim-trixie@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159

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
# Ordered from the least to the most often changed, as Docker rebuilds every
# layer above a changed one: the dependencies, which only a change to the lock
# or to pyproject.toml rebuilds; the bundle and the index, which a model
# promotion changes; and last the project's files, laid over the dependencies,
# so a code change rebuilds only that layer, which holds the healthcheck's
# script too, and the VM pulls only that one.
COPY --from=dependencies /app/.venv /app/.venv
COPY models/${MODEL_VARIANT} /app/models/${MODEL_VARIANT}
# FR-09's index, beside the bundle, where the API looks for it; `/health`
# reports its md5, which is what dvc.lock records for it.
COPY data/processed/comparables.parquet /app/models/comparables.parquet
COPY --from=builder /project/app /app

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
# while the API starts, then every 5 s; 3 failures in a row mark it unhealthy.
# After the start period, a probe whose two attempts of 6 s both go unanswered
# also ends the hung server, so that the restart policy brings it back; that
# needs an init in front of the server, which Compose's `init: true` adds
# (recommenditos/api/healthcheck.py, NFR-05, EDN-93). The module runs as a
# script, without `site` (-S) and isolated from the environment (-I): it needs
# only the standard library, and importing the package's __init__ and
# site-packages would cost a probe about twice its CPU time, every 5 s.
HEALTHCHECK --interval=5s --timeout=15s --start-period=60s --start-interval=1s --retries=3 \
    CMD ["python", "-I", "-S", "/app/healthcheck.py"]

# The launcher empties PROMETHEUS_MULTIPROC_DIR, then starts uvicorn with the app
# factory; WEB_CONCURRENCY sets the number of workers (one by default).
CMD ["python", "-m", "recommenditos.api", "--host", "0.0.0.0", "--port", "8000"]
