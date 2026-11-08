# The monitoring image: M6's jobs and the samples they read (docs/docs/monitoring.md).
#
# Built from the repository root, like the API image, with its own ignore file
# (monitoring/Dockerfile.dockerignore):
#     docker build -f monitoring/Dockerfile -t recommenditos-monitoring:local .
# or through Compose, `docker compose -f deploy/compose.yaml --profile replay build`.
# CD builds it from the same commit as the API image, pushes it under the same
# tag, and bakes in the samples of `dvc pull monitoring_samples`; without
# data/monitoring/ the build fails at the COPY below.
#
# It runs as an unprivileged user with a read-only root file system and no
# capabilities (deploy/compose.yaml): the replay writes nothing, the drift job
# (the stack's `monitor` service) only its volumes and /tmp. The environment
# comes from monitoring/uv.lock, apart from the project's own, because the drift
# job's Alibi Detect needs numpy<2 and pandas<3.

# The same uv and Python images as the API's Dockerfile, pinned by digest, which
# Dependabot moves together with the tag.
FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

FROM python:3.12-slim-trixie@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54 AS dependencies
COPY --from=uv /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app
# The locked dependencies without the dev group, into /app/.venv.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=monitoring/uv.lock,target=uv.lock \
    --mount=type=bind,source=monitoring/pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-dev --no-install-project
# Pillow out again: Alibi Detect 0.13 requires Pillow<11, which has dozens of
# fixed HIGH and CRITICAL findings, and only its image detectors and
# Matplotlib use it, none of which the jobs import. The stack's integration
# suite checks that it is gone and that a pass still runs
# (tests/integration/test_deployed_monitor.py).
RUN uv pip uninstall --python /app/.venv/bin/python pillow

FROM python:3.12-slim-trixie@sha256:4840d514c583a4d3d8cad3940e68dee6f0f6f3635c12fcd0a3d74c21a5b93a54
# The state of Debian's archive, which CD passes so that its cached apt layer is
# rebuilt whenever Debian publishes, as in the API's Dockerfile, which says how.
ARG DEBIAN_ARCHIVE_STATE
# zlib1g from trixie-security, as in the API's Dockerfile, which says why
# (CVE-2026-76844 and CVE-2026-85091) and when the line can go (#334). Not
# pinned to a version, for the reason the API's Dockerfile gives for libgomp1.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends --only-upgrade zlib1g \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --system --gid 10002 monitor \
    && useradd --system --uid 10002 --gid monitor --no-create-home --shell /usr/sbin/nologin monitor
# The mount point of the drift job's volume `monitor-results`, owned by the user
# the job runs as, the API's (10001, deploy/compose.yaml), so that Docker gives
# a new volume that owner when it first mounts it.
RUN mkdir -p /var/lib/recommenditos-monitor && chown 10001:10001 /var/lib/recommenditos-monitor
COPY --from=dependencies /app/.venv /app/.venv
# The package, compiled here, since the read-only root lets Python write no
# bytecode at run time.
COPY monitoring/recommenditos_monitoring /app/recommenditos_monitoring
RUN /app/.venv/bin/python -m compileall -q /app/recommenditos_monitoring
# The drift job's loop, which the stack's `monitor` service runs.
COPY monitoring/monitor-loop.sh /app/monitor-loop.sh
# The samples of the `monitoring_samples` stage: the replay's control and ES
# listings, the drift reference and the manifest with its reference date.
COPY data/monitoring /app/data/monitoring
# The evaluate stage's outputs of the commit the image is built from, the same
# commit as the API image's model, both committed to git: the test MdAPE and the
# full-input coverage the drift job compares each window's error and coverage
# with (#229), and the dvc.lock that names the model version of each variant's
# values, so that a window is judged against its own model's after a
# promotion (#307), and the test MdAPE per country, the market baseline of a
# window's MdAPE (#306).
COPY metrics.json /app/metrics.json
COPY reports/metrics/interval-coverage-by-make.csv /app/interval-coverage-by-make.csv
COPY reports/metrics/segments.csv /app/segments.csv
COPY dvc.lock /app/dvc.lock
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RECOMMENDITOS_MONITORING_DATA_DIR=/app/data/monitoring \
    RECOMMENDITOS_MONITOR_METRICS=/app/metrics.json \
    RECOMMENDITOS_MONITOR_COVERAGE=/app/interval-coverage-by-make.csv \
    RECOMMENDITOS_MONITOR_LOCK=/app/dvc.lock \
    RECOMMENDITOS_MONITOR_SEGMENTS=/app/segments.csv \
    # Alibi Detect imports Matplotlib, Numba and transformers, which want a
    # writable cache directory; the root is read-only, /tmp a tmpfs. Without
    # PyTorch or TensorFlow, which the job does not need, transformers would
    # warn about them on every pass.
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    HF_HOME=/tmp/huggingface \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1
WORKDIR /app
USER 10002:10002
ENTRYPOINT ["python", "-m", "recommenditos_monitoring"]
CMD ["--help"]
