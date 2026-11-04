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
# It runs as an unprivileged user and writes nothing: the stack starts it with a
# read-only root file system and no capabilities (deploy/compose.yaml). The
# environment comes from monitoring/uv.lock, apart from the project's own,
# because the drift job's Alibi Detect needs numpy<2 and pandas<3.

# The same uv and Python images as the API's Dockerfile, pinned by digest, which
# Dependabot moves together with the tag.
FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

FROM python:3.12-slim-trixie@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159 AS dependencies
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

FROM python:3.12-slim-trixie@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
RUN groupadd --system --gid 10002 monitor \
    && useradd --system --uid 10002 --gid monitor --no-create-home --shell /usr/sbin/nologin monitor
COPY --from=dependencies /app/.venv /app/.venv
# The package, compiled here, since the read-only root lets Python write no
# bytecode at run time.
COPY monitoring/recommenditos_monitoring /app/recommenditos_monitoring
RUN /app/.venv/bin/python -m compileall -q /app/recommenditos_monitoring
# The samples of the `monitoring_samples` stage: the replay's control and ES
# listings, the drift reference and the manifest with its reference date.
COPY data/monitoring /app/data/monitoring
ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RECOMMENDITOS_MONITORING_DATA_DIR=/app/data/monitoring
WORKDIR /app
USER 10002:10002
ENTRYPOINT ["python", "-m", "recommenditos_monitoring"]
CMD ["--help"]
