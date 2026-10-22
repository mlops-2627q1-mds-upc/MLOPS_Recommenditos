#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = MLOPS_Recommenditos
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = uv run python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff and hadolint (use `make format` to do formatting)
.PHONY: lint
lint: typecheck
	uv run ruff format --check
	uv run ruff check
	uv run pre-commit run hadolint --all-files

## Type-check recommenditos/ and tools/ with mypy (configured in pyproject.toml)
.PHONY: typecheck
typecheck:
	uv run mypy

## Format source code with ruff
# `--exit-zero`, so a finding `--fix` cannot fix still prints but does not stop
# make before `ruff format` runs; `make lint` is the check that fails on it.
.PHONY: format
format:
	uv run ruff check --fix --exit-zero
	uv run ruff format



## Run tests
.PHONY: test
test:
	uv run pytest

# The API image installs `[project] dependencies` alone (EDN-47), so this builds
# an environment with exactly that plus the `test` group and runs the serving
# path from it. `--no-sync` because a plain `uv run` would install the default
# groups into the environment this target exists to keep without them.
# `--noconftest` because tests/conftest.py imports whatever the rest of the suite
# needs, and the check must constrain the modules tests/test_serving.py imports
# and nothing else. No coverage, because one file's coverage of the package is
# not a number to read, and a JUnit file of its own, so the full suite's is kept.
SERVING_ENV = .venv-serving

## Run the serving tests from the runtime dependencies alone, as CI does
.PHONY: test-serving
test-serving:
	UV_PROJECT_ENVIRONMENT=$(SERVING_ENV) uv sync --locked --no-default-groups --group test
	UV_PROJECT_ENVIRONMENT=$(SERVING_ENV) RECOMMENDITOS_SERVING_RUNTIME=1 \
		uv run --no-sync pytest --noconftest tests/test_serving.py --no-cov \
		--junitxml=reports/junit-serving.xml


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Run the DVC pipeline (see dvc.yaml)
.PHONY: repro
repro:
	uv run dvc repro

## Write the synthetic test fixture to data/fixture/ (tests do not need it)
.PHONY: fixture
fixture:
	uv run python tests/fixtures/generate_fixture.py

## Show the pipeline's metrics (the SC-01 to SC-06 gate of NFR-01)
.PHONY: metrics
metrics:
	uv run dvc metrics show

# The API on the bundle in models/lgbm-basic (after `dvc pull`), or the one
# RECOMMENDITOS_MODEL_DIR names. Swagger UI at /docs, ReDoc at /redoc.
API_HOST ?= 127.0.0.1
API_PORT ?= 8000

## Write the API's OpenAPI document to docs/docs/openapi.json (after every API contract change)
.PHONY: openapi
openapi:
	$(PYTHON_INTERPRETER) -m tools.openapi_snapshot

## Run the API locally with uvicorn on http://127.0.0.1:8000 (docs at /docs)
.PHONY: api
api:
	uv run python -m recommenditos.api --host $(API_HOST) --port $(API_PORT)

# The serving stack of docs/docs/deployment.md (EDN-87): the API image with the
# pulled model, behind nginx, on http://localhost:8080. Needs Docker and `dvc pull`.
COMPOSE = docker compose -f deploy/compose.yaml
# The tag a deployment ships: the commit the image is built from.
IMAGE_TAG ?= $(shell git rev-parse --short HEAD)
SMOKE_URL ?= http://localhost:8080

## Build the API image recommenditos-api:<commit> for a deployment (docs/docs/deployment.md)
.PHONY: image
image:
	docker build -t recommenditos-api:$(IMAGE_TAG) .

## Build and start the API behind nginx on http://localhost:8080, waiting until healthy
.PHONY: stack
stack:
	$(COMPOSE) up -d --build --wait

## Stop the local stack and remove its containers
.PHONY: stack-down
stack-down:
	$(COMPOSE) down

# The integration suite of tests/integration/ against a running stack, through
# its proxy. Without coverage, which measures this process and not the API's, and
# with a JUnit file of its own, so the full suite's is kept. Against a remote URL
# the tests that need the Docker host are skipped. `make smoke`, the
# post-deployment check, always deselects them: a smoke run on the VM itself or
# through an SSH tunnel would otherwise kill the production API or inspect the
# containers of the machine it runs on.
STACK_URL ?= http://localhost:8080
INTEGRATION_MARKERS ?= integration

## Run the integration suite against a running stack: make test-integration [STACK_URL=...]
.PHONY: test-integration
test-integration:
	RECOMMENDITOS_STACK_URL=$(STACK_URL) uv run pytest tests/integration -m "$(INTEGRATION_MARKERS)" \
		--no-cov --junitxml=reports/junit-integration.xml

# The load test of docs/docs/load-testing.md (issue #143): Locust against a base
# URL, with the measurements taken on the Docker host of the stack it targets.
# `make loadtest-bodies` first, which needs `dvc pull data/processed/test.parquet`.
LOADTEST_URL ?= http://localhost:8080
LOADTEST_RATE ?= 20
LOADTEST_DURATION ?= 5m
LOADTEST_OUT ?= reports/loadtest/local
LOADTEST_ARGS ?=

## Write the load test's request bodies from held-out listings into data/interim/loadtest/
.PHONY: loadtest-bodies
loadtest-bodies:
	uv run python -m tools.loadtest.bodies

## Load test a stack and measure it: make loadtest [LOADTEST_URL=... LOADTEST_RATE=20 LOADTEST_DURATION=5m LOADTEST_OUT=...]
.PHONY: loadtest
loadtest:
	uv run python -m tools.loadtest.run --host $(LOADTEST_URL) --rate $(LOADTEST_RATE) \
		--duration $(LOADTEST_DURATION) --out $(LOADTEST_OUT) $(LOADTEST_ARGS)

## Check a deployed stack through its proxy: make smoke [SMOKE_URL=http://<vm>:<port>]
.PHONY: smoke
smoke:
	$(MAKE) test-integration STACK_URL=$(SMOKE_URL) INTEGRATION_MARKERS="integration and not docker_host"

## Lint every notebook with Pynblint, in its own locked environment (tools/pynblint-env)
.PHONY: notebook-lint
notebook-lint:
	uv run --project tools/pynblint-env --locked python tools/notebook_lint.py

# Expanded by make rather than by the shell: an unmatched glob would reach
# `jupyter execute` as a literal file name, which it fails on with a traceback.
NOTEBOOKS = $(wildcard notebooks/*.ipynb)

## Run every notebook top to bottom on the real data (needs the 548 MB source CSV; not in CI)
.PHONY: notebook-run
notebook-run:
ifeq ($(NOTEBOOKS),)
	@echo "No notebook in notebooks/ to run."
else
	uv run jupyter execute $(NOTEBOOKS)
endif


# initial (Milestones 1-3, max 15 pages) or final (Milestones 1-6, max 30 pages)
DELIVERABLE ?= initial

## Build draft report and EDN with writing guidance into reports/latex/build/
.PHONY: report
report:
	reports/latex/build.sh draft

## Build submission PDFs without guidance: make report-submit DELIVERABLE=initial|final
.PHONY: report-submit
report-submit:
	reports/latex/build.sh $(DELIVERABLE)

## Transfer the EDN entries of reports/edn.md into reports/latex/edn/ (after every change to it)
.PHONY: edn
edn:
	$(PYTHON_INTERPRETER) -m tools.edn_latex

## Same as `make report`, but inside Docker (no local LaTeX needed)
.PHONY: report-docker
report-docker:
	docker build -t mlops-latex reports/latex
	docker run --rm -v "$(CURDIR)/reports/latex:/report" mlops-latex draft


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
