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


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	uv run ruff format --check
	uv run ruff check

## Format source code with ruff
.PHONY: format
format:
	uv run ruff check --fix
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
