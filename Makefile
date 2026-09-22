#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = MLOPS_Recommenditos
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

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
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


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


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) recommenditos/dataset.py


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
