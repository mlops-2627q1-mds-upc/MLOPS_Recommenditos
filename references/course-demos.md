# Course demo repo - what matters for us

Notes from the official demo repo <https://github.com/mlops-2627q1-mds-upc/MLOps-2627q1-demos> (read at commit `1b96937`, September 2026).
The demo is an IMDB sentiment classifier (DistilBERT), so the code does not transfer 1:1, but it shows which tools and setup the teachers expect per milestone.
Re-check the repo before each milestone: at the time of writing it has no demos for GitHub Actions (M5b) or monitoring (M6) yet.

## Expected tools per milestone

| Milestone | Practice | Tools the demo uses | Demo guide |
|-----------|----------|---------------------|------------|
| M1 Inception | Requirements, model and dataset cards | Hugging Face [model card](https://huggingface.co/docs/hub/model-card-annotated) and [dataset card](https://github.com/huggingface/datasets/blob/main/templates/README_guide.md) templates | - |
| M2 Reproducibility | Project structure | Cookiecutter Data Science + uv | `docs/project-setup.md` |
| M2 Reproducibility | Code and data versioning | Git (GitHub Flow), DVC with a DagsHub remote | `docs/git-demo.md`, `docs/dvc-demo.md` |
| M2 Reproducibility | Experiment tracking | MLflow with DagsHub as tracking server | `docs/mlflow-demo.md` |
| M3 QA | Energy efficiency | CodeCarbon, emissions logged to MLflow | `docs/codecarbon-demo.md`, `src/modeling/train_with_code_carbon.py` |
| M3 QA | Static analysis | Ruff with Pylint rules (`PL`), Pynblint, Pylint, flake8 | `pyproject.toml` |
| M3 QA | Data and model testing | Pytest (+ pytest-cov), Great Expectations, Deepchecks for vision | `docs/pytest-demo.md`, `docs/great-expectations-demo.md` |
| M3 QA | Non-functional requirements (optional) | SHAP, AIF360, TrustML | - |
| M4 Deployment | ML system design | A cloud VM (FIB Virtech, AWS, GCP, Azure, Okteto) | `docs/deployment/` |
| M4 Deployment | API | FastAPI + Pydantic schemas, tested with Pytest and `httpx` | `docs/fastapi-demo.md` |
| M5 Packaging | Containers | Docker, Docker Compose, Kubernetes | `docs/docker-demo.md` |

The non-functional QA row (SHAP, AIF360, TrustML) is not in the grading table in `MLOps-lab.md`.
It is a cheap way to reach the "extended or innovative" (Excellent) grade for M3.
A broader list of MLOps tools the teachers point to: <https://agate-tangerine-725.notion.site/MLOps-tools-255624cb2156801e9a98db82fd911da2>.

## Setup details worth copying

### DVC (M2)

- Remote is DagsHub storage (100 GB free); connect the GitHub repo to DagsHub and pick the **HTTP** option for the DVC remote, not S3.
- The pipeline lives in `dvc.yaml` with stages `download -> preprocess -> validate-data -> split -> train`, each declaring its code files as `deps` so code changes trigger a rerun.
- `dvc repro` runs it; commit the resulting `dvc.lock`.
- `train` uses `foreach` to train several model variants from one stage definition.
- A file already tracked by Git has to be removed with `git rm --cached` before `dvc add`.
- `dvc import` (tracked) vs `dvc get` (untracked) for data coming from another repo.

### MLflow (M2)

- DagsHub doubles as the shared MLflow tracking server, configured via `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`.
- Credentials go in `.env` (gitignored), loaded with `python-dotenv` in `config.py`; a committed `.env.template` documents the variables.
- The demo pins `mlflow>=2.22,<3` for DagsHub compatibility; check whether that still applies before pinning.
- One experiment per model (`mlflow.set_experiment`), one run per training; use `autolog()` where the framework supports it, `log_param`/`log_metric`/`log_artifact` otherwise.
- Every DVC stage becomes its own MLflow run, so metrics from different stages end up in separate runs unless we group them deliberately.

### CodeCarbon (M3)

- Wrap only the `fit` call in an `EmissionsTracker` context manager (`tracking_mode="process"`, CSV output under a metrics dir, `on_csv_write="append"`).
- Read the last CSV row back and log it to MLflow (`log_metrics` for energy/emissions, `log_params` for hardware info) so emissions sit next to accuracy per run.

### Testing and linting (M3)

- Model tests: a fixture loads the trained model, one test asserts a metric threshold on the test set, parametrized tests check behaviour under small input perturbations (invariance tests).
- Data tests: Great Expectations file context in `gx/` with an expectation suite, validation definitions for raw and clean data, and a checkpoint that updates Data Docs; the validation runs as a DVC stage.
- API tests: FastAPI `TestClient` fixtures, happy path plus invalid input (e.g. input too long).
- Pytest config in `pyproject.toml`: `pythonpath = "."`, `testpaths = "tests"`, `--junitxml` report for CI and `--cov` HTML report under `reports/coverage`.
- The demo's ruff config enables `E, F, UP, B, SIM, I, PL`; ours only extends `I`, so it does not cover the Pylint-style checks the rubric asks for yet.

### API and deployment (M4)

- `src/api/api.py` holds the FastAPI app, `src/api/schemas.py` the Pydantic request/response models; Swagger UI at `/docs`, ReDoc at `/redoc`.
- The generic VM recipe: SSH key, clone the repo, `uv sync`, fill `.env`, `dvc pull` the model, run `uvicorn`, optionally put nginx in front as a reverse proxy.
- Free options compared in `docs/deployment/01_deploy_general.md`:

| Provider | Compute | Storage | Max memory | Duration |
|----------|---------|---------|------------|----------|
| AWS | 750 h/month | 5 GB S3 + 30 GB EBS | 1 GB | 12 months |
| Azure | 750 h/month | 5 GB Blob + 128 GB disks | 1-4 GB | 12 months |
| GCP | 750 h/month | 30 GB | 0.6 GB | 12 months + 300 USD |
| Okteto | Unlimited | 5 GB | 3 GB | Unlimited |
| FIB Virtech | Unlimited | 20 GB | 4 GB | 1 semester |

- The FIB Virtech VM is free for each team for the semester: <https://www.fib.upc.edu/en/fib/it-services/cloud-fib-teaching>.

### Docker (M5)

- The demo Dockerfile is minimal (`python` base image, `pip install -r requirements.txt`, `uvicorn` as `CMD`).
- We use uv and `pyproject.toml`, so our image should install from `uv.lock` instead; there is room to do better here (multi-stage build, non-root user, slim base).

## Papers behind the course

Cite these in the report where we discuss the course's MLOps process (already in `reports/latex/references.bib`):

- F. Lanubile, S. Martínez-Fernández, L. Quaranta. "Teaching MLOps in Higher Education through Project-Based Learning." SEET@ICSE 2023. doi:10.1109/ICSE-SEET58685.2023.00015
- F. Lanubile, S. Martínez-Fernández, L. Quaranta. "Training future ML engineers: a project-based course on MLOps." IEEE Software 2024. doi:10.1109/MS.2023.3310768
