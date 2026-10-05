# Recommenditos

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Used-car price estimation as an ML system in production.
This is our team project for *Machine Learning Systems in Production (MLOps)* at FIB-UPC, 2026/27.

## What we are building

We build and deploy a used-car price component: an ML model behind an API.
It covers two use cases:

- **Car valuation:** you describe a car you own, and the API estimates the price it would be listed at, with an explanation of what drives that price.
- **Purchase guidance:** you describe the car you want to buy, even only partially, and the API returns the price range you should expect to pay, plus comparable listings.

We train on the [AutoScout24 Car Listings Dataset (2025 snapshot)](https://zenodo.org/records/17643343): about 118,000 listings from 8 European countries.
The scope is used passenger cars of the makes with enough data behind them, which is mostly premium brands.
All Spanish listings are held out as a "new market", so we can show that our monitoring actually detects drift and that retraining fixes it.

The course grades how well we apply MLOps practices, not model accuracy.
So we focus on clean, reproducible and well-justified engineering: data and model versioning, experiment tracking, testing, a containerised API, CI/CD and monitoring.
The planned model is gradient boosting (LightGBM, with CatBoost as challenger), because it is strong on tabular data, trains in minutes on a CPU and gives exact SHAP explanations.

For the details, see:

- [Project brief](docs/docs/project-brief.md) - goal, data facts, modelling and architecture plan, open decisions.
- [Problem specification](docs/docs/problem-spec.md) - scope, target, features and success criteria.
- [Engineering Decision Notebook](reports/edn.md) - the decisions we made, the alternatives and why we chose what we chose.

## Team and contact

| Name | GitHub | E-mail | Role |
|------|--------|--------|------|
| Lukas Häußler | [@lukas2510](https://github.com/lukas2510) | <lukas.haeussler@estudiantat.upc.edu> | Product Owner, Scrum Master |
| Kevin Adameit | [@kadameit](https://github.com/kadameit) | <kevin.adameit@estudiantat.upc.edu> | Development Team |
| Urszula Sawczuk | [@ulasawczuk](https://github.com/ulasawczuk) | <urszula.wanda.sawczuk@estudiantat.upc.edu> | Development Team |
| Michal Dudek | [@michudud04](https://github.com/michudud04) | <michal.feliks.dudek@estudiant.upc.edu> | Development Team |
| Mark Atzberger | [@W11W11W11](https://github.com/W11W11W11) | <mark.welf.atzberger@estudiant.upc.edu> | Development Team |

For questions about the project, reach out by email or open an issue.
Day-to-day team communication happens on Discord, and we track our work on the [project board](https://github.com/orgs/mlops-2627q1-mds-upc/projects/1).

## Getting started

```bash
uv sync
uv run pre-commit install
uv run pre-commit install-hooks
```

[Getting started](docs/docs/getting-started.md) takes it from there: the DagsHub token, `.env`, `dvc pull` and your first MLflow run.

Before contributing, read [CONTRIBUTING.md](CONTRIBUTING.md).
It covers our Git workflow, data versioning with DVC and the checks a PR has to pass.

## Project organization

```
├── .env.template      <- The variables `.env` needs; `.env` itself is gitignored
├── LICENSE            <- Open-source license
├── Makefile           <- Convenience commands like `make lint` or `make test`
├── README.md          <- This file
├── data               <- Raw, interim and processed data, versioned with DVC
├── docs               <- MkDocs project with the project documentation
├── gx                 <- Great Expectations context, built by `configure_gx` (DVC output, not committed)
├── models             <- Trained and serialized models
├── notebooks          <- Jupyter notebooks, named `<number>.<version>-<initials>-<description>`
├── dvc.yaml           <- The pipeline; see docs/docs/pipeline.md
├── params.yaml        <- Everything the pipeline is configured by
├── metrics.json       <- The SC-01 to SC-06 gate result, written by `evaluate`
├── pyproject.toml     <- Serving dependencies, the dependency groups and tool configuration
├── references         <- Course material, data dictionaries and other references
├── reports            <- Report and EDN (LaTeX), metrics, generated figures
├── tests              <- Pytest test suite
├── tools              <- Repository tooling: the requirement matrix, the notebook lint and its environment
└── recommenditos      <- Source code of the package
    ├── config.py      <- Paths
    ├── schema.py      <- The data contract every stage reads and writes against
    ├── pipeline.py    <- Shared stage plumbing: params in, checked frames out
    ├── tracking.py    <- MLflow against DagsHub, and the setup check that proves it works
    ├── data           <- One module per data stage of the pipeline
    │   ├── download_raw_dataset.py    <- `download`
    │   ├── preprocess.py              <- `preprocess`
    │   ├── gx_context_configuration.py <- `configure_gx`
    │   ├── validate_data.py           <- `validate-data`
    │   ├── split_data.py              <- `split`
    │   ├── build_features.py          <- `features`
    │   └── synthetic.py               <- The generated test fixture
    ├── modeling
    │   ├── train.py   <- `train`
    │   ├── evaluate.py <- `evaluate`
    │   └── model.py   <- The estimators, and the load-and-predict seam
    └── plots.py       <- Visualizations
```

One module per pipeline stage rather than the flat `dataset.py` / `features.py`
of the template, so a stage's DVC dependencies name exactly the code it runs
([EDN-28](reports/edn.md)).
