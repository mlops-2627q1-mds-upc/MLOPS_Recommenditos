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
Day-to-day team communication happens on [Discord](https://discord.gg/uG2eACGng), and we track our work on the [project board](https://github.com/orgs/mlops-2627q1-mds-upc/projects/1).

## Getting started

```bash
uv sync
pre-commit install
```

To get the data, set up your DagsHub credentials once as described in [Data versioning](docs/docs/data-versioning.md#first-time-setup), then run `uv run dvc pull`.

Before contributing, read [CONTRIBUTING.md](CONTRIBUTING.md).
It covers our Git workflow, data versioning with DVC and the checks a PR has to pass.

## Project organization

```
├── LICENSE            <- Open-source license
├── Makefile           <- Convenience commands like `make lint` or `make test`
├── README.md          <- This file
├── data               <- Raw, interim and processed data, versioned with DVC
├── docs               <- MkDocs project with the project documentation
├── models             <- Trained and serialized models
├── notebooks          <- Jupyter notebooks, named `<number>-<initials>-<description>`
├── pyproject.toml     <- Package metadata and tool configuration
├── references         <- Course material, data dictionaries and other references
├── reports            <- Report and EDN (LaTeX), generated figures
├── tests              <- Pytest test suite
└── recommenditos      <- Source code of the package
    ├── config.py      <- Paths and configuration
    ├── dataset.py     <- Data download and preparation
    ├── features.py    <- Feature engineering
    ├── modeling
    │   ├── train.py   <- Model training
    │   └── predict.py <- Model inference
    └── plots.py       <- Visualizations
```
