Project brief
=============

What we are building, with which data, and how we plan to get there.
Keep this page up to date: it is the shared starting point for teammates and AI agents.

Status markers:

- **[decided]** agreed by the team.
- **[planned]** current plan, may change.
- **[open]** not decided yet; decisions that could reasonably go differently go through the EDN process in [AGENTS.md](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/AGENTS.md) before they are settled.

## 1. Project goal

We are team **Recommenditos** in *Machine Learning Systems in Production (MLOps)* at FIB-UPC, 2026/27.
We build and deploy a **used-car price component**: an ML model behind an API that

1. estimates the market price of a car the user describes, and
2. returns the price range a user should expect to pay for a car they want to buy.

The course grades how well we apply MLOps practices (reproducibility, QA, deployment, CI/CD, monitoring), not model accuracy.
"Excellent" requires extended or innovative use of each practice, so we prefer clean, well-justified engineering over model complexity.

## 2. Use cases **[planned]**

| ID | Use case | Input | Output |
|----|----------|-------|--------|
| UC1 | **Car valuation** | Full description of an owned car (make, model, version, registration date, mileage, power, fuel, transmission, equipment, ...) | Point estimate of the listing price (EUR) + explanation |
| UC2 | **Purchase guidance** | Partial description of a desired car (any subset of the fields) | Price range (e.g. 90 % interval) + comparable listings |

Out of scope for the core: free-text parsing via an external LLM (optional add-on, must never be required for the API to work).

Scope **[decided]**: used passenger cars of the makes with enough listings (currently 11, mostly premium brands) in 8 European countries.
The exact scope, target, features and success criteria live in the [problem specification](problem-spec.md).

## 3. Data

### 3.1 Main dataset **[decided]**

**AutoScout24 Car Listings Dataset (2025 snapshot)** by Muhammed Çelik.

- Zenodo (DVC source): <https://zenodo.org/records/17643343>, file `autoscout24_dataset_20251108.csv`, 548.6 MB, DOI `10.5281/zenodo.17643343`, version 1.0.0 (2025-11-18).
- Kaggle mirror: <https://www.kaggle.com/datasets/clkmuhammed/autoscout24-car-listings-dataset> (not verified).
- License: MIT. The author asks for a citation when publishing analyses, so we cite it in the dataset card and the report.

Verified facts (profiled on 2026-09-22):

- 118,382 listings, 75 columns, currency EUR, single scrape, no listing date column.
- 8 countries: DE 45,611, IT 23,957, NL 17,059, BE 9,582, **ES 8,015**, AT 7,213, FR 6,141, LU 789.
- 25 makes, heavily skewed: BMW 37,745, Porsche 25,511, Mercedes-Benz 19,400, Audi 15,469 (together 83 %).
  Mass-market brands are nearly absent (VW 352, Renault 60, Opel 60) or missing (Toyota, SEAT, Peugeot, Fiat, Skoda).
- Not only used cars: 4,252 new (`offer_type = N`), 3,702 pre-registered, 32,199 registered in 2025.
- 456 rows are `vehicle_type = Transporter`.
- Categorical labels are English, but `description` and `model_version` are free text in local languages (84,171 unique versions).
- Price ranges from 1 to 13.5M EUR (median 39,980); mileage from 0 to 2.57M km.

### 3.2 New-market drift scenario **[decided]**

Decision and reasoning: [EDN-03](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md).

We hold out all AutoScout24 `ES` listings (8,015, 6.8 % of the rows) as a "new market" the model has never seen.
Schema, scrape date and source portal stay the same, so any drift the monitoring detects on these listings comes from the market itself.

- The split stage of the DVC pipeline removes `ES` after deduplication and before the train/validation/calibration split, and writes it as its own versioned artefact.
  No `ES` row reaches training, validation or conformal calibration.
- In M6 we replay the `ES` listings against the API as simulated traffic.
  Alibi Detect should flag the input drift, and because every replayed listing has a price, we can also show the real MAE and interval coverage getting worse in Grafana.
  The same prices can serve as the delayed labels for `/feedback` (see 5).
- After the drift is confirmed, we retrain with `ES` included, which closes the monitoring feedback loop.
- **[planned]** `country` stays a feature, and the API accepts a country that is missing from training by treating it as unknown.
  An API test covers this case.
- **[planned, optional]** A synthetic drift scenario (e.g. shifted mileage or age) where we control exactly what changes, to show the detector reacts to a known cause.

**DataMarket, Spanish second-hand cars (free sample)**, <https://github.com/Data-Market/vehiculos-de-segunda-mano>, is not part of the pipeline.
It was the original drift set, and the supervisor (S. del Rey) asked us to check whether it is feasible given the feature mismatch.
It is not, for three reasons:

- Spain is not a new market: AutoScout24 already has 8,015 `ES` listings.
- 36.5 % of the DataMarket rows are makes the model never sees (Peugeot, Citroën, SEAT, ...), and the overlapping mass-market makes have under 400 training rows each.
- Market, time (2021 vs 2025), source portal and brand mix all shift at once, so a drift alarm cannot be attributed to any one of them.
  Example: BMW median price is 16,000 EUR in DataMarket vs 24,819 EUR in AutoScout24 `ES`.

Using it would also need a dedicated schema-mapping stage: 50,000 listings from one extraction (2021-01-15), Spanish labels, 21 columns of which only about 10 overlap, no equipment or description fields, `price_financed` 52.9 % and `power` 17.1 % missing, mileage up to 5M km, and free-text colour (3,565 variants).
It has no license file and is a sample of a commercial dataset, so we could not re-host it in a public DVC remote either.
The report describes this check as part of the data decisions.

### 3.3 Known data issues (handle in code, document in the dataset card)

- **Listing price is not transaction price.** We predict asking prices.
- **PII:** `vin`, `street`, `seller_company_name`, `zip`, exact coordinates, and probably contact details inside `description`.
  Drop or coarsen them during preprocessing.
  The raw file itself contains this PII, so **[open]**: pull it from Zenodo with `dvc import-url` instead of pushing a copy to our DagsHub remote (see [Data versioning](data-versioning.md)).
- **Leakage, never use as features:** `price_net` (derived from `price` and VAT), `price_vat_rate`; identifiers `id` and `vin` are not features either.
  `price_tax_deductible` is not known to a private user, so we exclude it; seller `ratings_*` only with justification.
- **Price in the description:** 36 % of descriptions contain a currency amount and about 7 % contain the exact listing price.
  Strip currency and number patterns before any text feature.
- **Duplicates:** `vin` is only 34 % filled, so VIN-based deduplication is not enough.
  A key on make, model, version, mileage, registration date, price and power finds 6,347 duplicate rows.
  Deduplicate **before** splitting.
- **Splits:** hold out `ES` first (see 3.2), then a grouped split (e.g. by dealer), not purely random.
  The dealer key comes from `seller_company_name`, which is PII, so hash it into a group id **before** the PII-removal stage.
  No listing date exists, so a temporal split is not possible.
- **Useless or empty fields:** `warranty` and `has_warranty` are 100 % empty; `had_accident` is True for 3 rows; `fuel_cons_city_l100_km` and `fuel_cons_highway_l100_km` are empty.
  The boolean flags likely encode "unknown" as False; there is no condition field.
- **Partly filled fields:** `nr_prev_owners` 55 %, `vin` 34 %, `price_net` 29 %, `production_year` 19 %, `electric_range_km` 11 %.
- **Outliers:** prices down to 1 EUR and up to 13.5M EUR; mileage up to 2.57M km.
  Great Expectations checks must cover these ranges.

## 4. Modelling plan **[planned]**

Target, features, metrics, baselines and success criteria (`SC-01` to `SC-05`) are defined in the [problem specification](problem-spec.md).

Experiment ladder (each step is one or more MLflow runs in the same experiment):

1. Median price per make/model/age bucket (sanity baseline).
2. Ridge regression on log-price (interpretable depreciation baseline).
3. **LightGBM** with basic features (main model).
4. LightGBM with all features (equipment, history flags) to quantify their value.
5. **CatBoost** as challenger (high-cardinality categoricals).
6. Optional: description-text features.
   The descriptions are multilingual, so embeddings would need a heavy multilingual model, which conflicts with the small CPU-only image.

Steps 1-3 (maybe 4) are the target for the first report; the rest is **[open]**.

Price ranges (UC2): **Conformalized Quantile Regression** with MAPIE (1.x API) on top of quantile LightGBM models (e.g. 5 % / 95 %).
Train with random masking of optional fields so the model produces wider intervals for partial inputs.
The calibration set must use the same masking, and the coverage guarantee is marginal (on average over all inputs), not per missing-field pattern.

Comparable listings: k-nearest-neighbour search over the processed listings, returned next to the prediction.

Explainability: SHAP (TreeExplainer) per prediction and globally.

Why gradient boosting: best-in-class on medium tabular data, native categoricals and missing values, trains in minutes on CPU, small artefacts, exact SHAP.
Alternatives considered: linear, kNN, random forest, tabular NNs, hierarchical Bayes, LLM zero-shot.

## 5. Target architecture **[planned]**

```
client --> FastAPI (model + SHAP + intervals)
             |-- /predict       (UC1)
             |-- /price-range   (UC2)
             |-- /comparables
             |-- /feedback      (reported sale prices; no real users, so simulated)
             '-- /health, /metrics (Prometheus)

         Storage for listings (processed, no PII), prediction log, feedback
         MLflow tracking + model registry
         Prometheus + Grafana (resources, latency, errors)
         Alibi Detect job (input drift on logged requests, interval coverage)
```

Everything runs via Docker Compose.
The API contract (Pydantic schemas) is the boundary: models can be swapped without changing clients.

Open points:

- Deployment target: FIB Virtech VM (free, 4 GB RAM, 20 GB disk, one semester) or another cloud.
  The full stack is tight on 4 GB.
- Storage: PostgreSQL only pays off if monitoring reads the prediction log; a lighter store may be enough.
- MLflow hosting: DagsHub (as in the course demo) or self-hosted.
- Feedback loop: simulate delayed labels from the held-out `ES` listings (see 3.2), or drop `/feedback`.

## 6. Tooling constraints

Checked against our `uv.lock` (numpy 2.4.6, pandas 3.0.6, typer 0.26.8, ipython 9.17.1) on 2026-09-22:

- **Alibi Detect 0.13** requires numpy<2 and pandas<3 and pulls in transformers, opencv and scikit-image.
  It must run in its own container, never in the API image.
  Its license is Business Source License 1.1 (free for non-production use); mention this in the report.
- **Pynblint 0.1.6** (last release August 2024) pins typer<0.13 and ipython<9.
  Run it isolated with `uvx pynblint`, not as a project dependency.
- **SHAP 0.52** requires Python 3.12+; we pin 3.11, so uv resolves an older SHAP unless we bump Python.
- **Static analysis:** we use ruff; enabling its Pylint rules (`PL`) covers the rubric's "Pylint or flake8".
- Keep the Docker image small and CPU-only: no deep-learning or GPU libraries without team agreement.

## 7. Milestones

| Milestone | Topic | Tools | Points |
|-----------|-------|-------|--------|
| M1 | Inception: problem, requirements, model and dataset cards, coordination | Hugging Face cards, GitHub Projects + Discord | 10 |
| M2 | Reproducibility: structure, versioning, experiment tracking | Cookiecutter DS, Git + GitHub Flow, DVC, MLflow | 25 |
| M3 | Quality assurance: energy, static analysis, data and model tests (optional: SHAP, AIF360, TrustML) | CodeCarbon, ruff/Pylint, Pynblint, Pytest, Great Expectations | 15 |
| M4 | Deployment: system design, API, API tests | FastAPI, Pytest, FIB VM / cloud | 25 |
| M5 | Packaging: containers, CI/CD | Docker, Docker Compose, GitHub Actions | 15 |
| M6 | Monitoring: resources, model performance, drift | Prometheus, Grafana, Alibi Detect | 10 |

Deliveries (via Atenea, 23:55):

- **1st report (M1-M3): 2026-10-13**, max 15 pages; presentation on 2026-10-14.
- **2nd report (M4-M6): 2026-12-08**, max 30 pages; presentation on 2026-12-09.

## 8. Decisions and the EDN

Recorded in [reports/edn.md](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md):

- EDN-01: dataset choice (AutoScout24).
- EDN-02: model family (gradient boosting, LightGBM as main model).
- EDN-03: new-market drift scenario (hold out AutoScout24 `ES`, drop DataMarket).
- EDN-04: used cars only.
- EDN-05: minimum listing support per make.
- EDN-06: success criteria.

Made in M1, still to be written up:

- Project choice (car price vs. route safety, recommender and grocery ideas).

**[open]**, to decide with the team:

- Deduplication key and split strategy.
- Feedback loop: simulated labels or no `/feedback` endpoint.

## 9. Reference links

- Course organisation: <https://github.com/mlops-2627q1-mds-upc>
- Cookiecutter Data Science: <https://cookiecutter-data-science.drivendata.org/>
- Model cards: <https://huggingface.co/docs/hub/model-cards>, dataset cards: <https://huggingface.co/docs/hub/datasets-cards>
- DVC: <https://dvc.org/doc>, MLflow: <https://mlflow.org/docs/latest/>
- CodeCarbon: <https://mlco2.github.io/codecarbon/>, Pynblint: <https://github.com/collab-uniba/pynblint>
- Pytest: <https://docs.pytest.org/>, Great Expectations: <https://docs.greatexpectations.io/>
- LightGBM: <https://lightgbm.readthedocs.io/>, CatBoost: <https://catboost.ai/docs/>, MAPIE: <https://mapie.readthedocs.io/>
- SHAP: <https://shap.readthedocs.io/>, AIF360: <https://aif360.readthedocs.io/>
- FastAPI: <https://fastapi.tiangolo.com/>, Docker Compose: <https://docs.docker.com/compose/>, GitHub Actions: <https://docs.github.com/actions>
- Prometheus: <https://prometheus.io/docs/>, Grafana: <https://grafana.com/docs/>, Alibi Detect: <https://docs.seldon.io/projects/alibi-detect/>
