Project brief
=============

What we are building, with which data, and how we plan to get there.
Keep this page up to date: it is the shared starting point for teammates and AI agents.

Status markers:

- **[decided]** agreed by the team.
- **[proposed]** proposed and pending team confirmation.
  The alternatives are in [Decisions pending confirmation](specification.md#decisions-pending-confirmation), and a confirmed proposal becomes **[decided]**.
- **[open]** not decided yet; decisions that could reasonably go differently go through the EDN process in [AGENTS.md](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/AGENTS.md) before they are settled.

These three are the whole set, and this is the only place they are defined.
Every other page links here instead of repeating them, so the definitions cannot drift apart.

## 1. Project goal

We are team **Recommenditos** in *Machine Learning Systems in Production (MLOps)* at FIB-UPC, 2026/27.
We build and deploy a **used-car price component**: an ML model behind an API that

1. estimates the market price of a car the user describes, and
2. returns the price range a user should expect to pay for a car they want to buy.

The course grades how well we apply MLOps practices (reproducibility, QA, deployment, CI/CD, monitoring), not model accuracy.
"Excellent" requires extended or innovative use of each practice, so we prefer clean, well-justified engineering over model complexity.

## 2. Use cases

| ID | Use case | Input | Output |
|----|----------|-------|--------|
| UC1 | **Car valuation** | Full description of an owned car (make, model, version, registration date, mileage, power, fuel, transmission, equipment, ...) | Point estimate of the listing price (EUR) + explanation |
| UC2 | **Purchase guidance** | Partial description of a desired car (any subset of the fields) | Price range (e.g. 90 % interval) + comparable listings |

Out of scope for the core: free-text parsing via an external LLM (optional add-on, must never be required for the API to work).

Scope **[decided]**: used passenger cars of the makes with enough listings (currently 11, mostly premium brands) in 8 European countries.
The exact scope, target, features and success criteria live in the [problem specification](problem-spec.md).
What the component must do and which qualities it must have is defined in the [requirements](requirements.md) (`FR-xx`, `NFR-xx`).
How each of them is realised (endpoints, input validation, latency, resources, privacy) is defined in the [specification](specification.md), under the same IDs.

## 3. Data

### 3.1 Main dataset **[decided]**

**AutoScout24 Car Listings Dataset (2025 snapshot)** by Muhammed Çelik.

- Zenodo (DVC source): <https://zenodo.org/records/17643343>, file `autoscout24_dataset_20251108.csv`, 548.6 MB, DOI `10.5281/zenodo.17643343`, version 1.0.0 (2025-11-18).
- Kaggle mirror: <https://www.kaggle.com/datasets/clkmuhammed/autoscout24-car-listings-dataset> (not verified).
- License: the upstream record contradicts itself. The Zenodo metadata field says MIT, while the author's own description on the same record says "You are welcome to use this dataset for research, educational, or analytical purposes", which reads as narrower than MIT.
  Both readings permit this project, so we document both in the dataset card instead of asking the author to resolve it.
  The author asks for a citation when publishing analyses, so we cite it in the dataset card and the report.

Verified facts (profiled on 2026-09-22, re-verified on 2026-09-29 against the Zenodo file with md5 `b23a122cc51baf7de39f449193ff0d28`):

- 118,382 listings, 75 columns, currency EUR, single scrape, no listing date column.
- 8 countries: DE 45,611, IT 23,957, NL 17,059, BE 9,582, **ES 8,015**, AT 7,213, FR 6,141, LU 789.
  15 rows have no `country_code` at all, and are also missing `seller_type`, `city`, `zip` and `street`.
- 25 makes, heavily skewed: BMW 37,745, Porsche 25,511, Mercedes-Benz 19,400, Audi 15,469 (together 83 %).
  Mass-market brands are nearly absent (VW 352, Renault 60, Opel 60) or missing (Toyota, SEAT, Peugeot, Fiat, Skoda).
- Not only used cars: 4,252 new (`offer_type = N`), 3,702 pre-registered, 32,199 registered in 2025.
  `offer_type` has a third value `A` in 3 rows.
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
  The same prices are the delayed labels posted to `/feedback` (see 5), **[decided]**, [EDN-13](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md).
- After the drift is confirmed, we retrain with `ES` included, which closes the monitoring feedback loop.
  **[decided]**, [EDN-12](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md): retraining and promotion are human-triggered, not automated; see [requirements](requirements.md) FR-15.
- **[decided]**, [EDN-18](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md): `country` stays a feature, and the API accepts a country that is missing from training by treating it as unknown, with a warning in the response ([requirements](requirements.md) FR-05).
  An API test covers this case.
- **Optional, if time allows:** a synthetic drift scenario (e.g. shifted mileage or age) where we control exactly what changes, to show the detector reacts to a known cause.
  Nobody is waiting on this, so it carries no status marker.
- Measured on 2026-09-23 while checking [requirements](requirements.md) NFR-11 ([EDN-14](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md), scripts in `reports/analysis/`): the `ES` replay is flagged within 100 requests in 200 of 200 trials, also when `country_code` is excluded. NFR-11 nevertheless states a window of 1,000, because the control clause and the "at least three changed properties" clause do not hold at 100 ([EDN-26](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
  Spain differs above all in listing completeness (`nr_prev_owners` missing in 95.5 % of `ES` rows vs 40.0 % in training, `nr_seats` 10.9 % vs 3.0 %) and in the `body_type`, `transmission` and `fuel_category` mix.
  The same run found that **held-out dealers shift as much as a new country**: on `make`, `model`, `gears`, `mileage_km_raw` and `age_years` a seller-grouped holdout differs from the training distribution as much as `ES` does, or more.
  That is the strongest argument for the seller-grouped split (see 3.3), and it is a risk for SC-04: per-segment error may partly reflect which dealers landed in which split.

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
  The raw file itself contains this PII. **[decided]** The `download` stage fetches it from the pinned Zenodo DOI and keeps it as a gitignored local cache under `data/external/`, so we do not track or re-publish the CSV (EDN-35, EDN-36, [Data versioning](data-versioning.md)). **[decided]** The Parquet the stage derives from it keeps the same PII columns, because preprocessing needs them to build the split's group key, and it is pushed to our DagsHub remote, which is public (EDN-20), so we accept that those columns are re-published in that format; the identical data is already public on Zenodo (EDN-34). The PII is removed in preprocessing rather than by withholding the file; see [Specification](specification.md) NFR-08.
- **Leakage, never use as features:** `price_net` (derived from `price` and VAT), `price_vat_rate`; identifiers `id` and `vin` are not features either.
  `price_tax_deductible` is not known to a private user, so we exclude it; seller `ratings_*` only with justification.
- **Price in the description:** about 7 % of all rows have the exact listing price in `description` (6.8 % matched as a whole number; a plain substring match gives 6.9 %, because it also counts a price that is only part of a longer number, see the [profiling notebook](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/notebooks/1.0-lh-dataset-card-profiling.ipynb)).
  How many contain any currency amount depends entirely on the pattern used (33.7 % for a currency token directly before or after a digit, as the profiling notebook matches it, 39 to 44 % for separator-formatted numbers depending on how the number is bounded), so we do not quote a single figure for it.
  Strip currency and number patterns before any text feature.
- **Duplicates:** `vin` is only 34 % filled, so VIN-based deduplication is not enough.
  A key on make, model, version, mileage, registration date, price and power finds 6,347 duplicate rows.
  Deduplicate **before** splitting.
- **Splits:** hold out `ES` first (see 3.2), then a grouped split (e.g. by dealer), not purely random.
  The dealer key comes from `seller_company_name`, which is PII, so hash it into a group id **before** the PII-removal stage.
  No listing date exists, so a temporal split is not possible.
- **Useless or empty fields:** `warranty` and `has_warranty` are 100 % empty; `had_accident` is True for 3 rows; `fuel_cons_city_l100_km` and `fuel_cons_highway_l100_km` are empty.
  There is no condition field.
- **Condition flags are one-sided:** 14,744 rows in the raw data are flagged as neither used, new nor pre-registered, and 18,108 of the 113,708 rows scoped by `offer_type` and `vehicle_type` have `is_used = False` while `offer_type = U`.
  A `True` in these flags is an assertion by the seller; a `False` only means the assertion is absent, so it must never be read as a "no" (EDN-23).
  This also limits the scope filter of EDN-04: pre-registered listings that carry no flag cannot be removed, which is accepted and documented rather than worked around (EDN-24).
- **Registration dates after the snapshot:** 164 listings are registered after 2025-11-08, the latest on 2026-11-01 and 137 of them in January 2026.
  Age computed as snapshot date minus registration date is negative for these rows, so preprocessing drops them, and the Great Expectations suites bound the date at the reference date: hard on the cleaned frame, with `mostly=0.99` on the raw one (EDN-22, [Data validation](pipeline.md#data-validation)).
- **Partly filled fields:** `nr_prev_owners` 55 %, `vin` 34 %, `price_net` 29 %, `production_year` 19 %, `electric_range_km` 11 %.
- **Outliers:** prices down to 1 EUR and up to 13.5M EUR; mileage up to 2.57M km.
  The price range is a scope filter in preprocessing and a hard expectation on the cleaned frame.
  The mileage range of FR-03 is a check on both frames that tolerates a share of 0.1 %, because three listings above 1,000,000 km survive preprocessing (EDN-72).

## 4. Modelling plan

Target, features, metrics, baselines and success criteria (`SC-01` to `SC-06`) are defined in the [problem specification](problem-spec.md).

Experiment ladder (each step is one or more MLflow runs in the same experiment):

1. Median price per make/model/age bucket (sanity baseline).
2. Ridge regression on log-price (interpretable depreciation baseline).
3. **LightGBM** with basic features (main model).
4. LightGBM with all features (equipment, history flags) to quantify their value.
5. **CatBoost** as challenger (high-cardinality categoricals).
6. Optional: description-text features.
   The descriptions are multilingual, so embeddings would need a heavy multilingual model, which conflicts with the small CPU-only image.

Steps 1-3 (maybe 4) are the target for the first report; the rest is **[open]**.

**Steps 1 to 4 are implemented** as the four variants of `train.variants` in `params.yaml`, one `train@<variant>` stage each, and each is one MLflow run in one experiment.
Measured on the real snapshot on 2026-10-05 (test-split MdAPE, fitted on the 11 supported makes and encoded in a vocabulary they alone decided, EDN-67, with early stopping deciding the tree count, EDN-70): step 1 12.16 %, step 2 9.52 %, step 3 6.81 %, step 4 6.26 %.
So the extended feature set is worth 0.55 pp on a fully described car - and 2.04 pp *worse* than the basic set on a request carrying only the ten fields FR-01 requires (10.30 % against 8.26 %), which is why **step 3 is the candidate** and step 4 is not ([EDN-62](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
The whole ladder fits in 36 seconds.
Numbers, hyperparameters and the caveats are in the [model card](model-card.md#the-experiment-ladder-as-measured): the tree count is decided by early stopping rather than by a binding budget and turned out to be worth 0.02 pp (EDN-70), and `learning_rate` and `num_leaves` are deliberately not tuned, because a sweep around them for step 3 found no point the validation split can tell apart (EDN-73). Later tuning follows the protocol in the [pipeline documentation](pipeline.md#tuning-a-hyperparameter).
Step 5, CatBoost, adds an `estimator` to the same mapping and a branch in `recommenditos/modeling/model.py`; nothing else has to change.

Price ranges (UC2): **Conformalized Quantile Regression** with MAPIE (1.x API) on top of quantile LightGBM models (e.g. 5 % / 95 %).
Train with random masking of optional fields so the model produces wider intervals for partial inputs.
The calibration set must use the same masking, and the coverage guarantee is marginal (on average over all inputs), not per missing-field pattern.

Whether the point model (UC1, steps 1-4 above) needs the same random masking as the interval models depends on how often each optional field is missing in training, measured on 2026-09-29 (**[decided]**, [EDN-15](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md), script in `reports/analysis/`):

- `body_type` is filled in **100.000 %** of the training listings and `seller_type` in **99.986 %** (14 of 97,889 rows), so a model trained on them effectively never learns a direction for their absence.
  Both are now required fields of `/predict` ([specification](specification.md) FR-01), which costs the user nothing: whoever owns the car knows the body type, and for UC1 the user is the seller.
- `nr_doors` (98.8 %), `nr_seats` (97.0 %) and `cylinders_volume_cc` (91.1 %) are missing rarely, so the signal for their absence is thin but real.
- `nr_prev_owners` (61.0 %), `gears` (62.5 %) and `drive_train` (76.4 %) carry enough natural missingness that native handling learns it.
  Only 30.7 % of training rows have every one of the six optional fields present, and 469 rows (0.5 %) have none of them, so the extreme case is in the training data too.

**[open]:** whether the three rarely-missing fields need the same random masking as the UC2 interval models, or whether native missing-value handling is enough.
This is a modelling decision, not an API one, and it is taken in M2/M3 once the pipeline exists and the effect can be measured.
Either way the outcome is bounded: SC-06 of the [problem specification](problem-spec.md#8-success-criteria) caps the MdAPE loss when an optional field is absent, and NFR-01's gate enforces it before a model is deployed.

Comparable listings: a filtered lookup over the processed listings (same make and model, close in age and mileage), not a learned nearest-neighbour model.
The exact window is defined in the [specification](specification.md) FR-09.

Explainability: SHAP (TreeExplainer) per prediction and globally.

Why gradient boosting: best-in-class on medium tabular data, native categoricals and missing values, trains in minutes on CPU, small artefacts, exact SHAP.
Alternatives considered: linear, kNN, random forest, tabular NNs, hierarchical Bayes, LLM zero-shot.

## 5. Target architecture

```
client --> reverse proxy --> FastAPI (model + SHAP + intervals)
                               |-- /predict       (UC1)
                               |-- /price-range   (UC2)
                               |-- /comparables
                               |-- /health, /metrics (Prometheus)
                               '-- /feedback      (internal only: the proxy refuses it
                                                   from outside; reported prices, no real
                                                   users, so replayed from the ES holdout)

         Storage for listings (processed, no PII), prediction log, feedback
         MLflow experiment tracking (DagsHub), not called by the API
         Prometheus + Grafana (resources, latency, errors)
         Better Uptime (external availability check, presentation windows)
         Alibi Detect job (input drift on logged requests, interval coverage)
```

Everything runs via Docker Compose.
The API contract (Pydantic schemas) is the boundary: models can be swapped without changing clients.
Endpoints, inputs and outputs are specified in the [specification](specification.md) (`FR-xx`, `NFR-xx`).
Model loading **[decided]**, [EDN-08](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md): the model is a DVC-tracked pipeline artifact baked into the API image at CI build time (`dvc pull`, pinned to the version `main` points to), not fetched from the MLflow registry at build or run time.
Promoting a model is a normal merge to `main`, matching GitHub Flow; MLflow stays the experiment-tracking and audit record of which run was chosen (see [specification](specification.md) FR-12).
Deployment target **[decided]**, [EDN-17](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md): the FIB Virtech VM the course provides (4 GB RAM, 20 GB disk, one semester), which is what every target in the [specification](specification.md) is written for.
It is tight on 4 GB, and that is planned for rather than left open: NFR-04 gives the drift job its own scheduled container, caps Prometheus by retention size and the logs by rotation, and keeps 4 GB of disk free.
Nobody has access yet; if that is still true at the M4a lab on 2026-10-21, we raise it with the teachers instead of planning on further.

Open points:

- Storage: PostgreSQL only pays off if monitoring reads the prediction log; a lighter store may be enough.
  Monitoring does read it: FR-14 joins the prediction log and the feedback labels, so the store has to support that join.
- MLflow hosting: DagsHub (as in the course demo) or self-hosted.

## 6. Tooling constraints

Checked against our `uv.lock` (numpy 2.4.6, pandas 3.0.6, typer 0.26.8, ipython 9.17.1) on 2026-09-22:

- **Alibi Detect 0.13** requires numpy<2 and pandas<3 and pulls in transformers, opencv and scikit-image.
  It must run in its own container, never in the API image.
  Its license is Business Source License 1.1 (free for non-production use); mention this in the report.
- **Pynblint 0.1.6** (last release August 2024) pins typer<0.13 and ipython<9, so it is never a project dependency.
  It pins typer but not click, and typer 0.12 crashes on click 8.2 and newer (released 2025-05-10) on every Python version, so a bare `uvx pynblint` does not run at all.
  **[decided]**, [EDN-64](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md): it runs from its own locked environment in `tools/pynblint-env/` (`pynblint==0.1.6`, `click<8.2`, a hash-verified `uv.lock` of its own) via `make notebook-lint`.
  It exits 0 whatever it finds, so `tools/notebook_lint.py` reads its JSON report and is the gate; the rules it enforces are the table in [Notebooks](notebooks.md#pynblint-rules) (EDN-65).
- **SHAP 0.52** requires Python 3.12+. **[decided]**, [EDN-09](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md): we bumped `requires-python` to `~=3.12.0` for this, checked against the full planned M2-M5 dependency set (`shap`, `mlflow`, `lightgbm`, `catboost`, `mapie`, `fastapi`, `great-expectations`, `dvc`, `pytest-cov`, `codecarbon`), which all resolve under 3.12 with no upper-bound conflicts.
  **[decided]**, [EDN-11](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md): `shap` itself is a training/notebook dependency only (global analysis, summary plots), never installed in the API image.
  It unconditionally pulls in `numba` and `llvmlite` (measured 189 MB) just to import the module, which a real serving image does not need: the API computes per-request explanations from the trained booster's own SHAP export instead (see [specification](specification.md) FR-08), verified bit-identical to `shap.TreeExplainer`.
- **Great Expectations 1.23.2** works on our pandas-3 frames for every expectation the suites use, each checked against a frame that satisfies it and one that breaks it ([EDN-68](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md), `reports/analysis/gx_probe.py`).
  Two gaps: it compares only a column's scalar type, so it cannot tell a nullable, extension or `object` dtype from its counterpart (`datetime64[us]` passes `datetime64[ns]`, `boolean` passes `bool`, `Float64` passes `float64`, `object` and `string[python]` pass `str`), and `schema.py` alone checks the declared dtype; and it cannot bound a text date, so the raw suite sees `registration_date` parsed.
  It is a pipeline dependency, not part of the API image.
- **Static analysis:** we use ruff; enabling its Pylint rules (`PL`) covers the rubric's "Pylint or flake8", and ruff lints the code cells of the notebooks too, while Pynblint checks their structure.
- Keep the Docker image small and CPU-only: no deep-learning or GPU libraries without team agreement.
- **Dependency groups** **[decided]**, [EDN-47](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md): `[project] dependencies` in `pyproject.toml` is the serving runtime only, and the API image installs that alone (`uv sync --no-default-groups`).
  Everything else is in PEP 735 groups (`pipeline`, `notebook`, `docs`, `test`, `dev`), and `default-groups = "all"` keeps a plain `uv sync` installing the whole environment.
  Measured from `uv.lock` on 2026-10-05: the runtime is 23 distributions, the project itself included, and 421 MB of site-packages (519 MB with bytecode), against 248 distributions and 815 MB (1,054 MB) for the full environment; pyarrow alone is 157 MB of it.
  FastAPI, uvicorn and pydantic join the runtime in M4; Great Expectations, CodeCarbon and `shap` go into groups.
  CI's `Serving runtime` job (`make test-serving`), a required check, runs the serving path from the runtime set and the test runner alone, so a training-only import in serving code fails the build.

## 7. Milestones

| Milestone | Topic | Tools | Points |
|-----------|-------|-------|--------|
| M1 | Inception: problem, requirements, model and dataset cards, coordination | Hugging Face cards, GitHub Projects + Discord | 10 |
| M2 | Reproducibility: structure, versioning, experiment tracking | Cookiecutter DS, Git + GitHub Flow, DVC, MLflow | 25 |
| M3 | Quality assurance: energy, static analysis, data and model tests (optional: SHAP, AIF360, TrustML) | CodeCarbon, ruff/Pylint, Pynblint, Pytest, Great Expectations | 15 |
| M4 | Deployment: system design, API, API tests | FastAPI, Pytest, FIB VM / cloud | 25 |
| M5 | Packaging: containers, CI/CD | Docker, Docker Compose, GitHub Actions | 15 |
| M6 | Monitoring: resources, model performance, drift | Prometheus, Grafana, Better Uptime, Alibi Detect | 10 |

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
- EDN-07: raw data hosting (import from Zenodo, never push to our own remote), amended by EDN-25.
- EDN-08: model loading (bake into the API image via `dvc pull` at CI build time, not the MLflow registry at runtime).
- EDN-09: bump to Python 3.12, to use real SHAP instead of a workaround.
- EDN-10: availability target replaced by recovery time plus a presentation-window commitment.
- EDN-11: `shap` kept out of the API image; serving uses the booster's native SHAP export instead.
- EDN-12: retraining and promotion are human-triggered, not automated.
- EDN-13: keep `/feedback`, but reachable only from inside the Compose network, so no TLS is needed.
- EDN-14: NFR-11's drift control is an i.i.d. sample of held-out listings, not a seller-grouped one.
- EDN-15: UC1 required fields, after measuring the fill rates, plus SC-06 for absent optional fields.
- EDN-16: separate `/predict` and `/price-range`, one endpoint per use case.
- EDN-17: deployment target is the FIB Virtech VM.
- EDN-18: unseen countries and models are accepted with a warning.
- EDN-19: the DagsHub repository the team uses as DVC remote and MLflow server.
- EDN-20: the DagsHub remote stays public.
- EDN-21: record both licence readings of the Zenodo record.
- EDN-22: drop listings registered after the reference date.
- EDN-23: read the condition flags as one-sided assertions.
- EDN-24: keep the pre-registered exclusion despite the unreliable flag.
- EDN-25: raw data acquisition (`dvc add` and push to our DagsHub remote), amended by EDN-36.
- EDN-26: NFR-11's window is 1,000 requests, and `model` is excluded from the drift comparison.
- EDN-27: the requirements and their specification are separate pages, sharing one set of `FR-xx`/`NFR-xx` IDs.
- EDN-28: one module per DVC stage, deviating from the flat Cookiecutter layout.
- EDN-29: the drift job runs at a significance level of 0.005, not 0.05.
- EDN-30: pipeline configuration lives in `params.yaml`, not in `config.py`.
- EDN-31: the processed-data contract is a hand-written `schema.py`, not Pandera.
- EDN-32: split proportions 60/10/10/20 and the project seed.
- EDN-33: a generated synthetic fixture, and a `download.source` parameter so the skeleton runs without the raw file.
- EDN-34: the derived raw Parquet keeps its PII columns and is pushed like any other stage output.
- EDN-35: the `download` stage owns `data/raw/`, and the published CSV is a local cache outside the DAG.
- EDN-36: the `dvc add` pointer of EDN-25 is retired, so we no longer host a copy of the raw CSV.

Made in M1, still to be written up:

- Project choice (car price vs. route safety, recommender and grocery ideas).

**[open]**, to decide with the team:

- Deduplication key and split strategy.
- Whether the point model needs random masking for the optional fields that are rarely missing in training, or whether native missing-value handling is enough (section 4); the field list itself is settled in EDN-15.

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
- Prometheus: <https://prometheus.io/docs/>, Grafana: <https://grafana.com/docs/>, Better Uptime: <https://betterstack.com/docs/uptime/start.html>, Alibi Detect: <https://docs.seldon.io/projects/alibi-detect/>
