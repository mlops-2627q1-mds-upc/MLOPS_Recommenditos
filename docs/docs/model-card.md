---
pretty_name: "Recommenditos used-car price model"
license: mit
tags:
- tabular
- automotive
- regression
- used-cars
- price-prediction
- lightgbm
- conformal-prediction
pipeline_tag: tabular-regression
metrics:
- mdape
- mae
- coverage
---

# Model Card for the Recommenditos Used-Car Price Model

Recommenditos estimates what a used passenger car would be **asked for** on a European online car portal.
Given a description of a car it returns a point estimate of the listing price (UC1), and for a partial description a price interval with comparable listings (UC2).
It is the model behind the project's API.

**Status: draft, no trained artefact yet.**
This is the initial model card, written in Milestone 1 before a model exists, so everything about training and results is a plan rather than a measurement.
It follows the [Hugging Face annotated model card template](https://huggingface.co/docs/hub/model-card-annotated) and is a living document: the placeholder results are replaced with measured values after Milestone 2 (first trained model) and Milestone 3 (model tests asserting each `SC-xx`).
Sections that describe an intention rather than a fact carry the status markers **[decided]**, **[proposed]** and **[open]**, defined in the [project brief](project-brief.md).

This card owns the trained model.
It does not repeat what other pages own, it links to them:

| Question | Page |
|---|---|
| What the model learns, on which data, and when it is good enough | [Problem specification](problem-spec.md) |
| What the component must do and which qualities it must have | [Requirements](requirements.md) |
| How each requirement is realised, in build-and-test detail | [Specification](specification.md) |
| Facts about the training data, its skew and its limitations | [Dataset card](dataset-card.md) |
| Goal, plan and open decisions | [Project brief](project-brief.md) |

## Model Details

### Model Description

Recommenditos is a supervised regression model over tabular car descriptions.
It is trained on `log(price)` and its predictions are transformed back with `exp`, which estimates the median asking price for a given car.
The same trained booster serves the point estimate, its explanation and, through separately calibrated quantile models, the price interval.

- **Developed by:** Team Recommenditos, *Machine Learning Systems in Production (MLOps)*, FIB-UPC, 2026/27.
  See [Model Card Authors](#model-card-authors).
- **Funded by:** nobody; this is unfunded coursework.
- **Model type:** supervised regression on `log(price)`, gradient-boosted decision trees.
- **Language(s):** not a language model.
  Categorical inputs use the English labels of the source data; the multilingual free-text fields are not used by the model (see [Out-of-Scope Use](#out-of-scope-use)).
- **License:** MIT, the licence of this repository.
  The training data carries its own terms, see the [dataset card](dataset-card.md#licensing).
- **Version:** none yet.
  From the first release on, the served version is the one `dvc.lock` on `main` points to, reported by `GET /health` and verified in CI ([specification](specification.md) FR-12, EDN-08).
- **Status:** planned, not trained.

### Model Sources

- **Repository:** <https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos>
- **Training data:** [dataset card](dataset-card.md)
- **Experiment tracking:** MLflow on DagsHub, from Milestone 2 (**[proposed]**, EDN-19, pending confirmation by the full team).
- **Model artefact:** DVC-tracked pipeline output, baked into the API image at CI build time (FR-12, EDN-08).

### Model family **[decided, EDN-02]**

- **Main model:** **LightGBM** gradient boosting on `log(price)`, with the basic and then the extended feature set ([experiment ladder](project-brief.md#4-modelling-plan) steps 3 and 4).
- **Challenger:** **CatBoost** (step 5), for high-cardinality categoricals.
- **Baselines:** B0, the median price per make, model and 2-year age bucket; B1, Ridge regression on `log(price)` ([problem specification](problem-spec.md#7-baselines)).

Gradient boosting was chosen because it is best-in-class on medium tabular data, handles categoricals and missing values natively, trains in minutes on CPU, produces small artefacts and supports exact SHAP explanations.
The alternatives considered, linear models, kNN, random forest, tabular neural networks, hierarchical Bayes and LLM zero-shot, and why each lost, are recorded in EDN-02.

## Uses

### Direct Use

- **UC1, car valuation:** a private seller describes a car they own and receives a point estimate of its listing price in EUR, with an explanation of the properties that drove it ([specification](specification.md) FR-06, FR-08).
- **UC2, purchase guidance:** a prospective buyer describes a car they are looking for, in as much or as little detail as they have, and receives a nominal 90 % price interval and a list of comparable listings (FR-07, FR-09).

The model reflects the market of one data snapshot, taken on 2025-11-08.
It estimates the **asking** price, not the transaction price, and it does not forecast future prices.

### Who is affected

The [requirements](requirements.md#who-the-component-is-for) name three roles, and a wrong answer does not cost them the same.

- **The seller** acts on a single number about a single car.
  An estimate that is 10 % low costs them real money once, and they have no way to notice it.
  This is the party the model can hurt most, which is why UC1 ships an explanation (FR-08) rather than a bare number.
- **The buyer** uses the interval as a sanity check against a price they were quoted.
  An interval that is too narrow makes a fair price look like a rip-off, and vice versa, which is what SC-05 bounds.
- **The team** operates the model and needs to be able to tell at any time whether it is still right, which is what the monitoring of FR-14 and NFR-11 is for.

### Scope of valid inputs

The scope is defined by the [problem specification](problem-spec.md#2-scope) and enforced by the API before anything reaches the model.

- **Used passenger cars only:** `offer_type = U`, not pre-registered, `vehicle_type = Car` (EDN-04).
- **Supported makes only:** makes with at least 300 listings in the cleaned used-car data, counted after removing the `ES` holdout and before the split, and computed by the pipeline rather than hard-coded (EDN-05).
  With the current data that is 11 makes covering 98.6 % of the used listings.
  A make outside that list is **refused, not guessed**, and the refusal names what is supported ([specification](specification.md) FR-04).
- **Markets:** trained on 7 countries (DE, IT, NL, BE, AT, FR, LU) with `country_code` as a feature.
  All `ES` listings are held out as the new-market drift scenario (EDN-03).
  The API accepts `ES` and passes it to the model as an unknown country, with a warning on the response, until the model is retrained with it (FR-05, EDN-18).
- **Required and optional inputs:** UC1 requires the ten facts a car's owner knows without research; UC2 requires only the make.
  Which fields those are, and why, is fixed in [specification](specification.md) FR-01 and FR-02 (EDN-15).
  Leaving an optional field out is allowed and bounded by SC-06.
- **Value ranges:** validated per field before prediction, so an implausible age, mileage or power is refused rather than extrapolated (FR-03).
  Training used prices between 500 EUR and 2M EUR.

The features themselves are listed in the [problem specification](problem-spec.md#4-features).

### Downstream Use

The model is not meant to be fine-tuned or embedded in another model.
It is consumed over the project's HTTP API, whose contract is the boundary, so the model can be replaced without changing anything a client sees ([specification](specification.md) FR-12).
Anyone reusing it outside that API has to reimplement the scope check (FR-04), the validation (FR-03) and the unseen-value warnings (FR-05), because the model itself does not refuse anything.

### Out-of-Scope Use

- Predicting **transaction or sale prices**.
  Only asking prices are observed.
- **Price forecasting** over time.
- New cars, pre-registered cars and transporters.
- Makes below the support threshold, and markets outside the 8 countries present in the data.
- Parsing free-text car descriptions.
  This stays an optional later experiment and must never become required for the API, because the descriptions are multilingual and about 7 % of them contain the listing price itself ([dataset card](dataset-card.md#known-data-issues)).
- **Any high-stakes automated decision**: financing, insurance, taxation, or a valuation of record.
  This is a guidance tool.

## Bias, Risks, and Limitations

- **Asking price is not transaction price.**
  Estimates are of the listed price and sit above what a car actually sells for, by an unknown and probably segment-dependent margin.
  Nothing in the data lets the model close that gap, and no evaluation here measures it.
- **Severe brand skew, and a gate that cannot see it.**
  BMW, Porsche, Mercedes-Benz and Audi are 82.9 % of the source data ([dataset card](dataset-card.md#bias-risks-and-limitations)).
  Mass-market brands are not simply out of scope: Volkswagen, Honda, Hyundai and Suzuki clear the 300-listing threshold and are served.
  But several supported makes sit so close to that threshold that SC-04 never checks them.
  A make needs about 2,500 in-scope listings to reach SC-04's 500 test rows at a 20 % test share, and **Honda (798), Hyundai (548), Aston Martin (360) and Volkswagen (348) fall short**, together 2.1 % of the training scope (listing counts measured 2026-09-29, `reports/analysis/make_support.py`).
  Four is a floor rather than a count: the split proportions are not pinned yet, and a four-way train, validation, calibration and test split leaves the test set below 20 %, at which point Volvo joins below a 15 % share and Suzuki below 13 %.
  Their rows still count towards the pooled criteria SC-01 to SC-03 and SC-05 to SC-06, but no criterion isolates them, so a failure specific to one of these makes cannot surface as a missed criterion.
  For these makes the per-segment half of the quality gate is blind by construction.
  Closing that gap, by raising the support threshold, lowering SC-04's segment size, or naming these makes as low-confidence in the response, is an open task for Milestone 3.
- **Classic and old cars.**
  In the reference run, cars older than 20 years are the only segment that breaches the per-segment target, at 15.3 % MdAPE against SC-04's 15 % limit.
  Closing this with the extended features or a scope change is an open task.
- **Reference-date drift.**
  Age is computed from the snapshot date in training and from the request date at serving, so served cars grow older than anything seen at the same registration date.
  This is expected, not accidental, and it is what the monitoring of Milestone 6 watches (FR-14).
- **Unknown encoded as False.**
  Several condition flags (`has_full_service_history`, `non_smoking`, `is_rental`) use `False` for both "no" and "not stated".
  They enter the model as one-sided "asserted" indicators and are never read as a denial (EDN-23), so the model cannot reward a car for the absence of a claim.
- **Single snapshot.**
  One scrape with no per-listing dates means no temporal validation, no seasonality and no trend.
  The model ages relative to the market from the day it is trained.
- **Market anchoring.**
  A price estimator trained on asking prices influences the asking prices it would later learn from.
  At any real scale this is a feedback loop, and the model would gradually be evaluated against a market it helped set.
  The project never reaches that scale, but the limitation belongs to the model, not to the deployment, so anyone reusing it inherits it.
- **Dealer concentration.**
  83.3 % of listings come from dealers, and a few large ones can dominate a segment.
  The seller-grouped split keeps this from inflating the measured quality, but it does not remove it from what the model learned.

### Fairness

SC-04 requires every segment with at least 500 test rows, including country and seller type, to stay within 15 % MdAPE.
That per-segment error parity is the fairness check this model gets, and it is a deliberate choice: classification-oriented fairness metrics such as AIF360's demographic parity do not transfer to a regression on asking prices, whereas equal error across market segments is the property a user would actually care about ([problem specification](problem-spec.md#8-success-criteria)).
The limitation of that choice is stated above: the segments most at risk of unequal treatment, the low-support makes, are exactly the ones too small for the criterion to apply.
The data carries no protected attributes of people, only of cars and of seller type, so no group-fairness analysis over persons is possible or intended.

### Recommendations

- Treat the output as a market reference, not as a guaranteed value, and never as an appraisal of record.
- Prefer the UC2 interval over the UC1 point estimate whenever the description is incomplete.
- Do not use the model for out-of-scope makes, out-of-scope markets or new cars; the API refuses these on purpose, so do not work around it.
- Treat answers for Volkswagen, Honda, Hyundai and Aston Martin as unvalidated until the gap above is closed.
- Read the estimate together with its explanation (FR-08) rather than on its own.

## How to Get Started with the Model

Filled in once the first model exists.
The intended entry point is the project's API, not the artefact ([specification](specification.md) FR-06 and FR-16):

```bash
curl -X POST http://<host>/predict \
  -H "Content-Type: application/json" \
  -d '{"make": "BMW", "model": "320d", "registration_date": "2019-03",
       "mileage_km_raw": 84000, "power_kw": 140, "fuel_category": "Diesel",
       "transmission": "Automatic", "country_code": "DE",
       "body_type": "Sedan", "seller_type": "Dealer"}'
```

The response carries the estimate in EUR, the model version, a request ID, any warnings and the explanation.
Every endpoint is documented with a request and a response example in the OpenAPI schema at `/docs` (FR-16).
To work with the artefact directly instead, clone the repository and run `dvc pull`; the exact command lands here with the first trained model.

## Training Details

### Training Data

The [AutoScout24 Car Listings Dataset (2025 snapshot)](dataset-card.md), a single scrape of 118,382 listings from 8 European markets, published on Zenodo under DOI [10.5281/zenodo.17643343](https://doi.org/10.5281/zenodo.17643343).

Not all of it is training data.
The scope filters, the `ES` holdout and the deduplication cut it down considerably, and the resulting row counts are reported in the [dataset card](dataset-card.md#how-this-project-uses-the-dataset) and the [problem specification](problem-spec.md#5-evaluation-protocol) rather than repeated here.
No column that identifies a person or an exact location reaches the model: the PII columns are dropped during preprocessing, so no model artefact, prediction log or comparable listing contains them ([requirements](requirements.md#2-non-functional-requirements) NFR-08).

### Training Procedure

#### Preprocessing

The pipeline is built in Milestone 2; these steps are fixed by decisions already taken.

1. **Scope.** Keep used passenger cars that are not pre-registered (EDN-04).
   The pre-registered filter reads a flag whose `False` also means "not stated", so a residual number of unmarked pre-registered listings stays in the data, knowingly (EDN-24).
2. **Drop impossible rows.** Listings registered after the 2025-11-08 reference date would have a negative age and are treated as data errors (EDN-22).
   Prices outside 500 EUR to 2M EUR are dropped as errors or collector outliers.
3. **Deduplicate**, before anything is split off, on a composite key rather than on `vin`, which is only 34 % filled ([dataset card](dataset-card.md#known-data-issues)).
4. **Hold out `ES`** as the new-market drift set (EDN-03).
   It never reaches training, validation or calibration.
5. **Restrict to supported makes** (EDN-05), computed from the cleaned data.
6. **Drop PII, identifiers, leaking and redundant columns** as listed in the [problem specification](problem-spec.md#4-features).
   `seller_company_name` survives only as a hash, as the split's grouping key.
7. **Split** into train, validation, calibration and test **grouped by seller**, so no seller appears in two sets.
   A random split would leak near-identical listings from one dealer across sets, and with no listing dates a temporal split is impossible.
   The proportions and the seed are pinned in `params.yaml`.
8. **Derive** age from the reference date and the registration date; encode categoricals natively rather than one-hot, which is why the model family was chosen.

Every stage runs under DVC, and a clean clone reproduces the same splits and metrics within 0.1 percentage points (NFR-06).

#### Feature space

The two feature sets are defined in the [problem specification](problem-spec.md#4-features).
What the data-dependent parts of them currently amount to is measured, not specified, because both thresholds are parameters the experiment ladder sweeps.
Measured on 2026-09-30 by [`reports/analysis/extended_features.py`](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/analysis/extended_features.py) over the scoped, deduplicated snapshot (105,405 listings, 61,180 of them in the training split), at `equipment_min_frequency` 0.01, `model_version_tokens` 1 and `model_version_min_frequency` 0.0005.

| Measurement | Value |
|---|---|
| Multi-hot equipment columns | 133 of the 146 distinct items in the training rows (comfort 43 of 45, entertainment 16 of 16, extra 38 of 49, safety 36 of 36) |
| `model_version` distinct raw values | 80,332, of which 78,738 survive folding case, accents and separators |
| `model_version` distinct leading tokens | 2,882 at one token, 18,248 at two |
| `model_version` levels above the floor | 279, covering 85.9 % of the training rows (167 and 78.1 % at a floor of 0.001) |
| `weight_kg` parsed range | 1 kg to 93,000 kg, median 1,810 kg; 30 rows below 500 kg and 9 above 4,000 kg |

Two of these are known weaknesses rather than results.

The `model_version` levels are coarse, and unevenly so.
The leading token names the body style for Audi and Porsche, and the engine letter for the German premium models that make up most of the data: `d`, `911` and `e` are the three most frequent levels, and 12.4 % of the training rows lead with an engine letter, which largely repeats `fuel_category` while discarding the trim.
Inside a single make and model the merging is heavy, with Porsche 992 level `911` covering 743 distinct raw trims and Audi A6 `avant` 473, and 329 pairs of levels survive where one is a prefix of the other.
This is accepted as a parameterised starting point; [EDN-41](problem-spec.md#decision-records) records the two finer alternatives that were measured and rejected.

`weight_kg` runs from 1 kg to 93,000 kg, so the column carries data-entry errors from the source.
The pipeline deliberately does not clean them: a value range is a data-quality rule and belongs to the Great Expectations suites (issue #25), not to the feature code.

#### Training

- **Target:** `log(price)`, predictions transformed back with `exp`.
- **Validation set:** early stopping and hyperparameter tuning.
  **Calibration set:** UC2 interval calibration only, never tuning.
- **Intervals (UC2):** Conformalized Quantile Regression with MAPIE (1.x) on quantile LightGBM models, trained with random masking of optional fields so that partial inputs produce wider intervals.
  The coverage guarantee is marginal, that is on average over all inputs, not per missing-field pattern.
- **Point model and missing fields [open]:** whether the point model needs the same random masking, or whether LightGBM's native missing handling suffices, is measured in Milestone 2 once the pipeline exists ([project brief](project-brief.md#4-modelling-plan)).
  Either way SC-06 bounds the outcome, and NFR-01's gate enforces it.
- **Comparables:** a filtered lookup over the processed listings, matching make and model within 2 years of age and 25 % of mileage, **not** a learned nearest-neighbour model.
  The filters are never relaxed to fill the list ([specification](specification.md) FR-09).

Hyperparameters, seeds and the chosen configuration are recorded here once the runs exist, and tracked in MLflow and `params.yaml`.

#### Speeds, Sizes, Times

Targets are set by the [requirements](requirements.md#2-non-functional-requirements); measured values replace them after Milestone 3.
The latency, image and memory targets are **[proposed]** (NFR-02 to NFR-04); NFR-10's training-time target is agreed.

| Property | Target | Measured |
|---|---|---|
| Training time, chosen configuration, no hyperparameter search | at most 15 minutes on a laptop CPU (NFR-10) | _TBD_ |
| API image, model and comparables index included | at most 1 GB, no GPU or deep-learning libraries (NFR-04) | _TBD_ |
| Resident memory of the API under load | below 1 GB (NFR-04) | _TBD_ |
| Inference latency, p95 | 200 ms for `/predict` including the explanation, 300 ms for `/price-range` and `/comparables` (NFR-02) | _TBD_ |

## Evaluation

### Testing Data, Factors and Metrics

- **Testing data:** the test split described under [Preprocessing](#preprocessing), held out and grouped by seller.
  The `ES` holdout is **not** a test set: it never contributes to a success criterion, it drives the drift scenario (EDN-03).
- **Factors:** results are disaggregated by make, country, seller type, fuel category and age bucket (0-1, 1-3, 3-6, 6-10, 10-20, over 20 years).
  Price buckets are reported too, but never used as a criterion, because they condition on the target, which is unknown at prediction time.
- **Metrics:** **MdAPE** as the primary metric, because it is robust to outliers and explains itself ("typically off by X %"); the share of predictions within 10 % and 20 %; MAE in EUR for context; MAPE for comparability only; and for UC2 the empirical interval coverage and the mean relative width.
  Definitions are in the [problem specification](problem-spec.md#6-metrics) and the [glossary](#glossary).

Every criterion is measured once per candidate model on the test set, logged to MLflow, and asserted by the model tests from Milestone 3.
A model is released only if it meets all six ([requirements](requirements.md#2-non-functional-requirements) NFR-01), which the gate in front of a promotion enforces (FR-15).

### Results

Filled with measured values after Milestones 2 and 3.
The criteria themselves are defined in the [problem specification](problem-spec.md#8-success-criteria) (EDN-06, with SC-06 added by EDN-15).

| ID | Criterion | Target | Measured | Pass? |
|----|-----------|--------|----------|-------|
| SC-01 | MdAPE | at most 9 % | _TBD_ | _TBD_ |
| SC-02 | Predictions within 20 % of the asking price | at least 85 % | _TBD_ | _TBD_ |
| SC-03 | MdAPE improvement over baseline B0 | at least 30 % lower | _TBD_ | _TBD_ |
| SC-04 | MdAPE per segment with at least 500 test rows | at most 15 % | _TBD_ | _TBD_ |
| SC-05 | Empirical coverage of the nominal 90 % intervals, full inputs and partial scenario P1 | 88 % to 92 % | _TBD_ | _TBD_ |
| SC-06 | MdAPE with each optional field masked, and with all masked at once | at most 1.5x the full-input MdAPE | _TBD_ | _TBD_ |

#### Reference values

A one-off exploratory run on 2026-09-22, not tracked in MLflow; the experiment ladder reproduces it.
Same scope as above without `ES`, deduplicated, 96,831 listings, 80/20 split grouped by seller.

| Model | MdAPE | Within 20 % |
|-------|-------|-------------|
| B0, median baseline | 11.9 % | 70.9 % |
| LightGBM, basic features, no tuning | 6.7 % | 90.6 % |

#### Summary

Nothing here is a result yet.
What the reference run says is that the targets are reachable but not free: an untuned LightGBM on the basic features already clears SC-01, SC-02 and SC-03 with margin, so the criteria are set where a model barely better than the median baseline still fails them.
The two open risks going into Milestone 2 are the over-20-years segment at 15.3 % against SC-04's 15 % limit, and the low-support makes that SC-04 cannot see at all.
SC-05 and SC-06 have no reference value, because the intervals are built in a later step and SC-06 is relative to the model's own full-input error by construction.

## Model Examination

Explanations are part of the product, not an afterthought: every valuation ships the properties that drove it (FR-08).

- **Per prediction:** the base price plus the five features with the largest absolute SHAP contribution, and the combined contribution of everything else.
  Because the model predicts `log(price)`, a contribution is reported as its multiplicative effect on the price and the effects multiply rather than add.
  They are complete, so recomputing the estimate from the reported effects reproduces it, which is what the API test asserts.
- **How it is computed:** from the trained booster's own SHAP export (LightGBM's `pred_contrib`, CatBoost's `ShapValues`), **not** the `shap` package.
  `shap` pulls in `numba` and `llvmlite`, measured at 189 MB, merely to import, which a CPU-only serving image should not carry (EDN-11).
  A training-time test asserts the booster's export matches `shap.TreeExplainer` bit for bit.
- **Globally:** `shap` is used offline, in training and notebooks, for summary plots and global feature importance.

## Environmental Impact

Measured with CodeCarbon from Milestone 3 and logged to MLflow next to the accuracy of each run (NFR-10).

| | |
|---|---|
| **Hardware type** | Laptop CPU for training, the course VM's CPU for serving. No GPU anywhere (NFR-04). |
| **Hours used** | Target: at most 15 minutes per training run of the chosen configuration, without hyperparameter search (NFR-10). _Measured: TBD._ |
| **Cloud provider** | None for training. Serving runs on the FIB Virtech VM provided by the course (EDN-17). |
| **Compute region** | Barcelona, Spain. |
| **Carbon emitted** | _TBD, per training run from CodeCarbon._ |

Serving energy is reported as an average per answer from the load test, not per individual request: CodeCarbon's granularity does not match single-digit-millisecond events, and a per-request tracker would eat into the latency budget of NFR-02.

## Technical Specifications

### Model Architecture and Objective

Gradient-boosted decision trees, LightGBM as the main model and CatBoost as the challenger, minimising squared error on `log(price)`.
Separate quantile models, conformally calibrated, produce the UC2 interval.
The point estimate and the bounds therefore come from different models, so the bounds are widened where needed to contain the estimate; widening only raises coverage, so the conformal guarantee still holds ([specification](specification.md) FR-07).

### Compute Infrastructure

#### Hardware

- **Training:** a laptop CPU.
  No GPU, and none of the planned steps needs one.
- **Serving:** the FIB Virtech VM, 4 GB RAM and 20 GB disk for the whole stack (EDN-17).
  The API container stays below 1 GB resident, and the drift job runs in its own scheduled container so its heavier dependency stack never competes with the API for memory (NFR-04).

#### Software

LightGBM and CatBoost for the model, MAPIE for the intervals, FastAPI and Pydantic for serving, DVC for data and artefact versioning, MLflow for tracking.
`shap` is a training and notebook dependency only and is never installed in the API image (EDN-11).
Exact versions are pinned in `uv.lock`; the constraints behind these choices are in the [project brief](project-brief.md#6-tooling-constraints).

## Glossary

- **Asking price:** the price a seller lists a car at.
  Not the price it sells for, which this data never observes.
- **MdAPE:** median absolute percentage error, the middle value of the per-listing relative errors.
  Chosen over MAPE because a handful of cheap cars can dominate the mean.
- **Nominal vs empirical coverage:** nominal is the confidence the interval claims, here 90 %; empirical is the share of test cars whose real price actually fell inside.
  SC-05 requires the two to agree within 2 points.
- **Marginal coverage:** the guarantee holds on average over all inputs, not separately for every pattern of missing fields.
- **Conformalized Quantile Regression:** a method that turns quantile predictions into intervals with a coverage guarantee, by calibrating them on data the model never trained on.
- **Segment:** a subset of the test set sharing one attribute value, for example all Audi cars or all cars 10 to 20 years old.
  SC-04 applies per segment.
- **Support threshold:** the minimum number of listings a make needs for the model to serve it at all, 300 (EDN-05).

## Citation

Cite the project by its repository, and the data by its Zenodo record, as the dataset's author asks.

**BibTeX:**

```bibtex
@misc{recommenditos_2026,
  author = {Häußler, Lukas and Adameit, Kevin and Sawczuk, Urszula and Dudek, Michal and Atzberger, Mark},
  title  = {Recommenditos: a used-car price model},
  year   = {2026},
  note   = {Machine Learning Systems in Production (MLOps), FIB-UPC},
  url    = {https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos}
}
```

**APA:**

Häußler, L., Adameit, K., Sawczuk, U., Dudek, M., & Atzberger, M. (2026).
*Recommenditos: a used-car price model* [Computer software].
Machine Learning Systems in Production (MLOps), FIB-UPC.
<https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos>

The dataset's own BibTeX and APA entries are in the [dataset card](dataset-card.md#citation).

## Decision records

The choices behind this page are recorded in [reports/edn.md](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md):

- EDN-02: model family, gradient boosting with LightGBM as the main model.
- EDN-03: hold out AutoScout24 `ES` as the new-market drift scenario.
- EDN-04: used cars only.
- EDN-05: minimum listing support per make.
- EDN-06: success criteria.
- EDN-08: model loading and promotion via DVC, not the MLflow registry.
- EDN-11: keep `shap` out of the API image; serving uses the booster's native SHAP export.
- EDN-12: retraining and promotion are human-triggered, not automated.
- EDN-15: UC1 required fields, and SC-06 for absent optional fields.
- EDN-17: deployment target is the FIB Virtech VM.
- EDN-18: unseen countries and models are accepted with a warning, not rejected.
- EDN-19: DagsHub as the DVC remote and the MLflow tracking server.
- EDN-22: drop listings registered after the age reference date.
- EDN-23: read the condition flags as one-sided assertions.
- EDN-24: keep the pre-registered exclusion although the flag behind it is unreliable.

## Model Card Authors

Team Recommenditos (UPC, MLOps 2026/27): @lukas2510, @kadameit, @ulasawczuk, @W11W11W11, @michudud04.

## Model Card Contact

Through the [project repository](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos), by opening an issue, or by e-mail to the team members listed in the [README](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos#team-and-contact).

## Maintenance

Retraining is started by a person reacting to a monitoring finding, never automatically, and a retrained model reaches users only after it has met every success criterion and a person has approved it (FR-15, EDN-12).
This card is updated in the same step, so it never describes a version that is not the one in service.

Concretely:

- **After Milestone 2:** the measured results per `SC-xx`, the chosen hyperparameters, the artefact size and the training time.
- **After Milestone 3:** the CodeCarbon figures, and confirmation that the model tests assert each criterion.
- **Whenever a model is promoted:** the version, the results and any change to the limitations above.
- **Once the first model exists:** move the modelling-plan parts of [project brief](project-brief.md#4-modelling-plan) into this card and leave a link behind in the brief.
