Model card: Recommenditos used-car price model
==============================================

!!! note "Draft — planned model, no trained artefact yet"
    This is the **initial draft** of the model card, written in M1 before a real model exists.
    It documents the *planned* modelling approach and follows the
    [Hugging Face model card template](https://huggingface.co/docs/hub/model-cards).
    It is a **living document**: the placeholder metrics below are filled with measured
    values after Milestone 2 (first trained model, [experiment ladder](../project-brief.md#4-modelling-plan-planned) steps 1-4)
    and Milestone 3 (model tests asserting each `SC-xx`).

This card owns everything about the trained model.
The **target, features and success criteria** the model is measured against live in the
[problem specification](../problem-spec.md); this card links to them instead of repeating them.
Facts about the data live in the dataset card.

## Model details

### Model description

Recommenditos is a used-car **asking-price** estimator for a European online car portal.
Given the description of a used passenger car it predicts the price the car would be listed at
(UC1), and for a partial description it returns a price interval and comparable listings (UC2).
It is the model behind the project's API.

- **Developed by:** Team Recommenditos, *Machine Learning Systems in Production (MLOps)*, FIB-UPC, 2026/27.
- **Model type:** supervised **regression** on `log(price)`, gradient-boosted decision trees (planned).
- **Language(s):** not a language model; categorical inputs use English labels. Free-text fields are not used by the core model.
- **License:** to be set with the repository license.
- **Task:** tabular regression (point estimate + prediction interval).

### Model type under consideration **[planned]**

Decision and reasoning: [EDN-02](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md).

- **Main model:** **LightGBM** gradient boosting on `log(price)`, with basic then extended features
  ([experiment ladder](../project-brief.md#4-modelling-plan-planned) steps 3-4).
- **Challenger:** **CatBoost** (step 5), for high-cardinality categoricals.
- **Baselines:** B0 = median price per make/model/age bucket; B1 = Ridge regression on `log(price)`
  ([problem spec §7](../problem-spec.md#7-baselines)).

Why gradient boosting: best-in-class on medium tabular data, native handling of categoricals and
missing values, trains in minutes on CPU, small artefacts, and exact SHAP explanations.
Alternatives considered (linear, kNN, random forest, tabular NNs, hierarchical Bayes, LLM zero-shot)
are recorded in the EDN.

### Model sources

- **Repository:** <https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos>
- **Problem specification:** [problem-spec.md](../problem-spec.md)
- **Project brief (modelling plan):** [project-brief.md §4](../project-brief.md#4-modelling-plan-planned)
- **Experiment tracking:** MLflow (added in M2).

## Uses

### Direct / intended use

- **UC1 — Car valuation:** a private buyer or seller describes an owned car (full description) and
  receives a **point estimate** of its listing price in EUR, with a SHAP explanation of the main
  price drivers.
- **UC2 — Purchase guidance:** a user describes a car they want to buy (any subset of the fields) and
  receives a **nominal 90 % price interval** and a list of comparable listings.

The model reflects the market of the data snapshot (AutoScout24, 2025-11-08). It estimates the
**asking** price, not the transaction price, and does not forecast future prices.

### Scope of valid inputs

The input scope is defined by the [problem spec §2](../problem-spec.md#2-scope) and enforced by the API:

- Used passenger cars only (`offer_type = U`, not pre-registered, `vehicle_type = Car`).
- **Supported makes only:** makes with ≥ 300 cleaned used-car listings (computed by the pipeline, not
  hard-coded; currently 11 makes covering 98.6 % of used listings). Out-of-scope makes are **rejected**,
  not guessed.
- Markets: trained on 7 countries (DE, IT, NL, BE, AT, FR, LU) with `country_code` as a feature.
  All `ES` listings are held out as the new-market drift scenario ([EDN-03](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)); the API accepts `ES` and treats it as an unknown country until the model is retrained.
- Training price range 500 EUR – 2M EUR.

Features are listed in [problem spec §4](../problem-spec.md#4-features).

### Out-of-scope use

- Predicting **transaction / sale prices** (only asking prices are observed).
- **Price forecasting** over time.
- New cars, pre-registered cars, transporters.
- Makes below the support threshold.
- Markets outside the 8 countries of the data.
- Parsing free-text car descriptions (optional later add-on; must never be required for the API).
- Any high-stakes automated decision (financing, insurance, taxation) — this is a guidance tool,
  not an appraisal of record.

## Bias, risks and limitations

- **Asking ≠ transaction price.** Estimates are of the listed price and tend to sit above the price a
  car actually sells for.
- **Brand skew.** The data is dominated by premium brands (BMW, Porsche, Mercedes-Benz, Audi);
  mass-market brands are out of scope, so the model does not generalise to them.
- **Classic / old cars.** In the reference run, cars **older than 20 years** are the only segment that
  breaches the per-segment target (MdAPE 15.3 % vs the SC-04 limit of 15 %). Closing this gap with the
  extended features or a scope change is an open task.
- **Reference-date drift.** Age is computed from the request date at serving vs the snapshot date in
  training, so served cars get older than anything seen at the same registration date — expected drift,
  watched in M6.
- **Unknown-encoded-as-False flags.** Several boolean fields (`has_full_service_history`, `non_smoking`,
  `is_rental`) encode "unknown" as False; the model cannot distinguish the two.
- **Single snapshot.** One scrape, no listing dates, so no temporal validation and no seasonality.

### Recommendations

Treat outputs as a market reference, not a guaranteed value. Prefer the UC2 interval over the UC1 point
estimate when inputs are incomplete. Do not use for out-of-scope makes, markets, or new cars.

## Training details

### Training data

See the dataset card (added later) and [problem spec §2-§5](../problem-spec.md#2-scope).
AutoScout24 Car Listings Dataset (2025 snapshot), ~118k listings, 8 European countries.
Deduplicated before splitting; `ES` held out; remaining data split into train / validation /
calibration / test **grouped by seller** (hashed) so no seller appears in two sets
([problem spec §5](../problem-spec.md#5-evaluation-protocol)).

### Training procedure **[planned]**

- **Target:** `log(price)`; predictions transformed back with `exp` (estimates the median price).
- **Validation set:** early stopping and tuning. **Calibration set:** UC2 interval calibration only.
- **Intervals (UC2):** Conformalized Quantile Regression (MAPIE 1.x) on quantile LightGBM models,
  trained with random masking of optional fields so partial inputs yield wider intervals. Coverage
  guarantee is marginal, not per missing-field pattern.
- **Comparables:** k-nearest-neighbour search over processed listings.
- **Explainability:** SHAP (TreeExplainer), per prediction and globally.

_Hyperparameters, seeds and compute are recorded here once the runs exist (M2); tracked in MLflow and `params.yaml`._

## Evaluation

### Testing data, factors and metrics

- **Test data & protocol:** [problem spec §5](../problem-spec.md#5-evaluation-protocol). All success
  criteria are measured once per candidate model on the test set, logged to MLflow, and asserted by the
  model tests (M3).
- **Metrics:** [problem spec §6](../problem-spec.md#6-metrics) — **MdAPE** (primary), share within
  ±10 % / ±20 %, MAE (EUR), MAPE (context only), interval coverage and mean relative width (UC2).
- **Segments:** make, country, seller type, fuel category, age bucket.

### Planned metrics vs success criteria

Filled with measured values against each `SC-xx` after M2/M3.
Criteria defined in [problem spec §8](../problem-spec.md#8-success-criteria) ([EDN-06](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).

| ID | Criterion | Target | Measured (M2/M3) | Pass? |
|----|-----------|--------|------------------|-------|
| [SC-01](../problem-spec.md#8-success-criteria) | MdAPE | ≤ 9 % | _TBD_ | _TBD_ |
| [SC-02](../problem-spec.md#8-success-criteria) | Predictions within ±20 % | ≥ 85 % | _TBD_ | _TBD_ |
| [SC-03](../problem-spec.md#8-success-criteria) | MdAPE improvement over baseline B0 | ≥ 30 % lower | _TBD_ | _TBD_ |
| [SC-04](../problem-spec.md#8-success-criteria) | MdAPE per segment (≥ 500 test rows) | ≤ 15 % | _TBD_ | _TBD_ |
| [SC-05](../problem-spec.md#8-success-criteria) | Empirical coverage of nominal 90 % intervals (full + partial P1) | 88–92 % | _TBD_ | _TBD_ |

**Reference values** (one-off exploratory run, 2026-09-22, not tracked; reproduced by the ladder in MLflow):

| Model | MdAPE | Within ±20 % |
|-------|-------|--------------|
| B0 (median baseline) | 11.9 % | 70.9 % |
| LightGBM, basic features, no tuning | 6.7 % | 90.6 % |

Known risk from that run: the *older-than-20-years* segment sits at 15.3 % MdAPE, just above SC-04.

## Environmental impact

Training is CPU-only and takes minutes; energy is measured with CodeCarbon in M3 and reported here once available.

## Technical specifications

- **Model architecture:** gradient-boosted decision trees (LightGBM main, CatBoost challenger) on `log(price)`.
- **Software:** see `uv.lock` and [project brief §6](../project-brief.md#6-tooling-constraints)
  (CPU-only, small Docker image; SHAP, MAPIE, LightGBM/CatBoost).
- **Serving:** FastAPI (`/predict`, `/price-range`, `/comparables`), see [project brief §5](../project-brief.md#5-target-architecture-planned).

## Citation

Cite the dataset (AutoScout24 Car Listings Dataset, DOI `10.5281/zenodo.17643343`) as required by its author; see the dataset card and report.

## Model card contact

Team Recommenditos — see the [README](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos#team-and-contact) or open an issue.

---

**Maintenance:** update after Milestones 2 and 3 with the measured results against each `SC-xx`.
Once the first model exists, move the modelling-plan parts of [project brief §4](../project-brief.md#4-modelling-plan-planned)
(model details, intervals, explainability) into this card and leave a link in the brief.
