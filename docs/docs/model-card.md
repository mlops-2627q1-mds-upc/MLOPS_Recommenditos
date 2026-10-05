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

**Status: the four ladder variants train, five of the six success criteria are measured, and nothing is released.**
The training procedure, the point metrics and SC-01 to SC-04 and SC-06 below are measured; SC-04 fails for every variant and SC-05 has no measurement until the UC2 intervals exist, so NFR-01's gate blocks, and the serving sections (latency, image size, explanations, intervals) are still a plan.
It follows the [Hugging Face annotated model card template](https://huggingface.co/docs/hub/model-card-annotated) and is a living document: the remaining placeholders are replaced as the `evaluate` stage's criteria and the API land.
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
- **Status:** `lgbm-basic` is the candidate, not released.
  All four variants of the ladder fit, and five of the six success criteria are now measured: SC-01 to SC-03 pass for both LightGBM variants, SC-04 fails for all four on cars over 20 years and the miss is accepted (EDN-62), SC-06 passes for the candidate and fails for `lgbm-extended`, and SC-05 has no measurement until the UC2 conformal intervals exist. NFR-01's gate therefore blocks. Every number below comes from the pipeline run the committed `dvc.lock` records, on the real snapshot, with one MLflow run per variant ([Experiment tracking](#experiment-tracking) says how to find them).

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

To work with the artefact directly instead, clone the repository, run `dvc pull` and load the bundle through its one seam:

```python
from recommenditos.modeling.model import load_model

model = load_model(Path("models/lgbm-basic"))
model.features  # the matrix columns the model consumes, in its own order
model.predict_eur(frame)  # EUR, float64, one row in one row out, indexed like `frame`
model.columns_of("equipment_comfort")  # the multi-hot columns one request field became
model.mask_absent(frame, ["equipment_comfort"])  # the frame with that field not given
```

`predict_eur` takes the feature matrix as it comes: it selects and casts what it needs, ignores extra columns such as `price`, and raises naming any column it needs and cannot find.
An empty frame gives an empty Series, because one row in one row out has to hold at no rows too.
A missing value is passed through unimputed and the estimator decides what to do with it (EDN-15).
What "missing" means for a field whose domain already represents absence is not left to the caller: `mask_absent` is the one implementation of it, and it is the one the API, SC-06's masking sweep and the tests all go through.
An omitted assertion flag is `False`, which is all the source can mean (EDN-23), and an omitted equipment list is the **empty** list, so every item of it is `False` too.
That second one is a decision rather than a reading of the data, and a distributional one: the raw field is always a list, so across the 105,405 scoped listings there is not one null in any of the four equipment fields, while the literal `[]` is 5.2 % to 8.0 % of them per field.
"Empty" is therefore a state the model was fitted on and "absent" is one it never saw (EDN-61).
The frame handed in is never modified.
`model.predict_log_price(frame)` is the log of the same number, which is what a future conformal step needs; `predict_interval_eur` is the reserved name for SC-05's intervals and does not exist yet.

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
5. **Restrict to supported makes** (EDN-05), computed by `split` from the cleaned data and applied by `features` to every frame, the `ES` holdout included, before any level or column is decided (EDN-48, EDN-67).
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
All at `equipment_min_frequency` 0.01, `model_version_tokens` 1 and `model_version_min_frequency` 0.0005.
What the training split decides is measured on 2026-10-05 by [`reports/analysis/served_vocabulary.py`](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/analysis/served_vocabulary.py) over the **60,378 training rows of the supported makes**, which is the population the vocabulary is fitted on (EDN-67); the script builds it over all 61,180 training rows too, and that figure is given beside it.
The properties of the raw columns are measured on 2026-09-30 by [`reports/analysis/extended_features.py`](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/analysis/extended_features.py) over the scoped, deduplicated snapshot (105,405 listings).

| Measurement | Value |
|---|---|
| `make` levels | 11, exactly the supported makes (25 over all training rows) |
| `model` levels | 437 (561 over all training rows) |
| Multi-hot equipment columns | 133 of the 146 distinct items in the served training rows (comfort 43 of 45, entertainment 16 of 16, extra 38 of 49, safety 36 of 36); the same 133 over all training rows |
| `model_version` levels above the floor | 278, covering 86.5 % of the served training rows (167 and 78.8 % at a floor of 0.001); 279 and 85.9 % over all training rows |
| Columns per matrix | 16 features in the basic set and 162 in the extended one, plus `price` and `log_price`; unchanged by the restriction, because no equipment item crossed the threshold either way |
| `model_version` distinct raw values | 80,332, of which 78,738 survive folding case, accents and separators |
| `model_version` distinct leading tokens | 2,882 at one token, 18,248 at two |
| `weight_kg` parsed range | 1 kg to 93,000 kg, median 1,810 kg; 30 rows below 500 kg and 9 above 4,000 kg |

Two of these are known weaknesses rather than results.

The `model_version` levels are coarse, and unevenly so.
The leading token names the body style for Audi and Porsche, and the engine letter for the German premium models that make up most of the data: `d`, `911` and `e` are the three most frequent levels, and 12.4 % of the training rows lead with an engine letter, which largely repeats `fuel_category` while discarding the trim.
Inside a single make and model the merging is heavy, with Porsche 992 level `911` covering 743 distinct raw trims and Audi A6 `avant` 473, and 329 pairs of levels survive where one is a prefix of the other.
These figures are EDN-41's, measured on 2026-09-30 over all training rows; the restriction to the supported makes leaves the per-model merges as they were, because every make named here is supported, and takes one level off the total.
This is accepted as a parameterised starting point; [EDN-41](problem-spec.md#decision-records) records the two finer alternatives that were measured and rejected.

`weight_kg` runs from 1 kg to 93,000 kg, so the column carries data-entry errors from the source.
The pipeline deliberately does not clean them in the feature code, where a value range does not belong.
The Great Expectations suites (issue #25) do not bound it either: the column is still text when they run, and no plausible range has been agreed, so the errors reach the extended feature set as they are.

#### Training

The `train` stage fits one variant of the [experiment ladder](project-brief.md#4-modelling-plan) per `dvc.yaml` stage, so `dvc repro train@lgbm-basic` retrains that variant alone.
Every variant is fitted only on the makes the API serves: `split` records them in `data/processed/supported_makes.json`, `features` restricts every frame to them before it builds the vocabulary, and `train` refuses a matrix holding any other make rather than filtering it again (EDN-48, EDN-67).
So no make the API refuses with a 422 is in the model's training data, and none decided the levels and columns the model is encoded in.

- **Target:** `log(price)`, predictions transformed back with `exp`.
  **No bias correction** is applied to the inverse transform (EDN-49).
  `exp(E[log P | x])` is the conditional median, which is what every metric here reports, and a Duan smearing correction targets the conditional mean instead, so it biases every prediction upward and away from the typical asking price a seller wants.
  Measured on the real snapshot when the decision was taken, with the LightGBM variants still at the former 1,000-tree budget, it makes every variant worse: smearing factor 1.0258 and MdAPE 9.52 % to 9.83 % for `b1`, 1.0052 and 6.83 % to 6.86 % for `lgbm-basic`, 1.0028 and 6.32 % to 6.37 % for `lgbm-extended`.
  `b0` needs no inverse transform at all, because it is fitted on `price` directly.
- **Validation set:** early stopping and hyperparameter tuning.
  **Calibration set:** UC2 interval calibration only, never tuning.
- **The prediction is bounded to the training price range.**
  `predict_eur` clips in log space before the exponential, to the observed minimum and maximum `price` of its own training rows, which are recorded in the bundle.
  On the real snapshot that range is 500 to 1,814,750 EUR.
  This is what makes the interface's "finite and strictly positive" true of the arithmetic rather than of a hope: `exp` overflows to `inf` above about 710 in log space and underflows to exactly `0.0` below -746, and an `inf` prediction would turn MdAPE into `nan` and pass the gate's comparison unnoticed.
  It also catches a real extrapolation rather than only a theoretical one: over the 19,665 test rows, B1's unbounded prediction leaves the range **once**, at **10,635,538 EUR**, while none of the other three variants leaves it at all (unbounded maxima 1,549,910 for B0, 1,055,373 for `lgbm-basic`, 731,089 for `lgbm-extended`).
  The bound is a guard rail and not a calibration, and it should not be read as one: it turns that 10.6 million into 1,814,750, which is the most expensive car the model was trained on rather than a sensible estimate for the car in question.
  Because the range is recorded in the bundle, the share of predictions sitting exactly on an edge can be reported instead of the clip hiding the pathology.

##### Hyperparameters as trained

All of these live in `params.yaml`, so a change reruns exactly the variant it affects.

| Variant | Ladder step | Estimator | Feature set | Parameters |
|---|---|---|---|---|
| `b0` | 1 | median of `price` per (make, model, 2-year age band), falling back to the make median and then the global median | basic | `age_bucket_years: 2` |
| `b1` | 2 | Ridge on `log(price)` | basic | `alpha: 1.0`, `min_category_rows: 5` |
| `lgbm-basic` | 3 | LightGBM | basic | `learning_rate: 0.05`, `num_leaves: 63`, `n_estimators: 5000`, `early_stopping_rounds: 50` |
| `lgbm-extended` | 4 | LightGBM | extended | as `lgbm-basic` |

The settings that are **not** parameters are modelling choices rather than knobs a sweep should touch, and they are the same for both LightGBM variants: `objective="regression"` (squared error on log price, because the log transform already handles the multiplicative error structure) and `metric="l1"` with `first_metric_only=True` for early stopping.
The absolute error in log space *is* the symmetric relative error, so the stopping point lines up with MdAPE, the metric the gate reads, rather than with a squared error nothing reports.
`n_estimators: 5000` is a ceiling and not a tree count: early stopping on the validation split decides how many trees the model keeps, and the booster is saved at that round, so the file *is* the model and no consumer has to remember an iteration argument (EDN-70).
On the real snapshot early stopping chooses round **1,243** for `lgbm-basic` and **1,647** for `lgbm-extended`, so the ceiling sits three times above the later of the two.
It used to be 1,000, and then it was the ceiling that ended both fits, not early stopping (issue #64).
5,000 rather than higher, because the ceiling is also the worst case of every cost that grows with the trees: a fit that used all of them would take about 80 s for `lgbm-extended`, write a booster of about 33 MB and spend about 105 ms on one row's SHAP export, against NFR-02's 200 ms for the whole of `/predict`.
`train` warns when the ceiling ends a fit, and `model.json` and the MLflow run record whether early stopping did (`early_stopped`, with the rounds run beside the trees kept), so a binding ceiling is a fact of the artefact rather than something read off a tree count.

Raising the ceiling is also what showed how closely the stopping metric tracks the gate's.
Between 1,000 trees and the round early stopping chose, the validation L1 in log space keeps falling - by 0.2 % for `lgbm-basic` and 0.6 % for `lgbm-extended` - while validation MdAPE is flat or marginally worse, 6.947 % to 6.956 % and 6.448 % to 6.453 % (`reports/analysis/early_stopping_ceiling_results.txt`).
L1 is a mean and MdAPE a median, so the last few hundred trees improve the tail of the errors rather than the typical one; on the test split they moved MdAPE by 0.02 pp and 0.07 pp (see [the ladder](#the-experiment-ladder-as-measured)).

B1's encoding is two decisions that look, at a glance, like rules of this project being broken, and both are recorded.
Its categoricals are **one-hot encoded**, which EDN-02 rules out for the tree models and which EDN-50 confines to B1: Ridge is linear, has no native categorical handling, and dropping the categoricals would leave the depreciation baseline unable to tell a Porsche from a Dacia.
Its numerics are **mean-filled with a per-feature missingness indicator**, which EDN-51 argues is not imputation under EDN-15: with an indicator for every feature the encoding is information-preserving, so the fill is a numerically neutral placeholder rather than a guess at the value.
`min_category_rows: 5` groups the levels below that many training rows into one shared column, which is also where an unseen level goes at serving time where such a group exists (EDN-18).
Folding is not free: measured on the real snapshot, no folding gives 503 encoded columns at 9.48 % MdAPE, 5 gives 404 at 9.52 %, 30 gives 294 at 9.71 % and 100 gives 187 at 10.61 %.
What the 0.04 pp buys is that a level seen in a handful of rows does not get a coefficient fitted on them, and 47 of the 100 levels folded at 5 are `model` values seen exactly once.
At 5 only `model` and `fuel_category` have an infrequent group at all; the other six categoricals encode an unseen or absent value as all zeros, which answers just as well and is measured per column in EDN-50.

##### Determinism

- **Seed:** `params.yaml`'s project-wide `seed`, passed to LightGBM as `random_state` so `feature_fraction_seed`, `bagging_seed` and `data_random_seed` all derive from it.
  Worth being exact about: with `feature_fraction` and `bagging_fraction` at their defaults of 1.0, **none of the four estimators consumes randomness at all** today.
  The seed is recorded for provenance and to make a future subsampled configuration reproducible; it is inert as configured, so a test asserting that a different seed changes the result would fail.
- **Threads:** `train.num_threads: 1`, never LightGBM's default of every core (EDN-53).
  The trees are already thread-independent - identical at 1, 2, 4 and 8 threads - but LightGBM writes the count into `booster.txt`, so a default would make the artefact's hash depend on the machine that ran `dvc repro`.
  1 rather than a larger pinned value because only 1 is a value every machine can honour.
  It costs something, and against a specific alternative: measured on the real snapshot as medians of three fits on an idle 8-core machine, `lgbm-extended` fits in 18.3 s at 1 thread against 12.8 s at 2 and 12.2 s at 4, so a *pinned* 2 or 4 would buy back about a third of the fit.
  Against LightGBM's default of every core, which is what the pin actually replaces, it is a gain rather than a cost: 18.3 s against 29.2 s at 8 threads, and 7.1 s against 17.7 s for `lgbm-basic`.
  It is also the steadiest value by a long way - three fits at 1 thread land within a second of each other, three at 8 span 26 s - so it is what makes a reported fit time mean anything.
  The same reversal is sharper on the 744-row test fixture (0.33 s at 1 thread against 28.6 s at 8), because OpenMP's overhead dominates whenever there is little work per thread - and the suite fits every variant on that fixture.
  These seconds are a controlled sweep for the ratios, not comparable with the ladder table's one-off wall-clock fit times below.
- **LightGBM:** `deterministic=True` and `force_row_wise=True`, so the histogram construction is not chosen by data size and thread count.
  Documented insurance rather than a measured fix, and worth being exact about: with both flags **off** the trees are still identical across 1, 2, 4 and 8 threads, so the flags are not what makes the result thread-independent here.
  They do change *which* trees are built, so they are not without consequence; they are simply not load-bearing for this property at this data size.
- **Ridge:** `solver="lsqr"`, scipy's single-threaded iterative solver, rather than the default `auto`, which may pick a dense solver and go through BLAS, whose reduction order depends on the thread count.
  Also insurance: the coefficients came out byte-identical across separate processes at 1 and 8 BLAS threads with `auto` too.
- **B0:** group medians only, and a median is order-independent, so the result does not depend on the row order Parquet hands back.
- What holds, and is asserted by `tests/test_model.py`: two fits of the same variant on the same rows give bit-identical predictions and a byte-identical payload file.
  `model.json` is deliberately excluded, because it carries `trained_at` and the MLflow run id, which are provenance and cannot be stable.
  What is **not** promised is bit-identical results across different LightGBM, scikit-learn or NumPy builds or across CPU architectures; NFR-06's "within 0.1 percentage points" is the promise that survives a toolchain change.

##### What a trained variant writes

`models/<variant>/`, one DVC-tracked directory per variant:

| File | Contents |
|---|---|
| `model.json` | The record: the estimator, the feature and target lists separately, the hyperparameters as trained, the training row counts and price range, the supported makes, the library versions and the MLflow run id. For the LightGBM variants also the trees kept (`best_iteration`), the rounds the fit ran (`boosting_rounds`) and whether early stopping or the `n_estimators` ceiling ended it (`early_stopped`); the three are null for `b0` and `b1`. |
| `feature_space.json` | A copy of the `features` stage's own artefact: the contract the matrices were written against and the vocabulary the training rows decided. |
| `booster.txt` | LightGBM variants: the native text format, saved at the early-stopped iteration. Not a pickle of the sklearn wrapper, because the text format survives a LightGBM upgrade, it is readable, and EDN-11's SHAP export needs a `Booster`. |
| `pipeline.joblib` | `b1` only: the fitted scikit-learn pipeline. |
| `lookup.parquet` | `b0` only: the whole model as one table a person can read, in EUR, with the row count behind each median. |

The feature space travels *with* the model rather than being looked up beside the matrices, because the bundle is baked into the API image on its own (EDN-08) and a code is a level's position: a model that encoded a request against one level order and scored it against another would be wrong with no error anywhere.
`recommenditos/modeling/model.py` is the only code that opens any of these files.

##### Experiment tracking

One MLflow experiment, `params.yaml`'s `train.mlflow_experiment` (`recommenditos-price`), and **one run per variant**, named after the variant.
`train` creates the run and records its id in `model.json`; `evaluate` resumes that id and appends the test metrics and the gate verdict rather than opening a run of its own.
That is deliberate: a run per DVC stage would scatter four variants over eight runs nothing joins, and the point of tracking is to be able to compare them.

Each run therefore carries the hyperparameters, the train and validation L1 in log space, the fit time, for the LightGBM variants the trees kept, the rounds run and `early_stopped` as 1 or 0 (so `metrics.early_stopped = 0` finds every fit the ceiling ended), the model artefact under `model/` and the test metrics with the six criteria.
Its parameters are every `params.yaml` key `dvc.yaml` declares for the `train` stage - the seed, `num_threads`, the estimator, the feature set and the hyperparameters, the last under the estimator's name - plus the variant's name and the shape of the training data; the experiment is the one declared key recorded as the run's experiment rather than as a parameter (NFR-14).
The emissions of the fit are not among them yet; they arrive with issue #38.
`train` logs **no metric in euros**: it must not touch the test set, and a train-set MdAPE would be a second implementation of the metric beside `evaluate`'s, so one run could carry two numbers that disagree.

Every run is tagged with `variant`, `estimator`, `feature_set` and `dvc_stage`, and - by the tracking seam, for NFR-14 - with what produced it: `git_commit` and `git_dirty` for the fit, one `train.deps.<path>` per dependency of its `train` stage holding the hash `dvc.lock` records for it, and the same for the evaluation that appended to it, `evaluate.git_commit`, `evaluate.git_dirty` and `evaluate.deps.<path>` (EDN-74).
[The DVC pipeline](pipeline.md#from-a-run-to-its-inputs) says what each tag identifies and how to get from a run back to its code, data and parameters.
The ladder this card reports is four runs of the `recommenditos-price` experiment, one per variant, produced by the `dvc repro` whose lock is committed.
The comparable view of one pipeline state is the experiment's own table filtered to that state's commit:

```
tags.git_commit = '<the full SHA of the run you want>'
```

with the columns `tags.variant`, `params.estimator`, `params.feature_set`, `metrics.mdape`, `metrics.within_20pct` and `metrics.fit_seconds`.
That filter is what picks one chain out of the experiment: the four runs of a `dvc repro` share one commit, and a run from another commit is a different pipeline.
Two things about it are worth knowing before the filter is believed, both checked on this ladder rather than assumed:

- **The commit is the one the pipeline ran from, not the one that holds its lock.**
  `dvc repro` runs at a commit and its `dvc.lock` and metrics are committed after it, so the runs name the parent of the commit that records them, and `tags.git_dirty` is `false` for all four: what DVC writes while it runs does not count as a change.
  The commit is on the branch of the pull request that re-ran the pipeline, so after the squash merge it is reached through that pull request.
- **The input tags are what ties a run to the committed lock.**
  For each of the four runs, every `train.deps.<path>` equals the hash the committed `dvc.lock` records for that dependency of `train@<variant>`, and every `evaluate.deps.<path>` the one under `evaluate`; `reports/analysis/run_provenance_check.py` checks that against the server, and its output is committed beside it.
  Runs from before 2026-10-05 carry a `dvc_lock_md5` tag instead, which never equals the lock that records them, and three of the four runs of the ladder before this one were tagged dirty only because DVC had rewritten `dvc.lock` under them.

The experiment also holds runs that are **not** pipeline runs: the test suite trains the same four variants on the synthetic fixture, and it logs to this experiment whenever a `.env` is present, under the same four run names.
Their metrics are an order of magnitude worse and they carry a different commit; they also carry no `train.deps.<path>` tags, because DVC did not run them, so since 2026-10-05 a run without those tags is not a pipeline run and its numbers are not the model's.

Training does **not** require credentials or a network.
With no `MLFLOW_TRACKING_URI` the stage logs one line, writes the model normally and records `mlflow.tracking_mode: "disabled"`, which is what lets CI and a fresh clone run the test suite; it never falls back to a local store, because since MLflow 3.16 that would mean a SQLite database in the repository root.
`RECOMMENDITOS_REQUIRE_TRACKING=1` turns any tracking failure into a failed stage, and that is the setting to use for the runs whose numbers are cited, so that a silent skip is not discovered while the report is being written.
- **Intervals (UC2):** Conformalized Quantile Regression with MAPIE (1.x) on quantile LightGBM models, trained with random masking of optional fields so that partial inputs produce wider intervals.
  The coverage guarantee is marginal, that is on average over all inputs, not per missing-field pattern.
- **Point model and missing fields [open]:** whether the point model needs the same random masking, or whether LightGBM's native missing handling suffices, is measured in Milestone 2 once the pipeline exists ([project brief](project-brief.md#4-modelling-plan)).
  Either way SC-06 bounds the outcome, and NFR-01's gate enforces it.
- **Comparables:** a filtered lookup over the processed listings, matching make and model within 2 years of age and 25 % of mileage, **not** a learned nearest-neighbour model.
  The filters are never relaxed to fill the list ([specification](specification.md) FR-09).

#### Speeds, Sizes, Times

Targets are set by the [requirements](requirements.md#2-non-functional-requirements); measured values replace them after Milestone 3.
The latency, image and memory targets are **[proposed]** (NFR-02 to NFR-04); NFR-10's training-time target is agreed.

| Property | Target | Measured |
|---|---|---|
| Training time, chosen configuration, no hyperparameter search | at most 15 minutes on a laptop CPU (NFR-10) | **8.6 s** for the candidate `lgbm-basic` (1,293 boosting rounds), 26.4 s for `lgbm-extended` (1,697 rounds), 36.4 s for the whole four-variant ladder, at `num_threads: 1` on the real snapshot (the tracked run of 2026-10-05). Read as an order of magnitude and not a benchmark: the exploratory fits of the identical two models on the same day took 8.0 s and 27.1 s, which is the size of the effect. The worst case the `n_estimators: 5000` ceiling allows is about 80 s for `lgbm-extended` (EDN-70) |
| Model artefact on disk | no target | 7.6 MB (`booster.txt`, 1,243 trees) for the candidate, 10.9 MB (1,647 trees) for `lgbm-extended`, plus 22 KB and 87 KB of metadata and feature space respectively. About 6.2 and 6.6 MB per 1,000 trees, so the ceiling's worst case is about 33 MB |
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

The criteria are defined in the [problem specification](problem-spec.md#8-success-criteria) (EDN-06, with SC-06 added by EDN-15).
Both LightGBM variants are shown, because the choice between them is what the criteria decided: **`lgbm-basic` is the candidate this card puts forward** (EDN-62), not the variant with the lower pooled MdAPE.
The numbers come from the pipeline run described under [the ladder](#the-experiment-ladder-as-measured), which is the one the committed `dvc.lock` records and the one the four MLflow runs hold.
It is a **measured candidate, not a release**: the gate blocks, so nothing is promoted, and promotion is a human decision in any case (EDN-12).

| ID | Criterion | Target | `lgbm-basic` (the candidate) | `lgbm-extended` |
|----|-----------|--------|------------------------------|-----------------|
| SC-01 | MdAPE | at most 9 % | 6.81 %, yes | 6.26 %, yes |
| SC-02 | Predictions within 20 % of the asking price | at least 85 % | 89.8 %, yes | 90.4 %, yes |
| SC-03 | MdAPE improvement over baseline B0 | at least 30 % lower | 44.0 % lower, yes | 48.6 % lower, yes |
| SC-04 | MdAPE per segment level with at least 500 test rows | at most 15 % | 16.77 % (`age_bucket=over 20`, 784 rows), **no** | 18.45 %, **no** |
| SC-05 | Empirical coverage of the nominal 90 % intervals, full inputs and partial scenario P1 | 88 % to 92 % | not measured, **no** | not measured, **no** |
| SC-06 | MdAPE with each optional field masked, and with all masked at once | at most 1.5x the full-input MdAPE | 1.21x, yes | **1.65x, no** |

`b0` and `b1` miss SC-01, SC-02 and SC-03 as well, and are baselines rather than candidates; their per-criterion numbers are in `reports/metrics/<variant>.json`.

**No variant is deployable**, because SC-04 fails for all four and SC-05 cannot be measured until the UC2 intervals exist, and the gate reports that rather than putting a model forward.
The candidate above is a documented judgement about which model *would* be released, not a computed one: `metrics.json`'s `best_variant` stays the lowest MdAPE overall, because that field means what it says, and `deployable_variant` stays null while anything blocks.
The two failures are the interesting half of this card, and neither is a surprise:

- **SC-04 fails on old cars, for every variant, and the extended features do not close it. The miss is accepted and recorded rather than worked around (EDN-62).** The [problem specification](problem-spec.md#8-success-criteria) already recorded cars over 20 years at 15.3 % as the known risk, and the pipeline measures them worse than that: **16.77 % for the candidate** and 18.45 % for `lgbm-extended`, over 784 test rows. Letting early stopping decide the tree count (EDN-70) moved the candidate 0.35 pp closer to the bar and not under it. The specification named two possible remedies, "the extended features or a scope change for classic cars", and measurement has now **ruled the first one out**: the extended set makes this segment 1.7 pp worse while improving the pooled figure by 0.55 pp, so the extra features do not carry information about old cars. That is a finding about the features, not a reason to move the bar: the threshold stays at 15 %, no upper age bound is added, and the criterion is reported as missed. Nothing is unblocked by softening it in any case, because SC-05 blocks the gate regardless. What is left, for a later ticket, is a scope change of the kind EDN-04 made for price, or a segment-specific model.
- **SC-06 passes for the candidate and fails on `lgbm-extended`, and there only when every optional field is absent at once.** Each of the 23 optional fields masked on its own costs at most 9.7 % of the MdAPE (`equipment_extra`, 1.10x), which is comfortably inside the bound; all 23 together cost 64.6 % (6.26 % to 10.30 %). `lgbm-basic` passes at 1.21x (6.81 % to 8.26 %), because it has six optional fields to lose rather than 23, and `b0` and `b1` pass at 1.00x and 1.16x.

    **This is what decided the candidate (EDN-62).** The two rows above describe the same request: only the ten fields FR-01 requires. On that request `lgbm-basic` answers at **8.26 %** and `lgbm-extended` at **10.30 %**, so the variant with the 0.55 pp pooled advantage is beaten by 2.04 pp as soon as the caller stops filling in the optional fields, which FR-01 says a caller need not do. The extended model's advantage is conditional on a well-filled request, which is precisely the fragility EDN-15 added SC-06 to expose, and a criterion a candidate misses is not something to serve. The worst single field is also worth naming per variant, because it is not the same one: `equipment_extra` for `lgbm-extended`, `cylinders_volume_cc` for `lgbm-basic` and `b0`, `nr_doors` for `b1` at 1.16x, which is above that variant's all-at-once ratio of 1.13x. SC-06 bounds the worst of both kinds of request, so a criterion measured only on the all-at-once scenario would have missed it.

SC-05 needs the UC2 conformal intervals, which are a later ticket.
The `evaluate` stage reports it as `not_measured` with the capability it checked for, and that blocks the gate rather than passing it (NFR-01).
The point-estimate half of the partial-input scenario P1 *is* measured now, and it is the strongest argument for building the intervals: the candidate goes from 6.81 % MdAPE to 19.67 % (2.9x) when only `make`, `model`, `registration_date` and `mileage_km_raw` are given, `lgbm-extended` to 19.02 % (3.0x) and `b1` to 2.5x its full-input value, while `b0` is unchanged because it reads only three columns. A point estimate that is typically 19 % off is not something to serve as a number, which is exactly what UC2's interval answers instead. P1 is reported beside the SC-06 sweep and is structurally excluded from it, because it masks required fields.

#### The per-segment breakdown, and the fairness reading

Every variant is cut by `make`, `country_code`, `seller_type`, `fuel_category` and age bucket, which gives **36 levels**, of which **26** clear the 500-row minimum SC-04 needs.
Six price buckets are reported beside them and can never enter a criterion, so the table `evaluate` writes is 42 rows per variant and 168 across the ladder (`reports/metrics/segments.csv`).
Ten levels are reported and excluded: nine below the row minimum (`Honda` 177, `Hyundai` 105, `Aston Martin` 84, `Volkswagen` 81, `LU` 147, `Others` 69, `LPG` 44, `CNG` 10, `Ethanol` 3) and one, `fuel_category=(missing)` with 5 rows, because it is the absence of a required input field and FR-01 answers such a request with a 422.

The fairness statement is the spread of MdAPE across the levels of a segment, since classification-oriented parity metrics do not apply to a regression (problem specification section 8).
For the candidate `lgbm-basic`, with `lgbm-extended`'s spread beside it because the two agree on where the disparity is:

| Segment | Qualifying levels | Best level | Worst level | Spread | Spread, `lgbm-extended` |
|---|---|---|---|---|---|
| `make` | 7 | Mercedes-Benz 5.23 % | Volvo 9.23 % (714 rows) | 4.0 pp | 4.4 pp |
| `country_code` | 6 | DE 6.37 % | FR 8.42 % (1,058 rows) | 2.0 pp | 2.7 pp |
| `seller_type` | 2 | Dealer 6.45 % | PrivateSeller 8.57 % (3,868 rows) | 2.1 pp | 2.4 pp |
| `fuel_category` | 5 | Electric/Diesel 5.74 % | Diesel 7.07 % (6,348 rows) | 1.3 pp | 1.3 pp |
| `age_bucket` | 6 | 0-1 years 5.55 % | over 20 years 16.77 % (784 rows) | 11.2 pp | 13.6 pp |

Read as a fairness check: the market segments are close to parity, and the one disparity that matters is **age**, not country or seller type.
A private seller is served about 2.1 pp worse than a dealer and a French listing 2.0 pp worse than a German one, both inside SC-04's bound; the owner of a car over 20 years old is served 11.2 pp worse than the owner of a new one and outside it.
The extended features widen four of the five spreads, `age_bucket`'s most, by 2.4 pp, and leave `fuel_category`'s where it is. That is the same finding SC-04 reports from the other direction: they buy pooled accuracy without buying it evenly.
The price buckets are deliberately not part of this statement: their profile is U-shaped (for the candidate, 22.2 % under 5,000 EUR and 6.3 % over 80,000 EUR against 5.8 % in the 40,000 to 80,000 EUR bucket), which is the regression-to-the-mean artefact of conditioning on the target rather than a finding about a market.

#### The experiment ladder, as measured

One run of the whole chain on the real snapshot on 2026-10-05 (`download.source: zenodo`, 118,382 raw listings, 105,405 after the scope and deduplication funnel), fitted on the 11 supported makes and encoded in the vocabulary their training rows decided (EDN-67): 60,378 training rows, 8,859 validation rows and 19,665 test rows, which are the served rows of the 61,180, 8,992 and 19,985 in the split.
It is the run the committed `dvc.lock` records, so `dvc pull` reproduces these artefacts exactly, and it is tracked: one MLflow run per variant, found with the `tags.git_commit` filter under [Experiment tracking](#experiment-tracking).
Every number is from the test split, at `train.num_threads: 1`, and MdAPE is the primary metric.
The bold figures are the lowest of the ladder; they are not the candidate, which SC-06 decided against them (EDN-62).

| Variant | Ladder step | Features | MdAPE | Within 20 % | Validation L1 (log price) | Trees | Fit time | Payload |
|---|---|---|---|---|---|---|---|---|
| `b0` | 1 | 16 | 12.16 % | 70.5 % | 0.1769 | - | 0.1 s | 19 KB |
| `b1` | 2 | 16 | 9.52 % | 80.3 % | 0.1406 | - | 1.3 s | 19 KB |
| `lgbm-basic` **(the candidate)** | 3 | 16 | 6.81 % | 89.8 % | 0.1003 | 1,243 | 8.6 s | 7.6 MB |
| `lgbm-extended` | 4 | 162 | **6.26 %** | **90.4 %** | 0.0966 | 1,647 | 26.4 s | 10.9 MB |

What the ladder says, and what it does not:

- **What ladder step 4 measured, stated as what it now is: the extended feature set buys pooled accuracy only when the caller fills in the optional fields, and loses more than it buys when they do not.** It is worth **0.55 pp** of MdAPE on a fully described car, and it is **2.04 pp worse** than the basic set on a request carrying only the ten fields FR-01 requires (10.30 % against 8.26 %, from the SC-06 sweep). It also costs 146 extra columns, 3.1x the fit time and a 90-second `features` stage. So the answer to "what are the extended features worth" is conditional on the request, and for a component whose contract lets a caller omit 23 of its 33 inputs the conditional half is the one that decides: the candidate is `lgbm-basic` (EDN-62). Step 4 did its job by producing a number that could have gone either way, and the criteria, not the pooled figure, are what read it.
- B0 reproduces the exploratory reference run below almost exactly (12.16 % against 11.9 %), and `lgbm-basic` likewise (6.81 % against 6.7 %), which is the cross-check that the pipeline is measuring what the notebook measured.
- **Early stopping decides the trees now, and the trees the old budget withheld were worth almost nothing (EDN-70).** At `n_estimators: 1000` both LightGBM variants ran out of budget before early stopping fired. With the ceiling at 5,000, early stopping chooses round 1,243 for `lgbm-basic` and 1,647 for `lgbm-extended`, and on the test split that moves MdAPE from 6.83 % to 6.81 % and from 6.32 % to 6.26 %, within 20 % by about 0.1 pp each, and no criterion's verdict for either variant. The training L1 falls much further than the validation L1 (0.0688 to 0.0655 against 0.1005 to 0.1003 for the candidate), so the extra trees mostly fit the training rows more closely. For `lgbm-extended` that comes with a larger dependence on its optional fields, SC-06 1.58x to 1.65x, and a worse old-car segment, 17.86 % to 18.45 %, while the candidate's old-car segment improves from 17.12 % to 16.77 %. What they cost is linear in the trees: 1.3x and 1.7x the boosting rounds and so the fit, 6.2 to 7.6 MB and 6.8 to 10.9 MB of booster, and one row's native SHAP export from 18.9 to 26.1 ms and from 22.3 to 35.4 ms. So the 1,000-tree models were a hair short of where their validation curve flattens, not far from it, and the pooled numbers this card used to quote were not a materially pessimistic bound. `learning_rate` and `num_leaves` are still the values the ladder was first fitted with; whether and how to tune them is the open half of issue #64.
- The whole four-variant ladder trains in **36.4 seconds** of fitting, against NFR-10's 15-minute budget, so nothing about the budget constrains the tuning. The figure is worth an order of magnitude of slack and not one second of precision: the exploratory fits of the same two LightGBM models on the same day took 8.0 s and 27.1 s against 8.6 s and 26.4 s here, because a fit at `num_threads: 1` competes with whatever else holds a core (EDN-53 measured the same variance from the other side).

#### Reference values

A one-off exploratory run on 2026-09-22, not tracked in MLflow; the experiment ladder reproduces it.
Same scope as above without `ES`, deduplicated, 96,831 listings, 80/20 split grouped by seller.

| Model | MdAPE | Within 20 % |
|-------|-------|-------------|
| B0, median baseline | 11.9 % | 70.9 % |
| LightGBM, basic features, no tuning | 6.7 % | 90.6 % |

#### Summary

The ladder reproduces the reference run, and the first three criteria pass with margin for both LightGBM variants: the candidate `lgbm-basic` is at 6.81 % MdAPE against SC-01's 9 %, 89.8 % within 20 % against SC-02's 85 %, and 44.0 % below the median baseline against SC-03's 30 %.
Nothing is released on that, because NFR-01's word is "every": SC-04 fails for every variant and SC-05 has no measurement until the UC2 intervals exist.
The chosen model is therefore `lgbm-basic` **as a candidate**, chosen over the lower pooled MdAPE of `lgbm-extended` because it is the one that meets SC-06 (EDN-62). Early stopping decides its tree count (EDN-70), and its other two hyperparameters are still the values it was first fitted with: the honest statement about it is that it is untuned in `learning_rate` and `num_leaves`, and that the tree count, the one knob measured so far, turned out to be worth 0.02 pp.

Three open risks going into the rest of Milestone 3, all three now measured rather than anticipated:

- **SC-04 is missed and the miss is accepted (EDN-62).** The over-20-years segment sat at 15.3 % in the reference run and the pipeline measures it at 16.77 % for the candidate, and the extended features are now ruled out as the remedy. A scope change or a segment-specific model is the open work.
- The low-support makes SC-04 cannot see at all, because they do not reach its 500-row minimum: `Honda` at 177 test rows down to `Ethanol` at 3, ten levels in all, reported in `reports/metrics/segments.csv` with the rule that excluded each one.
- **SC-06 did genuinely fail, on `lgbm-extended`**, which is what settled the candidate. It cannot be demonstrated on the synthetic fixture, whose generator derives price from a formula that ignores the optional columns, so the test suite drives it with a constructed degrading model instead.

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

Measured with CodeCarbon and logged to MLflow next to the accuracy of each run (NFR-10), which issue #38 implements; the tracked runs carry their fit time but no emissions figure yet.

| | |
|---|---|
| **Hardware type** | Laptop CPU for training, the course VM's CPU for serving. No GPU anywhere (NFR-04). |
| **Hours used** | Target: at most 15 minutes per training run of the chosen configuration, without hyperparameter search (NFR-10). Measured on the tracked run of 2026-10-05: 8.6 s of fitting for the candidate `lgbm-basic`, 36.4 s for the whole ladder. Fit time, and so energy, is linear in the boosting rounds: letting early stopping decide them (EDN-70) costs the candidate 1.3x the rounds for 0.02 pp of MdAPE. |
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
- EDN-48: `split` records the supported-make list and the stages that build model input apply it.
- EDN-67: `features` applies the supported-make list to every frame, the `ES` holdout included, before the vocabulary is built.
- EDN-49: no bias correction on the log-to-euro inverse transform.
- EDN-50: one-hot encoding for the Ridge baseline only, against EDN-02's "no one-hot".
- EDN-51: mean fill plus a per-feature missingness indicator for the Ridge numerics, which is not imputation.
- EDN-70: `n_estimators` is a ceiling of 5,000 that early stopping stays under, not a tree count.

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
