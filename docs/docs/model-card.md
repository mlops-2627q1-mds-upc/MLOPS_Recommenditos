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

**Status: the four ladder variants train, all six success criteria are measured for the candidate, it passes NFR-01's gate with SC-04 waived, and nothing is released yet.**
The training procedure, the point metrics, SC-01 to SC-06 and the UC2 price range below are measured; SC-04 fails for every variant and is waived by EDN-62, which accepted the miss, so the gate passes for `lgbm-basic` alone, and the serving sections on latency and image size are still a plan; the explanations are served (FR-08).
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
  All four variants of the ladder fit, and all six success criteria are measured for the candidate: SC-01 to SC-03 pass for both LightGBM variants, SC-04 fails for all four on cars over 20 years and the miss is an accepted limitation (EDN-62, [Bias, Risks, and Limitations](#bias-risks-and-limitations)), SC-06 passes for the candidate and fails for `lgbm-extended`, and SC-05 passes for the candidate, the one variant that carries the UC2 price range, and is not measured for the others. NFR-01's gate, with SC-04 waived by EDN-62, therefore passes for the candidate alone. Every number below comes from the pipeline run the committed `dvc.lock` records, on the real snapshot, with one MLflow run per variant ([Experiment tracking](#experiment-tracking) says how to find them).

### Model Sources

- **Repository:** <https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos>
- **Training data:** [dataset card](dataset-card.md)
- **Experiment tracking:** MLflow on DagsHub, from Milestone 2 (**[decided]**, EDN-19, confirmed by the full team on 2026-10-13).
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
  Four is the count at the committed split: the test share is pinned at 20 % in `params.yaml` so that Suzuki (824 test rows) and Volvo (714) stay above the 500, and below a test share of about 15 % Volvo would join the four, and Suzuki below 13 %.
  Their rows still count towards the pooled criteria SC-01 to SC-03 and SC-05 to SC-06, but no criterion isolates them, so a failure specific to one of these makes cannot surface as a missed criterion.
  For these makes the per-segment half of the quality gate is blind by construction.
  Closing that gap, by raising the support threshold, lowering SC-04's segment size, or naming these makes as low-confidence in the response, is an open task for Milestone 3.
- **Classic and old cars: an accepted limitation.**
  The candidate misses SC-04 for cars older than 20 years: its MdAPE there is 16.80 % over 784 test rows against the 15 % bound, and 18.2 % on average over 20 other splits (standard deviation 0.69 pp, range 17.10 % to 19.81 %), so the committed figure is the most favourable of 21 draws.
  The team accepted the miss and reports the criterion as missed (EDN-62), because an upper age bound would pass SC-04 by removing the rows it fails on and a higher threshold would fit the criterion to its result: the 15 % bound, the scope and the age breakdown stay as they are, and NFR-01's gate reports SC-04 as missed and waived by EDN-62 (`gate_waived` in `metrics.json`) rather than blocking on it.
  The extended features make the segment worse (18.45 %), the hyperparameter search kept the committed values after 34 fits (EDN-73), and the segment is over the bound on 20 of 20 other splits (EDN-32).
  What a user must know: an estimate for a car older than 20 years is less reliable than for a younger one, with half of these estimates off by more than 16.80 % of the listing price against 6.73 % over all cars, and 56.0 % of them within 20 % of it against 90.0 %.
  The limitation is accepted for this model and this delivery, and a scope change of the kind EDN-04 made for price or a segment-specific model would be a new decision.
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
  83.3 % of listings come from dealers, and a few large ones can dominate a segment: one dealer alone holds 25.2 % of the Mercedes-Benz listings ([measured](#how-comparable-the-sets-are-and-how-much-a-figure-moves-with-the-split)).
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
What "missing" means for a field whose domain already represents absence is not left to the caller: `mask_absent` is the one implementation of it, and it is the one the API, SC-06's masking sweep, the blanked copies of a fit (EDN-82) and the tests all go through.
An omitted assertion flag is `False`, which is all the source can mean (EDN-23), and an omitted equipment list is the **empty** list, so every item of it is `False` too.
That second one is a decision rather than a reading of the data, and a distributional one: the raw field is always a list, so across the 105,405 scoped listings there is not one null in any of the four equipment fields, while the literal `[]` is 5.2 % to 8.0 % of them per field.
"Empty" is therefore a state the model was fitted on and "absent" is one it never saw (EDN-61).
The frame handed in is never modified.
`model.predict_log_price(frame)` is the log of the same number.
`model.predict_interval_eur(frame)` returns UC2's nominal 90 % range for a bundle that `has_intervals`, today `lgbm-basic`: `lower_eur`, `upper_eur` and `median_eur`, the last from the 50 % quantile model (see [Price ranges](#price-ranges-uc2)).

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
   The pre-registered filter reads a flag whose `False` also means "not stated", so a residual number of unmarked pre-registered listings stays in the data, knowingly (EDN-04).
2. **Drop impossible rows.** Listings registered after the 2025-11-08 reference date would have a negative age and are treated as data errors.
   Prices outside 500 EUR to 2M EUR are dropped as errors or collector outliers.
3. **Deduplicate**, before anything is split off, on a composite key rather than on `vin`, which is only 34 % filled ([dataset card](dataset-card.md#known-data-issues)).
4. **Hold out `ES`** as the new-market drift set (EDN-03).
   It never reaches training, validation or calibration.
5. **Restrict to supported makes** (EDN-05), computed by `split` from the cleaned data and applied by `features` to every frame, the `ES` holdout included, before any level or column is decided.
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
What the training split decides is measured on 2026-10-05 by [`reports/analysis/served_vocabulary.py`](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/analysis/served_vocabulary.py) over the **60,378 training rows of the supported makes**, which is the population the vocabulary is fitted on (EDN-05); the script builds it over all 61,180 training rows too, and that figure is given beside it.
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
These figures were measured on 2026-09-30 over all training rows; the restriction to the supported makes leaves the per-model merges as they were, because every make named here is supported, and takes one level off the total.
This is accepted as a parameterised starting point; the archived entry EDN-41 in the [EDN archive](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn-archive.md) records the two finer alternatives that were measured and rejected.

`weight_kg` runs from 1 kg to 93,000 kg, so the column carries data-entry errors from the source.
The pipeline deliberately does not clean them in the feature code, where a value range does not belong.
The Great Expectations suites (issue #25) do not bound it either: the column is still text when they run, and no plausible range has been agreed, so the errors reach the extended feature set as they are.

#### Training

The `train` stage fits one variant of the [experiment ladder](project-brief.md#4-modelling-plan) per `dvc.yaml` stage, so `dvc repro train@lgbm-basic` retrains that variant alone.
Every variant is fitted only on the makes the API serves: `split` records them in `data/processed/supported_makes.json`, `features` restricts every frame to them before it builds the vocabulary, and `train` refuses a matrix holding any other make rather than filtering it again (EDN-05).
So no make the API refuses with a 422 is in the model's training data, and none decided the levels and columns the model is encoded in.

- **Target:** `log(price)`, predictions transformed back with `exp`.
  **No bias correction** is applied to the inverse transform.
  `exp(E[log P | x])` is the conditional median, which is what every metric here reports, and a Duan smearing correction targets the conditional mean instead, so it biases every prediction upward and away from the typical asking price a seller wants.
  Measured on the real snapshot when the decision was taken, with the LightGBM variants still at the former 1,000-tree budget, it makes every variant worse: smearing factor 1.0258 and MdAPE 9.52 % to 9.83 % for `b1`, 1.0052 and 6.83 % to 6.86 % for `lgbm-basic`, 1.0028 and 6.32 % to 6.37 % for `lgbm-extended`.
  `b0` needs no inverse transform at all, because it is fitted on `price` directly.
- **Validation set:** early stopping and hyperparameter tuning.
  **Calibration set:** UC2 interval calibration only, never tuning.
- **The prediction is bounded to the training price range.**
  `predict_eur` clips in log space before the exponential, to the observed minimum and maximum `price` of its own training rows, which are recorded in the bundle.
  On the real snapshot that range is 500 to 1,814,750 EUR.
  This is what makes the interface's "finite and strictly positive" true of the arithmetic rather than of a hope: `exp` overflows to `inf` above about 710 in log space and underflows to exactly `0.0` below -746, and an `inf` prediction would turn MdAPE into `nan` and pass the gate's comparison unnoticed.
  It also catches a real extrapolation rather than only a theoretical one: over the 19,665 test rows, B1's unbounded prediction leaves the range **once**, at **10,635,538 EUR**, and that of `lgbm-basic` once at the lower edge, at 482 EUR for a car listed at 500 EUR, while B0 and `lgbm-extended` stay inside it (unbounded maxima 1,549,910 for B0, 1,117,900 for `lgbm-basic`, 800,374 for `lgbm-extended`; measured on the models of 2026-10-13).
  The bound is a guard rail and not a calibration, and it should not be read as one: it turns that 10.6 million into 1,814,750, which is the most expensive car the model was trained on rather than a sensible estimate for the car in question.
  Because the range is recorded in the bundle, the share of predictions sitting exactly on an edge can be reported instead of the clip hiding the pathology.

##### Hyperparameters as trained

All of these live in `params.yaml`, so a change reruns exactly the variant it affects.

| Variant | Ladder step | Estimator | Feature set | Parameters |
|---|---|---|---|---|
| `b0` | 1 | median of `price` per (make, model, 2-year age band), falling back to the make median and then the global median | basic | `age_bucket_years: 2` |
| `b1` | 2 | Ridge on `log(price)` | basic | `alpha: 1.0`, `min_category_rows: 5` |
| `lgbm-basic` | 3 | LightGBM | basic | `learning_rate: 0.05`, `num_leaves: 63`, `n_estimators: 5000`, `early_stopping_rounds: 50`; `blanked_copies` with `all: 1` and `random: 1` (EDN-82) |
| `lgbm-extended` | 4 | LightGBM | extended | as `lgbm-basic`, without `blanked_copies` |

Both LightGBM variants also fit with `min_child_samples: 20`, the fewest training rows a leaf may hold, and `reg_lambda: 0.0`, no L2 penalty on the leaf values, which are LightGBM's own defaults and which `params.yaml` does not name: the keys are optional, so leaving one out fits exactly the booster that setting it to its default fits, and it appears in `params.yaml` only if a search adopts another value (EDN-73).
The searches of issues #88 and #117 adopted none, so every hyperparameter in the table is the one the models were first fitted with, now measured against its neighbours on the validation split rather than assumed (see "How the values were found" below).
The searches of issues #64 and #88 fitted `lgbm-basic` on its training rows without the blanked copies. Two sweeps of 2026-10-17, `blanked-lgbm-basic-lr-leaves` and `blanked-lgbm-basic-min-child-samples`, searched it again with them and kept every value, and the `reg_lambda` sweep of issue #117 also fitted it with them, as `train` does (EDN-73, EDN-82).

The settings that are **not** parameters are modelling choices rather than knobs a sweep should touch, and they are the same for both LightGBM variants: `objective="regression"` (squared error on log price, because the log transform already handles the multiplicative error structure) and `metric="l1"` with `first_metric_only=True` for early stopping.
The absolute error in log space *is* the symmetric relative error, so the stopping point lines up with MdAPE, the metric the gate reads, rather than with a squared error nothing reports.
`n_estimators: 5000` is a ceiling and not a tree count: early stopping on the validation split decides how many trees the model keeps, and the booster is saved at that round, so the file *is* the model and no consumer has to remember an iteration argument (EDN-73).
On the real snapshot early stopping chooses round **1,394** for `lgbm-basic` and **1,647** for `lgbm-extended`, so the ceiling sits three times above the later of the two.
It used to be 1,000, and then it was the ceiling that ended both fits, not early stopping (issue #64).
5,000 rather than higher, because the ceiling is also the worst case of every cost that grows with the trees: a fit that used all of them would take about 80 s for `lgbm-extended`, write a booster of about 33 MB and spend about 105 ms on one row's SHAP export, against NFR-02's 200 ms for the whole of `/predict`.
`train` warns when the ceiling ends a fit, and `model.json` and the MLflow run record whether early stopping did (`early_stopped`, with the rounds run beside the trees kept), so a binding ceiling is a fact of the artefact rather than something read off a tree count.

Raising the ceiling is also what showed how closely the stopping metric tracks the gate's.
Measured on 2026-10-05, before `lgbm-basic` was trained on blanked copies: between 1,000 trees and the round early stopping chose, the validation L1 in log space keeps falling - by 0.2 % for `lgbm-basic` and 0.6 % for `lgbm-extended` - while validation MdAPE is flat or marginally worse, 6.947 % to 6.956 % and 6.448 % to 6.453 % (`reports/analysis/early_stopping_ceiling_results.txt`).
L1 is a mean and MdAPE a median, so the last few hundred trees improve the tail of the errors rather than the typical one; on the test split they moved MdAPE by 0.02 pp and 0.07 pp (see [the ladder](#the-experiment-ladder-as-measured)).

`learning_rate: 0.05` and `num_leaves: 63` were **searched and kept**: no point of the grids around them is measurably better on the validation split (EDN-73).
The first sweep (issue #64) covered the candidate alone.
A grid of `learning_rate` 0.025, 0.05 and 0.1 against `num_leaves` 31, 63, 127 and 255 for the candidate, every point with early stopping deciding its trees, is flat on the validation split: validation MdAPE spans 6.88 % to 7.04 %, and no point is measurably better than the committed one.
The closest, `learning_rate` 0.025, is 0.0003 lower in validation L1 with a per-point 95 % interval of -0.0009 to +0.0003 under a paired bootstrap by seller group, and would cost 1.6x the trees; the two metrics do not even agree on which point is best (`reports/analysis/tuning_sweep_results.txt`).
That holds simultaneously as well: widened by the max-statistic bootstrap so that the intervals of all eleven comparisons hold together, its interval is -0.0011 to +0.0005, and only two points, both at `learning_rate` 0.1, remain measurably worse.
Any tuning, the search of issue #88 included, follows the protocol in the [pipeline documentation](pipeline.md#tuning-a-hyperparameter): `recommenditos/modeling/tune.py` fits every point of a grid on the train and validation splits only, measures and tracks each fit in the MLflow experiment `recommenditos-price-tuning`, and adopts a value only when its simultaneous interval of the validation L1 difference lies below zero.

**How the values were found (issue #88, EDN-73).**
The first sweep (issue #64) covered `lgbm-basic` alone, so `lgbm-extended`'s `learning_rate` and `num_leaves` were the values it was first fitted with, carried over from the candidate rather than measured on its own feature set.
Issue #88 extends the search to both variants: EDN-73's grid of `learning_rate` and `num_leaves` for each, then a second grid of `min_child_samples` 5, 10, 20, 50 and 100 with the first grid's winner fixed, every fit measured by CodeCarbon and logged as its own run, under the same adoption rule.
The search ran on 2026-10-06, and all four sweeps kept their reference, so `params.yaml` is unchanged and the committed test metrics are those of the configuration it kept; the search never read the test split.
Every one of its 34 fits was ended by early stopping, not by the 20,000-tree ceiling.
Each sweep's record - the results, the readable table and CodeCarbon's row for every fit - is in the MLflow experiment `recommenditos-price-tuning` on DagsHub, as the `results/` artifacts of the run tagged `sweep_role` `summary` and `sweep` with the sweep's id below, not in the repository (EDN-73).

- **`lgbm-basic`, `learning_rate` x `num_leaves`: keep** (sweep `88-lgbm-basic-lr-leaves`).
  It reproduces EDN-73's twelve fits exactly, with the same best rounds, validation L1, validation MdAPE and intervals as `reports/analysis/tuning_sweep_results.txt`, which checks the search module against the recorded evidence.
  The lowest validation L1 is `(0.025, 63)` again, 0.00031 below the committed point with a simultaneous interval of -0.00109 to +0.00047; validation MdAPE spans 6.88 % to 7.04 %.
- **`lgbm-basic`, `min_child_samples`: keep 20** (sweep `88-lgbm-basic-min-child-samples`).
  The lowest validation L1 is 10, 0.00043 below 20 with a simultaneous interval of -0.00124 to +0.00038, and 100 is measurably worse; validation MdAPE spans 6.92 % to 7.03 %.
- **`lgbm-extended`, `learning_rate` x `num_leaves`: keep** (sweep `88-lgbm-extended-lr-leaves`), the first time this variant's values were measured on its own feature set.
  The lowest validation L1 is `(0.025, 127)`, 0.00040 below the committed point with a simultaneous interval of -0.00150 to +0.00070; validation MdAPE spans 6.25 % to 6.69 %, and all four points at `learning_rate` 0.1 and `(0.05, 31)` are measurably worse.
  `(0.025, 255)` shows what the rule guards against: it is 0.20 pp better in validation MdAPE, 6.25 % against 6.45 %, with a per-point interval of -0.385 % to -0.060 %, but MdAPE is not the selection metric, its L1 difference is unresolved even point by point (-0.00146 to +0.00069), and its MdAPE interval, widened to hold for all eleven points at once, spans zero (-0.427 % to +0.028 %).
  The best-looking of eleven noisy comparisons looks better than it is, which is the winner's curse.
- **`lgbm-extended`, `min_child_samples`: keep 20** (sweep `88-lgbm-extended-min-child-samples`).
  20 has the lowest validation L1 of the five values, no other value is resolved either way, and validation MdAPE spans 6.45 % to 6.49 %.

**L2 regularisation (issue #117, EDN-73).**
`reg_lambda` shrinks a leaf of n training rows towards zero by about n / (n + lambda), so it damps the predictions that rest on few listings, where the errors concentrate.
It was swept at 0, 1, 10, and 100 for each variant on 2026-10-20, under the same rule with one change: the interval is corrected across all sweeps decided on this validation split (Bonferroni), and these were the eighth and ninth, so each held at 1 - 0.05/9, about 99.4 %.
Its 8 fits took 339 s and 2.25 Wh, and the test split was not read.

- **`lgbm-basic`, `reg_lambda`: keep 0** (sweep `117-lgbm-basic-reg-lambda`).
  The lowest validation L1 is 1, 0.00001 below 0 with a simultaneous interval of -0.00089 to +0.00086. 10 and 100 are worse but not measurably, and validation MdAPE spans 7.00 % to 7.05 %.
- **`lgbm-extended`, `reg_lambda`: keep 0** (sweep `117-lgbm-extended-reg-lambda`).
  0 has the lowest validation L1, 1 is 0.00014 above it with a simultaneous interval of -0.00090 to +0.00118, and validation MdAPE spans 6.44 % to 6.52 %.

So no neighbour of the committed values is measurably better, for either variant, and these values are the search's result, not a default nobody checked.

B1's encoding is two decisions that look, at a glance, like rules of this project being broken, and both are recorded.
Its categoricals are **one-hot encoded**, which EDN-02 rules out for the tree models and which is confined to B1: Ridge is linear, has no native categorical handling, and dropping the categoricals would leave the depreciation baseline unable to tell a Porsche from a Dacia.
Its numerics are **mean-filled with a per-feature missingness indicator**, which is not imputation in the sense of EDN-15: with an indicator for every feature the encoding is information-preserving, so the fill is a numerically neutral placeholder rather than a guess at the value.
`min_category_rows: 5` groups the levels below that many training rows into one shared column, which is also where an unseen level goes at serving time where such a group exists (EDN-18).
Folding is not free: measured on the real snapshot, no folding gives 503 encoded columns at 9.48 % MdAPE, 5 gives 404 at 9.52 %, 30 gives 294 at 9.71 % and 100 gives 187 at 10.61 %.
What the 0.04 pp buys is that a level seen in a handful of rows does not get a coefficient fitted on them, and 47 of the 100 levels folded at 5 are `model` values seen exactly once.
At 5 only `model` and `fuel_category` have an infrequent group at all; the other six categoricals encode an unseen or absent value as all zeros, which answers just as well and is measured per column in the archived entry EDN-50 of the [EDN archive](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn-archive.md).

##### Determinism

- **Seed:** `params.yaml`'s project-wide `seed`, passed to LightGBM as `random_state` so `feature_fraction_seed`, `bagging_seed` and `data_random_seed` all derive from it.
  Worth being exact about: with `feature_fraction` and `bagging_fraction` at their defaults of 1.0, **none of the four estimators consumes randomness at all** today.
  The seed is recorded for provenance and to make a future subsampled configuration reproducible; it is inert as configured, so a test asserting that a different seed changes the result would fail.
- **Threads:** `train.num_threads: 1`, never LightGBM's default of every core.
  The trees are already thread-independent - identical at 1, 2, 4 and 8 threads - but LightGBM writes the count into `booster.txt`, so a default would make the artefact's hash depend on the machine that ran `dvc repro`.
  1 rather than a larger pinned value because only 1 is a value every machine can honour.
  It costs something, and against a specific alternative: measured on the real snapshot as medians of three fits on an idle 8-core machine, `lgbm-extended` fits in 18.3 s at 1 thread against 12.8 s at 2 and 12.2 s at 4, so a *pinned* 2 or 4 would buy back about a third of the fit.
  Against LightGBM's default of every core, which is what the pin actually replaces, it is a gain rather than a cost: 18.3 s against 29.2 s at 8 threads, and 7.1 s against 17.7 s for `lgbm-basic`.
  It is also the steadiest value by a long way - three fits at 1 thread land within a second of each other, three at 8 span 26 s - so it is what makes a reported fit time mean anything.
  The same reversal is sharper on the 744-row test fixture (0.33 s at 1 thread against 28.6 s at 8), because OpenMP's overhead dominates whenever there is little work per thread - and the suite fits every variant on that fixture.
  These seconds are a controlled sweep for the ratios, not comparable with the ladder table's one-off wall-clock fit times below.
  The setting pins LightGBM's boosting and nothing else in the span `train` measures, so the measured fit uses more CPU time than wall time (issue #113, `reports/analysis/fit_cpu_threads.py`, 2026-10-22).
  The boosting of the point model and of the price range's quantile models, and MAPIE's calibration, run on one thread: at most 0.08 s of CPU time on any other thread in any of them.
  The training and validation L1 that `fit_variant` writes into the model record are predicted with `Booster.predict`, which ignores the setting and uses LightGBM's default of one OpenMP thread per core: 16.2 s of CPU time on other threads in 2.5 s for `lgbm-extended`, and 10.0 s in 1.5 s for `lgbm-basic`.
  Those threads do useful work and change no number: at `OMP_NUM_THREADS=1` the two L1 are identical, and the step takes 15.4 s and 9.5 s instead.
  `b1` runs on OpenBLAS's pool of one thread per core, in the Ridge solver and in the predictions, and gains no time from it (the Ridge item below says what it changes): 4.2 s of CPU time on other threads in a span of 1.32 s, against 1.09 s at `OPENBLAS_NUM_THREADS=1`.
  CodeCarbon's own sampling adds at most about 1 % of CPU time: the threads that ended inside the span used 1.78 s with the tracker against 0.03 s without it for `lgbm-basic` (1.0 %), and 0.28 s against 0.05 s for `lgbm-extended` (0.5 %).
  The measured CPU time over wall time is therefore 4.2 for `b1`, 1.07 for `lgbm-basic`, whose span is mostly its single-threaded price range, and 1.65 for `lgbm-extended`; with both variables at 1 it is 1.00 for all three.
- **LightGBM:** `deterministic=True` and `force_row_wise=True`, so the histogram construction is not chosen by data size and thread count.
  Documented insurance rather than a measured fix, and worth being exact about: with both flags **off** the trees are still identical across 1, 2, 4 and 8 threads, so the flags are not what makes the result thread-independent here.
  They do change *which* trees are built, so they are not without consequence; they are simply not load-bearing for this property at this data size.
- **Ridge:** `solver="lsqr"`, scipy's iterative solver, rather than the default `auto`, which may pick a dense solver.
  It does not make `b1` thread-independent: LSQR's vector operations go through OpenBLAS, whose pool has one thread per core unless `OPENBLAS_NUM_THREADS` says otherwise, and `train.num_threads` does not reach it.
  Fitted in separate processes at 1, 2, 4 and 8 OpenBLAS threads, the coefficients differ at every count, while two fits at 8 are byte-identical and equal the committed `models/b1` (`reports/analysis/ridge_blas_threads.py`, 2026-10-22).
  So `b1`'s `pipeline.joblib`, and its hash in `dvc.lock`, depend on the core count of the machine that fits it; its validation L1 moves by at most 0.00002 between the counts.
- **B0:** group medians only, and a median is order-independent, so the result does not depend on the row order Parquet hands back.
- What holds, and is asserted by `tests/test_model.py`: two fits of the same variant on the same rows give bit-identical predictions and a byte-identical payload file.
  `model.json` is deliberately excluded, because it carries `trained_at` and the MLflow run id, which are provenance and cannot be stable.
  What is **not** promised is bit-identical results across different LightGBM, scikit-learn or NumPy builds or across CPU architectures; NFR-06's "within 0.1 percentage points" is the promise that survives a toolchain change.

##### What a trained variant writes

`models/<variant>/`, one DVC-tracked directory per variant:

| File | Contents |
|---|---|
| `model.json` | The record: the estimator, the feature and target lists separately, the hyperparameters as trained, the blanked copies the fit added and the fields they omit (`blanked_copies`), the training row counts (`n_train_rows` for the listings, `n_fit_rows` with their copies) and price range, the supported makes, the library versions and the MLflow run id. For the LightGBM variants also the trees kept (`best_iteration`), the rounds the fit ran (`boosting_rounds`) and whether early stopping or the `n_estimators` ceiling ended it (`early_stopped`); the three are null for `b0` and `b1`. |
| `feature_space.json` | A copy of the `features` stage's own artefact: the contract the matrices were written against and the vocabulary the training rows decided. |
| `booster.txt` | LightGBM variants: the native text format, saved at the early-stopped iteration. Not a pickle of the sklearn wrapper, because the text format survives a LightGBM upgrade, it is readable, and EDN-11's SHAP export needs a `Booster`. |
| `pipeline.joblib` | `b1` only: the fitted scikit-learn pipeline. |
| `lookup.parquet` | `b0` only: the whole model as one table a person can read, in EUR, with the row count behind each median. |
| `interval_{lower,upper,median}.txt` | A variant with a price range, today `lgbm-basic`: the 5 %, 95 % and 50 % quantile boosters, in the same text format. `model.json` then holds their record under `intervals`: the quantiles, the copies they were fitted on, the trees each kept, the two conformity offsets MAPIE computed and the calibration rows behind them (EDN-85). |

The feature space travels *with* the model rather than being looked up beside the matrices, because the bundle is baked into the API image on its own (EDN-08) and a code is a level's position: a model that encoded a request against one level order and scored it against another would be wrong with no error anywhere.
`recommenditos/modeling/model.py` is the only code that opens any of these files.

##### Experiment tracking

One MLflow experiment, `params.yaml`'s `train.mlflow_experiment` (`recommenditos-price`), and **one run per variant**, named after the variant.
`train` creates the run and records its id in `model.json`; `evaluate` resumes that id and appends the test metrics and the gate verdict rather than opening a run of its own.
That is deliberate: a run per DVC stage would scatter four variants over eight runs nothing joins, and the point of tracking is to be able to compare them.

Each run therefore carries the hyperparameters, the train and validation L1 in log space, the fit time, the energy and emissions of the fit, for the LightGBM variants the trees kept, the rounds run and `early_stopped` as 1 or 0 (so `metrics.early_stopped = 0` finds every fit the ceiling ended), the model artefact under `model/` and the test metrics with the six criteria.
Its parameters are every `params.yaml` key `dvc.yaml` declares for the `train` stage - the seed, `num_threads`, the estimator, the feature set, the blanked copies per kind (zero for a variant that names none), the request contract and the hyperparameters, the last under the estimator's name - plus the variant's name and the shape of the training data; the experiment is the one declared key recorded as the run's experiment rather than as a parameter (NFR-14).
The energy metrics are `energy_kwh` (with its `cpu_energy_kwh` and `ram_energy_kwh` halves), `emissions_kg_co2eq`, `fit_seconds` and `fit_cpu_seconds`, and the parameters prefixed `codecarbon.` say how they were produced, including the CPU power method: those are facts about the machine rather than `params.yaml` keys, except `train.energy`'s three settings, which are declared. See [Environmental Impact](#environmental-impact) for what the figure is.
`train` logs **no metric in euros**: it must not touch the test set, and a train-set MdAPE would be a second implementation of the metric beside `evaluate`'s, so one run could carry two numbers that disagree.

Every run is tagged with `variant`, `estimator`, `feature_set` and `dvc_stage`, and - by the tracking seam, for NFR-14 - with what produced it: `git_commit` and `git_dirty` for the fit, one `train.deps.<path>` per dependency of its `train` stage holding the hash `dvc.lock` records for it, and the same for the evaluation that appended to it, `evaluate.git_commit`, `evaluate.git_dirty` and `evaluate.deps.<path>`.
[The DVC pipeline](pipeline.md#from-a-run-to-its-inputs) says what each tag identifies and how to get from a run back to its code, data and parameters.
The ladder this card reports is four runs of the `recommenditos-price` experiment, one per variant, produced by the `dvc repro` whose lock is committed.
They were fitted on 2026-10-13 at commit `30c1e6c`, on the branch `feature/train-with-blanked-copies`, and the run id in each `models/<variant>/model.json` names them.
The comparable view of one pipeline state is the experiment's own table filtered to that state's commit:

```
tags.git_commit = '<the full SHA of the run you want>'
```

with the columns `tags.variant`, `params.estimator`, `params.feature_set`, `metrics.mdape`, `metrics.within_20pct`, `metrics.fit_seconds` and `metrics.energy_kwh`.
That filter is what picks one chain out of the experiment: the four runs of a `dvc repro` share one commit, and a run from another commit is a different pipeline.
Two things about it are worth knowing before the filter is believed, both checked on this ladder rather than assumed:

- **The commit is the one the pipeline ran from, not the one that holds its lock.**
  `dvc repro` runs at a commit and its `dvc.lock` and metrics are committed after it, so the runs carry the commit they were made from, which is an earlier commit of the same branch than the one that records them: `30c1e6c` for this ladder, the commit before `64bca73`, which holds its lock.
  `tags.git_dirty` is `false` for all four: what DVC writes while it runs does not count as a change.
  That commit is on the branch `feature/train-with-blanked-copies` and not on `main`, so after the squash merge it is reached through that branch's pull request (`git fetch origin pull/121/head`).
- **The input tags are what ties a run to the committed lock.**
  For each of the four runs, every `train.deps.<path>` equals the hash the committed `dvc.lock` records for that dependency of `train@<variant>`, and every `evaluate.deps.<path>` the one under `evaluate`; `reports/analysis/run_provenance_check.py` checks that against the server, and its output is committed beside it.
  Runs from before 2026-10-05 carry a `dvc_lock_md5` tag instead, which never equals the lock that records them, and three of the four runs of the last ladder made before that date were tagged dirty only because DVC had rewritten `dvc.lock` under them.

The experiment also holds runs that are **not** pipeline runs: the test suite trains the same four variants on the synthetic fixture, and it logs to this experiment whenever a `.env` is present, under the same four run names.
Their metrics are an order of magnitude worse and they carry a different commit; they also carry no `train.deps.<path>` tags, because DVC did not run them, so since 2026-10-05 a run without those tags is not a pipeline run and its numbers are not the model's.
A pipeline run is not always a committed one either: the experiment keeps the ladders of earlier pipeline states, among them two passes of 2026-10-12 at commits `96875d0` and `8301e07` of pull request #109's branch, so the commit above and the run ids in `models/<variant>/model.json` are what name the committed chain.

Training does **not** require credentials or a network.
With no `MLFLOW_TRACKING_URI` the stage logs one line, writes the model normally and records `mlflow.tracking_mode: "disabled"`, which is what lets CI and a fresh clone run the test suite; it never falls back to a local store, because since MLflow 3.16 that would mean a SQLite database in the repository root.
`RECOMMENDITOS_REQUIRE_TRACKING=1` turns any tracking failure into a failed stage, and that is the setting to use for the runs whose numbers are cited, so that a silent skip is not discovered while the report is being written.
- **Intervals (UC2) [decided, EDN-85]:** Conformalized Quantile Regression with MAPIE 1.6 on quantile LightGBM models at 5 %, 95 % and 50 %, fitted with `lgbm-basic`'s hyperparameters on its training rows plus one copy with every field but `make` omitted and one with a random subset omitted, so that partial inputs produce wider intervals.
  MAPIE computes the two conformity offsets on the 8,987 calibration rows with copies drawn the same way, in the `train` stage only; the bundle stores the boosters and the offsets, and serving adds them up without MAPIE.
  The coverage guarantee is marginal, that is on average over all inputs, not per missing-field pattern.
- **Point model and missing fields [decided, EDN-82]:** `lgbm-basic` is fitted on its 60,378 training rows plus two copies of each, 181,134 rows in all: one copy with all six optional input fields set to what an omitted field is, and one with a random subset of them omitted.
  A request carrying only the fields FR-01 requires was almost absent from the training rows (324 of 60,378 have all six optional fields empty), and LightGBM's native missing handling answered it at 8.57 % validation MdAPE; with the copies it is 7.19 %, and a complete request is not measurably worse (`reports/analysis/blanked_fields_results.txt`).
  `lgbm-extended` is trained without copies and does not become the candidate: with them it was worse than `lgbm-basic` as then trained on a request giving the basic set's six optional fields and no others (7.12 % against 6.96 %), which the rule fixed before the measurement excludes.
  That measurement is one validation split and one fit per configuration.
  @lukas2510, the repository owner, adopted this on 2026-10-13 on that measurement, and the test split was read once, for the adopted configuration ([Results](#results)).
  SC-06 bounds the outcome, and NFR-01's gate enforces it.
- **Comparables:** a filtered lookup over the processed training and validation listings, matching make and model within 2 years of age and 25 % of mileage, **not** a learned nearest-neighbour model.
  The `comparables` stage writes the index, 70,172 listings in 1.87 MiB, and the API image carries it beside the bundle ([API docs](api.md#comparables)).
  The filters are never relaxed to fill the list ([specification](specification.md) FR-09).

#### Speeds, Sizes, Times

Targets are set by the [requirements](requirements.md#2-non-functional-requirements); measured values replace them after Milestone 3.
The latency, image and memory targets are **[proposed]** (NFR-02 to NFR-04); NFR-10's training-time target is agreed.

| Property | Target | Measured |
|---|---|---|
| Training time, chosen configuration, hyperparameter search not included | at most 15 minutes on a laptop CPU (NFR-10) | **176.8 s** for the candidate `lgbm-basic` in the tracked run of 2026-10-20, of which 158.6 s are UC2's price range (three quantile models over 181,134 rows and MAPIE's calibration, `interval_fit_seconds` in MLflow) and the rest the point model; 26.5 s for `lgbm-extended` and 205.3 s for the whole four-variant ladder, at `num_threads: 1` on the real snapshot. Before the price range, the tracked run of 2026-10-13 fitted the point model alone in 19.0 s (1,444 boosting rounds over 181,134 rows, the listings and their blanked copies), `lgbm-extended` in 30.5 s (1,697 rounds) and the ladder in 51.6 s, on a laptop that other work kept busy. Read as an order of magnitude and not a benchmark: fits of the identical `lgbm-extended` between 2026-10-05 and 2026-10-13 took 27.1 s to 58.5 s, depending on what else held a core, which is the size of the effect. The worst case the `n_estimators: 5000` ceiling allows is about 80 s for `lgbm-extended` at the rate measured for the archived entry EDN-70, and about 90 s at this run's |
| Hyperparameter search | no budget: run once per tuning decision, outside `dvc repro`, and reported apart from the final fit (NFR-10, EDN-73) | **391.3 s** of fitting and 3.6071 Wh for the 34 fits of issue #88's search: 112.3 s and 1.0864 Wh over 17 fits for `lgbm-basic`, 279.1 s and 2.5207 Wh over 17 fits for `lgbm-extended`, at `num_threads: 1` on an i7-1360P laptop, so its fit times are not comparable with the i5-10210U figures above. Issue #117's `reg_lambda` search added 339.4 s and 2.2468 Wh over 8 fits on the same laptop (the `results/` record of each sweep's summary run in MLflow) |
| Model artefact on disk | no target | 8.7 MB (`booster.txt`, 1,394 trees) for the candidate's point model, plus 36.4 MB for its three quantile boosters (4.1, 3.8 and 28.5 MB for 612, 578 and 4,774 trees), 10.9 MB (1,647 trees) for `lgbm-extended`, plus 22 KB and 87 KB of metadata and feature space respectively. About 6.2 and 6.6 MB per 1,000 trees, so the ceiling's worst case is about 33 MB |
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
A model is released only if it meets all six ([requirements](requirements.md#2-non-functional-requirements) NFR-01), except a criterion an EDN entry accepted as missed and `params.yaml` lists under `evaluate.waived_criteria`, which the gate reports as waived; the gate in front of a promotion enforces it (FR-15).

### How comparable the sets are, and how much a figure moves with the split

The split assigns whole sellers by a hash of the seed and the seller key and stratifies on nothing, so the four sets are comparable only as far as the draw makes them.
**That was measured, and the split is kept as it is (EDN-32).**
What changed is the reporting: this section states the distribution of every set, and a figure of the committed split is given with its spread over 20 other splits.
It was enough to keep the split because the three things that could have argued against it did not show: the committed draw is better balanced than most, the pooled criteria hold on every split tried, and the one criterion that fails, SC-04, fails on every one of them.
A stratified-group split would have narrowed the largest make gap from 4.86 pp to 2.04 pp, and would have moved 52.6 % of the rows between sets as soon as the `ES` listings join the pool, which is the retrain FR-15 compares against the current model; the hash split moves none.
Choosing another seed for a better mix was rejected as well, since it is a new split picked by its outcome.

This is what the committed draw looks like, over every row the `split` stage wrote (the served makes are 98.6 % of them):

| | train | validation | calibration | test | pool |
|---|---|---|---|---|---|
| rows | 61,180 | 8,992 | 9,169 | 19,985 | 99,326 |
| seller groups | 16,907 | 2,810 | 2,848 | 5,870 | 28,435 |
| BMW | 33.73 % | 33.91 % | 37.66 % | 36.12 % | 34.59 % |
| Porsche | 22.47 % | 22.29 % | 23.71 % | 22.57 % | 22.59 % |
| Mercedes-Benz | 14.95 % | 14.28 % | 8.67 % | 11.11 % | 13.54 % |
| Audi | 12.27 % | 12.20 % | 11.80 % | 12.16 % | 12.20 % |
| median price (EUR) | 39,750 | 37,000 | 35,945 | 36,949 | 38,699 |
| older than 20 years | 3.98 % | 4.26 % | 4.36 % | 4.07 % | 4.06 % |

The other seven supported makes are within 0.9 pp of their pool share in every set, and the price quantiles from the 10th to the 90th are in [`split_strategy_results.txt`](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/analysis/split_strategy_results.txt), which holds every number of this section.

- **Mercedes-Benz is the one make that is visibly unbalanced, and one dealer is why.**
  The largest seller group holds 3,383 listings, 3.41 % of the pool, and 3,382 of them are Mercedes-Benz: a quarter (25.2 %) of all Mercedes-Benz rows belong to one dealer.
  It landed in train, and without it train's Mercedes-Benz share is 9.97 % and the pool's is 10.49 %, so the gap between train and the other sets is that dealer raising train rather than calibration being short.
  The remaining 1.8 pp in calibration is the next tier: of the 20 dealers with the most Mercedes-Benz listings, one landed in calibration, against ten in train, three in validation and six in test.
  No grouped split can balance this make, because that dealer has to go somewhere whole, and in a 10 % set it would make the set one third Mercedes-Benz.
- **The committed draw is an ordinary one, on the favourable side.**
  Over 1,000 seeds of the same assignment, the largest distance of any make's share in any set from its pool share is 9.07 pp on average and 6.30 pp at the median; the committed split is at 4.86 pp, and 72 % of the seeds are less balanced.
  What decides it is where the largest dealer lands: train in 61 % of the seeds (mean gap 5.09 pp), test in 19 % (10.30 pp), and validation or calibration in the remaining 20 % (20 pp).
  The calibration set's 8.67 % Mercedes-Benz is around the 21st percentile of its own distribution, and its median price at the 17th.
- **A classifier cannot tell train from test once sellers are held out.**
  One LightGBM classifier on the model's features and the price, asked to separate train rows from test rows, reaches an AUC of 0.495 on sellers it did not see.
  For sellers assigned at random that is nearly a tautology, so it confirms the assignment rather than adding evidence about it.
  The same classifier reaches 0.599 when rows are held out at random, which is the dealer signature a row-wise split would have leaked (EDN-14) and not a difference between the sets.

How much a reported figure depends on which sellers landed where was measured by repeating the split with seeds 1 to 20 and refitting the candidate `lgbm-basic` on each, through the `split` and `features` stages and the `train` stage's own fit, with the committed hyperparameters and model seed.
The seeds were fixed before any of them was run, and none of them is a candidate split: this measures noise, it does not select a seed.
The 20 splits share one pool of listings, so the spread describes how a figure moves with the assignment and is not a confidence interval for new data.

| `lgbm-basic`, test set | committed split | mean of 20 other splits | standard deviation | range |
|---|---|---|---|---|
| MdAPE (SC-01) | 6.73 % | 6.72 % | 0.26 pp | 6.22 % to 7.02 % |
| within 20 % (SC-02) | 89.99 % | 90.11 % | 0.70 pp | 89.08 % to 91.39 % |
| MdAPE, older than 20 years (SC-04) | 16.80 % | 18.18 % | 0.69 pp | 17.10 % to 19.81 % |
| test rows older than 20 years | 784 | 773 | 31 | 731 to 833 |

- **The verdict on the pooled criteria is stable.**
  SC-01 and SC-02 hold on all 20 splits, with the worst MdAPE 2.0 pp under the bound, and the committed 6.73 % is within a tenth of a standard deviation of the mean.
  Most of that spread is again the largest dealer: on the five splits that put it in test the MdAPE is 6.32 % on average, and on the other fifteen it is 6.85 %: that dealer's listings are then about 15 % of the test set, and the lower figure implies they are priced more accurately than the rest.
  The committed split has that dealer in train, so its figure is of the less flattering kind.
- **SC-04's miss on old cars is not an artefact of the draw.**
  The segment is over the 15 % bound on 20 of 20 splits, and it is the worst qualifying level on every one.
  The committed 16.80 % comes with a mean of 18.2 % over the 20 other splits and is lower than every one of them, the most favourable of 21 draws, so the figure this card reports understates the miss by about 1.4 pp against the typical split rather than overstating it.
- **A tuning difference has to be read against this.**
  The validation MdAPE of the committed hyperparameters has a standard deviation of 0.40 pp over the splits, and a range of 1.22 pp, against the 0.16 pp the whole grid of EDN-73 spans on the committed one.
  Compared on the same split the noise is much smaller: the grid's lowest-MdAPE point, 255 leaves, is lower than the committed 63 on 17 of 20 splits, by 0.08 pp on average, and higher in validation L1, the metric the tuning protocol decides by, on 9 of 20.
  A gain of that size is consistent in sign and a third of the noise of the test figure it would be reported in.

Two things this does not cover.
A temporal split would be the stronger test for a price model and is impossible, because the snapshot carries no listing dates (see [Bias, Risks, and Limitations](#bias-risks-and-limitations)).
Whether the UC2 intervals cover each make equally, Mercedes-Benz above all since the calibration set holds the least of it, is measured on the test set in [Coverage per make](#coverage-per-make).

### Results

The criteria are defined in the [problem specification](problem-spec.md#8-success-criteria) (EDN-06, with SC-06 added by EDN-15).
Both LightGBM variants are shown, because the choice between them is what the criteria decided: **`lgbm-basic` is the candidate this card puts forward** (EDN-62), not the variant with the lower pooled MdAPE.
The numbers come from the pipeline run described under [the ladder](#the-experiment-ladder-as-measured), which is the one the committed `dvc.lock` records and the one the four MLflow runs hold.
It is a **measured candidate, not a release**: the gate passes for it with SC-04 waived, and promotion is a human decision (EDN-12).

| ID | Criterion | Target | `lgbm-basic` (the candidate) | `lgbm-extended` |
|----|-----------|--------|------------------------------|-----------------|
| SC-01 | MdAPE | at most 9 % | 6.73 %, yes | 6.26 %, yes |
| SC-02 | Predictions within 20 % of the asking price | at least 85 % | 90.0 %, yes | 90.4 %, yes |
| SC-03 | MdAPE improvement over baseline B0 | at least 30 % lower | 44.6 % lower, yes | 48.6 % lower, yes |
| SC-04 | MdAPE per segment level with at least 500 test rows | at most 15 % | 16.80 % (`age_bucket=over 20`, 784 rows), **no** | 18.45 %, **no** |
| SC-05 | Empirical coverage of the nominal 90 % intervals, full inputs and partial scenario P1 | 88 % to 92 % | 90.2 % full, 91.0 % P1, yes | no price range, not measured, **no** |
| SC-06 | MdAPE with each optional field masked, and with all masked at once | at most 1.5x the full-input MdAPE | 1.02x, yes | **1.65x, no** |

These are the figures of the committed split: over 20 other splits the candidate's test MdAPE is 6.72 % on average (standard deviation 0.26 pp) and its MdAPE for cars older than 20 years is 18.2 %, and the SC-04 figure above is the most favourable of 21 draws ([How comparable the sets are, and how much a figure moves with the split](#how-comparable-the-sets-are-and-how-much-a-figure-moves-with-the-split)).

`b0` and `b1` miss SC-01, SC-02 and SC-03 as well, and are baselines rather than candidates; their per-criterion numbers are in `reports/metrics/<variant>.json`.

**`lgbm-basic` is the one deployable variant**: SC-04 fails for all four and is waived by EDN-62, `lgbm-extended` misses SC-06, `b0` and `b1` miss SC-01 to SC-03, and the three without a price range cannot meet SC-05.
`metrics.json` names it as `deployable_variant`, lists the waiver as `waived_criteria: sc04 (EDN-62)` and gives each variant's `gate_waived` with the measured SC-04 value, so the waiver is read beside the verdict; `best_variant` stays the lowest MdAPE overall, because that field means what it says.
The two failures are the interesting half of this card, and neither is a surprise:

- **SC-04 fails on old cars, for every variant, and the extended features do not close it.**
  The miss is an accepted limitation of this model (EDN-62), stated in full under [Bias, Risks, and Limitations](#bias-risks-and-limitations).
  The [problem specification](problem-spec.md#8-success-criteria) already recorded cars over 20 years at 15.3 % as the known risk, and the pipeline measures them worse than that: **16.80 % for the candidate** (18.2 % on average over 20 other splits) and 18.45 % for `lgbm-extended`, over 784 test rows.
  Letting early stopping decide the tree count (EDN-73) moved the candidate 0.35 pp closer to the bar and not under it, and the blanked copies of EDN-82 left it there, at 16.80 % after 16.77 %.
  The specification named two possible remedies, "the extended features or a scope change for classic cars", and measurement has **ruled the first one out**: the extended set makes this segment 1.6 pp worse while improving the pooled figure by 0.48 pp, so the extra features do not carry information about old cars.
  The hyperparameter search kept the committed values (EDN-73), so it leaves the figure where it is.
  That is a finding about the features and the model, not a reason to move the bar: the threshold stays at 15 %, no upper age bound is added, and the criterion is reported as missed.
- **SC-06 passes for the candidate and fails on `lgbm-extended`, and there only when every optional field is absent at once.** Each of the 23 optional fields masked on its own costs at most 9.7 % of the MdAPE (`equipment_extra`, 1.10x), which is comfortably inside the bound; all 23 together cost 64.6 % (6.26 % to 10.30 %). `lgbm-basic` passes at 1.02x (6.73 % to 6.86 %): it has six optional fields to lose rather than 23, and it is trained on copies of its rows with those fields omitted (EDN-82), without which it was at 1.21x. `b0` and `b1` pass at 1.00x and 1.16x.

    **This is what decided the candidate (EDN-62), and it compares the two variants as they are trained.** The two rows above describe the same request: only the ten fields FR-01 requires. On that request `lgbm-basic` answers at **6.86 %** and `lgbm-extended` at **10.30 %**, so the variant with the 0.48 pp pooled advantage is beaten by 3.44 pp as soon as the caller stops filling in the optional fields, which FR-01 says a caller need not do. Most of that gap is training and not what the extended features cost: `lgbm-basic` is fitted on blanked copies of its rows and `lgbm-extended` is not, and with the same copies `lgbm-extended` answered that request at 7.14 % on the validation split, against 10.79 % without them and 7.19 % for `lgbm-basic` (EDN-82). As it is trained, the extended model's advantage is conditional on a well-filled request, which is precisely the fragility EDN-15 added SC-06 to expose, and a criterion a candidate misses is not something to serve. The worst single field is also worth naming per variant, because it is not the same one: `equipment_extra` for `lgbm-extended`, `drive_train` for `lgbm-basic` at 1.01x, `cylinders_volume_cc` for `b0`, `nr_doors` for `b1` at 1.16x, which is above that variant's all-at-once ratio of 1.13x. SC-06 bounds the worst of both kinds of request, so a criterion measured only on the all-at-once scenario would have missed it.

#### Price ranges (UC2)

SC-05 is measured on the conformal intervals of the `lgbm-basic` bundle (EDN-85), on the 19,665 test rows.
The nominal 90 % range covers **90.2 %** of the prices for full inputs and **91.0 %** for P1, both inside the 88 % to 92 % band, so the criterion passes.
Its mean width, relative to the 50 % quantile model, is 0.46 for a full input and 0.93 for P1, so leaving out everything but make, model, registration date and mileage doubles the range, as FR-02 asks.
The guarantee is marginal over the mix of complete and partial requests the calibration copies make up, so the two scenarios landing in the band is a measurement, not a property of the method.
The variants without a price range report SC-05 as `not_measured`, which blocks the gate for them rather than passing it (NFR-01).

The point-estimate half of the partial-input scenario P1 is reported beside the SC-06 sweep: the candidate goes from 6.73 % MdAPE to 16.23 % (2.4x) when only `make`, `model`, `registration_date` and `mileage_km_raw` are given, `lgbm-extended` to 19.02 % (3.0x) and `b1` to 2.5x its full-input value, while `b0` is unchanged because it reads only three columns. A point estimate that is typically 16 % off is not something to serve as a number, which is exactly what UC2's interval answers instead.
The typical price `/price-range` reports beside the range is the point model's when the request carries every field FR-01 requires, and the 50 % quantile model's otherwise (FR-07, EDN-85): the point model has never seen a required field missing, and on the calibration split its MdAPE on a make-only request is 98.5 % against 40.1 % for the 50 % model, while on a full request it is 0.21 pp better than that model (`reports/analysis/price_range_point_estimate_results.txt`). P1 is reported beside the SC-06 sweep and is structurally excluded from it, because it masks required fields.


#### Coverage per make

SC-05 gates the pooled coverage only.
`evaluate` also writes the coverage and the mean relative width of the range per make and scenario to `reports/metrics/interval-coverage-by-make.csv`, with the number of test rows of each make, and nothing in that table enters the gate.
A make with fewer than `evaluate.success_criteria.sc05_min_make_rows` (500) test rows is reported and not judged, because the binomial standard error of a coverage over `n` rows at 90 % is `sqrt(0.9 * 0.1 / n)`, 1.3 points at 500 rows against a band of plus or minus 2 points, and 3.3 points at 81.
The standard error assumes independent rows, which the listings of one dealer are not, so it is a lower bound on the noise.
Measured on the 19,665 test rows for `lgbm-basic`, the only variant with a price range:

| Make | Test rows | Full inputs: coverage | Full inputs: mean width | P1: coverage | P1: mean width |
| --- | --- | --- | --- | --- | --- |
| All makes (SC-05) | 19,665 | 90.2 % | 0.46 | 91.0 % | 0.93 |
| BMW | 7,219 | 90.3 % | 0.46 | 91.2 % | 0.92 |
| Porsche | 4,510 | 90.8 % | 0.44 | 90.2 % | 0.90 |
| Audi | 2,431 | 90.4 % | 0.38 | **95.4 %** | 0.86 |
| Mercedes-Benz | 2,220 | 90.8 % | 0.35 | **93.9 %** | 0.92 |
| Alfa Romeo | 1,300 | 91.2 % | 0.59 | 90.5 % | 1.10 |
| Suzuki | 824 | **85.9 %** | 0.59 | **81.7 %** | 0.87 |
| Volvo | 714 | 90.2 % | 0.80 | **87.1 %** | 1.10 |
| Honda | 177 | 84.8 % (not judged) | 0.62 | 88.7 % (not judged) | 0.96 |
| Hyundai | 105 | 85.7 % (not judged) | 0.62 | 85.7 % (not judged) | 2.02 |
| Aston Martin | 84 | 86.9 % (not judged) | 0.76 | 79.8 % (not judged) | 1.33 |
| Volkswagen | 81 | 84.0 % (not judged) | 0.73 | 75.3 % (not judged) | 1.17 |

Bold marks a judged make outside 88 % to 92 %.
Five judged cells are outside the band, Suzuki in both scenarios and Audi, Mercedes-Benz and Volvo on P1, and the four makes under 500 rows sit between 75 % and 89 % with standard errors of 2.3 to 3.3 points, so they are neither cleared nor convicted.

- **Mercedes-Benz** is inside the band for full inputs (90.8 %, 0.8 points above nominal, 1.2 standard errors) and outside it for P1: 93.9 %, 3.9 points above nominal, 6.1 standard errors.
  The range is too wide for this make when little is known about the car, which is the safe direction for a buyer using it as a sanity check, and its mean width on P1 (0.92) is the pool's (0.93), so it is not a narrower range that was paid for with the extra coverage.
  Among the supported makes the calibration listings hold 8.85 % Mercedes-Benz against 11.29 % of the test rows ([the split analysis](#how-comparable-the-sets-are-and-how-much-a-figure-moves-with-the-split) has the pool-wide shares); this table shows the effect is a coverage that is high rather than low.
  It does not show that the make's share in calibration is the cause: Audi, with 12.0 % of the calibration listings against 12.4 % of the test rows among the supported makes, deviates further in the same direction.
- **Suzuki** is the make the pooled figure hides: 4.1 points below nominal for full inputs and 8.3 points below on P1, with 824 test rows, so the shortfall is well outside the noise.
- The pooled P1 figure of 91.0 % is the weighted result of over-coverage for Audi and Mercedes-Benz and under-coverage for Suzuki and Volvo.
  The pooled calibration stays as a stated limitation (EDN-89): per-make calibration was weighed and rejected.
  The range is calibrated on the pooled calibration set, and a reader of a Suzuki or Mercedes-Benz range on partial input should read the table above rather than the nominal 90 %.

#### The per-segment breakdown, and the fairness reading

Every variant is cut by `make`, `country_code`, `seller_type`, `fuel_category` and age bucket, which gives **36 levels**, of which **26** clear the 500-row minimum SC-04 needs.
Six price buckets are reported beside them and can never enter a criterion, so the table `evaluate` writes is 42 rows per variant and 168 across the ladder (`reports/metrics/segments.csv`).
Ten levels are reported and excluded: nine below the row minimum (`Honda` 177, `Hyundai` 105, `Aston Martin` 84, `Volkswagen` 81, `LU` 147, `Others` 69, `LPG` 44, `CNG` 10, `Ethanol` 3) and one, `fuel_category=(missing)` with 5 rows, because it is the absence of a required input field and FR-01 answers such a request with a 422.

The fairness statement is the spread of MdAPE across the levels of a segment, since classification-oriented parity metrics do not apply to a regression (problem specification section 8).
For the candidate `lgbm-basic`, with `lgbm-extended`'s spread beside it because the two agree on where the disparity is:

| Segment | Qualifying levels | Best level | Worst level | Spread | Spread, `lgbm-extended` |
|---|---|---|---|---|---|
| `make` | 7 | Mercedes-Benz 5.33 % | Volvo 9.06 % (714 rows) | 3.7 pp | 4.4 pp |
| `country_code` | 6 | DE 6.33 % | FR 8.08 % (1,058 rows) | 1.8 pp | 2.7 pp |
| `seller_type` | 2 | Dealer 6.36 % | PrivateSeller 8.82 % (3,868 rows) | 2.5 pp | 2.4 pp |
| `fuel_category` | 5 | Electric/Diesel 5.66 % | Diesel 6.87 % (6,348 rows) | 1.2 pp | 1.3 pp |
| `age_bucket` | 6 | 0-1 years 5.48 % | over 20 years 16.80 % (784 rows) | 11.3 pp | 13.6 pp |

`reports/figures/error-by-segment.pdf` draws the candidate's 26 qualifying levels against the bound, built by the `compare-segments` stage from the same record.

Read as a fairness check: the market segments are close to parity, and the one disparity that matters is **age**, not country or seller type.
A private seller is served about 2.5 pp worse than a dealer and a French listing 1.8 pp worse than a German one, both inside SC-04's bound; the owner of a car over 20 years old is served 11.3 pp worse than the owner of a new one and outside it.
The extended features widen four of the five spreads, `age_bucket`'s most, by 2.3 pp, and narrow `seller_type`'s by less than 0.1 pp. That is the same finding SC-04 reports from the other direction: they buy pooled accuracy without buying it evenly.
The price buckets are deliberately not part of this statement: their profile is U-shaped (for the candidate, 24.3 % under 5,000 EUR and 6.1 % over 80,000 EUR against 5.8 % in the 40,000 to 80,000 EUR bucket), which is the regression-to-the-mean artefact of conditioning on the target rather than a finding about a market.

#### The experiment ladder, as measured

One run of the whole chain on the real snapshot on 2026-10-05 (`download.source: zenodo`, 118,382 raw listings, 105,405 after the scope and deduplication funnel), fitted on the 11 supported makes and encoded in the vocabulary their training rows decided (EDN-05): 60,378 training rows, 8,859 validation rows and 19,665 test rows, which are the served rows of the 61,180, 8,992 and 19,985 in the split.
`lgbm-basic` is fitted on those training rows plus two blanked copies of each, 181,134 rows (EDN-82).
It is the run the committed `dvc.lock` records, so `dvc pull` reproduces these artefacts exactly, and it is tracked: one MLflow run per variant, found with the `tags.git_commit` filter under [Experiment tracking](#experiment-tracking).
Every number is from the test split, at `train.num_threads: 1`, and MdAPE is the primary metric.
The bold figures are the lowest of the ladder; they are not the candidate, which SC-06 decided against them (EDN-62).

| Variant | Ladder step | Features | MdAPE | Within 20 % | Validation L1 (log price) | Trees | Fit time | Payload |
|---|---|---|---|---|---|---|---|---|
| `b0` | 1 | 16 | 12.16 % | 70.5 % | 0.1769 | - | 0.1 s | 19 KB |
| `b1` | 2 | 16 | 9.52 % | 80.3 % | 0.1406 | - | 2.0 s | 19 KB |
| `lgbm-basic` **(the candidate)** | 3 | 16 | 6.73 % | 90.0 % | 0.1004 | 1,394 | 19.0 s | 8.7 MB |
| `lgbm-extended` | 4 | 162 | **6.26 %** | **90.4 %** | 0.0966 | 1,647 | 30.5 s | 10.9 MB |

What the ladder says, and what it does not:

- **What ladder step 4 measured, stated as what it now is: as the two variants are trained, the extended one buys pooled accuracy only when the caller fills in the optional fields, and loses more than it buys when they do not.** It is worth **0.48 pp** of MdAPE on a fully described car, and it is **3.44 pp worse** than `lgbm-basic` on a request carrying only the ten fields FR-01 requires (10.30 % against 6.86 %, from the SC-06 sweep). That second figure is not the price of the extended features: `lgbm-basic` is fitted on blanked copies of its rows and `lgbm-extended` is not, and trained with the same copies it was level with `lgbm-basic` on that request on the validation split, 7.14 % against 7.19 % (EDN-82). It also costs 146 extra columns, 1.6x the fit time and a 90-second `features` stage. So what the extended variant is worth, as it is trained, is conditional on the request, and for a component whose contract lets a caller omit 23 of its 33 inputs the conditional half is the one that decides: the candidate is `lgbm-basic` (EDN-62). Step 4 did its job by producing a number that could have gone either way, and the criteria, not the pooled figure, are what read it.
- B0 reproduces the exploratory reference run below almost exactly (12.16 % against 11.9 %), and `lgbm-basic` likewise (6.73 % against 6.7 %), which is the cross-check that the pipeline is measuring what the notebook measured.
- **Early stopping decides the trees now, and the trees the old budget withheld were worth almost nothing (EDN-73).** At `n_estimators: 1000` both LightGBM variants ran out of budget before early stopping fired.
  This was measured on 2026-10-05, before `lgbm-basic` was trained on blanked copies, so the candidate's figures in this bullet are those of the model without them.
  With the ceiling at 5,000, early stopping chose round 1,243 for `lgbm-basic` and 1,647 for `lgbm-extended`, and on the test split that moved MdAPE from 6.83 % to 6.81 % and from 6.32 % to 6.26 %, within 20 % by about 0.1 pp each, and no criterion's verdict for either variant.
  The training L1 falls much further than the validation L1 (0.0688 to 0.0655 against 0.1005 to 0.1003 for the candidate), so the extra trees mostly fit the training rows more closely.
  For `lgbm-extended` that comes with a larger dependence on its optional fields, SC-06 1.58x to 1.65x, and a worse old-car segment, 17.86 % to 18.45 %, while the candidate's old-car segment improves from 17.12 % to 16.77 %.
  What they cost is linear in the trees: 1.3x and 1.7x the boosting rounds and so the fit, 6.2 to 7.6 MB and 6.8 to 10.9 MB of booster, and one row's native SHAP export from 18.9 to 26.1 ms and from 22.3 to 35.4 ms.
  So the 1,000-tree models were a hair short of where their validation curve flattens, not far from it, and the pooled numbers this card used to quote were not a materially pessimistic bound.
  `learning_rate` and `num_leaves` were then swept for the candidate on the validation split and left where they were, because no point of the grid was measurably better (EDN-73), and issue #88's search, over both variants and `min_child_samples` as well, kept every value for the same reason (see [Hyperparameters as trained](#hyperparameters-as-trained)).
- The whole four-variant ladder trains in **51.6 seconds** of fitting, against NFR-10's 15-minute budget, so nothing about the budget constrains the tuning. The figure is worth an order of magnitude of slack and not one second of precision: other fits of `lgbm-extended` between 2026-10-05 and 2026-10-12, with identical trees, took 27.1 s to 58.5 s against 30.5 s here, because a fit at `num_threads: 1` competes with whatever else holds a core.

#### Reference values

A one-off exploratory run on 2026-09-22, not tracked in MLflow; the experiment ladder reproduces it.
Same scope as above without `ES`, deduplicated, 96,831 listings, 80/20 split grouped by seller.

| Model | MdAPE | Within 20 % |
|-------|-------|-------------|
| B0, median baseline | 11.9 % | 70.9 % |
| LightGBM, basic features, no tuning | 6.7 % | 90.6 % |

#### Summary

The ladder reproduces the reference run, and the first three criteria pass with margin for both LightGBM variants: the candidate `lgbm-basic` is at 6.73 % MdAPE against SC-01's 9 %, 90.0 % within 20 % against SC-02's 85 %, and 44.6 % below the median baseline against SC-03's 30 %.
SC-04 fails for every variant, and SC-05 passes for the candidate, whose price range covers 90.2 % of full inputs and 91.0 % of P1 requests.
The gate passes `lgbm-basic` with SC-04 waived, because the team accepted that miss (EDN-62); releasing it stays a person's decision in a reviewed pull request (EDN-12).
The chosen model is therefore `lgbm-basic` **as a candidate**, chosen over the lower pooled MdAPE of `lgbm-extended` because it is the one that meets SC-06 (EDN-62).
Early stopping decides its tree count (EDN-73), which turned out to be worth 0.02 pp, and its `learning_rate`, `num_leaves` and `min_child_samples` are the values it was first fitted with because the search around them found nothing the validation split can tell apart, for either LightGBM variant (EDN-73).
So the honest statement about it is not that it is tuned to an optimum but that the obvious knobs have been searched and kept, because none of them buys a resolvable gain.

Three risks going into the rest of Milestone 3, all three now measured rather than anticipated:

- **SC-04 is missed on cars older than 20 years, and the miss is an accepted limitation, not open work (EDN-62).**
  The candidate is at 16.80 % against the 15 % bound (18.2 % on average over 20 other splits), NFR-01's gate reports it as waived by EDN-62, and [Bias, Risks, and Limitations](#bias-risks-and-limitations) states it in full.
- The low-support makes SC-04 cannot see at all, because they do not reach its 500-row minimum: `Honda` at 177 test rows down to `Ethanol` at 3, ten levels in all, reported in `reports/metrics/segments.csv` with the rule that excluded each one.
- **SC-06 did genuinely fail, on `lgbm-extended`**, which is what settled the candidate. It cannot be demonstrated on the synthetic fixture, whose generator derives price from a formula that ignores the optional columns, so the test suite drives it with a constructed degrading model instead.

## Model Examination

Explanations are part of the product, not an afterthought: every valuation ships the properties that drove it (FR-08).

- **Per prediction:** the base price plus the five features with the largest absolute SHAP contribution, and the combined contribution of everything else.
  A feature is named by the request field it comes from, so the age is `registration_date` and an equipment list, many columns in the model, is one effect ([API docs](api.md#the-explanation)).
  Because the model predicts `log(price)`, a contribution is reported as its multiplicative effect on the price and the effects multiply rather than add.
  They are complete, so recomputing the estimate from the reported effects reproduces it, which is what the API test asserts.
- **How it is computed:** from the trained booster's own SHAP export (LightGBM's `pred_contrib`, CatBoost's `ShapValues`), **not** the `shap` package.
  `shap` pulls in `numba` and `llvmlite`, measured at 189 MB, merely to import, which a CPU-only serving image should not carry (EDN-11).
  A training-time test asserts the booster's export matches `shap.TreeExplainer` bit for bit; for LightGBM `shap` calls that same export, so the test also checks that each row's values add up to the model's log price.
- **Globally:** `shap` is used offline, in training and notebooks, for summary plots and global feature importance.
- **Behaviour under a changed input:** `tests/test_model_behaviour.py` asserts the direction of the three inputs whose effect on a price has a known sign, on aggregate, because boosted trees are not monotone row by row: the change has to move more predictions its way than the other.
  Measured on 2026-10-13 over the 8,859 validation rows, 20,000 km more lowers 92.1 % of the predictions of `lgbm-basic` and raises 7.4 %, and lowers 96.3 % and raises 3.3 % for `lgbm-extended`; a first registration one year earlier lowers 87.2 % and raises 12.8 %, and 92.7 % and 7.2 %; 20 kW more raises 78.7 % and lowers 19.7 %, and 77.3 % and 20.9 %.
  The median of the per-row ratios of the changed to the unchanged prediction is 0.952 and 0.952 for the mileage step, 0.950 and 0.953 for the year, and 1.043 and 1.027 for the power step.
  The same tests hold every prediction bit-identical when a column that is not a feature changes (`vin`, `seller_group_id`, `price`, `log_price`).
- **Behaviour under a negligible change:** a tree model is a step function, so a car standing on a split changes its price with one kilometre.
  10 km more moves 3.4 % of the 8,859 validation predictions of `lgbm-basic` by more than 1 % and 1.6 % of those of `lgbm-extended`, the largest move being 26 % and 15 %, and the test bounds that share at 5 %.
  A first registration one day earlier moves no prediction of either, because the snapshot's registration dates are a month apart and a day crosses no split, so the test asks that of the fixture-trained models only.
- **A car whose price is known:** requested with the required fields alone, a BMW 116, a BMW 320 and a Porsche 992 are priced at 0.81, 0.97 and 0.97 times the median price of the comparable training listings by `lgbm-basic`, and at 0.74, 0.94 and 1.10 times by `lgbm-extended`.
  The test bounds the factor at 1.5 either way.

## Environmental Impact

CodeCarbon records the energy and emissions of every fit, and only of the fit, into `reports/emissions/<variant>.csv` and into the variant's MLflow run beside its accuracy (NFR-10).
Since the price range (EDN-85), `lgbm-basic`'s fit includes its three quantile models and MAPIE's calibration: 0.592 Wh over 176.8 s in the tracked run of 2026-10-20, of which 158.6 s are the price range, against 0.098 Wh over 26.5 s for `lgbm-extended`, so `reports/figures/energy-vs-error.pdf` compares the candidate's UC1 and UC2 models together with `lgbm-extended`'s point model alone.
The figures below without a date are from the run of 2026-10-13, before the price range.
**The figure is an estimate, not a reading.**
RAPL, the CPU's energy counter, is root-only on Linux since kernel 5.10, so on every machine we train on CodeCarbon falls back to its `cpu_load` model: 10 W for the RAM plus the CPU's TDP times the share of its logical CPUs the fit kept busy.
On the i5-10210U the ladder is trained on, that is 15 W / 8 x the fit's CPU time plus 10 W x its wall time, so a single-threaded fit is charged 11.75 to 11.80 W for every second it runs, and the RAM constant is most of it.
The real fits are charged 13 to 15 W for every second in the ladder run of 2026-10-13, because parts of the measured span use more than one thread although `train.num_threads` is 1: the CPU time of the two LightGBM fits is 1.6 and 1.7 times their wall time, and that of `b1` 3.8 times.
The extra CPU time is the training and validation L1 the model record carries, which LightGBM predicts on one thread per core, and for `b1` OpenBLAS's threads behind the Ridge solver; [Determinism](#determinism) has the measurement.
The energies of two variants therefore follow their fit times closely (1.7 against 1.6 for `lgbm-extended` over `lgbm-basic`), which is a fair comparison of their cost when both come from one run on one machine, and nothing more: the same fit is charged more on a busy laptop, because it takes longer, and about 80 W per second on a GitHub runner, whose four vCPUs CodeCarbon gives a quarter of a 280 W server TDP each.
Emissions are energy times the Spanish grid's 174.05 g CO2eq/kWh, pinned rather than geolocated so that an identical fit has identical emissions wherever it runs.
Checked once against the RAPL counter on an idle laptop (`reports/analysis/rapl_validation.py`, 2026-10-06, at the committed hyperparameters, which the search of issue #88 later kept, and with `lgbm-basic` fitted without its blanked copies), with an idle package draw of 6.6 W and three fits of each variant.
The estimate's total is about 70 % of the counter's package energy for the three fitted variants and 112 % for `b0`, whose whole fit is about 1 J, and its CPU half is 27 % to 41 % of the energy above idle.
The ranking and the ratios survive: `lgbm-extended` costs 2.97 times `lgbm-basic` by the counter and 2.97 times by the estimate.
One variant's package energy varies by a factor of 1.3 to 2.1 between repeats, so only medians are quoted.

The table is the tracked ladder run of 2026-10-13 on the i5-10210U laptop (8 logical CPUs, 15 GB RAM).
Each fit time and energy is one measurement, on a laptop that other work shared: the committed table records a machine-wide CPU utilisation (`machine_cpu_utilization_pct`) of 22.8 % and 27.2 % during the two LightGBM fits.
Fit time and energy vary with that from run to run: `lgbm-extended`, whose trees are byte-identical to those of the run of 2026-10-12, took 30.53 s here and 42.37 s then.
`lgbm-basic`'s fit covers 181,134 rows, three times the 60,378 that `reports/figures/energy-vs-error.csv` gives as its `n_train_rows`, because that column counts the listings and not their blanked copies (EDN-82).
The hyperparameter search of issue #88 kept every value, so it is also the cost of the final configuration.

| Variant | Fit time | Energy | Emissions | MdAPE, every field given | MdAPE, only the fields FR-01 requires |
|---|---|---|---|---|---|
| `b0` | 0.09 s | 0.0003 Wh | 0.06 mg CO2eq | 12.2 % | 12.2 % |
| `b1` | 2.05 s | 0.0086 Wh | 1.49 mg CO2eq | 9.5 % | 10.7 % |
| `lgbm-basic` | 18.98 s | 0.0683 Wh | 11.88 mg CO2eq | 6.7 % | 6.9 % |
| `lgbm-extended` | 30.53 s | 0.1137 Wh | 19.80 mg CO2eq | 6.3 % | 10.3 % |

`reports/figures/energy-vs-error.pdf` draws the same rows, built by the `compare-energy` stage from the artefacts rather than by hand.
On this run `lgbm-extended` costs 1.7 times the energy of `lgbm-basic` for 0.48 percentage points of MdAPE, which is 0.05 Wh more in absolute terms and negligible on its own.
It earns that only when the user supplies the extra fields: given only the fields FR-01 requires it is worse than `lgbm-basic` (10.3 % against 6.9 %).
The whole ladder is 0.191 Wh, 33.2 mg CO2eq and 51.6 s of fitting, against NFR-10's 15 minutes.
The search of issue #88 changed no value, so these numbers and the conclusion stand for the final configuration.

The hyperparameter search of issue #88 is measured the same way, fit by fit, and kept apart from these figures: its CodeCarbon rows (`results/emissions.csv`) and its totals (the JSON record beside them under `results/`) are artifacts of each sweep's summary run in the MLflow experiment `recommenditos-price-tuning`, not files in the repository, because the table above is the cost of the models that were chosen, and a search is the cost of choosing them (NFR-10, EDN-73).
It ran on 2026-10-06 on another laptop, a 13th Gen Intel Core i7-1360P (16 logical CPUs) under Windows, with CodeCarbon 3.3.1's estimate and the same pinned 174.05 g CO2eq/kWh.

| Sweep | Fits | Fit time | Energy | Emissions |
|---|---|---|---|---|
| `88-lgbm-basic-lr-leaves` | 12 | 82.6 s | 0.7932 Wh | 138.05 mg CO2eq |
| `88-lgbm-basic-min-child-samples` | 5 | 29.7 s | 0.2932 Wh | 51.03 mg CO2eq |
| `lgbm-basic`, total | 17 | 112.3 s | 1.0864 Wh | 189.09 mg CO2eq |
| `88-lgbm-extended-lr-leaves` | 12 | 201.6 s | 1.8397 Wh | 320.20 mg CO2eq |
| `88-lgbm-extended-min-child-samples` | 5 | 77.5 s | 0.6810 Wh | 118.52 mg CO2eq |
| `lgbm-extended`, total | 17 | 279.1 s | 2.5207 Wh | 438.73 mg CO2eq |
| **Search, total** | **34** | **391.3 s** | **3.6071 Wh** | **627.81 mg CO2eq** |

On its own machine the search for `lgbm-basic` cost about 20 times one fit of the configuration it searched, which had no blanked copies (0.0549 Wh), and the search for `lgbm-extended` about 16 times one of its own (0.1584 Wh).
Set against the ladder's 0.191 Wh above, the whole search is about 19 times the final ladder, but that ratio crosses two machines, one of them busy with other work, so it is an order of magnitude and nothing finer: the same fit is charged differently on each, and the search's reference fit of `lgbm-extended` took 16.0 s against the ladder's 30.53 s.
Either way it is a one-off cost of choosing the values, paid once per tuning decision, while each retraining of the product pays only the final fit.

The `reg_lambda` search of issue #117 ran on 2026-10-20 on the same i7-1360P, measured and recorded the same way.
The laptop had little free memory, and its reference fit of `lgbm-extended` took 84.0 s against 16.0 s on 2026-10-06, so its fit times say more about the machine's load than about the fits.

| Sweep | Fits | Fit time | Energy | Emissions |
|---|---|---|---|---|
| `117-lgbm-basic-reg-lambda` | 4 | 77.3 s | 0.6248 Wh | 108.75 mg CO2eq |
| `117-lgbm-extended-reg-lambda` | 4 | 262.1 s | 1.6220 Wh | 282.30 mg CO2eq |
| **Search, total** | **8** | **339.4 s** | **2.2468 Wh** | **391.05 mg CO2eq** |

| | |
|---|---|
| **Hardware type** | Laptop CPU for training, the course VM's CPU for serving. No GPU anywhere (NFR-04). |
| **Hours used** | Target: at most 15 minutes per training run of the chosen configuration; a hyperparameter search is reported apart (NFR-10, EDN-73). Measured on the tracked run of 2026-10-13: 19.0 s of fitting for the candidate `lgbm-basic`, 51.6 s for the whole ladder; issue #88's search, outside the budget, took 391.3 s of fitting on another laptop. Fit time, and so energy, is linear in the boosting rounds: letting early stopping decide them cost the candidate 1.3x the rounds for 0.02 pp of MdAPE when EDN-73 measured it, before the blanked copies. |
| **Cloud provider** | None for training. Serving runs on the FIB Virtech VM provided by the course (EDN-17). |
| **Compute region** | Barcelona, Spain. |
| **Carbon emitted** | 0.191 Wh and 33.2 mg CO2eq for the whole ladder on 2026-10-13, measured while other work shared the laptop (0.216 Wh on 2026-10-12 and 0.157 Wh on 2026-10-06, when `lgbm-basic` was fitted without its blanked copies), with the hyperparameters the search kept, per variant in the table above, plus 3.6071 Wh and 627.81 mg CO2eq once for the search itself: CodeCarbon's estimate (`cpu_load`) against the Spanish grid. |

Serving energy is reported as an average per answer from the load test, not per individual request: CodeCarbon's granularity does not match single-digit-millisecond events, and a per-request tracker would eat into the latency budget of NFR-02.
Whether the VM exposes an energy counter is not known yet, so on the VM the figure is expected to be an estimate of the same kind.

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

- EDN-02: the model family is gradient boosting with LightGBM.
- EDN-03: the Spanish listings are held out as the new-market scenario.
- EDN-04: the model covers used passenger cars only.
- EDN-05: a make is supported from 300 listings.
- EDN-06: a model is deployable only when it meets the success criteria.
- EDN-08: the model is baked into the API image.
- EDN-11: serving explains predictions without the `shap` package.
- EDN-12: retraining and promotion are started by a person.
- EDN-15: `/predict` requires ten fields, with missing inputs bounded by SC-06.
- EDN-17: the component is deployed on the FIB Virtech VM.
- EDN-18: unseen countries and models are accepted with a warning.
- EDN-19: the DagsHub repository belongs to a team organisation.
- EDN-23: condition flags are read as one-sided assertions.
- EDN-32: the split is 60/10/10/20 by seller, with the snapshot date as seed.
- EDN-62: `lgbm-basic` is the candidate model.
- EDN-73: hyperparameters change only under a written tuning protocol.
- EDN-82: `lgbm-basic` is trained on blanked copies of its listings.
- EDN-85: serving adds the stored conformity offsets to the quantile models, without MAPIE, and a partial price-range request takes its typical price from their median.

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
