Problem specification
=====================

What the model learns, on which data, and when it is good enough.
This page owns the ML framing: the [requirements](requirements.md), the [specification](specification.md), the dataset card, the model card and the report link here instead of repeating it.
For the overall plan see the [project brief](project-brief.md); for facts about the data see the dataset card.

## 1. Problem statement

Private buyers and sellers of used cars want to know what a car is worth on the market without searching and comparing listings by hand.
We learn this from listings: given the description of a used car, the model predicts the **asking price** it would have on a European online car portal.

This is a supervised **regression** problem with two outputs:

- **UC1 (car valuation):** a point estimate of the asking price for a fully described car.
- **UC2 (purchase guidance):** a price interval (nominal 90 %) for a partially described car.

The model reflects the market of the data snapshot (AutoScout24, 2025-11-08).
It does not forecast future prices.

## 2. Scope

### In scope

- **Used passenger cars only:** `offer_type = U`, not pre-registered, `vehicle_type = Car` ([EDN-04](#decision-records)).
  The pre-registered filter can only read `is_preregistered`, whose `False` also covers "not stated", so a residual number of unmarked pre-registered listings stays in the training data ([EDN-04](#decision-records)).
- **Supported makes:** makes with at least **300 listings** in the cleaned used-car data, counted after removing the `ES` holdout and before the split ([EDN-05](#decision-records)).
  The pipeline computes this list; it is not hard-coded.
  With the current data these are 11 makes covering 98.6 % of the used listings: BMW, Porsche, Mercedes-Benz, Audi, Alfa Romeo, Suzuki, Volvo, Honda, Hyundai, Aston Martin and Volkswagen.
  The model is fitted and evaluated on these makes only, and every feature matrix - the `ES` holdout's included - holds no other, because the `features` stage restricts each frame before it decides any level or column ([EDN-05](#decision-records)).
  The split itself stays lossless, so the listings of the other makes are still in its frames.
- **Markets:** the model is trained on 7 countries (DE, IT, NL, BE, AT, FR, LU), with the country as a feature.
  All `ES` listings are held out as the new-market drift scenario ([EDN-03](#decision-records), project brief section 3.2); they join the training data only after the drift is confirmed and the model is retrained.
  The component accepts `ES` and treats it as an unknown country until then.
  A consequence worth stating, because it looks like a defect: `country_code` is encoded over the levels the training rows hold, so the holdout's own country is not a level and its feature matrix has **no `country_code` at all**, in any row.
  That is the scenario rather than a bug - the model genuinely has never seen that market - and it is the treatment [EDN-18](#decision-records) requires of an unseen value, which is why the `features` stage logs a warning naming any column a split leaves entirely empty instead of dropping or renaming it.
- **Price range used for training:** 500 EUR to 2M EUR; listings outside it are treated as data errors or collector outliers.
- **Registration date up to the reference date:** 164 listings in the raw file are registered after the 2025-11-08 snapshot, which would give them a negative age.
  114 of them are new cars and fall outside the used-car scope anyway; the 27 that survive the scope filters are treated as data errors and dropped.

### Out of scope

- New cars, pre-registered cars and transporters.
- Makes below the support threshold: the component rejects them instead of guessing.
- Transaction (sale) prices: we only observe asking prices.
- Price forecasting over time.
- Parsing free-text car descriptions (e.g. via an LLM).
- Markets outside the 8 countries of the data.

## 3. Target

- **Target:** asking price in EUR (`price`).
- **Training target:** `log(price)`.
  Price errors are roughly proportional to the price, and the log keeps expensive cars from dominating the loss.
- Predictions are transformed back with `exp`, which estimates the median price for a given car; this matches the median-based primary metric (section 6).

## 4. Features

The features describe the car the way a user can describe it.
`make` is always required, because the scope check depends on it; which other inputs the API requires is defined in the [specification](specification.md).

### Basic feature set (experiment ladder step 3)

| Group | Features |
|-------|----------|
| Identity | `make`, `model`, `body_type` |
| Age and usage | age (reference date minus `registration_date`), `mileage_km_raw`, `nr_prev_owners` |
| Technical | `power_kw`, `fuel_category`, `transmission`, `drive_train`, `gears`, `cylinders_volume_cc`, `nr_seats`, `nr_doors` |
| Market | `country_code`, `seller_type` |

The reference date for age is the snapshot date (2025-11-08) in training and the request date in serving.
As time passes, served cars therefore get older than anything seen in training at the same registration date; this is expected drift, watched in M6.

### Extended feature set (experiment ladder step 4)

Added on top of the basic set, to measure what they are worth:

- Equipment: the four equipment lists (`equipment_comfort`, `equipment_entertainment`, `equipment_extra`, `equipment_safety`) as multi-hot features.
  An item becomes its own column only above a configured minimum share of the training rows, so the matrix does not grow a tail of near-constant columns.
  The vocabulary is decided by the training rows alone and versioned as an artefact, so the same columns are built for every split and for a request.
  How many items that keeps is measured, not specified: the [model card](model-card.md#feature-space) carries the count with the date it was measured, because the experiment ladder sweeps the threshold.
- History flags: `has_full_service_history`, `non_smoking`, `is_rental`.
  These are one-sided: a `True` is an assertion by the seller, a `False` only means the assertion is absent, not that the opposite holds.
  They therefore enter the model as plain "asserted" indicators and are never read as a denial ([EDN-23](#decision-records)); the dataset card documents the same.
- Appearance: `body_color`, `paint_type`, `upholstery`, `upholstery_color`.
- Further technical data: `model_version` (normalised), `weight_kg` (parsed out of its text form, `'1,945 kg'`), `cylinders`, `electric_range_km`, `envir_standard`, `original_market`.
  `model_version` is free text with over 80,000 distinct values across the scoped listings, so normalising means: fold case and accents, collapse the separators, keep a configured number of leading tokens, and keep only the levels above a configured minimum share of the training rows.
  Everything below that floor is missing, which is how any unknown value is treated ([EDN-15](#decision-records)).
  How many levels that leaves, how much of the column they cover and how coarse they are is measured rather than specified, for the same reason: both numbers are parameters the ladder sweeps, and the [model card](model-card.md#feature-space) carries them with their date.

### Excluded columns

| Columns | Reason |
|---------|--------|
| `price_net`, `price_vat_rate` | Leakage: derived from the target. |
| `price_tax_deductible`, `price_negotiable` | Seller-side listing options tied to the price, not part of the car description. |
| `id`, `vin`, `german_hsn_tsn` | Identifiers. |
| `street`, `zip`, `city`, `latitude`, `longitude`, `seller_company_name` | PII or exact location. `seller_company_name` is only used, hashed, as the split group key. |
| `ratings_average`, `ratings_count`, `ratings_recommend_percentage` | Describe the seller, not the car; unknown to a user. |
| `description` | Contains the listing price in about 7 % of rows and is multilingual. Only an optional later experiment (ladder step 6), after stripping prices. |
| `warranty`, `has_warranty`, `fuel_cons_city_l100_km`, `fuel_cons_highway_l100_km` | Empty. |
| `had_accident` | True in only 3 rows. |
| `price_currency`, `offer_type`, `is_new`, `vehicle_type` | Constant after scoping. |
| `is_used` | Contradicts `offer_type`: False in 18,108 of the 113,708 rows scoped by `offer_type` and `vehicle_type`, so it is not a usable negative. |
| `is_preregistered` | Defines the scope filter (EDN-04), so it cannot also be a feature. |
| `mileage_km`, `power_hp`, `body_color_original`, `primary_fuel`, `seller_is_dealer` | Duplicate or overlap another column (as text, other unit, free-text variant, a fuel grade that overlaps `fuel_category` and is only half filled or, for `seller_is_dealer`, exactly `seller_type`). |
| `production_year`, `electric_range_city_km`, fuel consumption and CO2 columns | Sparse (0.5-39 % filled) and rarely known by users. |

## 5. Evaluation protocol

- Listings are deduplicated **before** anything is split off.
- All `ES` listings are removed next and stored as a separate drift set ([EDN-03](#decision-records)).
  They are not used to evaluate the success criteria; their error and interval coverage are tracked by the monitoring in M6.
- The remaining listings are split into train, validation, calibration and test sets **grouped by seller** (hashed dealer name; location for private sellers), so no seller appears in two sets.
  The exact split is documented in the dataset card and fixed by a seed in `params.yaml`.
- The validation set is used for early stopping and tuning, the calibration set only to calibrate the UC2 intervals.
- All success criteria are measured once per candidate model on the test set, logged to MLflow, and asserted by the model tests (M3).

## 6. Metrics

| Metric | Use |
|--------|-----|
| **MdAPE** (median absolute percentage error) | Primary metric: robust to outliers, easy to explain ("typically off by X %"). |
| Share of predictions within ±10 % and ±20 % | User-facing: how often the estimate is "close enough". |
| MAE in EUR | Absolute error, for context. |
| MAPE | Reported for comparability with other work; not a criterion, because cheap cars dominate it. |
| Interval coverage and mean relative width (UC2) | Whether the nominal 90 % intervals hold, and how useful (narrow) they are. |

Segments reported for every candidate model: make, country, seller type, fuel category, age bucket (0-1, 1-3, 3-6, 6-10, 10-20, over 20 years).
Price buckets are reported too, but they are not used in success criteria because they condition on the target, which is unknown at prediction time.

## 7. Baselines

- **B0:** median price per make, model and 2-year age bucket, falling back to the make median and then the global median.
- **B1:** Ridge regression on `log(price)` (interpretable depreciation baseline).

## 8. Success criteria

A model is good enough to deploy when it meets all of the following on the test set ([EDN-06](#decision-records), SC-06 added by [EDN-15](#decision-records)), except a criterion an EDN entry has accepted as missed, which NFR-01's gate then reports as waived (SC-04, [EDN-62](#decision-records)):

| ID | Criterion |
|----|-----------|
| SC-01 | MdAPE ≤ 9 %. |
| SC-02 | At least 85 % of predictions within ±20 % of the asking price. |
| SC-03 | MdAPE at least 30 % lower than baseline B0. |
| SC-04 | Every **level** of every segment of section 6 with at least 500 test rows (price buckets excluded) has MdAPE ≤ 15 %. |
| SC-05 | Nominal 90 % intervals reach an empirical coverage between 88 % and 92 %, both for full inputs and for the partial-input scenario P1. |
| SC-06 | With each optional input field masked on its own, and with all of them masked at once, the point model's MdAPE stays at or below 1.5 times its full-input MdAPE. |

**P1** is the partial-input scenario SC-05 and [FR-02](specification.md) both refer to: only `make`, `model`, `registration_date` and `mileage_km_raw` are given, and every other input field is absent.
It is stated here as a named field list because two requirements and one criterion depend on it.

SC-04 is a statement about a **level** (`make=BMW`) rather than about a segmenting variable, since `make` as a whole holds every test row.
A level that represents the absence of a *required* input field is reported with its metrics but excluded from the criterion, because FR-01 refuses such a request with a 422, so the level cannot occur at serving time at all: it is a data-quality finding, not a population the deployed component can be asked about.
For `make` and `body_type`, the two required fields the published file fills in every row, neither contract rules a gap out, because that is a measured fill rate and not a structural property ([#78](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/78)). The raw suite asserts both filled up to `validate.required_filled_mostly`, so a scrape with a gap there fails `validate-data` with the rule named, and does not fail `download` with a contract error.
When no level reaches the row minimum, SC-04 is reported as **not measured** rather than as met.
"Every level satisfies P" is vacuously true over an empty set, and reporting that as a pass would claim a check nobody ran.

SC-05 is measured on the UC2 conformal intervals, which the `lgbm-basic` bundle carries ([EDN-85](#decision-records)).
For a variant whose bundle carries none, the `evaluate` stage reports SC-05 as not measured, which **blocks** [NFR-01](requirements.md)'s gate rather than passing it: no model is deployable without intervals whose coverage has been checked.
The metrics artefact reports separately how many candidates meet every criterion that *could* be measured, so an outstanding criterion is visible as such instead of reading as a model that failed.

SC-06 covers what SC-04 and SC-05 do not: the point estimate for a request that leaves an optional field out, which the API explicitly allows ([specification](specification.md) FR-01).
The optional fields are the **input fields** of the evaluated model's feature set that FR-01 does not require, which is 6 fields for the basic set and 23 for the extended one.
An input field is matched to the feature column or columns derived from it, so `registration_date` is a required input even though the model consumes the derived `age_years`, and one equipment input field covers every multi-hot column built from it.
Without that matching the criterion would mask `age_years`, the most important feature in the model, as if it were optional.
"Masked" means the field carries what the API sends when it is omitted, and what that is per field is fixed by [EDN-23](#decision-records) rather than being one missing value for everything: an omitted assertion flag is `False`, an omitted equipment list is empty, and everything else is absent.
That the two are the same operation is a constraint on the API rather than an assumption: the request model defaults an omitted equipment list to `[]` ([specification](specification.md) FR-01), so what SC-06 masks is what a caller omitting the field actually sends.
The criterion is relative to the model's own full-input MdAPE, like SC-03 is relative to the baseline, because there is no reference value for it yet and inventing an absolute threshold would be guesswork.
SC-06 only fixes the observable outcome, and the mechanism that keeps it is part of the candidate's training: `lgbm-basic` is fitted on its training rows plus two copies of each, one with every optional field set to what an omitted field is and one with a random subset of them ([EDN-82](#decision-records)).
LightGBM's native missing handling alone left a request carrying only the required fields at 1.21 times the full-input MdAPE, and with the copies it is at 1.02 times.

SC-04's per-segment breakdown (including country and seller type) also serves as a basic fairness check across market segments.
Classification-oriented fairness metrics (e.g. AIF360's demographic parity) do not directly apply to this regression task; per-segment error parity is the task-appropriate equivalent.
The disparity is the spread of MdAPE across the levels of a segment, and the fairness statement is that no qualifying level exceeds SC-04's bound.
Price buckets are not part of that statement: they condition on the target, so their U-shaped error profile is a regression-to-the-mean artefact rather than a finding about a market.

### Reference values

The targets are set from a one-off exploratory run on 2026-09-22 (not tracked; the experiment ladder reproduces it in MLflow).
Same scope as above (without `ES`), deduplicated, 80/20 split grouped by seller, 96,831 listings:

| Model | MdAPE | Within ±20 % |
|-------|-------|--------------|
| B0 (median baseline) | 11.9 % | 70.9 % |
| LightGBM, basic features, no tuning | 6.7 % | 90.6 % |

- SC-01 and SC-02 leave a margin to these values, so they hold across seeds and splits and still rule out a model barely better than B0.
- SC-03: the exploratory LightGBM is 44 % better than B0.
- SC-04: all segments stay at or below 10.4 % except **cars older than 20 years, at 15.3 %**.
  The reference run named this as the known risk of the basic feature set, with the extended features or a scope change for classic cars as the possible remedies.
  The pipeline has since measured the segment at 16.80 % for the candidate model over 784 test rows (18.2 % on average over 20 other splits, see the [model card](model-card.md#how-comparable-the-sets-are-and-how-much-a-figure-moves-with-the-split)), and it has ruled the extended features out as the remedy: they make this segment 1.6 pp worse (18.45 %) while improving the pooled figure.
  The miss is accepted: SC-04 is recorded as missed, the 15 % threshold and the scope stay as they are, and the release gate reports it as waived by that entry rather than blocking on it ([EDN-62](#decision-records), NFR-01).
  The hyperparameter search kept the committed values ([EDN-73](#decision-records)), and the segment is over the bound on 20 of 20 other splits ([EDN-32](#decision-records)).
  The [model card](model-card.md#bias-risks-and-limitations) states the limitation in full.
- SC-05 has no reference value from the exploratory run; measured by the pipeline on the test split, the candidate's intervals cover 90.2 % of full inputs and 91.0 % of P1 requests (EDN-85).
  SC-06 has none either, because it is relative to the model's own full-input MdAPE by construction.
  The fill rates behind SC-06 were measured on 2026-09-29 ([EDN-15](#decision-records)): in the training scope `body_type` is filled in 100.000 % of the listings and `seller_type` in 99.986 % (14 of 97,889 rows missing), `nr_doors`, `nr_seats` and `cylinders_volume_cc` in 91 to 99 %, and `nr_prev_owners`, `gears` and `drive_train` in 61 to 76 %.

## Decision records

The choices behind this page are recorded in [reports/edn.md](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md):

- EDN-03: the Spanish listings are held out as the new-market scenario.
- EDN-04: the model covers used passenger cars only.
- EDN-05: a make is supported from 300 listings.
- EDN-06: a model is deployable only when it meets the success criteria.
- EDN-15: `/predict` requires ten fields, with missing inputs bounded by SC-06.
- EDN-23: condition flags are read as one-sided assertions.
- EDN-32: the split is 60/10/10/20 by seller, with the snapshot date as seed.
- EDN-58: an unmeasured success criterion blocks the release gate.
- EDN-62: `lgbm-basic` is the candidate model.
- EDN-73: hyperparameters change only under a written tuning protocol.
- EDN-82: `lgbm-basic` is trained on blanked copies of its listings.
- EDN-85: serving adds the stored conformity offsets to the quantile models, without MAPIE.
