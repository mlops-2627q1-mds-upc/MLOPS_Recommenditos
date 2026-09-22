Problem specification
=====================

What the model learns, on which data, and when it is good enough.
This page owns the ML framing: the [requirements](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/14), the dataset card, the model card and the report link here instead of repeating it.
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
- **Supported makes:** makes with at least **300 listings** in the cleaned used-car data, counted after removing the `ES` holdout and before the split ([EDN-05](#decision-records)).
  The pipeline computes this list; it is not hard-coded.
  With the current data these are 11 makes covering 98.6 % of the used listings: BMW, Porsche, Mercedes-Benz, Audi, Alfa Romeo, Suzuki, Volvo, Honda, Hyundai, Aston Martin and Volkswagen.
- **Markets:** the model is trained on 7 countries (DE, IT, NL, BE, AT, FR, LU), with the country as a feature.
  All `ES` listings are held out as the new-market drift scenario ([EDN-03](#decision-records), project brief section 3.2); they join the training data only after the drift is confirmed and the model is retrained.
  The component accepts `ES` and treats it as an unknown country until then.
- **Price range used for training:** 500 EUR to 2M EUR; listings outside it are treated as data errors or collector outliers.

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
`make` is always required, because the scope check depends on it; which other inputs the API requires is defined in the [requirements](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/14).

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
- History flags: `has_full_service_history`, `non_smoking`, `is_rental`.
  "False" may mean "unknown" in these flags; the dataset card documents this.
- Appearance: `body_color`, `paint_type`, `upholstery`, `upholstery_color`.
- Further technical data: `model_version` (normalised), `weight_kg`, `cylinders`, `electric_range_km`, `envir_standard`, `original_market`.

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
| `price_currency`, `offer_type`, `is_used`, `is_new`, `is_preregistered`, `vehicle_type` | Constant after scoping. |
| `mileage_km`, `power_hp`, `body_color_original`, `primary_fuel` | Duplicate another column (as text, other unit, free-text variant or finer fuel label). |
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

A model is good enough to deploy when it meets all of the following on the test set ([EDN-06](#decision-records)):

| ID | Criterion |
|----|-----------|
| SC-01 | MdAPE ≤ 9 %. |
| SC-02 | At least 85 % of predictions within ±20 % of the asking price. |
| SC-03 | MdAPE at least 30 % lower than baseline B0. |
| SC-04 | Every segment of section 6 with at least 500 test rows (price buckets excluded) has MdAPE ≤ 15 %. |
| SC-05 | Nominal 90 % intervals reach an empirical coverage between 88 % and 92 %, both for full inputs and for the partial-input scenario P1 (only make, model, registration date and mileage given). |

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
  This is a known risk for the basic feature set; the extended features or a scope change for classic cars must close it.
- SC-05 has no reference value yet; the intervals are built in a later step.

## Decision records

The choices behind this page are recorded in [reports/edn.md](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md):

- EDN-03: hold out AutoScout24 `ES` as the new-market drift scenario.
- EDN-04: used cars only.
- EDN-05: minimum listing support per make.
- EDN-06: success criteria.
