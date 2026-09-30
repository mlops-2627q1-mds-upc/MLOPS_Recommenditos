---
pretty_name: "AutoScout24 Car Listings Dataset (2025 snapshot)"
language:
- en
- de
- it
- nl
- fr
- es
tags:
- tabular
- automotive
- regression
- used-cars
- price-prediction
task_categories:
- tabular-regression
size_categories:
- 100K<n<1M
license: mit
---

# Dataset Card for AutoScout24 Car Listings Dataset (2025 snapshot)

A tabular snapshot of 118,382 vehicle listings scraped from AutoScout24, one of Europe's largest online car marketplaces.
Each row is one listing described by 75 columns covering pricing, technical specifications, energy and emissions data, equipment, condition flags, and seller and location attributes.
It is the training dataset for the Recommenditos used-car price component.

All figures on this page were measured on the file itself (see [Provenance of the figures](#provenance-of-the-figures)), not copied from the publisher's description.

## Dataset Details

### Dataset Description

The publisher collected publicly listed vehicle offers from AutoScout24 and structured them into a single CSV file.
The publisher states that the data was validated and normalised using Pydantic and Pandas; we have not independently verified that claim.

- **Curated by:** Muhammed Çelik (ORCID [0009-0001-2685-1263](https://orcid.org/0009-0001-2685-1263)), who compiled and published the dataset. The underlying listings originate from AutoScout24 sellers and dealers.
- **Language:** The column names and categorical labels are English. The free-text `description` and `model_version` fields are in the seller's local language, so German, Italian, Dutch, French and Spanish all occur.
- **Snapshot date:** 2025-11-08, taken from the published file name. There is no per-listing date column, so this single date applies to every row.
- **License:** MIT, with a caveat. See [Licensing](#licensing).

### Dataset Sources

- **Zenodo (the source our pipeline uses):** <https://zenodo.org/records/17643343>, DOI [10.5281/zenodo.17643343](https://doi.org/10.5281/zenodo.17643343), version 1.0.0, file `autoscout24_dataset_20251108.csv`, 548.6 MB, md5 `b23a122cc51baf7de39f449193ff0d28`.
- **Kaggle mirror:** <https://www.kaggle.com/datasets/clkmuhammed/autoscout24-car-listings-dataset>, DOI [10.34740/KAGGLE/DS/8683897](https://doi.org/10.34740/KAGGLE/DS/8683897).

We pull the file from Zenodo rather than Kaggle because the Zenodo record pins an immutable version with a checksum, which is what makes the data reproducible without re-hosting a copy ourselves.
See [Data versioning](data-versioning.md).

## Uses

### Direct Use

- Regression modelling of listing price from technical, equipment and location features.
- Feature selection and feature-importance analysis for price drivers.
- Unsupervised clustering by technical specification or equipment profile.
- Exploratory market analysis: price, mileage and age relationships, fuel-efficiency and CO2 patterns by fuel type, equipment popularity, regional price comparisons.

### Out-of-Scope Use

- **Real transaction prices.** The `price` field is the seller's asking price, not a confirmed sale price. Do not use it to model actual transaction values without treating that gap explicitly.
- **Live or current market state.** This is a single snapshot taken on 2025-11-08 with no per-listing date, so it supports neither trend analysis nor real-time pricing decisions.
- **Claims about the European used-car market as a whole.** Four premium brands make up 82.9 % of the rows and mass-market brands are nearly absent, so this sample is not representative of the wider market. See [Bias, Risks, and Limitations](#bias-risks-and-limitations).
- **Identifying or contacting individuals.** The file includes `vin`, `seller_company_name`, `city`, `street` and exact `latitude` and `longitude`. It is not an appropriate basis for locating or contacting specific sellers.

## Dataset Structure

**Format:** CSV (UTF-8) · **Rows:** 118,382 · **Columns:** 75 · **Currency:** EUR for every row · **Splits:** none provided, the upstream file is a single CSV

### Columns

| Group | Fields |
|---|---|
| Identifiers | `id` |
| Free text | `description` (96.3 % filled, multilingual) |
| Ratings | `ratings_average`, `ratings_count`, `ratings_recommend_percentage` |
| Pricing | `price_currency`, `price`, `price_tax_deductible`, `price_negotiable`, `price_net`, `price_vat_rate` |
| Vehicle identity and body | `vin`, `make`, `model`, `model_version`, `german_hsn_tsn`, `mileage_km_raw`, `mileage_km`, `registration_date`, `production_year`, `vehicle_type`, `body_type`, `nr_seats`, `nr_doors`, `body_color`, `paint_type`, `body_color_original`, `upholstery`, `upholstery_color` |
| Drivetrain and mechanicals | `power_kw`, `power_hp`, `transmission`, `gears`, `drive_train`, `cylinders`, `cylinders_volume_cc`, `weight_kg` |
| Fuel, energy and emissions | `has_particle_filter`, `fuel_category`, `primary_fuel`, `electric_range_km`, `electric_range_city_km`, `fuel_cons_comb_l100_km`, `fuel_cons_city_l100_km`, `fuel_cons_highway_l100_km`, `co2_emission_grper_km`, `fuel_cons_comb_l100_wltp_km`, `fuel_cons_electric_comb_l100_wltp_km`, `co2_emission_grper_wltp_km` |
| Equipment (free-text lists) | `equipment_comfort`, `equipment_entertainment`, `equipment_extra`, `equipment_safety` |
| Condition flags | `is_used`, `is_new`, `is_preregistered`, `had_accident`, `has_full_service_history`, `non_smoking`, `nr_prev_owners`, `is_rental` |
| Miscellaneous | `envir_standard`, `original_market`, `offer_type` |
| Location | `country_code`, `zip`, `city`, `street`, `latitude`, `longitude` |
| Seller | `seller_is_dealer`, `seller_type`, `seller_company_name`, `has_warranty`, `warranty` |

### Distribution

| Country | Listings |
|---|---|
| DE | 45,611 |
| IT | 23,957 |
| NL | 17,059 |
| BE | 9,582 |
| ES | 8,015 |
| AT | 7,213 |
| FR | 6,141 |
| LU | 789 |
| missing | 15 |

There are 25 makes in total, but the distribution is extremely skewed:

| Make | Listings | Share |
|---|---|---|
| BMW | 37,745 | 31.9 % |
| Porsche | 25,511 | 21.5 % |
| Mercedes-Benz | 19,400 | 16.4 % |
| Audi | 15,469 | 13.1 % |
| Alfa Romeo | 7,806 | 6.6 % |
| Suzuki | 4,592 | 3.9 % |
| Volvo | 3,643 | 3.1 % |
| all others (18 makes) | 4,216 | 3.6 % |

Other key distributions:

- **Offer type:** 114,127 used (`U`), 4,252 new (`N`), 3 other (`A`). 3,702 of the `U` rows are additionally flagged `is_preregistered`, and 32,199 cars were first registered in 2025.
- **Vehicle type:** 117,926 `Car` and 456 `Transporter`.
- **Seller:** 98,565 dealer listings and 19,802 private listings, across 17,141 distinct company names.
- **Price:** 1 to 13,500,000 EUR, median 39,980. Eight rows are priced below 100 EUR.
- **Mileage:** 0 to 2,570,000 km, median 56,400.
- **Registration date:** up to 2026-11-01, which is after the snapshot date. See [Known data issues](#known-data-issues).

### Completeness

Twenty-two columns are fully populated. The columns sparse enough to matter for modelling are:

| Column | Filled |
|---|---|
| `warranty`, `has_warranty`, `fuel_cons_city_l100_km`, `fuel_cons_highway_l100_km` | 0.0 % |
| `electric_range_city_km` | 0.5 % |
| `fuel_cons_electric_comb_l100_wltp_km` | 4.2 % |
| `electric_range_km` | 10.8 % |
| `production_year` | 18.8 % |
| `co2_emission_grper_km` | 21.0 % |
| `fuel_cons_comb_l100_wltp_km` | 25.6 % |
| `price_vat_rate` | 25.7 % |
| `german_hsn_tsn` | 26.0 % |
| `price_net` | 28.7 % |
| `vin` | 34.0 % |
| `fuel_cons_comb_l100_km` | 35.3 % |
| `original_market` | 38.5 % |
| `co2_emission_grper_wltp_km` | 39.2 % |
| `primary_fuel` | 50.7 % |
| `nr_prev_owners` | 54.7 % |

Some of this is structural rather than missing data: the electric-range and WLTP columns only apply to the matching vehicle types.

### Known data issues

These are the issues we found when profiling the file, and each one needs handling in the pipeline.

- **Asking price, not transaction price.** `price` is what the seller asked for, never what the car sold for.
- **Four columns are completely empty:** `warranty`, `has_warranty`, `fuel_cons_city_l100_km` and `fuel_cons_highway_l100_km`. A fifth, `had_accident`, is `True` in only 3 of 118,382 rows, so it carries no usable signal.
- **The condition flags contradict each other.** 14,744 rows are flagged as neither used, new nor pre-registered, and 18,446 of the 114,127 `offer_type = 'U'` rows have `is_used = False`. `False` in these boolean flags most likely encodes "unknown" rather than "no", so they cannot be read as reliable negatives. The same caution applies to `has_full_service_history`, `non_smoking` and `is_rental`.
- **164 listings are registered after the snapshot date,** the latest being 2026-11-01 and 137 of them falling in January 2026. Any age feature computed as "snapshot date minus registration date" is negative for these rows.
- **Duplicate listings.** `vin` is only 34.0 % filled, so deduplicating on it is not enough. A composite key of make, model, version, mileage, registration date, price and power finds 6,347 duplicate rows. There are no exact full-row duplicates.
- **The listing price leaks into the free text.** About 6.9 % of all rows have the exact listing price inside `description`, and many more contain some currency amount. Prices must be stripped before any text feature is built.
- **Redundant column pairs.** `mileage_km` duplicates `mileage_km_raw` as text, `power_hp` duplicates `power_kw` in another unit, `body_color_original` is a free-text variant of `body_color`, and `primary_fuel` is a finer-grained `fuel_category`. Use one of each pair, not both.
- **Outliers at both ends.** Prices from 1 EUR and mileages up to 2,570,000 km are present and need explicit range checks.
- **15 rows have no seller or location data at all,** missing `country_code`, `seller_type`, `city`, `zip` and `street`, although they still carry coordinates.
- **Self-reported and unverified.** `had_accident`, `has_full_service_history`, `non_smoking` and the equipment lists are all seller-declared and independently unchecked. Treat them as noisy.

## How this project uses the dataset

The full framing lives in the [problem specification](problem-spec.md); this section records only what directly concerns the data.

- **Scope.** Used passenger cars that are not pre-registered, which is `offer_type = 'U'`, `vehicle_type = 'Car'` and `is_preregistered = False`, leaving 110,013 of the 118,382 rows (113,708 before the pre-registered filter). Preprocessing additionally drops the listings registered after the snapshot date and keeps only prices between 500 EUR and 2,000,000 EUR.
- **Excluded as leakage.** `price_net` and `price_vat_rate` are derived from the target. `price_tax_deductible` and `price_negotiable` are seller-side listing options tied to the price rather than properties of the car.
- **Excluded as identifiers or PII.** `id`, `vin`, `german_hsn_tsn`, `street`, `zip`, `city`, `latitude`, `longitude` and `seller_company_name`. The last is used only in hashed form, as the grouping key for the split.
  That hash is a grouping key and not an anonymisation: it is unsalted, so a dictionary built from this published file maps every group id back to the company name, or to the `country_code`, `zip` and `city` of a private seller, that it came from. No artefact carries those columns, but nor are the ids opaque, and this file is where that is stated ([EDN-35](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
- **Row funnel.** Each rule is applied to what the rule above it left, so these are the counts the `preprocess` stage reports (measured 2026-09-30, `reports/analysis/preprocess_funnel.py`): 118,382 as read, 110,013 after the used-passenger-car scope, 109,986 after dropping the listings registered after the snapshot, 109,911 after the price range and **105,405** after deduplication.
  Deduplication removes 4,506 rows, not the 6,347 duplicates the raw file holds, and the date rule removes 27 of the 164 post-snapshot listings, because in both cases the rules above them delete the rest first.
- **Splits.** Listings are deduplicated first, and every remaining `ES` listing (6,079 of the 8,015 in the raw file, 7,545 of them still present before deduplication) is then held out as a simulated new market for drift monitoring, never reaching training, validation or calibration. The remainder is split into train, validation, calibration and test grouped by seller, so that no seller appears in two splits. A purely random split would leak near-identical listings from the same dealer across splits, and no listing date exists, so a temporal split is not possible.
- **Proportions and seed.** Pinned in `params.yaml`: 60 % train, 10 % validation, 10 % calibration and 20 % test of the non-`ES` listings, with seed `20251108`.
  The test share is 20 % because SC-04 only checks a segment once it holds 500 test rows, and below about 15 % Volvo and then Suzuki drop under that bar.
- **Not yet fixed.** The realised split sizes, which follow once the preprocessing rules run on the real snapshot rather than on the synthetic fixture.

## Dataset Creation

### Curation Rationale

To turn unstructured, web-based AutoScout24 listings into a validated tabular dataset suitable for price modelling and automotive market analysis.

### Source Data

#### Data Collection and Processing

The publisher collected the data from publicly available AutoScout24 listings and states that it was processed and validated with Pydantic for schema validation and Pandas for tabular processing.
The exact scraping method, the date range over which listings were collected, and the reason for this particular country and brand coverage are not documented by the publisher.

#### Who are the source data producers?

The listings are written by AutoScout24 sellers, both private individuals and commercial dealerships, across the eight European markets present in the file.

### Annotations

There is no manual annotation process.
Every field is either seller-declared at listing time or computed by the publisher's processing pipeline.

#### Personal and Sensitive Information

The raw file carries a real re-identification risk and is handled accordingly.

- `vin` uniquely identifies a specific physical vehicle.
- `seller_company_name`, `city`, `street` and exact `latitude` and `longitude` identify a dealership, or for the 19,802 private listings, a specific location.
- The free-text `description` may contain seller-inserted contact details.

In this project the raw file is tracked with `dvc add` and pushed to our own DVC remote, so the team pulls one copy of it instead of each member re-downloading it from Zenodo (EDN-25).
That means we re-host the personal data ourselves, on a remote that is public (EDN-20).
We record this as an accepted risk rather than a solved problem: the marginal exposure is small, because the identical file is already publicly downloadable from the pinned Zenodo DOI under the same licence, but we are publishers of that personal data in our own right.
The PII columns are dropped in preprocessing, so no processed dataset, prediction log, comparable listing or model artefact contains them.
See [Data versioning](data-versioning.md).

Anyone else redistributing or deploying from this data should drop or hash `vin`, `street` and the exact coordinates, and treat `seller_company_name` as sensitive.
A plain hash of any of these is a pseudonym and not a removal, because the values can be enumerated from this file: our own group key is measurably invertible that way (EDN-35), and a hash that has to resist that needs a secret key.

## Bias, Risks, and Limitations

- **Severe brand skew.** BMW, Porsche, Mercedes-Benz and Audi account for 82.9 % of all rows. Mass-market brands are almost absent, with 352 Volkswagen, 60 Renault and 60 Opel listings, and Toyota, SEAT, Peugeot, Fiat and Škoda do not appear at all. A model trained on this data will be unreliable for ordinary mass-market cars, and this is the single biggest limitation of the dataset.
- **Geographic skew.** Germany alone is 38.5 % of the rows, and Germany, Italy and the Netherlands together are 73.2 %. Luxembourg contributes 789 listings.
- **Not purely used cars.** 4,252 listings are new (`offer_type = 'N'`) and a further 3,702, all of them inside the used bucket, are flagged as pre-registered; 456 rows are light commercial vehicles rather than passenger cars.
- **Asking price, not transaction price.** The target is what sellers hoped to get, which sits above realised sale prices by an unknown and probably segment-dependent margin.
- **Snapshot bias.** One point in time with no per-listing date means no trend analysis, and the data ages relative to any deployed model.
- **Unverified self-reported fields.** Condition and history flags are seller claims, and the `False` values are ambiguous between "no" and "unknown".
- **Dealer concentration.** 83.3 % of listings come from dealers across 17,141 company names, so a handful of large dealers can dominate individual segments.

### Recommendations

- Treat the price as an asking price everywhere it is reported, including in any write-up.
- Restrict claims to the brands and countries that are actually well represented.
- Handle missingness explicitly rather than imputing silently, especially for the structurally sparse columns.
- Group by seller when splitting, so that model quality is not overstated by near-duplicate listings from the same dealer.
- Strip currency amounts from `description` before using it as a text feature.

## Licensing

The upstream record is internally inconsistent, and both statements come from the same Zenodo record.

- The structured license field on the Zenodo record is **MIT**, which is what the frontmatter of this card reports.
- The author's own description text states: "You are welcome to use this dataset for research, educational, or analytical purposes."

MIT is permissive and would allow commercial use, while the prose restricts it to research, education and analysis.
We have not asked the author to resolve this, because both readings clearly permit what this project does, which is non-commercial academic work.
Anyone intending commercial use should clarify it with the author first.

The author asks for a citation when analyses are published, which we honour in this card and in the project report.

## Citation

We cite the Zenodo record, because it identifies the exact file and version used, and the Kaggle DOI, because it is the citation the author asks for.

**BibTeX:**

```bibtex
@misc{celik_autoscout24_zenodo_2025,
  author    = {Çelik, Muhammed},
  title     = {autoscout24\_dataset\_20251108.csv},
  version   = {1.0.0},
  year      = {2025},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.17643343},
  url       = {https://doi.org/10.5281/zenodo.17643343}
}

@misc{celik_autoscout24_kaggle_2025,
  author    = {Çelik, Muhammed},
  title     = {AutoScout24 Car Listings Dataset},
  year      = {2025},
  publisher = {Kaggle},
  doi       = {10.34740/KAGGLE/DS/8683897},
  url       = {https://www.kaggle.com/ds/8683897}
}
```

**APA:**

Çelik, M. (2025). *autoscout24_dataset_20251108.csv* (Version 1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.17643343

## Provenance of the figures

Every number on this page was measured on the Zenodo file `autoscout24_dataset_20251108.csv`, verified against the published checksum md5 `b23a122cc51baf7de39f449193ff0d28`, and profiled on 2026-09-29.
The publisher's own description gives round figures such as "~120K listings" and was written with AI assistance, as the Zenodo record itself notes, so we treat it as a claim rather than as evidence and report our own measurements instead.

## Dataset Card Authors

Team Recommenditos (UPC, MLOps 2026/27): @lukas2510, @kadameit, @ulasawczuk, @W11W11W11, @michudud04.

## Dataset Card Contact

Through the [project repository](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos), by opening an issue.
