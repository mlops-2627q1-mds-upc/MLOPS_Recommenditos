---
pretty_name: "AutoScout24 Car Listings Dataset (2025 Snapshot)"
language:
- en
tags:
- tabular
- automotive
- regression
- feature-engineering
- eda
task_categories:
- tabular-regression
size_categories:
- 100K<n<1M
license: Dataset is allowed to be used for research, educational or analytical purposes.
---

# Dataset Card for AutoScout24 Car Listings Dataset (2025 Snapshot)

A structured, ~120K-row snapshot of vehicle listings scraped from AutoScout24, one of Europe's largest online car marketplaces, covering pricing, technical specifications, energy/emissions data, equipment, and seller/location attributes. Built for vehicle price prediction, market segmentation, and feature-importance analysis.

## Dataset Details

### Dataset Description

This dataset contains publicly listed vehicle offers from AutoScout24, collected and structured into a single tabular file. Each row is one listing, described by ~70 fields spanning pricing, vehicle specifications, fuel/energy data, equipment, seller information, and location. The publisher states the data was validated and normalized using Pydantic and Pandas.

- **Curated by:** Kaggle user `clkmuhammed` (compiled and published the dataset); underlying listings originate from AutoScout24 sellers and dealers
- **Shared by [optional]:** `clkmuhammed` on Kaggle
- **Language:** English (per the dataset's own metadata); the free-text `description` field may contain other European languages, since AutoScout24 operates across multiple non-English-speaking markets
- **License:** Dataset is allowed to be used for research, educational or analytical purposes.

### Dataset Sources [optional]

- **Repository:** https://www.kaggle.com/datasets/clkmuhammed/autoscout24-car-listings-dataset

## Uses

### Direct Use

- Regression modeling of vehicle price from technical, equipment, and location features
- Feature selection / feature-importance analysis for price drivers
- Unsupervised clustering by technical specification or equipment profile
- Exploratory market analysis: price–mileage–age relationships, fuel-efficiency and CO₂ patterns by fuel type, equipment popularity, regional price comparisons

### Out-of-Scope Use

- **Real transaction prices:** the `price` field is the seller's *asking* price, not a confirmed sale price — don't use this to model actual transaction values without treating that gap explicitly
- **Live/current market state:** this is a single point-in-time snapshot (labeled "2025"); it will not reflect the current market and shouldn't be used for real-time pricing decisions
- **Representativeness claims:** the exact per-country / per-market breakdown isn't documented here, so claims about "the European used-car market" as a whole should be checked against the dataset's actual geographic distribution first
- **Identifying or contacting individuals:** the dataset includes `vin`, `seller_company_name`, `city`, `street`, and `latitude`/`longitude` — this is not an appropriate basis for locating or contacting specific sellers

## Dataset Structure

**Format:** CSV (UTF-8) · **Records:** ~120,000 listings · **Currency:** EUR · **Splits:** none provided (single file)

| Group | Fields |
|---|---|
| Identifiers | `id`, `description` |
| Ratings | `ratings_average`, `ratings_count`, `ratings_recommend_percentage` |
| Pricing | `price_currency`, `price`, `price_tax_deductible`, `price_negotiable`, `price_net`, `price_vat_rate` |
| Vehicle identity & body | `vin`, `make`, `model`, `model_version`, `german_hsn_tsn`, `mileage_km_raw`, `mileage_km`, `registration_date`, `production_year`, `vehicle_type`, `body_type`, `nr_seats`, `nr_doors`, `body_color`, `paint_type`, `body_color_original`, `upholstery`, `upholstery_color` |
| Drivetrain & mechanicals | `power_kw`, `power_hp`, `transmission`, `gears`, `drive_train`, `cylinders`, `cylinders_volume_cc`, `weight_kg` |
| Fuel / energy / emissions | `has_particle_filter`, `fuel_category`, `primary_fuel`, `electric_range_km`, `electric_range_city_km`, `fuel_cons_comb_l100_km`, `fuel_cons_city_l100_km`, `fuel_cons_highway_l100_km`, `co2_emission_grper_km`, `fuel_cons_comb_l100_wltp_km`, `fuel_cons_electric_comb_l100_wltp_km`, `co2_emission_grper_wltp_km` |
| Equipment (free text / lists) | `equipment_comfort`, `equipment_entertainment`, `equipment_extra`, `equipment_safety` |
| Condition flags | `is_used`, `is_new`, `is_preregistered`, `had_accident`, `has_full_service_history`, `non_smoking`, `nr_prev_owners`, `is_rental` |
| Miscellaneous | `envir_standard`, `original_market`, `offer_type` |
| Location | `country_code`, `zip`, `city`, `street`, `latitude`, `longitude` |
| Seller | `seller_is_dealer`, `seller_type`, `seller_company_name`, `has_warranty`, `warranty` |

Notes for modeling:
- Several fields are conditional on vehicle type and will be sparse/mostly-null for non-matching rows — e.g. `electric_range_km` and `electric_range_city_km` only apply to (partial) electric vehicles; `vin` and `german_hsn_tsn` may be inconsistently populated.
- `mileage_km_raw` vs `mileage_km` and the WLTP vs non-WLTP consumption/emissions pairs suggest parallel raw/cleaned fields — check for redundancy before using both.

## Dataset Creation

### Curation Rationale

To turn unstructured, web-based AutoScout24 listings into a validated, ML-ready tabular dataset for price modeling and automotive market analysis.

### Source Data

#### Data Collection and Processing

Collected from publicly available AutoScout24 listings; the publisher states the data was processed and validated using Pydantic (schema validation) and Pandas (tabular processing) for consistency. Exact scraping method, date range, and country coverage are not specified in the available page content. [More Information Needed]

#### Who are the source data producers?

The listings themselves are created by AutoScout24 sellers — both private individuals and commercial dealerships — across the European markets AutoScout24 operates in.

### Annotations

No manual annotation process is described. All fields appear to be either seller-declared at listing time or computed/normalized by the publisher's processing pipeline. [More Information Needed] — not explicitly confirmed by the publisher.

#### Personal and Sensitive Information

This dataset carries real re-identification risk and should be handled carefully:
- `vin` (Vehicle Identification Number) uniquely identifies a specific physical vehicle
- `seller_company_name`, `city`, `street`, and precise `latitude`/`longitude` can identify a dealership or, in the case of private sellers, a specific location
- The free-text `description` field could contain seller-inserted contact details

Consider dropping or hashing `vin`, `street`, and exact coordinates, and treating `seller_company_name` as sensitive, especially for any public redistribution or model deployment.

## Bias, Risks, and Limitations

- **Asking price ≠ transaction price:** as with most scraped marketplace data, `price` reflects what the seller asked, not what the car sold for.
- **Snapshot bias:** a single 2025 snapshot cannot support trend/time-series analysis on its own.
- **Unverified self-reported fields:** `had_accident`, `has_full_service_history`, `non_smoking`, and similar flags are seller-declared and not independently verified — treat as noisy labels.
- **Unknown geographic distribution:** market/country coverage isn't documented here, so the sample may be skewed toward AutoScout24's largest markets (e.g., Germany, Austria) rather than evenly spread across Europe — should be checked empirically.
- **Missing-value structure:** many columns are conditional on vehicle type or listing completeness (see Dataset Structure notes) and will need explicit missingness handling.

### Recommendations

Before modeling: check the actual license on the Kaggle page, profile per-column missingness and the country/market distribution, decide how to handle `vin`/location fields for privacy, and be explicit in any writeup that `price` is an asking price rather than a settled transaction price.

## Citation

**BibTeX:**
```
@misc{clkmuhammed_autoscout24_2025,
  author       = {clkmuhammed},
  title        = {AutoScout24 Car Listings Dataset (2025 Snapshot)},
  year         = {2025},
  publisher    = {Kaggle},
  url          = {https://www.kaggle.com/datasets/clkmuhammed/autoscout24-car-listings-dataset}
}
```

**APA:**
clkmuhammed. (2025). *AutoScout24 Car Listings Dataset (2025 Snapshot)* [Data set]. Kaggle. https://www.kaggle.com/datasets/clkmuhammed/autoscout24-car-listings-dataset
