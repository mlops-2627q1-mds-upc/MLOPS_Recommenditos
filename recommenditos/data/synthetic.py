"""A synthetic stand-in for the raw AutoScout24 snapshot.

The real file is 548 MB and carries PII, so nothing here is sampled from it:
everything below is generated from the public catalogues in the dataset card
(make and model names, the category value sets) plus a seeded random number
generator.

EDN-25 does track the raw file with DVC and push it to our remote, so this is
not about the file being unavailable. It is that a fixture is committed to Git,
and a sample of real listings would put personal data in the repository itself
rather than behind a DVC pointer (NFR-08). Generating also makes the whole test
suite runnable without the download and without DagsHub credentials, which is
what lets the stage tickets be built in parallel.

It is used in two places:

- `tests/` build every fixture from it, so a test needs no data access.
- the `download` stage produces it while `download.source` is `synthetic`,
  which is what keeps `dvc repro` green before #33 implements the real
  acquisition.

The frame is deliberately awkward in the same ways the real file is. Every
edge case the pipeline rules exist for is guaranteed present whatever the row
count, so a stage ticket can assert against it rather than hope for it:

===========================  ===============================================
Edge case                    The rule it exercises
===========================  ===============================================
Registered after the snapshot  EDN-22, negative age
Registered on the snapshot     EDN-22, the bound is inclusive
Duplicate on the dedup key     preprocess step 6, 7-column key
An `ES` listing                EDN-03, the drift holdout
A make far below support       EDN-05, the 300-listing threshold
Price below and above range    the params price range
`offer_type = N`               EDN-04, used cars only
`is_preregistered = True`      EDN-24
`vehicle_type = Transporter`   EDN-04, passenger cars only
`is_used = False` under `U`    EDN-23, the one-sided flags
A null registration date       age is missing, not zero
A private seller               the split's group-key fallback
===========================  ===============================================
"""

import numpy as np
import pandas as pd

from recommenditos.schema import RAW_SCHEMA

# The snapshot date of the real file. Rows are generated relative to it so the
# "registered after the snapshot" case means the same thing here as there.
SNAPSHOT_DATE = pd.Timestamp("2025-11-08")

# How many rows are appended to guarantee the edge cases above.
EDGE_CASE_ROWS = 15

#: The composite key preprocessing deduplicates on (params.yaml,
#: `preprocess.dedup_key`). Repeated here because the edge cases have to be
#: built so that only the row meant to be a duplicate is one, and this module
#: takes no params file. `tests/test_data.py` asserts the two are equal, because
#: a copy that has drifted builds a fixture for a rule the pipeline no longer has.
DEDUP_KEY = (
    "make",
    "model",
    "model_version",
    "mileage_km_raw",
    "registration_date",
    "price",
    "power_kw",
)

# Make shares from the dataset card, renormalised. The real distribution is
# extremely skewed (four makes are 83 % of the file) and the skew is what the
# support threshold of EDN-05 exists for, so the fixture keeps it. Model names
# are public; no listing is copied.
_MAKES: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    ("BMW", 0.319, ("1 Series", "3 Series", "5 Series", "X3", "X5", "X3 M", "i4")),
    ("Porsche", 0.215, ("911", "718", "Macan", "Cayenne", "Panamera", "991")),
    ("Mercedes-Benz", 0.164, ("A 180", "C 220", "E 300", "GLC 300", "S 500", "Vito")),
    ("Audi", 0.131, ("A1", "A3", "A4", "A6", "Q3", "Q5", "e-tron")),
    ("Alfa Romeo", 0.066, ("Giulia", "Stelvio", "Giulietta", "Tonale")),
    ("Suzuki", 0.039, ("Swift", "Vitara", "Ignis", "S-Cross")),
    ("Volvo", 0.031, ("XC40", "XC60", "XC90", "V60")),
    ("Honda", 0.013, ("Civic", "CR-V", "Jazz")),
    ("Hyundai", 0.009, ("i20", "i30", "Tucson")),
    ("Volkswagen", 0.007, ("Golf", "Polo", "Tiguan")),
    # Deliberately far below any support threshold, so the unsupported-make
    # path of EDN-05 and FR-04 has something to reject.
    ("Bugatti", 0.006, ("Chiron",)),
)

# Country shares from the dataset card. ES is 6.8 % there and is held out whole.
_COUNTRIES: tuple[tuple[str, float], ...] = (
    ("DE", 0.385),
    ("IT", 0.202),
    ("NL", 0.144),
    ("BE", 0.081),
    ("ES", 0.068),
    ("AT", 0.061),
    ("FR", 0.052),
    ("LU", 0.007),
)

_BODY_TYPES = ("Compact", "Sedan", "Station wagon", "Coupe", "Convertible", "Off-Road/Pick-up")
_FUEL_CATEGORIES = ("Gasoline", "Diesel", "Electric", "Electric/Gasoline", "LPG", "Others")
_TRANSMISSIONS = ("Manual", "Automatic", "Semi-automatic")
_DRIVE_TRAINS = ("Front Wheel Drive", "Rear Wheel Drive", "4WD")
_BODY_COLORS = ("Black", "Grey", "White", "Blue", "Red", "Silver", "Green")
_UPHOLSTERY = ("Cloth", "Full leather", "Part leather", "Velour", "Other")
_UPHOLSTERY_COLORS = ("Black", "Grey", "Beige", "Brown", "Blue")
_PAINT_TYPES = ("Metallic", "Others")
_ENVIR_STANDARDS = ("Euro 4", "Euro 5", "Euro 6", "Euro 6b", "Euro 6d", "Euro 6d-TEMP")
_ORIGINAL_MARKETS = ("Germany", "Italy", "Netherlands", "Belgium", "France", "Austria")
_EQUIPMENT = {
    "equipment_comfort": (
        "Air conditioning",
        "Cruise control",
        "Keyless entry",
        "Parking sensors",
    ),
    "equipment_entertainment": ("Bluetooth", "Apple CarPlay", "Android Auto", "Digital radio"),
    "equipment_extra": ('Alloy wheels (18")', "Tow bar", "Roof rails", "Spoiler"),
    "equipment_safety": ("ABS", "Blind spot monitor", "Lane departure warning", "Alarm system"),
}

# Roughly the fill rates the dataset card reports for the columns whose
# missingness the model is meant to learn from (EDN-15). Anything not listed is
# always present.
_FILL_RATES = {
    "model": 0.988,
    "model_version": 0.993,
    "registration_date": 0.998,
    "mileage_km_raw": 0.995,
    "nr_seats": 0.959,
    "nr_doors": 0.988,
    "body_color": 0.899,
    "paint_type": 0.780,
    "upholstery": 0.774,
    "upholstery_color": 0.726,
    "power_kw": 0.986,
    "transmission": 0.993,
    "gears": 0.635,
    "drive_train": 0.766,
    "cylinders": 0.790,
    "cylinders_volume_cc": 0.896,
    "weight_kg": 0.797,
    "fuel_category": 0.999,
    "electric_range_km": 0.108,
    "nr_prev_owners": 0.547,
    "envir_standard": 0.651,
    "original_market": 0.385,
    "country_code": 0.999,
    "seller_type": 0.999,
}

# How often each boolean is True, as a share of rows. Named rather than
# inlined so the numbers read as the distribution choices they are - and
# because a bare literal in a comparison is a lint finding in this package.
_TRUE_RATES = {
    "equipment_item": 0.50,
    "has_full_service_history": 0.45,
    "has_particle_filter": 0.20,
    "is_rental": 0.03,
    "is_used": 0.84,
    "non_smoking": 0.40,
    "price_negotiable": 0.15,
    "price_tax_deductible": 0.25,
    "seller_is_dealer": 0.83,
}

# How many distinct sellers per 100 rows. The real file has 17,141 company
# names over 118,382 listings, so a seller carries about 7 listings; the same
# ratio here is what makes a seller-grouped split actually constrain anything.
_LISTINGS_PER_SELLER = 7


def generate_raw_listings(n_rows: int = 2000, *, seed: int = 20251108) -> pd.DataFrame:
    """A RAW_SCHEMA-valid synthetic snapshot of `n_rows` listings.

    The same seed always gives the same frame, including the edge-case rows,
    which are appended last so their row positions are stable.
    """
    # Each edge case is built from its own body row, so there have to be at
    # least as many body rows as edge cases.
    minimum = 2 * EDGE_CASE_ROWS
    if n_rows < minimum:
        raise ValueError(f"n_rows must be at least {minimum}, got {n_rows}")

    rng = np.random.default_rng(seed)
    body = _random_listings(rng, n_rows - EDGE_CASE_ROWS)
    frame = pd.concat([body, _edge_cases(body)], ignore_index=True)
    frame["id"] = [f"synthetic-{index:08d}" for index in range(len(frame))]
    return RAW_SCHEMA.conform(frame)


def _random_listings(rng: np.random.Generator, n_rows: int) -> pd.DataFrame:
    makes = _draw_makes(rng, n_rows)
    columns = {
        "id": [""] * n_rows,
        "description": _maybe_null(
            rng, pd.Series([f"Listing {i}." for i in range(n_rows)]), 0.963
        ),
        **_seller_columns(rng, n_rows),
        **_vehicle_columns(rng, n_rows, makes),
        **_drivetrain_columns(rng, n_rows),
        **_equipment_columns(rng, n_rows),
        **_flag_columns(rng, n_rows),
        **_location_columns(rng, n_rows),
    }
    frame = pd.DataFrame(columns)
    return _price_columns(rng, frame)


def _draw_makes(rng: np.random.Generator, n_rows: int) -> pd.Series:
    names = [make for make, _, _ in _MAKES]
    weights = np.array([share for _, share, _ in _MAKES])
    return pd.Series(rng.choice(names, size=n_rows, p=weights / weights.sum()))


def _vehicle_columns(rng, n_rows, makes: pd.Series) -> dict[str, pd.Series]:
    models_by_make = {make: models for make, _, models in _MAKES}
    models = pd.Series([rng.choice(models_by_make[make]) for make in makes])
    # Registrations are always the first of a month in the real file, spread
    # over roughly 25 years before the snapshot.
    months_old = rng.integers(0, 300, size=n_rows)
    dates = pd.Series(
        [(SNAPSHOT_DATE - pd.DateOffset(months=int(m))).replace(day=1) for m in months_old]
    )
    mileage = np.round(rng.gamma(2.0, 30000.0, size=n_rows) + months_old * 500.0, -2)
    return {
        "make": makes,
        "model": _maybe_null(rng, models, _FILL_RATES["model"]),
        "model_version": _maybe_null(
            rng,
            pd.Series(
                [
                    f"{m} {v}"
                    for m, v in zip(
                        models, rng.choice(("Style", "Sport", "GT"), n_rows), strict=True
                    )
                ]
            ),
            _FILL_RATES["model_version"],
        ),
        "german_hsn_tsn": _maybe_null(rng, pd.Series(["0000/ABC"] * n_rows), 0.26),
        "mileage_km_raw": _maybe_null(rng, pd.Series(mileage), _FILL_RATES["mileage_km_raw"]),
        "mileage_km": pd.Series([f"{int(km):,} km" for km in mileage]),
        "registration_date": _maybe_null(
            rng, dates.dt.strftime("%Y-%m-%d"), _FILL_RATES["registration_date"]
        ),
        "production_year": _maybe_null(rng, pd.Series(dates.dt.year.astype("float64")), 0.188),
        "vehicle_type": pd.Series(["Car"] * n_rows),
        "body_type": pd.Series(rng.choice(_BODY_TYPES, n_rows)),
        "nr_seats": _maybe_null(
            rng, pd.Series(rng.choice((2.0, 4.0, 5.0, 7.0), n_rows)), _FILL_RATES["nr_seats"]
        ),
        "nr_doors": _maybe_null(
            rng, pd.Series(rng.choice((2.0, 3.0, 5.0), n_rows)), _FILL_RATES["nr_doors"]
        ),
        "body_color": _maybe_null(
            rng, pd.Series(rng.choice(_BODY_COLORS, n_rows)), _FILL_RATES["body_color"]
        ),
        "paint_type": _maybe_null(
            rng, pd.Series(rng.choice(_PAINT_TYPES, n_rows)), _FILL_RATES["paint_type"]
        ),
        "body_color_original": _maybe_null(
            rng, pd.Series(rng.choice(_BODY_COLORS, n_rows)), 0.626
        ),
        "upholstery": _maybe_null(
            rng, pd.Series(rng.choice(_UPHOLSTERY, n_rows)), _FILL_RATES["upholstery"]
        ),
        "upholstery_color": _maybe_null(
            rng,
            pd.Series(rng.choice(_UPHOLSTERY_COLORS, n_rows)),
            _FILL_RATES["upholstery_color"],
        ),
    }


def _drivetrain_columns(rng, n_rows) -> dict[str, pd.Series]:
    power_kw = np.round(rng.gamma(4.0, 40.0, size=n_rows) + 40.0)
    weight = np.round(rng.normal(1650.0, 300.0, size=n_rows), -1).clip(700.0, 3000.0)
    empty = pd.Series([np.nan] * n_rows, dtype="float64")
    return {
        "power_kw": _maybe_null(rng, pd.Series(power_kw), _FILL_RATES["power_kw"]),
        "power_hp": _maybe_null(
            rng, pd.Series(np.round(power_kw * 1.36)), _FILL_RATES["power_kw"]
        ),
        "transmission": _maybe_null(
            rng, pd.Series(rng.choice(_TRANSMISSIONS, n_rows)), _FILL_RATES["transmission"]
        ),
        "gears": _maybe_null(
            rng, pd.Series(rng.choice((5.0, 6.0, 7.0, 8.0), n_rows)), _FILL_RATES["gears"]
        ),
        "drive_train": _maybe_null(
            rng, pd.Series(rng.choice(_DRIVE_TRAINS, n_rows)), _FILL_RATES["drive_train"]
        ),
        "cylinders": _maybe_null(
            rng, pd.Series(rng.choice((3.0, 4.0, 6.0, 8.0), n_rows)), _FILL_RATES["cylinders"]
        ),
        "cylinders_volume_cc": _maybe_null(
            rng,
            pd.Series(np.round(rng.uniform(900.0, 5000.0, n_rows), -1)),
            _FILL_RATES["cylinders_volume_cc"],
        ),
        "weight_kg": _maybe_null(
            rng, pd.Series([f"{int(w):,} kg" for w in weight]), _FILL_RATES["weight_kg"]
        ),
        "has_particle_filter": pd.Series(rng.random(n_rows) < _TRUE_RATES["has_particle_filter"]),
        "fuel_category": _maybe_null(
            rng, pd.Series(rng.choice(_FUEL_CATEGORIES, n_rows)), _FILL_RATES["fuel_category"]
        ),
        "primary_fuel": _maybe_null(rng, pd.Series(["Super E10 95"] * n_rows), 0.507),
        "electric_range_km": _maybe_null(
            rng,
            pd.Series(np.round(rng.uniform(30.0, 600.0, n_rows))),
            _FILL_RATES["electric_range_km"],
        ),
        "electric_range_city_km": empty.copy(),
        "fuel_cons_comb_l100_km": _maybe_null(
            rng, pd.Series(np.round(rng.uniform(3.0, 15.0, n_rows), 1)), 0.353
        ),
        # Empty in every row of the real file.
        "fuel_cons_city_l100_km": empty.copy(),
        "fuel_cons_highway_l100_km": empty.copy(),
        "co2_emission_grper_km": _maybe_null(
            rng, pd.Series(np.round(rng.uniform(80.0, 320.0, n_rows))), 0.210
        ),
        "fuel_cons_comb_l100_wltp_km": _maybe_null(
            rng, pd.Series(np.round(rng.uniform(3.0, 15.0, n_rows), 1)), 0.256
        ),
        "fuel_cons_electric_comb_l100_wltp_km": _maybe_null(
            rng, pd.Series(np.round(rng.uniform(12.0, 30.0, n_rows), 1)), 0.042
        ),
        "co2_emission_grper_wltp_km": _maybe_null(
            rng, pd.Series(np.round(rng.uniform(80.0, 320.0, n_rows))), 0.392
        ),
        "envir_standard": _maybe_null(
            rng, pd.Series(rng.choice(_ENVIR_STANDARDS, n_rows)), _FILL_RATES["envir_standard"]
        ),
        "original_market": _maybe_null(
            rng, pd.Series(rng.choice(_ORIGINAL_MARKETS, n_rows)), _FILL_RATES["original_market"]
        ),
    }


def _equipment_columns(rng, n_rows) -> dict[str, pd.Series]:
    columns = {}
    for name, items in _EQUIPMENT.items():
        drawn = rng.random((n_rows, len(items))) < _TRUE_RATES["equipment_item"]
        # Repr of a Python list, never null: an empty list is the string "[]",
        # exactly as the real file encodes it.
        columns[name] = pd.Series(
            [repr([item for item, keep in zip(items, row, strict=True) if keep]) for row in drawn]
        )
    return columns


def _flag_columns(rng, n_rows) -> dict[str, pd.Series]:
    return {
        # A False in these is "not asserted", not "no" (EDN-23).
        "is_used": pd.Series(rng.random(n_rows) < _TRUE_RATES["is_used"]),
        "is_new": pd.Series([False] * n_rows),
        "is_preregistered": pd.Series([False] * n_rows),
        "had_accident": pd.Series([False] * n_rows),
        "has_full_service_history": pd.Series(
            rng.random(n_rows) < _TRUE_RATES["has_full_service_history"]
        ),
        "non_smoking": pd.Series(rng.random(n_rows) < _TRUE_RATES["non_smoking"]),
        "is_rental": pd.Series(rng.random(n_rows) < _TRUE_RATES["is_rental"]),
        "nr_prev_owners": _maybe_null(
            rng,
            pd.Series(rng.choice((0.0, 1.0, 2.0, 3.0), n_rows)),
            _FILL_RATES["nr_prev_owners"],
        ),
        "offer_type": pd.Series(["U"] * n_rows),
    }


def _location_columns(rng, n_rows) -> dict[str, pd.Series]:
    names = [code for code, _ in _COUNTRIES]
    weights = np.array([share for _, share in _COUNTRIES])
    countries = pd.Series(rng.choice(names, size=n_rows, p=weights / weights.sum()))
    return {
        "country_code": _maybe_null(rng, countries, _FILL_RATES["country_code"]),
        # Synthetic stand-ins for the PII columns, so a test that asserts none
        # of them survives preprocessing has something to look for.
        "zip": pd.Series([f"{int(z):05d}" for z in rng.integers(1000, 99999, n_rows)]),
        "city": pd.Series(rng.choice(("Springfield", "Rivertown", "Lakeside"), n_rows)),
        "street": pd.Series([f"Example Street {int(n)}" for n in rng.integers(1, 200, n_rows)]),
        "latitude": pd.Series(np.round(rng.uniform(36.0, 54.0, n_rows), 5)),
        "longitude": pd.Series(np.round(rng.uniform(-9.0, 18.0, n_rows), 5)),
        "vin": _maybe_null(rng, pd.Series([f"SYNTHETICVIN{i:05d}" for i in range(n_rows)]), 0.34),
    }


def _seller_columns(rng, n_rows) -> dict[str, pd.Series]:
    n_sellers = max(2, n_rows // _LISTINGS_PER_SELLER)
    seller_index = rng.integers(0, n_sellers, size=n_rows)
    is_dealer = rng.random(n_rows) < _TRUE_RATES["seller_is_dealer"]
    company = pd.Series(
        [
            f"Example Motors {int(i):04d}" if dealer else None
            for i, dealer in zip(seller_index, is_dealer, strict=True)
        ],
        dtype="str",
    )
    return {
        "seller_is_dealer": pd.Series(is_dealer),
        "seller_type": _maybe_null(
            rng,
            pd.Series(["Dealer" if dealer else "PrivateSeller" for dealer in is_dealer]),
            _FILL_RATES["seller_type"],
        ),
        "seller_company_name": company,
        "ratings_average": _maybe_null(
            rng, pd.Series([f"4,{int(d)}" for d in rng.integers(0, 10, n_rows)]), 0.646
        ),
        "ratings_count": _maybe_null(
            rng, pd.Series(rng.integers(1, 500, n_rows).astype("float64")), 0.646
        ),
        "ratings_recommend_percentage": _maybe_null(
            rng, pd.Series(rng.integers(0, 101, n_rows).astype("float64")), 0.646
        ),
        "has_warranty": pd.Series([np.nan] * n_rows, dtype="float64"),
        "warranty": pd.Series([np.nan] * n_rows, dtype="float64"),
    }


def _price_columns(rng: np.random.Generator, frame: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(frame)
    # Price falls with age and mileage, so the baselines have something to
    # learn and a metric on the fixture is not pure noise.
    age_years = (
        (SNAPSHOT_DATE - pd.to_datetime(frame["registration_date"])).dt.days.fillna(3650) / 365.25
    ).to_numpy()
    mileage = frame["mileage_km_raw"].fillna(100_000).to_numpy()
    power = frame["power_kw"].fillna(100).to_numpy()
    base = 8000.0 + power * 180.0
    price = base * np.exp(-0.09 * age_years - mileage / 900_000.0)
    price = np.round(price * rng.lognormal(0.0, 0.18, size=n_rows), -1).clip(600.0, 1_900_000.0)
    frame["price"] = price
    frame["price_currency"] = pd.Series(["EUR"] * n_rows)
    frame["price_tax_deductible"] = pd.Series(
        rng.random(n_rows) < _TRUE_RATES["price_tax_deductible"]
    )
    frame["price_negotiable"] = pd.Series(rng.random(n_rows) < _TRUE_RATES["price_negotiable"])
    frame["price_net"] = _maybe_null(rng, pd.Series(np.round(price * 0.81, -1)), 0.287)
    frame["price_vat_rate"] = _maybe_null(rng, pd.Series([19.0] * n_rows), 0.257)
    return frame


def _edge_cases(body: pd.DataFrame) -> pd.DataFrame:
    """The rows that guarantee every pipeline rule has something to act on.

    Each row is built from a *different* body row and then given a price
    nobody else has, for two reasons that both bite hard if ignored:

    - Cloning one body row would make most of these duplicates of it on the
      seven-column deduplication key, so preprocessing would delete the very
      cases the fixture exists to provide. Only row 1 is meant to be a
      duplicate, and it is made one deliberately.
    - Cloning one body row would also copy its `country_code` into all of
      them. If that row happened to be Spanish, `split` would divert the whole
      block into the holdout and the train, validation, calibration and test
      frames would contain none of these cases at all.
    - That price has to be inside the training price range. A sentinel above
      the ceiling is unique too, but then the price rule deletes every row
      carrying it, and only the price-specific cases below reach the interim
      frame - which is the first bullet's failure in another disguise.
    """
    rows = [body.iloc[index].copy() for index in range(EDGE_CASE_ROWS)]
    for index, row in enumerate(rows):
        # A price nobody else has, so no row below collides on the dedup key by
        # accident, and one *inside* the training price range, so a row is not
        # deleted by the price rule when it exists to exercise another one.
        # Body prices are whole multiples of ten, so a distinct fraction of a
        # cent makes the uniqueness structural rather than lucky. The
        # price-specific cases overwrite it again.
        row["price"] = float(row["price"]) + (index + 1) / 100
        # Never the holdout country unless the row is the holdout case.
        row["country_code"] = "DE"

    # 1. Registered after the snapshot: age would be negative (EDN-22).
    rows[0]["registration_date"] = "2026-01-01"
    # 2. The one intended duplicate: identical to a plain body listing on all
    #    seven columns of the deduplication key, different outside it, which is
    #    what a real duplicate listing looks like. The source is a body row and
    #    not another edge case for two reasons. Deduplication runs last, so a
    #    pair whose partner an earlier rule deletes never reaches it - copying
    #    row 0 made the pair post-snapshot, and the date rule removed both
    #    before the rule this case exists for ever saw them. And the body row
    #    carries the lower `id`, so it is the copy deduplication keeps, which
    #    leaves every other edge case below intact. At the minimum row count
    #    this body row is also the one the last edge case is built from; that
    #    case overwrites its own price, so the pair is still exactly one.
    source = body.iloc[-1]
    for column in DEDUP_KEY:
        rows[1][column] = source[column]
    rows[1]["non_smoking"] = not bool(source["non_smoking"])
    # 3. An `ES` listing for the drift holdout (EDN-03).
    rows[2]["country_code"] = "ES"
    # 4. A make far below any support threshold (EDN-05, FR-04).
    rows[3]["make"] = "Bugatti"
    rows[3]["model"] = "Chiron"
    # 5. and 6. Outside the training price range at both ends.
    rows[4]["price"] = 250.0
    rows[5]["price"] = 3_000_000.0
    # 7. A new car, dropped by the used-car scope (EDN-04).
    rows[6]["offer_type"] = "N"
    rows[6]["is_new"] = True
    rows[6]["is_used"] = False
    # 8. Pre-registered, dropped by the scope even though the flag is
    #    unreliable (EDN-24).
    rows[7]["is_preregistered"] = True
    # 9. Not a passenger car (EDN-04).
    rows[8]["vehicle_type"] = "Transporter"
    # 10. No registration date: age is missing, not zero.
    rows[9]["registration_date"] = None
    # 11. A private seller, so the split's group-key fallback is exercised.
    rows[10]["seller_company_name"] = None
    rows[10]["seller_is_dealer"] = False
    rows[10]["seller_type"] = "PrivateSeller"
    # The one-sided flag of EDN-23: used by `offer_type`, not by `is_used`.
    rows[10]["is_used"] = False
    # 12. No seller and no location at all, as 15 rows of the real file have.
    #     Every such listing falls into one shared group, which is the safe
    #     direction but worth having something to assert against.
    for column in ("seller_company_name", "country_code", "zip", "city", "seller_type"):
        rows[11][column] = None
    rows[11]["seller_is_dealer"] = False
    rows[11]["country_code"] = None
    # 13. and 14. Exactly on each price bound, so the range rule's
    #     inclusive-or-exclusive edge is testable rather than assumed.
    rows[12]["price"] = 500.0
    rows[13]["price"] = 2_000_000.0
    # 15. Registered exactly on the snapshot date, so the date rule's bound is
    #     testable the same way. No body row can provide it: they are all
    #     registered on the first of a month and the snapshot is the 8th.
    rows[14]["registration_date"] = SNAPSHOT_DATE.strftime("%Y-%m-%d")

    return pd.DataFrame(rows).reset_index(drop=True)


def _maybe_null(rng: np.random.Generator, values: pd.Series, fill_rate: float) -> pd.Series:
    """`values` with a share of `1 - fill_rate` blanked out, as the real file is."""
    if fill_rate >= 1.0:
        return values
    keep = rng.random(len(values)) < fill_rate
    return values.where(pd.Series(keep, index=values.index))
