"""Extended feature set: how coarse the `model_version` levels are, and what `weight_kg` covers.

Two questions the `features` stage had to answer and could not answer from the specification.

1. `model_version` is free text. Keeping its leading token and then a frequency floor is cheap,
   but how much does it merge, and would prefixing the token with make and model, or keeping two
   tokens after an engine letter, merge less? The docstring of `build_features.model_version` and
   the EDN entry for the decision both cite these numbers.
2. `weight_kg` is text (`'1,945 kg'`). Parsing it made its value range visible for the first
   time, which is the follow-up issue #25 needs in order to add a range check.

Scope matches the evaluation protocol: used cars only (EDN-04), listings registered after the age
reference date dropped (EDN-22), the training price range, deduplicated before any split. The `ES`
holdout and the split are reproduced here too, because the vocabulary is decided by the training
rows alone. The raw dataset is not in the repo (NFR-08, EDN-07), see the README.
"""

import hashlib
import re
import sys
import unicodedata

import pandas as pd

CSV = sys.argv[1]
SNAPSHOT = pd.Timestamp("2025-11-08")  # age reference date, problem specification section 4
SEED = 20251108
HOLDOUT_COUNTRY = "ES"
TRAIN_SHARE = 0.60
BUCKETS = 10_000

EQUIPMENT_COLUMNS = [
    "equipment_comfort",
    "equipment_entertainment",
    "equipment_extra",
    "equipment_safety",
]
EQUIPMENT_MIN_FREQUENCY = 0.01
MODEL_VERSION_MIN_FREQUENCY = 0.0005
DEDUP_KEY = [
    "make",
    "model",
    "model_version",
    "mileage_km_raw",
    "registration_date",
    "price",
    "power_kw",
]
# The leading token is one of these for most BMW and Mercedes listings, where it names the engine
# rather than the trim. Option C below keeps two tokens for exactly those.
ENGINE_LETTERS = {"c", "d", "e", "h", "i", "t"}

SEPARATOR = re.compile(r"[^0-9a-z.]+")
LONE_DOT = re.compile(r"(?<!\d)\.|\.(?!\d)")
WEIGHT = re.compile(r"^\s*([0-9]{1,3}(?:,[0-9]{3})*)\s*kg\s*$")
LOOSE_WEIGHT = re.compile(r"^\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*kg\s*$")


def normalise(text, tokens=None):
    plain = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().casefold()
    words = SEPARATOR.sub(" ", LONE_DOT.sub(" ", plain)).split()
    return " ".join(words if tokens is None else words[:tokens])


def leading(values, tokens):
    normalised = values.dropna().map(lambda text: normalise(text, tokens))
    return normalised[normalised != ""]


def train_rows(frame):
    """The training split, reproduced as `split_data.assign_split` computes it."""
    remaining = frame[frame["country_code"] != HOLDOUT_COUNTRY]
    location = (
        remaining["country_code"].fillna("??").astype(str)
        + "|"
        + remaining["zip"].fillna("?????").astype(str)
        + "|"
        + remaining["city"].fillna("?").astype(str)
    )
    key = remaining["seller_company_name"].where(
        remaining["seller_company_name"].notna(), "private|" + location
    )
    group = key.map(lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16])

    def bucket(value):
        digest = hashlib.sha256(f"{SEED}|{value}".encode()).hexdigest()
        return int(digest[:8], 16) % BUCKETS / BUCKETS

    return remaining[group.map(bucket) < TRAIN_SHARE]


def frequent(values, min_frequency, denominator):
    counts = values.value_counts()
    return set(counts[counts > min_frequency * denominator].index)


def load(path):
    frame = pd.read_csv(path, low_memory=False)
    raw_rows = len(frame)
    frame = frame[
        (frame["offer_type"] == "U")
        & (~frame["is_preregistered"].astype("boolean").fillna(False))
        & (frame["vehicle_type"] == "Car")
    ]
    registration = pd.to_datetime(frame["registration_date"], errors="coerce")
    frame = frame[~(registration > SNAPSHOT)]
    frame = frame[(frame["price"] >= 500) & (frame["price"] <= 2_000_000)]
    frame = frame.drop_duplicates(subset=DEDUP_KEY)
    print(f"raw rows: {raw_rows:,}")
    print(f"scoped, deduplicated rows: {len(frame):,}")
    return frame


def report_model_version(scoped, training):
    print("\n" + "=" * 78)
    print("model_version: what the leading token is worth")
    print("=" * 78)
    raw = scoped["model_version"].dropna()
    print(f"distinct raw values over the scoped listings: {raw.nunique():,}")
    print(f"distinct after folding case, accents and separators: {raw.map(normalise).nunique():,}")
    for tokens in (1, 2):
        print(f"distinct after keeping {tokens} leading token(s): {leading(raw, tokens).nunique():,}")

    rows = len(training)
    one = leading(training["model_version"], 1).reindex(training.index)
    two = leading(training["model_version"], 2).reindex(training.index)
    variants = {
        "A (chosen): leading token": one,
        "B: make|model|leading token": (
            training["make"].fillna("?").astype(str)
            + "|"
            + training["model"].fillna("?").astype(str)
            + "|"
            + one.astype(str)
        ).where(one.notna()),
        "C: two tokens when the first is an engine letter": one.where(~one.isin(ENGINE_LETTERS), two),
        "D: two tokens throughout": two,
    }
    print(f"\nthe {rows:,} training rows decide the levels; threshold {MODEL_VERSION_MIN_FREQUENCY}")
    print(f"{one.isin(ENGINE_LETTERS).mean():.1%} of them lead with an engine letter")
    for label, values in variants.items():
        levels = frequent(values, MODEL_VERSION_MIN_FREQUENCY, rows)
        kept = values.where(values.isin(levels))
        merged = (
            pd.DataFrame(
                {
                    "make": training["make"],
                    "model": training["model"],
                    "level": kept,
                    "raw": training["model_version"],
                }
            )
            .dropna(subset=["level"])
            .groupby(["make", "model", "level"])["raw"]
            .nunique()
        )
        prefix_pairs = sum(
            1 for short in levels for long in levels if short != long and long.startswith(short)
        )
        print(f"\n  {label}")
        print(f"    levels above the threshold: {len(levels):,}")
        print(f"    training rows covered: {kept.notna().mean():.1%}")
        print(f"    worst merge inside one make and model: {merged.max():,} distinct raw trims")
        print(f"    median merge: {merged.median():.0f}, mean: {merged.mean():.1f}")
        print(f"    make+model+level groups over 50 raw trims: {int((merged > 50).sum()):,}")
        print(f"    level pairs where one is a prefix of the other: {prefix_pairs:,}")
        if label.startswith("A"):
            print("    the six worst merges:")
            for (make, model, level), count in merged.sort_values(ascending=False).head(6).items():
                print(f"      {make} {model}, level {level!r}: {count:,} distinct raw trims")
            print(f"    three most frequent levels: {list(kept.value_counts().head(3).index)}")

    print("\n  A at the alternative thresholds the ladder can sweep:")
    for threshold in (0.001, 0.002):
        levels = frequent(one, threshold, rows)
        print(
            f"    threshold {threshold}: {len(levels):,} levels, "
            f"{one.isin(levels).mean():.1%} of the training rows covered"
        )


def report_equipment(training):
    print("\n" + "=" * 78)
    print(f"equipment: what the threshold of {EQUIPMENT_MIN_FREQUENCY} keeps")
    print("=" * 78)
    total_items = total_kept = 0
    for column in EQUIPMENT_COLUMNS:
        values = training[column].dropna()
        counts = pd.Series(
            [item for value in values for item in set(eval(value))]  # noqa: S307
        ).value_counts()
        kept = counts[counts > EQUIPMENT_MIN_FREQUENCY * len(training)]
        total_items += len(counts)
        total_kept += len(kept)
        print(f"  {column}: {len(counts)} distinct items, {len(kept)} above the threshold")
    print(f"  total: {total_kept} of {total_items} items become a column")


def report_weight(scoped):
    print("\n" + "=" * 78)
    print("weight_kg: the range the parse made visible (follow-up for issue #25)")
    print("=" * 78)
    text = scoped["weight_kg"].dropna().astype(str)
    print(f"non-null values in the scoped set: {len(text):,}")
    for label, pattern in (("the stage's pattern", WEIGHT), ("a loose pattern", LOOSE_WEIGHT)):
        unmatched = sorted(set(text[~text.str.match(pattern)]))
        print(f"  values {label} rejects: {len(unmatched)} {unmatched[:5]}")
    parsed = pd.to_numeric(
        text.str.extract(WEIGHT, expand=False).str.replace(",", "", regex=False)
    )
    print(f"  min {parsed.min():,.0f} kg, max {parsed.max():,.0f} kg")
    print(f"  rows below 500 kg: {int((parsed < 500).sum())}")
    print(f"  rows above 4,000 kg: {int((parsed > 4000).sum())}")
    print(
        f"  0.1 % quantile {parsed.quantile(0.001):,.0f} kg, median {parsed.median():,.0f} kg, "
        f"99.9 % quantile {parsed.quantile(0.999):,.0f} kg"
    )
    heaviest = parsed.sort_values(ascending=False).head(6).index
    print(f"  the six heaviest raw values: {text[heaviest].tolist()}")
    print("\n  what a locale flip would do to a loose pattern, which is why it is not used:")
    for value in ("194,5 kg", "1.945 kg", "1,2,3 kg", "1,9450 kg", "1,945 kg", "894 kg"):
        loose = LOOSE_WEIGHT.match(value)
        tight = WEIGHT.match(value)
        read_as = float(loose.group(1).replace(",", "")) if loose else None
        print(
            f"    {value!r:12} loose -> {read_as if read_as is not None else 'refused'}"
            f"   the stage -> {float(tight.group(1).replace(',', '')) if tight else 'refused'}"
        )


scoped = load(CSV)
training = train_rows(scoped)
print(f"training rows: {len(training):,}")
report_model_version(scoped, training)
report_equipment(training)
report_weight(scoped)
