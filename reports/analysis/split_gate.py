"""The `split` stage's ratio gate: the constants, their false-alarm budget, and the sizes.

Question: what should the gate of `recommenditos/data/split_data.py` allow? It judges a
realised seller-grouped split against four rules, and three of them carry a number that has
to come from a measurement rather than from taste:

- `DISPERSION_SIGMAS`, how far a share may scatter from its ratio in units of the assignment
  noise. Budget: at most 1 seed in 1,000 may turn the stage red on a pool that satisfies the
  concentration limit. A constant chosen without measuring this is what the first version of
  the stage shipped: it claimed 200 seeds reach at worst 3.5 sigma, which was an artefact of
  stopping at 200.
- `MIN_SHARE_OF_RATIO`, the floor. Budget: no seed may trip it on the real snapshot, and it
  must reject a set at well under its intended size.
- `MAX_GROUP_SHARE`, the largest share of the pool one seller is expected to hold. The
  dispersion bound is a statement about many small independent groups, so it is only
  meaningful while this holds.

It also reports the realised split sizes, the group count, the largest dealer and the
supported-make list, which the dataset card states as fact, and the two figures the
downstream make filter is judged on.

Everything is computed by calling the stage's own functions, so this cannot drift from what
the pipeline does. That is the difference from re-deriving the formula here, which is how a
gate and its justification come apart.

Scope matches the evaluation protocol, problem specification sections 2 and 5: used cars
only (EDN-04), listings registered after the age reference date dropped (EDN-22), the
training price range, deduplicated before any split. The raw dataset is not in the repo
(NFR-08, EDN-07), see the README.

Run from the repository root, so that `recommenditos` is importable:

    python reports/analysis/split_gate.py cars.csv [seeds]

`seeds` defaults to 2000, which is what the committed output was produced with and takes
about 25 minutes. The sweep is per unique seller group rather than per row, which is the
same assignment because `assign_split` maps the key alone.
"""

import sys

import pandas as pd

from recommenditos.data.preprocess import hash_seller_group
from recommenditos.data.split_data import (
    DISPERSION_SIGMAS,
    MAX_DISPERSION_BOUND,
    MAX_GROUP_SHARE,
    MIN_SHARE_OF_RATIO,
    SPLIT_NAMES,
    assign_split,
    check_ratios,
    cross_holdout_sellers,
    dispersion_bounds,
    realised_shares,
    supported_makes,
)
from recommenditos.data.synthetic import generate_raw_listings

CSV = sys.argv[1]
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

# params.yaml, `split` and `seed`. Copied rather than loaded so that the committed output
# stays attached to the values it was measured at, the way nfr11_check.py keeps P_VAL.
RATIOS = {"train": 0.60, "validation": 0.10, "calibration": 0.10, "test": 0.20}
PROJECT_SEED = 20251108
HOLDOUT_COUNTRY = "ES"
MIN_LISTINGS_PER_MAKE = 300
PRICE_MIN = 500
PRICE_MAX = 2_000_000
SNAPSHOT = pd.Timestamp("2025-11-08")

SCOPE_COLS = [
    "make",
    "model",
    "model_version",
    "mileage_km_raw",
    "registration_date",
    "price",
    "power_kw",
    "offer_type",
    "is_preregistered",
    "vehicle_type",
    "country_code",
    "seller_company_name",
    "zip",
    "city",
]
DEDUP_KEY = [
    "make",
    "model",
    "model_version",
    "mileage_km_raw",
    "registration_date",
    "price",
    "power_kw",
]


def scope(frame):
    """The rows `preprocess` leaves behind, with the split's grouping key."""
    frame = frame[
        (frame["offer_type"] == "U")
        & (~frame["is_preregistered"].astype("boolean").fillna(False))
        & (frame["vehicle_type"] == "Car")
    ]
    registered = pd.to_datetime(frame["registration_date"], errors="coerce")
    frame = frame[~(registered > SNAPSHOT)]
    frame = frame[(frame["price"] >= PRICE_MIN) & (frame["price"] <= PRICE_MAX)]
    frame = frame.drop_duplicates(subset=DEDUP_KEY).copy()
    frame["seller_group_id"] = hash_seller_group(frame)
    return frame


def sweep(group_sizes, seeds):
    """Per-seed worst sigma and smallest realised share, over unique groups."""
    keys = pd.Series(list(group_sizes.index))
    counts = group_sizes.to_numpy()
    total = int(group_sizes.sum())
    # One sigma per set, from the same expression `dispersion_bounds` uses.
    one_sigma = {
        name: bound / DISPERSION_SIGMAS
        for name, bound in dispersion_bounds(group_sizes, RATIOS, max_bound=1.0).items()
    }
    rows = []
    for seed in seeds:
        assigned = assign_split(keys, RATIOS, seed)
        sizes = dict.fromkeys(SPLIT_NAMES, 0)
        for name, count in zip(assigned, counts, strict=True):
            sizes[name] += int(count)
        shares = realised_shares(sizes)
        worst_sigma, worst_where, least = 0.0, None, 9.9
        for name in SPLIT_NAMES:
            sigma = abs(shares[name] - RATIOS[name]) / one_sigma[name]
            if sigma > worst_sigma:
                worst_sigma, worst_where = sigma, f"{name} at {shares[name]:.2%}"
            least = min(least, shares[name] / RATIOS[name])
        rows.append({"seed": seed, "sigma": worst_sigma, "where": worst_where, "least": least})
    return pd.DataFrame(rows), total


def quantile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def report_pool(label, group_sizes, seeds):
    print(f"\n--- {label}")
    sweeps, total = sweep(group_sizes, seeds)
    largest = int(group_sizes.max())
    bounds = dispersion_bounds(group_sizes, RATIOS)
    print(f"  pool {total:,} rows in {len(group_sizes):,} groups")
    print(f"  largest group {largest:,} rows, {largest / total:.2%} of the pool")
    print(f"  dispersion bounds at {DISPERSION_SIGMAS} sigma, capped at {MAX_DISPERSION_BOUND:.0%}:")
    for name in SPLIT_NAMES:
        print(f"    {name:12}{bounds[name] * 100:6.2f} pp")
    print(f"  over {len(sweeps):,} seeds:")
    worst = sweeps.loc[sweeps["sigma"].idxmax()]
    print(f"    worst sigma            {worst['sigma']:.3f}  (seed {worst['seed']}, {worst['where']})")
    for q in (0.5, 0.9, 0.99, 0.999):
        print(f"    sigma at quantile {q:<6} {quantile(sweeps['sigma'], q):.3f}")
    print(f"    smallest realised share as a fraction of its ratio: {sweeps['least'].min():.3f}")
    tripped = int((sweeps["sigma"] > DISPERSION_SIGMAS).sum())
    floored = int((sweeps["least"] < MIN_SHARE_OF_RATIO).sum())
    print(f"    seeds tripping the dispersion rule at {DISPERSION_SIGMAS}: {tripped} of {len(sweeps):,}")
    print(f"    seeds tripping the floor at {MIN_SHARE_OF_RATIO}: {floored} of {len(sweeps):,}")
    return sweeps


def main():
    seeds = range(1, SEEDS + 1)
    interim = scope(pd.read_csv(CSV, usecols=SCOPE_COLS, low_memory=False))
    held = interim["country_code"] == HOLDOUT_COUNTRY
    holdout, pool = interim[held], interim[~held]
    groups = pool["seller_group_id"]
    group_sizes = groups.value_counts()

    print("=" * 78)
    print("The realised split on the real snapshot")
    print("=" * 78)
    print(f"  {len(interim):,} listings survive preprocessing")
    print(f"  {len(holdout):,} are the {HOLDOUT_COUNTRY} holdout ({len(holdout) / len(interim):.2%})")
    print(f"  {len(pool):,} are left to split, in {len(group_sizes):,} seller groups")
    print(f"  rows with no country at all, which are split rather than held out: "
          f"{int(interim['country_code'].isna().sum())}")

    assigned = assign_split(groups, RATIOS, PROJECT_SEED)
    sizes = {name: int((assigned == name).sum()) for name in SPLIT_NAMES}
    report = check_ratios(sizes, group_sizes, RATIOS)
    print(f"\n  at the project seed {PROJECT_SEED}, which the gate accepts:")
    print(f"  {'set':12}{'rows':>10}{'share':>9}{'configured':>12}{'of ratio':>10}{'bound':>9}")
    for name in SPLIT_NAMES:
        print(
            f"  {name:12}{sizes[name]:>10,}{report['shares'][name] * 100:>8.2f}%"
            f"{RATIOS[name] * 100:>11.0f}%{report['shares'][name] / RATIOS[name]:>10.3f}"
            f"{report['bounds'][name] * 100:>8.2f}pp"
        )

    # Why "four sigma" had no calibrated meaning: the share is not a sum of many
    # comparable terms. Var(share) is proportional to sum(n_i^2), so one group's share of
    # the variance is n_i^2 / sum(n_i^2).
    squares = (group_sizes.astype("int64") ** 2).sum()
    print(
        f"\n  the largest dealer alone accounts for "
        f"{int(group_sizes.max()) ** 2 / int(squares):.1%} of the variance of any set's share, "
        f"so the share's distribution is dominated by one Bernoulli"
    )

    print("\n" + "=" * 78)
    print("The supported makes (EDN-05), counted after the holdout and before the split")
    print("=" * 78)
    makes = supported_makes(pool["make"], MIN_LISTINGS_PER_MAKE)
    covered = sum(int(entry["listings"]) for entry in makes)
    print(f"  {len(makes)} of {pool['make'].nunique()} makes reach {MIN_LISTINGS_PER_MAKE} listings")
    print(f"  they cover {covered:,} of the {len(pool):,} split rows ({covered / len(pool):.2%})")
    for entry in makes:
        print(f"    {entry['make']:16}{entry['listings']:>8,}")

    # What the downstream filter costs, which is the figure #36, #37 and #39 need.
    listed = {entry["make"] for entry in makes}
    in_scope = pool[pool["make"].isin(listed)]
    scoped_assigned = assign_split(in_scope["seller_group_id"], RATIOS, PROJECT_SEED)
    scoped_sizes = {name: int((scoped_assigned == name).sum()) for name in SPLIT_NAMES}
    scoped_shares = realised_shares(scoped_sizes)
    moved = max(abs(scoped_shares[name] - report["shares"][name]) for name in SPLIT_NAMES)
    print(
        f"\n  applying the list to the split sets moves every realised share by at most "
        f"{moved * 100:.2f} pp ({len(pool):,} rows to {len(in_scope):,})"
    )
    holdout_supported = int(holdout["make"].isin(listed).sum())
    print(
        f"  of the {len(holdout):,} holdout rows, {holdout_supported:,} are a supported make "
        f"and {len(holdout) - holdout_supported:,} are not"
    )

    print("\n" + "=" * 78)
    print("The holdout is selected per row, so a cross-border dealer is in both")
    print("=" * 78)
    crossing = cross_holdout_sellers(holdout["seller_group_id"], groups)
    print(f"  sellers listing both in {HOLDOUT_COUNTRY} and outside it: {len(crossing):,}")
    print(f"  holdout rows they hold: {int(holdout['seller_group_id'].isin(crossing).sum()):,}")
    print(f"  split rows they hold:   {int(groups.isin(crossing).sum()):,}")
    print(
        "  hash_seller_group keys a dealer by its company name, and no name in this snapshot\n"
        "  appears both inside and outside the holdout country, so the conflict between\n"
        "  EDN-03 and EDN-14 does not bind here. It does on the synthetic fixture, whose\n"
        "  dealer names are drawn independently of the country: see the pools below."
    )

    print("\n" + "=" * 78)
    print("Reordering SPLIT_NAMES is not a cosmetic change")
    print("=" * 78)
    # The order decides which hash bucket belongs to which set, so the tuple is part of
    # the split's identity and an FR-15 before-and-after comparison depends on it.
    import recommenditos.data.split_data as stage

    original = stage.SPLIT_NAMES
    try:
        stage.SPLIT_NAMES = ("test", "calibration", "validation", "train")
        reordered = assign_split(groups, RATIOS, PROJECT_SEED)
    finally:
        stage.SPLIT_NAMES = original
    per_seller = pd.DataFrame(
        {"g": groups.to_numpy(), "before": assigned.to_numpy(), "after": reordered.to_numpy()}
    ).drop_duplicates("g")
    print(
        f"  reversing the tuple moves {(per_seller['after'] != per_seller['before']).mean():.1%} "
        f"of sellers to another set"
    )

    print("\n" + "=" * 78)
    print(f"The constants, against a sweep of {SEEDS:,} seeds per pool")
    print("=" * 78)
    print(f"  DISPERSION_SIGMAS     {DISPERSION_SIGMAS}")
    print(f"  MIN_SHARE_OF_RATIO    {MIN_SHARE_OF_RATIO}")
    print(f"  MAX_GROUP_SHARE       {MAX_GROUP_SHARE}")
    print(f"  MAX_DISPERSION_BOUND  {MAX_DISPERSION_BOUND}")
    print(
        "\n  Budget: at most 1 seed in 1,000 may turn the stage red on a pool that satisfies\n"
        "  MAX_GROUP_SHARE. The three pools below are the ones the project runs on: the real\n"
        "  snapshot, and the two synthetic fixtures the tests use."
    )

    all_sweeps = [report_pool("real snapshot", group_sizes, seeds)]
    for rows in (2000, 20000):
        fixture = scope(generate_raw_listings(rows))
        fixture_pool = fixture[fixture["country_code"] != HOLDOUT_COUNTRY]
        all_sweeps.append(
            report_pool(
                f"synthetic fixture, {rows:,} raw rows",
                fixture_pool["seller_group_id"].value_counts(),
                seeds,
            )
        )
        fixture_crossing = cross_holdout_sellers(
            fixture[fixture["country_code"] == HOLDOUT_COUNTRY]["seller_group_id"],
            fixture_pool["seller_group_id"],
        )
        print(
            f"  cross-border sellers: {len(fixture_crossing):,}, holding "
            f"{int(fixture['seller_group_id'].isin(fixture_crossing).sum()):,} rows in all"
        )

    worst = max(float(sweeps["sigma"].max()) for sweeps in all_sweeps)
    trials = sum(len(sweeps) for sweeps in all_sweeps)
    tripped = sum(int((sweeps["sigma"] > DISPERSION_SIGMAS).sum()) for sweeps in all_sweeps)
    print("\n" + "=" * 78)
    print("Verdict")
    print("=" * 78)
    print(f"  worst sigma over all {trials:,} seed-pool trials: {worst:.3f}")
    print(
        f"  trials the dispersion rule would have failed at {DISPERSION_SIGMAS}: {tripped}, "
        f"a false-alarm rate of {tripped / trials:.4%} against a budget of 0.1%"
    )
    print(f"  margin above the worst observed: {DISPERSION_SIGMAS - worst:.2f} sigma")
    print(
        f"  the superseded 4.0 would have failed "
        f"{sum(int((s['sigma'] > 4.0).sum()) for s in all_sweeps)} of them, which is what makes it "
        f"unfit: a seed sweep is a routine thing to run."
    )
    real, small = float(all_sweeps[0]["least"].min()), float(all_sweeps[1]["least"].min())
    print(
        f"\n  The floor at {MIN_SHARE_OF_RATIO} sits below the smallest share any seed produced"
        f" on the real\n  snapshot ({real:.3f}), and it rejects a set at 0.58 of its intended"
        f" size, which the derived\n  bound alone accepted."
    )
    print(
        f"\n  On the 1,862-row fixture pool the smallest is {small:.3f}, because 608 groups"
        f" scatter far\n  more than 28,435, so the floor would fire there on"
        f" {int((all_sweeps[1]['least'] < MIN_SHARE_OF_RATIO).sum())} of {SEEDS:,} seeds. The"
        f" floor is a\n  statement about a pool of the snapshot's granularity, and"
        f" tests/test_split.py sweeps that\n  pool against a looser floor, naming this"
        f" measurement, rather than pretending otherwise."
    )


if __name__ == "__main__":
    main()
