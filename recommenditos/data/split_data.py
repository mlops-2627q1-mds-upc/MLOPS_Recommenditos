"""`split` stage: hold Spain out, count the supported makes, split by seller.

Three steps, in this order, because each one changes what the next one sees.

1. The `ES` holdout leaves first, as its own artefact (EDN-03). M6 replays it
   against the running API as a new market, so no `ES` row may reach train,
   validation, calibration or test.
2. The supported makes are counted on what is left, before the split (EDN-05),
   and written as an artefact. The API reads the list from the model metadata
   and rejects a make outside it (FR-04); a hard-coded list would be a second
   copy of the same fact, free to disagree with the data.
3. The rest is split grouped by `seller_group_id`, whole sellers at a time.
   Held-out dealers shift the feature distribution as much as a different
   country does (EDN-14), so a random split would leak one dealer into train
   and test and flatter every metric we report.

This stage partitions; it does not filter. Every row of the interim frame lands
in exactly one of the five artefacts, which is what makes "the split loses no
row" a testable invariant instead of a hope, and what keeps EDN-05's threshold
revisitable without re-running the split. The supported-make list is therefore
a *record* here, and `features` is where it is applied, to every frame and
before the vocabulary is built (EDN-48, EDN-67): problem-spec section 5
describes the split as removing nothing but `ES`, while section 2 scopes the
*model* to the supported makes. Which stage applies the list, which ones check
it and which tests hold each guarantee is written down in
`docs/docs/pipeline.md` under "The supported makes", not only in a pull
request, because a squash-merge commit message is not somewhere anyone looks.

The holdout is selected per row, so "every `ES` row is held out" wins over "no
seller appears in two sets" for a dealer that lists in `ES` and elsewhere. That
conflict is real and is resolved deliberately rather than silently: see
`cross_holdout_sellers`.

Two properties the bucketing below is built for, both worth more than tighter
ratios. It is order-independent, so a clean clone reproduces the same split from
`seed` and the group key alone (NFR-06). And it is stable along one axis only:
*adding rows or sellers* leaves the sellers already there where they were, which
is what lets the retrain-with-`ES` run of FR-15 be compared against the current
model instead of being confounded by sellers that moved between train and test.
It is not stable against a change to `seed`, to the ratios, or to the order of
`SPLIT_NAMES`: each of those reshuffles most sellers (measured: reordering the
tuple moves 80.0 % of them on the real snapshot). So an FR-15 comparison holds
only while those three are untouched, and changing one of them means the
before-and-after metrics are no longer comparable.
"""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from recommenditos.config import INTERIM_DATA_DIR, PARAMS_FILE, PROCESSED_DATA_DIR
from recommenditos.pipeline import load_params, read_frame, write_frame, write_json
from recommenditos.schema import INTERIM_SCHEMA, PROCESSED_SCHEMA

#: The sets `split` writes, besides the country holdout.
#:
#: Changing this tuple is not a local edit. The order decides which hash bucket
#: belongs to which set, so reordering it moves 80.0 % of sellers between sets
#: (measured on the real snapshot) and invalidates any FR-15 before-and-after
#: comparison. Adding a set also needs a ratio in params.yaml and an `outs`
#: entry in dvc.yaml, which names the five files one by one.
SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "calibration", "test")

#: The holdout's artefact name. `dvc.yaml` and the `features` stage name the
#: same file, so it is spelled out rather than derived from `holdout_country`: a
#: changed country would otherwise produce a differently named artefact that the
#: DAG does not know about, and `dvc repro` would report a missing output
#: instead of a holdout nobody reads.
HOLDOUT_NAME = "holdout_es"

#: The make list, under the name the API's metadata (FR-04) is fed from.
MAKES_FILE = "supported_makes.json"

#: The smallest share of its configured ratio a set may come out at.
#:
#: This is the rule that catches an undersized set, and it is deliberately
#: unconditional: it is a fraction of the *configured* ratio, so it does not
#: widen when the data turns out to be concentrated. The earlier gate derived
#: its whole bound from the realised group sizes, which meant the bound grew in
#: exactly the case that makes a split untrustworthy - one dealer holding a
#: tenth of the pool bought itself a 14 pp allowance - and a call with three
#: empty sets passed.
#:
#: 0.75 comes from a stated budget: no seed may trip it on data that satisfies
#: `MAX_GROUP_SHARE`. Over 2,000 seeds on the real snapshot the smallest share
#: any set reached was 0.820 of its ratio, so 0.75 sits 9 % below the worst
#: measured case, and it rejects the calibration set at 0.58 of its intended
#: size that the derived bound alone accepted. See
#: `reports/analysis/split_gate.py`.
MIN_SHARE_OF_RATIO = 0.75

#: The largest share of the pool one seller group is expected to hold.
#:
#: The dispersion bound below is only meaningful while this holds, because it
#: is a statement about many small independent groups. Checking it separately
#: is what turns the old failure mode inside out: concentration now *fires* the
#: gate instead of widening it. On the real snapshot the largest dealer holds
#: 3.41 % of the pool, so 5 % leaves room for the snapshot to grow without a
#: false alarm while a mis-set `group_key` - grouping by `make` puts 34.6 % in
#: one group, by `country_code` more - is far above it.
MAX_GROUP_SHARE = 0.05

#: The widest the derived dispersion bound may ever get, in share points.
#:
#: Without a cap the bound is unbounded: grouping by `country_code` derived a
#: 101 pp allowance, which no realised share can miss. 0.15 is above the widest
#: bound any pool the project runs on needs (the 1,865-row fixture pool asks
#: 13.2 pp at `DISPERSION_SIGMAS`), so it never binds on a pool this project
#: splits, and it stops the pathological cases from being vacuous.
MAX_DISPERSION_BOUND = 0.15

#: How many standard deviations of assignment noise a realised share may sit
#: from its ratio before the *dispersion* rule fires.
#:
#: Sellers are assigned independently, so a set's share is a sum over sellers
#: and its standard deviation follows from the group sizes: see
#: `dispersion_bounds`. The multiple is not a normal-theory quantile, and it
#: must not be read as one - 81 % of the variance of the train share comes from
#: the single largest dealer, so the share's distribution is dominated by one
#: Bernoulli and "four sigma" would have no calibrated meaning here.
#:
#: It is chosen from a measured false-alarm budget instead: at most 1 seed in
#: 1,000 may turn the stage red on a pool that satisfies `MAX_GROUP_SHARE`.
#: Over 2,000 seeds the worst case is 4.25 sigma on the real snapshot, 4.19 on
#: the 20,000-row fixture pool and 3.69 on the 1,865-row one, and the real
#: snapshot's 99.9th percentile is 4.16. 5.0 clears every one of those 6,000
#: trials, so the measured false-alarm rate is 0, with 0.75 sigma of margin
#: above the worst observed. The previous 4.0 did not: seeds 1 to 400 already
#: reach 3.92 and the sweep to 2,000 reaches 4.25, so the next `dvc exp` seed
#: sweep would have turned the stage red on a sound split.
#:
#: This rule is not what catches an undersized set - `MIN_SHARE_OF_RATIO` is.
#: It catches *scatter*: shares that move more than independent assignment can
#: explain, which is the signature of a broken bucketing rather than of a set
#: that came out small. See `reports/analysis/split_gate.py`.
DISPERSION_SIGMAS = 5.0

#: How far the ratios may sum away from 1 before the split refuses to run.
_RATIO_SUM_EPSILON = 1e-9

#: Resolution of the group hash. A group's bucket is its hash modulo this,
#: so the ratios are honoured to within one bucket.
_BUCKETS = 10_000

app = typer.Typer()


def assign_split(groups: pd.Series, ratios: dict[str, float], seed: int) -> pd.Series:
    """Map each group key to one of `SPLIT_NAMES`, deterministically.

    Hashing the group rather than shuffling the rows means the assignment
    depends on the seed and the key alone: the same seller lands in the same
    set whatever order the frame arrives in, and no seller is ever split.
    """
    missing = sorted(set(SPLIT_NAMES) - set(ratios))
    if missing:
        raise ValueError(f"no ratio given for {', '.join(missing)}")

    total = sum(ratios[name] for name in SPLIT_NAMES)
    if abs(total - 1.0) > _RATIO_SUM_EPSILON:
        # Without this a one-character edit in params.yaml silently produces an
        # empty calibration set, so the UC2 intervals calibrate on nothing, or
        # an empty test set, so every success criterion is computed on no rows.
        raise ValueError(f"the split ratios must sum to 1, but {ratios} sums to {total}")

    if groups.isna().any():
        # The interim contract declares the group key non-nullable and the read
        # refuses a null, so this is unreachable through the stage. It is here
        # because the failure would be silent rather than loud: the key is
        # stringified, so `None` and `float("nan")` would land in different
        # sets, and the pool the bounds are derived from drops nulls while the
        # realised sizes count them.
        raise ValueError(
            f"{int(groups.isna().sum()):,} row(s) carry no group key, so they cannot be "
            f"grouped; a null seller key is a bug in the stage that wrote the frame"
        )

    edges: list[tuple[str, float]] = []
    cumulative = 0.0
    for name in SPLIT_NAMES:
        cumulative += ratios[name]
        edges.append((name, cumulative))

    def bucket(key: str) -> str:
        digest = hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()
        position = int(digest[:8], 16) % _BUCKETS / _BUCKETS
        for name, edge in edges:
            if position < edge:
                return name
        # Only reachable through floating-point slack at the very top of the
        # range, because the ratios are checked above to sum to 1.
        return SPLIT_NAMES[-1]

    return groups.map(bucket)


def supported_makes(makes: pd.Series, min_listings: int) -> list[dict[str, object]]:
    """The makes with at least `min_listings` listings, largest first (EDN-05).

    Each entry carries the count that admitted it, so a reader sees why every
    make is in the list and the smallest count shows that none was let in below
    the threshold.

    Ties are broken by name, so the artefact depends on the counts alone and not
    on which make happened to appear first in the frame (NFR-06).
    """
    counts = makes.value_counts()
    supported = counts[counts >= min_listings]
    if supported.empty:
        # An empty list makes FR-04 reject every request, and nothing
        # downstream would notice: the pipeline would go on training a model
        # the API refuses to serve a single prediction from.
        largest = f"{int(counts.iloc[0]):,}" if len(counts) else "none at all"
        raise ValueError(
            f"no make reaches {min_listings} listings in {len(makes):,} rows, so the "
            f"supported-make list would be empty; the largest make has {largest}"
        )
    ordered = sorted(supported.items(), key=lambda entry: (-entry[1], entry[0]))
    return [{"make": str(make), "listings": int(listings)} for make, listings in ordered]


def write_supported_makes(
    makes: list[dict[str, object]], path: Path, min_listings: int, counted_over: int
) -> None:
    """Write the make list with the threshold and the population it was counted over.

    Without those two numbers a reader sees `{"make": "Volkswagen",
    "listings": 348}` and cannot tell whether 348 cleared the bar or what it is
    348 out of, so the artefact cannot in fact be audited on its own. They are
    not a second source of truth that can drift from params.yaml: they are
    emitted by the same call that applied them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "min_listings_per_make": int(min_listings),
        "counted_over_rows": int(counted_over),
        "supported_makes": makes,
    }
    write_json(document, path)
    logger.success(f"Wrote {len(makes)} supported make(s) to {path}.")


def realised_shares(sizes: dict[str, int]) -> dict[str, float]:
    """Each set's share of the rows that were split, at the sizes it came out at."""
    total = sum(sizes.values())
    if not total:
        raise ValueError("no row was assigned to any set, so there are no shares to report")
    return {name: size / total for name, size in sizes.items()}


def dispersion_bounds(
    group_sizes: pd.Series,
    ratios: dict[str, float],
    max_bound: float = MAX_DISPERSION_BOUND,
) -> dict[str, float]:
    """How far each set's share may scatter from its ratio, from the group sizes.

    Sellers are assigned independently of each other, so with group sizes `n_i`
    over a pool of `N` rows the realised share of a set with ratio `p` has
    standard deviation `sqrt(p (1 - p) * sum(n_i^2)) / N`. The bound is
    `DISPERSION_SIGMAS` of that, capped at `max_bound`.

    The cap is the point. Uncapped, this quantity is derived from the realised
    concentration, so it grows in exactly the case where a split deserves more
    suspicion rather than less: a single group holding the whole pool derives a
    196 pp allowance, and a mis-set `group_key` derived 101 pp. The cap keeps
    the quantity useful as what it is - a scatter check - while
    `MIN_SHARE_OF_RATIO` and `MAX_GROUP_SHARE` carry the rules that an
    undersized or concentrated split has to fail.
    """
    pool = int(group_sizes.sum())
    if not pool:
        raise ValueError("there are no groups to split, so no dispersion bound can be derived")
    spread = math.sqrt(int((group_sizes.astype("int64") ** 2).sum())) / pool
    return {
        name: min(
            DISPERSION_SIGMAS * spread * math.sqrt(ratios[name] * (1 - ratios[name])),
            max_bound,
        )
        for name in SPLIT_NAMES
    }


@dataclass(frozen=True)
class GateLimits:
    """The three declared numbers `check_ratios` judges a realised split against.

    One object rather than three loose arguments, so the gate's calibration can
    be read, passed and overridden as a single thing, and so a test that opens
    up one rule to observe another says so at the call site.
    """

    min_share_of_ratio: float = MIN_SHARE_OF_RATIO
    max_group_share: float = MAX_GROUP_SHARE
    max_dispersion_bound: float = MAX_DISPERSION_BOUND


#: What the stage judges itself against. A bare `check_ratios` call uses it, so
#: the gate cannot be weakened by a caller that forgets to pass a limit.
DECLARED_LIMITS = GateLimits()


def check_ratios(
    sizes: dict[str, int],
    group_sizes: pd.Series,
    ratios: dict[str, float],
    limits: GateLimits = DECLARED_LIMITS,
) -> dict[str, dict[str, float]]:
    """Refuse a split whose realised sizes are not the ones that were asked for.

    Returns the realised shares, the deviations and the bounds in one mapping,
    so the caller logs the numbers this function already computed instead of
    deriving them a second time and risking a different answer.

    This has to fail rather than warn: the sets it distorts are the ones every
    reported number rests on - an undersized test set makes SC-04 stop checking
    segments, an undersized calibration set widens every UC2 interval - and
    nobody reads a share in a log line on a green run.

    Four rules, each named in the failure so the message says which one fired,
    and all of them reported together rather than only the first:

    - `non-empty`, because a set with no rows is the worst case and the
      cheapest to state.
    - `floor`, a set at less than `limits.min_share_of_ratio` of its configured
      share. Unconditional, and the rule that catches an undersized set.
    - `concentration`, one group above `limits.max_group_share` of the pool.
      This is the declared expectation the dispersion bound rests on; when it
      breaks, the bound is not trustworthy and that, rather than the bound, is
      what the stage should say.
    - `dispersion`, a share scattered further than independent assignment can
      explain, bounded by `dispersion_bounds`.

    `limits` defaults to the declared constants, so a bare call is gated:
    nothing about the safety of this function depends on its caller remembering
    to pass one.
    """
    shares = realised_shares(sizes)
    bounds = dispersion_bounds(group_sizes, ratios, limits.max_dispersion_bound)
    deviations = {name: shares[name] - ratios[name] for name in SPLIT_NAMES}
    least = limits.min_share_of_ratio

    broken: list[str] = []

    empty = [name for name in SPLIT_NAMES if not sizes[name]]
    if empty:
        broken.append(f"[non-empty] {', '.join(empty)} came out with no rows at all")

    # An empty set is reported as empty and not also as below the floor: the
    # second statement adds nothing to the first.
    under = [name for name in SPLIT_NAMES if sizes[name] and shares[name] < ratios[name] * least]
    if under:
        broken.append(
            "[floor] "
            + "; ".join(
                f"{name} came out at {shares[name]:.1%}, "
                f"{shares[name] / ratios[name]:.0%} of its configured {ratios[name]:.0%}, "
                f"where {least:.0%} is the least allowed"
                for name in under
            )
        )

    pool = int(group_sizes.sum())
    if pool:
        largest = int(group_sizes.max())
        if largest / pool > limits.max_group_share:
            broken.append(
                f"[concentration] one group holds {largest:,} of the {pool:,} pooled rows "
                f"({largest / pool:.1%}), above the {limits.max_group_share:.0%} this gate "
                f"is calibrated for, so no grouped split can honour the ratios and the "
                f"dispersion bound cannot be trusted; check that `group_key` names the "
                f"seller column"
            )

    scattered = [name for name in SPLIT_NAMES if abs(deviations[name]) > bounds[name]]
    if scattered:
        broken.append(
            "[dispersion] "
            + "; ".join(
                f"{name} came out at {shares[name]:.1%} against {ratios[name]:.0%}, off by "
                f"{deviations[name]:+.1%} where {bounds[name]:.1%} is allowed"
                for name in scattered
            )
        )

    if broken:
        raise ValueError(
            "the realised split is not the one that was asked for: "
            + " | ".join(broken)
            + ". Whole sellers are assigned at a time, so a different `seed` in params.yaml "
            "moves them; a miss that survives a seed change is a property of the data or of "
            "`group_key`, not of the seed."
        )

    return {"shares": shares, "deviations": deviations, "bounds": bounds}


def cross_holdout_sellers(holdout_groups: pd.Series, pool_groups: pd.Series) -> set[str]:
    """The sellers that are in the country holdout *and* in the split, if any.

    Two invariants of this stage cannot both hold for a dealer that lists in
    the holdout country and elsewhere: "every `ES` row is held out" (EDN-03)
    and "no seller appears in two sets" (EDN-14). The holdout is selected per
    row, so the first one wins, and this function is how the cost of that
    choice is reported instead of being discovered later.

    `ES` wins because EDN-03 defines the holdout as the `ES` listings and
    prices its cost as exactly those rows: holding out whole sellers would put
    non-`ES` rows into an artefact that M6 replays as Spanish traffic, which
    changes what the drift measurement measures, and it would make the holdout
    a different size than the 6,079 rows the dataset card and EDN-03 state. On
    the real snapshot the conflict never arises - `hash_seller_group` keys a
    dealer by its company name, and no dealer name in the snapshot appears both
    in `ES` and outside it - so holding out whole sellers would buy nothing
    there while costing EDN-03 its definition.

    It does arise on the synthetic fixture, whose dealer names are drawn
    independently of the country, and EDN-14's finding is why that matters
    rather than being a curiosity: dealer-level shift is as large as
    country-level shift, so a replay containing dealers the model trained on
    understates drift and confounds both NFR-11's flagging and FR-15's retrain
    comparison. That is a real limitation of the fixture as a drift-rehearsal
    stand-in, which is why the count is reported rather than hidden.
    """
    return set(holdout_groups.unique()) & set(pool_groups.unique())


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "listings.parquet",
    output_dir: Path = PROCESSED_DATA_DIR,
    params_path: Path = PARAMS_FILE,
):
    params = load_params(params_path)
    split_params = params["split"]
    frame = read_frame(input_path, INTERIM_SCHEMA)
    group_key = split_params["group_key"]

    # 1. The holdout leaves first, so nothing below can see an `ES` row (EDN-03).
    is_holdout = frame["country_code"] == split_params["holdout_country"]
    holdout, remaining = frame[is_holdout], frame[~is_holdout]
    if remaining.empty:
        # Without this the stage dies inside the make count, whose message
        # blames the support threshold for a frame that holds nothing but `ES`.
        raise ValueError(
            f"every one of the {len(frame):,} interim rows carries "
            f"{split_params['holdout_country']}, so the holdout takes the whole frame and "
            f"there is nothing left to split"
        )

    # 2. The supported makes are an artefact, never a hard-coded list: the API
    #    reads them from the model metadata (FR-04). Counted here, after the
    #    holdout and before the split, so neither an `ES` listing nor the luck
    #    of which set a seller landed in can decide whether a make is in scope.
    makes = supported_makes(remaining["make"], split_params["min_listings_per_make"])
    covered = sum(int(entry["listings"]) for entry in makes)

    # 3. Whole sellers at a time, so no seller appears in two sets (EDN-14).
    groups = remaining[group_key]
    assigned = assign_split(groups, split_params["ratios"], params["seed"])
    subsets = {name: remaining[assigned == name] for name in SPLIT_NAMES}
    sizes = {name: len(subset) for name, subset in subsets.items()}
    group_sizes = groups.value_counts()

    # Assign, check, then write, so that a rejected split leaves the previous
    # run's artefacts intact. Checking after the writes means the stage fails
    # with train.parquet, test.parquet and supported_makes.json already
    # overwritten, which is the one state nobody can recover from.
    report = check_ratios(sizes, group_sizes, split_params["ratios"])

    write_frame(holdout, output_dir / f"{HOLDOUT_NAME}.parquet", PROCESSED_SCHEMA)
    write_supported_makes(
        makes,
        output_dir / MAKES_FILE,
        split_params["min_listings_per_make"],
        len(remaining),
    )
    for name, subset in subsets.items():
        write_frame(subset, output_dir / f"{name}.parquet", PROCESSED_SCHEMA)

    logger.info(
        f"{split_params['holdout_country']} holdout: {len(holdout):,} rows, "
        f"{len(holdout) / max(len(frame), 1):.1%} of the interim frame; "
        f"{len(remaining):,} rows left to split."
    )
    logger.info(
        f"{len(makes)} make(s) reach {split_params['min_listings_per_make']} listings, "
        f"covering {covered:,} of the {len(remaining):,} rows being split "
        f"({covered / max(len(remaining), 1):.1%})."
    )
    logger.info(f"{len(group_sizes):,} seller groups, largest {int(group_sizes.iloc[0]):,} rows.")
    for name in SPLIT_NAMES:
        logger.info(
            f"{name}: {sizes[name]:,} rows, {report['shares'][name]:.1%} of the split pool, "
            f"{report['deviations'][name]:+.1%} against the configured "
            f"{split_params['ratios'][name]:.0%} (within {report['bounds'][name]:.1%})."
        )

    crossing = cross_holdout_sellers(holdout[group_key], groups)
    if crossing:
        in_holdout = int(holdout[group_key].isin(crossing).sum())
        in_split = int(groups.isin(crossing).sum())
        logger.warning(
            f"{len(crossing):,} seller(s) list both in "
            f"{split_params['holdout_country']} and outside it, so they appear in the "
            f"holdout ({in_holdout:,} rows) and in a split set ({in_split:,} rows). "
            f"Every holdout-country row is held out, which is the rule EDN-03 fixes, so "
            f"the holdout is not seller-disjoint from the split. A replay of it therefore "
            f"contains dealers the model trained on, which understates drift (EDN-14)."
        )
    else:
        logger.info(
            f"No seller lists both in {split_params['holdout_country']} and outside it, so "
            f"the holdout is seller-disjoint from the split as well."
        )


if __name__ == "__main__":
    app()
