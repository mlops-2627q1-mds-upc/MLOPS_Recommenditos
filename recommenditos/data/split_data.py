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
a *record* here, and the stages that turn a set into model input are where it
has to be applied: problem-spec section 5 describes the split as removing
nothing but `ES`, while section 2 scopes the *model* to the supported makes.
Issue #35's pull request records why it was read that way and what follows for
#36, #37 and #39 - a gate measured over makes the API rejects reports a number
the product can never deliver.

Two properties the bucketing below is built for, both worth more than tighter
ratios. It is order-independent, so a clean clone reproduces the same split from
`seed` and the group key alone (NFR-06). And it is stable as the data grows: a
seller keeps its set when rows are added, so the retrain-with-`ES` run of FR-15
can be compared against the current model instead of being confounded by
sellers that moved between train and test.
"""

import hashlib
import json
import math
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from recommenditos.config import INTERIM_DATA_DIR, PARAMS_FILE, PROCESSED_DATA_DIR
from recommenditos.pipeline import load_params, read_frame, write_frame
from recommenditos.schema import INTERIM_SCHEMA, PROCESSED_SCHEMA

#: The sets `split` writes, besides the country holdout. Downstream stages
#: iterate this, so adding a set is a change here and in params.yaml only.
SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "calibration", "test")

#: The holdout's artefact name. `dvc.yaml` and the `features` stage name the
#: same file, so it is spelled out rather than derived from `holdout_country`: a
#: changed country would otherwise produce a differently named artefact that the
#: DAG does not know about, and `dvc repro` would report a missing output
#: instead of a holdout nobody reads.
HOLDOUT_NAME = "holdout_es"

#: The make list, under the name the API's metadata (FR-04) is fed from.
MAKES_FILE = "supported_makes.json"

#: How many standard deviations a realised share may sit from its ratio.
#:
#: Whole sellers are assigned at a time, so a set's share is a sum over sellers
#: assigned independently of each other, and how far it can honestly land from
#: its ratio follows from the group sizes rather than from taste: see
#: `ratio_tolerances`. Four sigma is the widest deviation the measurement leaves
#: room for - over 200 seeds the worst case reaches 3.5 sigma on the real
#: snapshot and 3.1 on the synthetic fixture - so a sound split never trips it
#: while a set that came out at a fraction of its size still does.
RATIO_SIGMAS = 4.0

#: How far the ratios may sum away from 1 before the split refuses to run.
_RATIO_TOLERANCE = 1e-9

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
    if abs(total - 1.0) > _RATIO_TOLERANCE:
        # Without this a one-character edit in params.yaml silently produces an
        # empty calibration set, so the UC2 intervals calibrate on nothing, or
        # an empty test set, so every success criterion is computed on no rows.
        raise ValueError(f"the split ratios must sum to 1, but {ratios} sums to {total}")

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

    Each entry carries the count that admitted it, so the artefact can be
    audited on its own: a reader sees why every make is in the list, and the
    smallest count shows that none was let in below the threshold. The threshold
    stays in params.yaml, which `dvc.lock` ties to this exact artefact, instead
    of being copied in here where the copy could drift from the value that was
    actually applied.

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


def write_supported_makes(makes: list[dict[str, object]], path: Path) -> None:
    """Write the make list under the one key FR-04's metadata is read from."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"supported_makes": makes}, indent=2) + "\n", encoding="utf-8")
    logger.success(f"Wrote {len(makes)} supported make(s) to {path}.")


def realised_shares(sizes: dict[str, int]) -> dict[str, float]:
    """Each set's share of the rows that were split, at the sizes it came out at."""
    total = sum(sizes.values())
    if not total:
        raise ValueError("no row was assigned to any set, so there are no shares to report")
    return {name: size / total for name, size in sizes.items()}


def ratio_tolerances(group_sizes: pd.Series, ratios: dict[str, float]) -> dict[str, float]:
    """How far each set's share may sit from its ratio, from the group sizes alone.

    Sellers are assigned independently of each other, so with group sizes `n_i`
    over a pool of `N` rows the realised share of a set with ratio `p` has
    standard deviation `sqrt(p (1 - p) * sum(n_i^2)) / N`. The tolerance is
    `RATIO_SIGMAS` of that, which makes it a property of the data rather than a
    number someone liked: it tightens as the pool gains sellers and widens when
    one seller is large enough that no grouped split could place it neatly.

    A fixed percentage cannot do this job. To hold on the 1,873-row synthetic
    fixture it would have to allow 6.8 points, which is the worst of 200 seeds
    there, and that width would then accept a calibration set at a third of its
    intended size on the real snapshot, where the group sizes allow 4.5 points
    and the worst of 200 seeds reaches 4.4. Neither bound is narrow, and that is
    the honest answer rather than a slack one: one dealer holds 3.4 % of the real
    pool, so no seller-grouped split can place it without moving a share.
    """
    pool = int(group_sizes.sum())
    if not pool:
        raise ValueError("there are no groups to split, so no tolerance can be derived")
    spread = math.sqrt(int((group_sizes.astype("int64") ** 2).sum())) / pool
    return {
        name: RATIO_SIGMAS * spread * math.sqrt(ratios[name] * (1 - ratios[name]))
        for name in SPLIT_NAMES
    }


def check_ratios(
    sizes: dict[str, int], group_sizes: pd.Series, ratios: dict[str, float]
) -> dict[str, float]:
    """Refuse a split whose realised shares miss their ratios by more than the data allows.

    Returns the deviations, so the caller can log what it got as well as what
    it wanted. This has to fail rather than warn: the sets it distorts are the
    ones every reported number rests on - an undersized test set makes SC-04
    stop checking segments, an undersized calibration set widens every UC2
    interval - and nobody reads a share in a log line on a green run.
    """
    shares = realised_shares(sizes)
    tolerances = ratio_tolerances(group_sizes, ratios)
    deviations = {name: shares[name] - ratios[name] for name in SPLIT_NAMES}
    over = [name for name in SPLIT_NAMES if abs(deviations[name]) > tolerances[name]]
    if over:
        missed = "; ".join(
            f"{name} came out at {shares[name]:.1%} against {ratios[name]:.0%}, "
            f"off by {deviations[name]:+.1%} where {tolerances[name]:.1%} is allowed"
            for name in over
        )
        raise ValueError(
            f"the realised split misses its ratios by more than the group sizes allow: "
            f"{missed}. Whole sellers are assigned at a time, so a different `seed` in "
            f"params.yaml moves them; a miss this large that survives a seed change means "
            f"one seller holds more rows than a set is meant to."
        )
    return deviations


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR / "listings.parquet",
    output_dir: Path = PROCESSED_DATA_DIR,
    params_path: Path = PARAMS_FILE,
):
    params = load_params(params_path)
    split_params = params["split"]
    frame = read_frame(input_path, INTERIM_SCHEMA)

    # 1. The holdout leaves first, so nothing below can see an `ES` row (EDN-03).
    is_holdout = frame["country_code"] == split_params["holdout_country"]
    holdout, remaining = frame[is_holdout], frame[~is_holdout]
    write_frame(holdout, output_dir / f"{HOLDOUT_NAME}.parquet", PROCESSED_SCHEMA)
    logger.info(
        f"{split_params['holdout_country']} holdout: {len(holdout):,} rows, "
        f"{len(holdout) / max(len(frame), 1):.1%} of the interim frame; "
        f"{len(remaining):,} rows left to split."
    )

    # 2. The supported makes are an artefact, never a hard-coded list: the API
    #    reads them from the model metadata (FR-04). Counted here, after the
    #    holdout and before the split, so neither an `ES` listing nor the luck
    #    of which set a seller landed in can decide whether a make is in scope.
    makes = supported_makes(remaining["make"], split_params["min_listings_per_make"])
    write_supported_makes(makes, output_dir / MAKES_FILE)
    covered = sum(int(entry["listings"]) for entry in makes)
    logger.info(
        f"{len(makes)} make(s) reach {split_params['min_listings_per_make']} listings, "
        f"covering {covered:,} of the {len(remaining):,} rows being split "
        f"({covered / max(len(remaining), 1):.1%})."
    )

    # 3. Whole sellers at a time, so no seller appears in two sets (EDN-14).
    groups = remaining[split_params["group_key"]]
    assigned = assign_split(groups, split_params["ratios"], params["seed"])
    sizes: dict[str, int] = {}
    for name in SPLIT_NAMES:
        subset = remaining[assigned == name]
        write_frame(subset, output_dir / f"{name}.parquet", PROCESSED_SCHEMA)
        sizes[name] = len(subset)

    group_sizes = groups.value_counts()
    deviations = check_ratios(sizes, group_sizes, split_params["ratios"])
    shares = realised_shares(sizes)
    tolerances = ratio_tolerances(group_sizes, split_params["ratios"])
    logger.info(f"{len(group_sizes):,} seller groups, largest {int(group_sizes.iloc[0]):,} rows.")
    for name in SPLIT_NAMES:
        logger.info(
            f"{name}: {sizes[name]:,} rows, {shares[name]:.1%} of the split pool, "
            f"{deviations[name]:+.1%} against the configured "
            f"{split_params['ratios'][name]:.0%} (within {tolerances[name]:.1%})."
        )


if __name__ == "__main__":
    app()
