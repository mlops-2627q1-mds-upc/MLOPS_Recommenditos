"""`comparables` stage: the index of real listings that FR-09 returns for UC2.

`POST /v1/comparables` answers a car description with up to 20 listings of the
same make and model, so that a user can check a price range against the market.
Those listings come from this index, which ships in the API image beside the
model (FR-12, EDN-08): a versioned pipeline output, not a query against a
database.

The index holds the `train` and `validation` listings and nothing else
(`SOURCES`). The `ES` holdout is M6's new market (EDN-03), and the `test` and
`calibration` rows are where SC-01 to SC-05 are measured, so a listing from any
of them must stay unseen by everything the API serves. The stage reads no other
file, and it refuses a row from the holdout country in case a changed `split`
ever let one through.

The rows are cut to `COMPARABLES_SCHEMA`: the key features and the price, and
no PII column (NFR-08), because the contract selects its columns. A row without
a `make` is dropped, since `make` is the one field every request carries and
such a row could never be returned. Nothing else is filtered: the windows that
make a listing comparable are the API's (`recommenditos/api/comparables.py`),
so the index answers any request the API accepts.

`age_years` is the listing's age on the snapshot date (`reference_date`),
computed by the function the `features` stage uses, so it is the age the
listing's price was asked at. The API compares it with the age of the
requested car today.

The rows are sorted by make, model, age, mileage and price, so the file is the
same bytes on every run and a tie between two listings the API cannot otherwise
order is broken the same way every time.
"""

from collections.abc import Mapping
from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from recommenditos.config import PARAMS_FILE, PROCESSED_DATA_DIR
from recommenditos.data.build_features import age_years
from recommenditos.pipeline import load_params, read_frame, write_frame
from recommenditos.schema import DERIVED_COLUMNS, PROCESSED_SCHEMA

app = typer.Typer()

#: The `split` frames the index is built from (FR-09). Never `test`,
#: `calibration` or the `ES` holdout: see the module docstring.
SOURCES: tuple[str, ...] = ("train", "validation")

#: The index's file name in `data/processed/`, the API's default comparables file.
INDEX_FILE = "comparables.parquet"

#: The index's contract: a selection of the processed columns, the key features
#: a user judges a listing by and the ones the API filters and sorts on. Because
#: `conform` selects, no PII column (NFR-08) and not the `seller_group_id`
#: pseudonym can reach the index. `registration_date` is the year and month as a
#: request spells it. It lives here rather than in `recommenditos/schema.py`
#: because every stage depends on that module, so a contract added there would
#: rerun the whole pipeline for one new artefact.
COMPARABLES_SCHEMA = (
    PROCESSED_SCHEMA.select(
        (
            "make",
            "model",
            "model_version",
            "registration_date",
            "mileage_km_raw",
            "power_kw",
            "fuel_category",
            "transmission",
            "body_type",
            "seller_type",
            "country_code",
            "price",
        ),
        name="comparables",
    )
    .with_dtype("registration_date", "str", name="comparables")
    .extend(DERIVED_COLUMNS, name="comparables")
)

#: The order of the rows, which also breaks the API's remaining ties.
_SORT_KEYS: tuple[str, ...] = ("make", "model", "age_years", "mileage_km_raw", "price")


def build_index(
    frames: Mapping[str, pd.DataFrame], reference_date: str, holdout_country: str
) -> pd.DataFrame:
    """The comparables index of `frames`, which are the `split` frames of `SOURCES`.

    Raises `ValueError` for a frame outside `SOURCES` and for a row of the
    holdout country, because either would put a listing in the index that
    FR-09 keeps out of it.
    """
    unexpected = sorted(set(frames) - set(SOURCES))
    if unexpected:
        raise ValueError(
            f"the comparables index is built from {', '.join(SOURCES)} only (FR-09), "
            f"not from {', '.join(unexpected)}"
        )
    frame = pd.concat([frames[name] for name in SOURCES if name in frames], ignore_index=True)
    held_out = int((frame["country_code"] == holdout_country).sum())
    if held_out:
        raise ValueError(
            f"{held_out:,} row(s) of {', '.join(SOURCES)} carry the holdout country "
            f"{holdout_country}, which no artefact but the holdout may hold (EDN-03)"
        )
    frame = frame[frame["make"].notna()].copy()
    frame["age_years"] = age_years(frame["registration_date"], reference_date)
    frame["registration_date"] = frame["registration_date"].dt.strftime("%Y-%m")
    frame = frame.sort_values(list(_SORT_KEYS), kind="stable", na_position="last")
    return COMPARABLES_SCHEMA.conform(frame.reset_index(drop=True))


@app.command()
def main(
    input_dir: Path = PROCESSED_DATA_DIR,
    output_path: Path = PROCESSED_DATA_DIR / INDEX_FILE,
    params_path: Path = PARAMS_FILE,
):
    params = load_params(params_path)
    frames = {
        name: read_frame(input_dir / f"{name}.parquet", PROCESSED_SCHEMA) for name in SOURCES
    }
    index = build_index(frames, params["reference_date"], params["split"]["holdout_country"])
    write_frame(index, output_path, COMPARABLES_SCHEMA)
    pairs = index[["make", "model"]].drop_duplicates()
    logger.info(
        f"{len(index):,} listings of {pairs['make'].nunique()} makes and {len(pairs):,} "
        f"make-model pairs, {output_path.stat().st_size / 2**20:.1f} MiB on disk."
    )


if __name__ == "__main__":
    app()
