"""`features` stage: turn the split frames into one feature matrix per set.

STUB. It derives `age_years` and selects the columns of the requested feature
set, so every downstream stage reads a schema-valid matrix. This stage has no
equivalent in the course demo, whose text model needs no feature engineering;
it is our addition and the report says so.

Issue #36 adds the encoding: the four equipment lists as multi-hot columns, a
normalised `model_version`, and `weight_kg` parsed out of its text form
(`'1,945 kg'`). The multi-hot columns are data-dependent, so the stage will add
them to the contract with `Schema.extend` rather than to `schema.py`.

Two rules that must survive that work, both from EDN-02 and EDN-15:
categoricals stay categorical - no one-hot encoding, LightGBM and CatBoost
handle them natively - and missing values are never imputed, because the
missingness itself carries signal.
"""

from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from recommenditos.config import PARAMS_FILE, PROCESSED_DATA_DIR
from recommenditos.data.split_data import SPLIT_NAMES
from recommenditos.pipeline import load_params, read_frame, write_frame
from recommenditos.schema import PROCESSED_SCHEMA, feature_schema

#: The `ES` holdout gets features too: M6 replays it against the API.
FEATURE_INPUTS: tuple[str, ...] = (*SPLIT_NAMES, "holdout_es")

_DAYS_PER_YEAR = 365.25

app = typer.Typer()


def age_years(registration_date: pd.Series, reference_date: str) -> pd.Series:
    """Age in years at `reference_date`, null where the registration date is.

    The one place age is computed. The API calls it with the request date, so
    the feature cannot drift between training and serving (problem-spec 4).
    """
    reference = pd.Timestamp(reference_date)
    return (reference - pd.to_datetime(registration_date)).dt.days / _DAYS_PER_YEAR


@app.command()
def main(
    feature_set: str = typer.Argument(..., help="a key of features.sets in params.yaml"),
    input_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR / "features",
    params_path: Path = PARAMS_FILE,
):
    params = load_params(params_path)
    columns = params["features"]["sets"][feature_set]
    schema = feature_schema(columns, name=f"features-{feature_set}")

    logger.warning(
        "STUB: the equipment multi-hot encoding, the normalised model_version and the "
        "weight_kg parse of issue #36 are not applied yet."
    )

    for name in FEATURE_INPUTS:
        frame = read_frame(input_dir / f"{name}.parquet", PROCESSED_SCHEMA)
        frame["age_years"] = age_years(frame["registration_date"], params["reference_date"])
        write_frame(frame, output_dir / feature_set / f"{name}.parquet", schema)


if __name__ == "__main__":
    app()
