"""`validate-data` stage: the data gate. A violation fails the pipeline.

Today it enforces the *structural* contract of `recommenditos.schema` on the
raw and the interim frame. Issue #25 adds the Great Expectations suites, which
enforce the *value* rules on top: the training price range, the mileage range,
the non-null columns, the supported make list, and `registration_date <=
reference date` as a hard expectation on the processed data and with
`mostly=0.99` on the raw data.

Whatever #25 adds, this stage keeps the property the course demo's equivalent
lacks: **a failing expectation fails the stage.** The demo only logs the number
of failures and leaves the actual gate to `tests/test_data.py`, which means a
`dvc repro` can produce a model from data that never passed validation.
"""

from pathlib import Path

from loguru import logger
import typer

from recommenditos.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from recommenditos.pipeline import read_frame
from recommenditos.schema import INTERIM_SCHEMA, RAW_SCHEMA, SchemaError

app = typer.Typer()


@app.command()
def main(
    raw_path: Path = RAW_DATA_DIR / "listings.parquet",
    interim_path: Path = INTERIM_DATA_DIR / "listings.parquet",
):
    failures = []
    for path, schema in ((raw_path, RAW_SCHEMA), (interim_path, INTERIM_SCHEMA)):
        try:
            read_frame(path, schema)
        except SchemaError as error:
            failures.append(f"{path}: {error}")
        else:
            logger.success(f"{path} satisfies the {schema.name!r} contract.")

    if failures:
        joined = "\n\n".join(failures)
        raise SchemaError(f"data validation failed for {len(failures)} artefact(s):\n\n{joined}")

    logger.warning(
        "STUB: only the structural contract is checked. The Great Expectations value "
        "rules arrive with issue #25."
    )


if __name__ == "__main__":
    app()
