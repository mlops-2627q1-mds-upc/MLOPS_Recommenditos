"""`validate-data` stage: the data gate. A violation fails the pipeline.

Today it enforces the *structural* contract of `recommenditos.schema` on the
raw and the interim frame. Issue #25 adds the Great Expectations suites, which
enforce the value rules on top: the training price range, the mileage range,
the non-null columns, the supported make list, and `registration_date <=
reference date` as a hard expectation on the cleaned frame and with
`mostly=0.99` on the raw one.

The cleaned frame this stage checks is the **interim** one. Issue #25 calls it
"the processed frame", but the split has not run at this point in the DAG, so
no `data/processed/` artefact exists yet; interim is the frame that carries the
rules the suite is about.

Two properties the course demo's equivalent does not have, and that should
survive whatever #25 adds:

- **A failure fails the stage.** The demo logs how many expectations failed and
  leaves the real gate to `tests/test_data.py`, so a `dvc repro` there can
  produce a model from data that never passed validation.
- **The gate is in the graph.** The demo's stage declares no `outs`, which makes
  it a leaf nothing depends on, so `dvc repro split` would skip it. This one
  writes its summary and `split` depends on that file, so the pipeline cannot
  reach training without passing through here.
"""

import json
from pathlib import Path

from loguru import logger
import typer

from recommenditos.config import INTERIM_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR
from recommenditos.pipeline import read_frame
from recommenditos.schema import INTERIM_SCHEMA, RAW_SCHEMA, SchemaError

app = typer.Typer()


@app.command()
def main(
    raw_path: Path = RAW_DATA_DIR / "listings.parquet",
    interim_path: Path = INTERIM_DATA_DIR / "listings.parquet",
    summary_path: Path = REPORTS_DIR / "data-validation" / "summary.json",
):
    summary: dict[str, dict] = {}
    failures = []

    for path, schema in ((raw_path, RAW_SCHEMA), (interim_path, INTERIM_SCHEMA)):
        try:
            frame = read_frame(path, schema)
        except SchemaError as error:
            summary[schema.name] = {"passed": False, "rows": None}
            failures.append(f"{path}: {error}")
        else:
            summary[schema.name] = {"passed": True, "rows": len(frame)}
            logger.success(f"{path} satisfies the {schema.name!r} contract.")

    # Written before the raise, so a failed run leaves a readable record of
    # which artefact broke rather than only a traceback in the DVC log.
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if failures:
        joined = "\n\n".join(failures)
        raise SchemaError(f"data validation failed for {len(failures)} artefact(s):\n\n{joined}")

    logger.warning(
        "STUB: only the structural contract is checked. The Great Expectations value "
        "rules arrive with issue #25."
    )


if __name__ == "__main__":
    app()
