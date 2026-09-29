"""`configure_gx` stage: build the Great Expectations context, suites and checkpoint.

STUB. Issue #25 implements it: a file data context under `gx/`, one suite for
the raw frame and one for the processed frame, a validation definition per
frame and a checkpoint that updates the Data Docs.

The stage exists now so the pipeline's shape is settled and #25 fills one module
without touching anyone else's. Two things it should do differently from the
course demo, both found while building this skeleton:

- **Give the stage an `outs`.** The demo's `configure_gx` declares none, which
  makes it a disconnected node in the DVC graph: nothing depends on it, so DVC
  does not guarantee it runs before `validate-data`. Declaring what it writes
  under `gx/` connects the two.
- **Make it idempotent.** Adding a data source that already exists raises
  `DataContextError`, so the second `dvc repro` on a machine fails unless the
  script gets-or-creates. The committed config is also UUID-laden and differs
  between runs, so `gx/` is checked-in configuration, not something to
  regenerate and diff in CI.

Structural validation does not wait for this stage: `validate_data` already
checks both frames against `recommenditos.schema`. Great Expectations adds the
value rules on top (ranges, fill rates, the supported make list).
"""

from loguru import logger
import typer

app = typer.Typer()


@app.command()
def main():
    logger.warning(
        "STUB: the Great Expectations context is configured by issue #25. Until then "
        "`validate-data` enforces the structural contract from recommenditos.schema."
    )


if __name__ == "__main__":
    app()
