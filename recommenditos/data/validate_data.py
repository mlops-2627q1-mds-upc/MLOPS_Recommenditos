"""`validate-data` stage: the data gate. A violation fails the pipeline.

Two layers, in this order, for each of the raw and the interim frame:

1. The **structural** contract of `recommenditos.schema`, enforced by reading
   the frame. A break here is a bug in our code.
2. The **Great Expectations** suite `configure_gx` built for that frame, run
   through its checkpoint. A break here is a change in the data. What the suites
   assert, and why each rule sits where it does, is the docstring of
   `recommenditos.data.gx_context_configuration`.

The Data Docs are rendered once, after both frames, from the results the two
checkpoint runs stored.

A frame that fails its contract is not handed to its suite: the suite's
expectations assume the contract holds, and their failures would only repeat the
contract's in a less precise form.

The cleaned frame this stage checks is the **interim** one. Issue #25 calls it
"the processed frame", but the split has not run at this point in the DAG, so
no `data/processed/` artefact exists yet; interim is the frame that carries the
rules the suite is about, and `split` only moves its rows between files.

Three properties the course demo's equivalent does not have:

- **A failure fails the stage.** The demo logs how many expectations failed and
  leaves the real gate to `tests/test_data.py`, so a `dvc repro` there can
  produce a model from data that never passed validation.
- **Every frame's result is read.** The demo reads the first validation result
  of its checkpoint run and drops the others, so its clean-data suite could fail
  without anything noticing.
- **The gate is in the graph.** The demo's stage declares no `outs`, which makes
  it a leaf nothing depends on, so `dvc repro split` would skip it. This one
  writes its summary and `split` depends on that file, so the pipeline cannot
  reach training without passing through here.

The summary is git-tracked, so it is written to be read in a diff: no run id, no
timestamp, the contract expectations as a count and every rule with what it
found. A data change then shows up in the pull request as the counts that moved.
The full validation results and the Data Docs, which carry the run time, are
DVC outputs beside it.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import great_expectations as gx
from great_expectations.data_context.data_context.context_factory import project_manager
from great_expectations.execution_engine import pandas_execution_engine
from loguru import logger
import pandas as pd
import typer

from recommenditos.config import INTERIM_DATA_DIR, RAW_DATA_DIR
from recommenditos.data.gx_context_configuration import (
    CONTRACT,
    GX_DIR,
    INTERIM,
    KIND,
    RAW,
    RAW_AS_VALIDATED,
    VALIDATION_SUMMARY_FILE,
)
from recommenditos.pipeline import read_frame, write_json
from recommenditos.schema import INTERIM_SCHEMA, RAW_SCHEMA, Schema, SchemaError

#: How many decimals of `unexpected_percent` the summary keeps. Enough to tell
#: 164 rows from 165 in 118,382, few enough that a diff stays readable.
_PERCENT_DECIMALS = 4

#: The result fields a rule reports, where Great Expectations computes them.
_COUNTS = ("element_count", "missing_count", "unexpected_count", "unexpected_percent")

app = typer.Typer()


class DataValidationError(ValueError):
    """A frame broke its contract or failed an expectation of its suite."""


def load_context(context_dir: Path):
    """The context `configure_gx` built, or a failure that says to build it.

    Checked first because `gx.get_context` creates an empty context wherever it
    is pointed, so a missing store would otherwise be replaced by a new one
    inside `configure_gx`'s output, and the stage would fail later, on a
    checkpoint that does not exist, with a message about neither.
    """
    if not (context_dir / "great_expectations.yml").exists():
        raise DataValidationError(
            f"no Great Expectations context in {context_dir}; build it with "
            f"`dvc repro configure_gx` (or `dvc pull gx`) before validating"
        )
    return gx.get_context(mode="file", context_root_dir=context_dir)


def _as_validated(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """The frame in the representation its suite was written for."""
    if name == RAW:
        # Only the date is cast, column by column, so the 75-column frame is
        # not copied whole for one column's sake.
        column = "registration_date"
        cast = RAW_AS_VALIDATED.select([column], name=RAW).conform(frame[[column]])
        return frame.assign(**{column: cast[column]})
    return frame


def _raised(exception_info: dict) -> list[str]:
    """The messages of every exception Great Expectations caught for one result.

    It reports a failure to evaluate as a failed result, which on its own reads
    like a data problem. The info is either one record or one per metric.
    """
    records = [exception_info] if "raised_exception" in exception_info else exception_info.values()
    return [
        str(record.get("exception_message"))
        for record in records
        if isinstance(record, dict) and record.get("raised_exception")
    ]


def _rule_entry(result) -> dict[str, Any]:
    """One rule's result, as the summary records it."""
    config = result.expectation_config.to_json_dict()
    entry: dict[str, Any] = {"expectation": config["type"]}
    entry.update({key: value for key, value in config["kwargs"].items() if key != "batch_id"})
    entry["success"] = bool(result.success)
    for key in _COUNTS:
        value = result.result.get(key)
        if value is None:
            continue
        entry[key] = (
            round(float(value), _PERCENT_DECIMALS) if key.endswith("percent") else int(value)
        )
    observed = result.result.get("observed_value")
    if isinstance(observed, int | float) and not isinstance(observed, bool):
        entry["observed_value"] = observed
    errors = _raised(result.exception_info or {})
    if errors:
        entry["error"] = "; ".join(sorted(set(errors)))
    return entry


def _describe(entry: dict[str, Any]) -> str:
    """A rule's result as one line of a log or an error message."""
    column = entry.get("column", "the table")
    found = ""
    if "unexpected_count" in entry:
        found = f": {entry['unexpected_count']:,} unexpected ({entry['unexpected_percent']} %)"
    elif "observed_value" in entry:
        found = f": observed {entry['observed_value']:,}"
    if "error" in entry:
        found += f", not evaluated: {entry['error']}"
    return f"{entry['expectation']} on {column}{found}"


@contextmanager
def _without_batch_fingerprint() -> Iterator[None]:
    """Stop Great Expectations hashing the whole frame into the batch's markers.

    It fingerprints every in-memory batch whose shallow `memory_usage` is below
    `HASH_THRESHOLD`, and a pandas-3 text column is shallow, so the raw frame
    always qualifies. Measured on the real snapshot, the stage takes 14 s and
    peaks at 4.4 GB with the fingerprint, and 11 s and 2.2 GB without it, most
    of which is the raw frame itself (reports/analysis/gx_fingerprint_cost.py).
    The fingerprint is only a marker in the stored result, and what was
    validated is already pinned by `dvc.lock`. Scoped rather than set once at
    import, so it cannot change how Great Expectations behaves for anyone else
    in the same process.
    """
    previous = pandas_execution_engine.HASH_THRESHOLD
    pandas_execution_engine.HASH_THRESHOLD = 0
    try:
        yield
    finally:
        pandas_execution_engine.HASH_THRESHOLD = previous


def validate_frame(context, name: str, frame: pd.DataFrame) -> dict[str, Any]:
    """Run `name`'s checkpoint on `frame` and summarise what it found.

    Public, so a test can hand a suite a frame the contract would have refused
    first - a PII column, say - and see that the suite refuses it too.

    A suite that checks nothing passes any frame, so an empty one is refused, and
    so is a run that returns fewer results than its suite holds: Great
    Expectations reports an expectation it could not evaluate as a failure, so a
    result that is missing altogether is the one case that would read as a pass.

    `context` is made Great Expectations' current project first. A checkpoint
    looks its suite and validation definition up through a process-wide project,
    which is whichever context was loaded last, so with two contexts in one
    process - a notebook, the test suite - the other one's stores would be read
    and the run would fail on a freshness check, or validate against the wrong
    suite.
    """
    expected = len(context.suites.get(name).expectations)
    if not expected:
        raise DataValidationError(
            f"the {name!r} suite has no expectations, so it would pass any frame; "
            f"rebuild it with `dvc repro configure_gx`"
        )
    project_manager.set_project(context)
    with _without_batch_fingerprint():
        run = context.checkpoints.get(name).run(
            batch_parameters={"dataframe": _as_validated(name, frame)}
        )
    (validation,) = run.run_results.values()
    if len(validation.results) != expected:
        raise DataValidationError(
            f"the {name!r} suite has {expected} expectations but {len(validation.results)} "
            f"results came back, so the missing ones would read as passes"
        )
    contract = [r for r in validation.results if r.expectation_config.meta.get(KIND) == CONTRACT]
    rules = [r for r in validation.results if r.expectation_config.meta.get(KIND) != CONTRACT]
    return {
        "passed": bool(validation.success),
        "rows": len(frame),
        "contract": {
            "expectations": len(contract),
            "failed": sorted(
                _describe(_rule_entry(result)) for result in contract if not result.success
            ),
        },
        # Sorted, because Great Expectations returns results in the order its
        # metric graph resolved them rather than the suite's, and a diff of the
        # summary should move only when a result does.
        "rules": sorted(
            (_rule_entry(result) for result in rules),
            key=lambda entry: (entry.get("column", ""), entry["expectation"]),
        ),
    }


def failures(entry: dict[str, Any]) -> list[str]:
    """Every expectation a frame's summary entry records as failed."""
    if "error" in entry:
        return [entry["error"]]
    failed = list(entry["contract"]["failed"])
    failed.extend(_describe(rule) for rule in entry["rules"] if not rule["success"])
    return failed


@app.command()
def main(
    raw_path: Path = RAW_DATA_DIR / "listings.parquet",
    interim_path: Path = INTERIM_DATA_DIR / "listings.parquet",
    summary_path: Path = VALIDATION_SUMMARY_FILE,
    context_dir: Path = GX_DIR,
) -> dict[str, Any]:
    context = load_context(context_dir)
    frames: tuple[tuple[str, Path, Schema], ...] = (
        (RAW, raw_path, RAW_SCHEMA),
        (INTERIM, interim_path, INTERIM_SCHEMA),
    )

    summary: dict[str, dict[str, Any]] = {}
    for name, path, schema in frames:
        try:
            frame = read_frame(path, schema)
        except SchemaError as error:
            summary[name] = {"passed": False, "rows": None, "error": f"{path}: {error}"}
            continue
        summary[name] = validate_frame(context, name, frame)
        for rule in summary[name]["rules"]:
            if rule["success"] and rule.get("unexpected_count"):
                logger.warning(f"{name}: {_describe(rule)}, within its tolerance")

    # Both before the raise, so a failed run leaves a readable record of which
    # rule broke, and Data Docs that show a sample of the failing values, rather
    # than only a traceback in the DVC log.
    context.build_data_docs()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(summary, summary_path)

    failed = {name: failures(entry) for name, entry in summary.items() if not entry["passed"]}
    if failed:
        listed = "\n".join(
            f"  - {name}: {problem}" for name, problems in failed.items() for problem in problems
        )
        raise DataValidationError(
            f"data validation failed for {len(failed)} frame(s); the Data Docs show a "
            f"sample of the failing values:\n{listed}"
        )
    for name, entry in summary.items():
        logger.success(
            f"{name}: {entry['rows']:,} rows pass the contract and all "
            f"{entry['contract']['expectations'] + len(entry['rules'])} expectations."
        )
    return summary


if __name__ == "__main__":
    app()
