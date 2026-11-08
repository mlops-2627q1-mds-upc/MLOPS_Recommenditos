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
The full validation results and the Data Docs are DVC outputs beside it.

All three are byte-stable: a rerun on the same data and the same `gx/` writes
the same bytes, so `dvc.lock` moves only when the data or the suites do (#289).
Great Expectations dates every result, its path and the Data Docs page drawn
from it with the moment of the run, so the stage names the run after the
snapshot it validates instead (`run_id`), with `reference_date` as its time, and
writes its times in UTC whatever the machine's time zone. `configure_gx` covers
the rest: the results store's id and the Data Docs' element ids.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import datetime as dt
from pathlib import Path
from typing import Any

import great_expectations as gx
from great_expectations.core.run_identifier import RunIdentifier
from great_expectations.data_context.data_context.context_factory import project_manager
from great_expectations.execution_engine import pandas_execution_engine
from great_expectations.render.renderer import site_index_page_renderer
from loguru import logger
import pandas as pd
import typer

from recommenditos.config import INTERIM_DATA_DIR, PARAMS_FILE, RAW_DATA_DIR
from recommenditos.data.gx_context_configuration import (
    CONTRACT,
    GX_DIR,
    INTERIM,
    KIND,
    RAW,
    RAW_AS_VALIDATED,
    VALIDATION_SUMMARY_FILE,
)
from recommenditos.pipeline import load_params, read_frame, write_json
from recommenditos.schema import INTERIM_SCHEMA, RAW_SCHEMA, Schema, SchemaError

#: How many decimals of `unexpected_percent` the summary keeps. Enough to tell
#: 164 rows from 165 in 118,382, few enough that a diff stays readable.
_PERCENT_DECIMALS = 4

#: The result fields a rule reports, where Great Expectations computes them.
_COUNTS = ("element_count", "missing_count", "unexpected_count", "unexpected_percent")

#: How Great Expectations writes a batch's load time.
_LOAD_TIME_FORMAT = "%Y%m%dT%H%M%S.%fZ"

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


def run_id(params: dict[str, Any]) -> RunIdentifier:
    """The run every result of the stage is stored under: the snapshot it checks.

    Great Expectations names a run by the moment it starts, by default, and
    writes that moment into the result's path, its `run_id`, its
    `validation_time` and the Data Docs page drawn from it, so every rerun on the
    same data would write a different result. The snapshot's date is what a
    result is about; when it was computed is in the commit that recorded it.
    """
    snapshot = dt.datetime.fromisoformat(params["reference_date"]).replace(tzinfo=dt.UTC)
    return RunIdentifier(run_name=f"snapshot-{snapshot:%Y-%m-%d}", run_time=snapshot)


class _UTCZone:
    """`tzlocal` as the Data Docs index renderer sees it: always UTC."""

    @staticmethod
    def get_localzone() -> dt.tzinfo:
        return dt.UTC


@contextmanager
def _great_expectations_in_utc() -> Iterator[None]:
    """Make Great Expectations write UTC where it would write the local zone.

    Two places, in Great Expectations 1.24.1, put the machine's time zone into
    what the stage writes: `RunIdentifierSchema` converts a run's time to the
    local zone (`set_run_time_tz(None)`) before it stores a result, and the
    Data Docs index renderer prints each run's time in `tzlocal`'s zone. The
    same run would then read `+01:00` on a laptop in Barcelona, `+02:00` there in
    summer and `+00:00` on a CI runner. Both calls are patched for the length of
    the block, on every platform, and put back afterwards, as
    `_without_batch_fingerprint` does; the stage's test runs it in two zones,
    so an upgrade that moves either call shows there.
    """
    set_run_time_tz = RunIdentifier.set_run_time_tz
    local_zone = site_index_page_renderer.tzlocal

    def set_run_time_tz_utc(self: RunIdentifier, tz: dt.timezone | None) -> None:
        set_run_time_tz(self, tz or dt.UTC)

    # `setattr`, because mypy refuses a direct assignment to a method or a
    # module, which is the point of the patch.
    setattr(RunIdentifier, "set_run_time_tz", set_run_time_tz_utc)  # noqa: B010
    setattr(site_index_page_renderer, "tzlocal", _UTCZone)  # noqa: B010
    try:
        yield
    finally:
        setattr(RunIdentifier, "set_run_time_tz", set_run_time_tz)  # noqa: B010
        setattr(site_index_page_renderer, "tzlocal", local_zone)  # noqa: B010


def _store_with_the_run_time(context, run) -> None:
    """Store each result again with the batch's load time set to the run's time.

    The load time is the one moment left in a result once the run has a fixed
    id: the pandas engine stamps a batch with the clock when it receives the
    frame, and the stamp is shown on the frame's Data Docs page.
    """
    for key, validation in run.run_results.items():
        markers = validation.meta.get("batch_markers") or {}
        if "ge_load_time" not in markers:
            continue
        markers["ge_load_time"] = run.run_id.run_time.strftime(_LOAD_TIME_FORMAT)
        context.validation_results_store.set(key, validation)


def validate_frame(
    context, name: str, frame: pd.DataFrame, run: RunIdentifier | None = None
) -> dict[str, Any]:
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

    `run` is the run to store the result under; without one, Great Expectations
    names it by the moment it starts, which is enough for a check that reads
    only the summary.
    """
    expected = len(context.suites.get(name).expectations)
    if not expected:
        raise DataValidationError(
            f"the {name!r} suite has no expectations, so it would pass any frame; "
            f"rebuild it with `dvc repro configure_gx`"
        )
    project_manager.set_project(context)
    with _without_batch_fingerprint(), _great_expectations_in_utc():
        checkpoint_run = context.checkpoints.get(name).run(
            batch_parameters={"dataframe": _as_validated(name, frame)}, run_id=run
        )
        if run is not None:
            _store_with_the_run_time(context, checkpoint_run)
    (validation,) = checkpoint_run.run_results.values()
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
    params_path: Path = PARAMS_FILE,
) -> dict[str, Any]:
    context = load_context(context_dir)
    run = run_id(load_params(params_path))
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
        summary[name] = validate_frame(context, name, frame, run)
        for rule in summary[name]["rules"]:
            if rule["success"] and rule.get("unexpected_count"):
                logger.warning(f"{name}: {_describe(rule)}, within its tolerance")

    # Both before the raise, so a failed run leaves a readable record of which
    # rule broke, and Data Docs that show a sample of the failing values, rather
    # than only a traceback in the DVC log.
    with _great_expectations_in_utc():
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
