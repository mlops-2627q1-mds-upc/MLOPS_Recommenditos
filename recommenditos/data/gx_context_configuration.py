"""`configure_gx` stage: build the Great Expectations context, suites and checkpoints.

What it builds, as a file data context under `gx/`:

- one pandas data source with two dataframe assets, `raw` and `interim`;
- one expectation suite per asset, written from `recommenditos.schema` and
  `params.yaml` rather than by hand;
- one validation definition per asset, and one checkpoint per asset;
- a Data Docs site, rendered from the stored validation results.

`validate-data` runs the two checkpoints and then renders the Data Docs once.
This stage only builds them, so a change to a rule reruns validation without
anyone touching the store by hand.

Five things it does differently from the course demo, each for a reason found
while building it:

- **The store is a build product, not configuration.** Great Expectations
  assigns a fresh UUID to the context, the data source, every asset, batch
  definition, suite, expectation, validation definition and checkpoint each time
  it saves one, and it overwrites an id passed in. Measured on 1.23.2: two builds
  from the same code differ in every id, and so does a second `add_or_update`
  over an existing store. A committed `gx/` would therefore change on every
  rebuild and conflict between any two branches that rebuilt it. So `gx/` is
  this stage's DVC output, cached and gitignored like `models/`, and the source
  of truth that is reviewed is this module. The demo commits its store.
- **The stage has an `outs`, and `validate-data` depends on it.** The demo's
  declares none, which leaves it a disconnected node DVC does not have to run
  first. Here the DAG cannot reach validation without this stage.
- **It rebuilds from scratch, so it is idempotent.** Adding a data source that
  already exists raises `DataContextError`, which is why the demo's script
  fails the second time it runs on a machine; get-or-create would avoid that but
  leave behind a suite or an expectation this module no longer defines. DVC
  deletes the output before running the stage anyway, so `main` does the same
  when it is run by hand.
- **The assets are dataframes, not Parquet paths.** The demo's Parquet assets
  write the absolute path of the author's checkout into `great_expectations.yml`,
  so its store only works on that machine. A dataframe asset holds no path, lets
  `validate-data` validate the frame its contract has just accepted rather than
  read the file a second time, and lets a test hand it the synthetic fixture in
  memory. The price is one checkpoint per frame: `Checkpoint.run` takes one set
  of batch parameters for all of its validation definitions, so it cannot be
  given two different frames in one run. For the same reason the checkpoints
  carry no Data Docs action: it would render the whole site once per frame,
  and `validate-data` renders it once, after both.
- **Running a checkpoint never writes into `gx/`.** The validation results and
  the Data Docs go to `reports/data-validation/`, which is `validate-data`'s
  output. Inside `gx/` they would modify this stage's output every time
  validation ran, and `dvc status` would report it changed after every run.

What the suites check, and where each rule of issue #25 lives:

- **The contract, generated from `recommenditos.schema`.** The ordered column
  list, every column's dtype and every non-nullable column, for both frames.
  `Schema.validate` already enforces all of it before the suites run, so these
  expectations pass whenever they are reached. They are here because #25 puts
  the schema in the suite, which makes the Data Docs a complete statement of
  what each frame is. Generating them keeps `schema.py` the only place the
  contract is written down. They are weaker than it, and do not replace it:
  Great Expectations compares only a column's scalar type, so it cannot tell a
  nullable, extension or `object` dtype from its counterpart. Measured on
  1.23.2, `datetime64[us]` passes `datetime64[ns]`, a nullable `boolean` passes
  `bool`, `Float64` passes `float64`, and an `object` or `string[python]` text
  column passes `str`. The dtype a contract declares is checked by `schema.py`
  alone (EDN-31 found the datetime unit gap in Pandera too).
- **The interim column list is the PII guarantee.** It names neither a column
  NFR-08 forbids nor `is_used`, which EDN-23 excludes as a feature, so a frame
  carrying either fails it.
- **`registration_date` at or before `reference_date` (EDN-22).** Hard on the
  interim frame, because preprocessing has dropped the later ones by then. On
  the raw frame with `validate.raw_mostly`, so a future scrape carrying many
  such rows is flagged instead of silently cleaned. Great Expectations cannot
  compare the raw file's text dates with a date: its bounds are typed as numbers
  or dates, a text bound is parsed into a date, and comparing that with a text
  column raises. So the raw suite validates the raw frame with that one column
  parsed, by the same cast the interim contract applies (`RAW_AS_VALIDATED`).
- **The training price range**, hard on the interim frame, from the same
  `preprocess` bounds the scope filter uses. Not on the raw frame: the published
  file runs from 1 EUR to 13.5M EUR, and the range is a filter preprocessing
  applies, so the raw data breaks it by design.
- **The mileage range**, from `validate`, with the same tolerance on both
  frames, `validate.mileage_mostly`. It is a check rather than a filter:
  preprocessing does not apply it, so the published file's three readings above
  1,000,000 km reach the cleaned frame, and the rule reports them on both frames
  while still failing on a systematic break.
- **The fill rate of the columns the published file fills in every row**, from
  `validate.required_filled_mostly`, on both frames: `make`, `body_type` and the
  four equipment lists. Both contracts leave them nullable, because that is a
  measured fill rate and not a structural property (#78). On the raw frame the
  rule reports a gap in a scrape, with the rule named and the row count, in
  this stage rather than as a contract error in `download`; on the cleaned
  frame it reports preprocessing emptying a column.
- **No vacuous pass.** Every column rule holds over no rows and Great
  Expectations skips missing values, so an empty frame, or a bounded column
  that has emptied out, would pass every rule that reads it. Both suites
  therefore require `validate.min_rows` rows, and `registration_date` and
  `mileage_km_raw` filled up to `validate.filled_mostly`; `price` is already
  non-null by contract. `validate-data` also refuses a suite with no
  expectations and a run that returns fewer results than its suite holds.
- **The condition flags** (EDN-23): `has_full_service_history`, `non_smoking`
  and `is_rental` are checked to be boolean and non-null, which the contract
  expectations already do. Nothing checks that `is_used` agrees with
  `offer_type`, deliberately: a `False` there means "not asserted", so the rule
  would fail on the published file by design (18,108 of its 113,708 used
  passenger-car rows).

Not here, although issue #25 lists it: **the supported-make list.** `split`
computes it after this stage has run, and the processed sets deliberately still
contain every make, because `split` is a lossless partition and `features` is
the stage that applies the list (EDN-48, EDN-67). No frame this stage can see is
supposed to satisfy it. The guarantee is enforced where the list is applied, and
checked by `train` and `evaluate` (docs/docs/pipeline.md, "The supported makes").
"""

import os
from pathlib import Path
import shutil
from typing import Any

import great_expectations as gx
from great_expectations.data_context.types.base import ProgressBarsConfig
import great_expectations.expectations as gxe
from loguru import logger
import pandas as pd
import typer

from recommenditos.config import PARAMS_FILE, PROJ_ROOT, REPORTS_DIR
from recommenditos.pipeline import load_params
from recommenditos.schema import INTERIM_SCHEMA, RAW_SCHEMA, Schema

#: The context, this stage's DVC output.
#:
#: The paths of the two validation stages live here rather than in
#: `recommenditos/config.py`, unlike the other stages' roots, because every
#: stage depends on `config.py`: adding them there would rerun the whole
#: pipeline, from the 548 MB download to the four model fits, for a change
#: that concerns two stages. `tests/conftest.py` repeats them for its guard,
#: and a test holds the two copies equal.
GX_DIR = PROJ_ROOT / "gx"

#: `validate-data`'s outputs, kept apart from `GX_DIR` so that running the
#: suites never modifies the stage output that defines them: the git-tracked
#: summary, and the validation results and the Data Docs rendered from them.
VALIDATION_DIR = REPORTS_DIR / "data-validation"
VALIDATION_SUMMARY_FILE = VALIDATION_DIR / "summary.json"
VALIDATION_RESULTS_DIR = VALIDATION_DIR / "results"
DATA_DOCS_DIR = VALIDATION_DIR / "data-docs"

#: The one data source, and the two frames it validates. Each name is also the
#: name of that frame's asset, batch definition, suite, validation definition
#: and checkpoint, so a result found in the store or the Data Docs says which
#: frame it belongs to without a lookup table.
DATASOURCE = "listings"
RAW = "raw"
INTERIM = "interim"
FRAMES: tuple[str, ...] = (RAW, INTERIM)

#: Where an expectation comes from, in its `meta`. `validate-data` reports the
#: contract expectations as a count and the rules one by one, because the first
#: kind passes whenever it is reached and the second is what a reader of the
#: summary is looking for.
KIND = "kind"
CONTRACT = "contract"
RULE = "rule"

#: The raw frame as the raw suite sees it: the published file with
#: `registration_date` parsed, because Great Expectations cannot bound a text
#: date (see the module docstring). The cast is the interim contract's own, so
#: the raw suite flags exactly the rows the interim rule then drops, and every
#: other column is the raw contract unchanged.
RAW_AS_VALIDATED: Schema = RAW_SCHEMA.with_dtype(
    "registration_date", INTERIM_SCHEMA.column("registration_date").dtype, name=RAW
)

#: The nullable columns a bounded rule reads. `price` is bounded too, and is
#: non-null in both contracts already.
BOUNDED_NULLABLE_COLUMNS: tuple[str, ...] = ("registration_date", "mileage_km_raw")

#: The columns the published file fills in every row, which both contracts leave
#: nullable because that is a measured fill rate (#78). Each suite asserts them
#: filled up to `validate.required_filled_mostly`.
REQUIRED_FILLED_COLUMNS: tuple[str, ...] = (
    "make",
    "body_type",
    "equipment_comfort",
    "equipment_entertainment",
    "equipment_extra",
    "equipment_safety",
)

app = typer.Typer()


# --------------------------------------------------------------------------
# The suites
# --------------------------------------------------------------------------


def contract_expectations(schema: Schema) -> list[gxe.Expectation]:
    """The ordered column list, each column's dtype and each non-null column.

    Each column's description from `schema.py` becomes its expectation's note,
    so the Data Docs say why a column is there - "One-sided assertion
    (EDN-23)", "PII; survives only as the hashed group key" - and not only what
    type it has.
    """
    meta = {KIND: CONTRACT}
    expectations: list[gxe.Expectation] = [
        gxe.ExpectTableColumnsToMatchOrderedList(column_list=list(schema.names), meta=meta)
    ]
    for column in schema.columns:
        expectations.append(
            gxe.ExpectColumnValuesToBeOfType(
                column=column.name, type_=column.dtype, notes=column.description, meta=meta
            )
        )
    for column in schema.columns:
        if not column.nullable:
            expectations.append(gxe.ExpectColumnValuesToNotBeNull(column=column.name, meta=meta))
    return expectations


def _registered_by_the_reference_date(params: dict[str, Any], *, mostly: float) -> gxe.Expectation:
    return gxe.ExpectColumnValuesToBeBetween(
        column="registration_date",
        max_value=pd.Timestamp(params["reference_date"]).to_pydatetime(),
        mostly=mostly,
        notes=(
            "EDN-22: age is the reference date minus the registration date, so a later "
            "date makes it negative. Preprocessing drops these listings; the raw suite "
            "tolerates them up to `validate.raw_mostly`, so that a scrape carrying many "
            "is flagged rather than silently cleaned."
        ),
        meta={KIND: RULE},
    )


def _mileage_in_range(params: dict[str, Any]) -> gxe.Expectation:
    """The same rule on both frames: a check on the data, not a scope filter."""
    bounds = params["validate"]
    return gxe.ExpectColumnValuesToBeBetween(
        column="mileage_km_raw",
        min_value=bounds["mileage_min_km"],
        max_value=bounds["mileage_max_km"],
        mostly=bounds["mileage_mostly"],
        notes=(
            "The odometer range FR-03 accepts at serving time, as a check rather than a "
            "filter: the published file has three readings above it, the highest "
            "2,570,000 km, and preprocessing keeps them, so the rule tolerates a few "
            "and fails on a systematic break such as a scrape in another unit."
        ),
        meta={KIND: RULE},
    )


def _required_filled(params: dict[str, Any], *, where: str) -> list[gxe.Expectation]:
    """The fill rate of every column the published file fills in every row."""
    mostly = params["validate"]["required_filled_mostly"]
    return [
        gxe.ExpectColumnValuesToNotBeNull(
            column=column,
            mostly=mostly,
            notes=(
                f"The published file fills this column in every row, but that is a measured "
                f"fill rate, so the contract leaves it nullable and this rule asserts it up "
                f"to `validate.required_filled_mostly`. {where}"
            ),
            meta={KIND: RULE},
        )
        for column in REQUIRED_FILLED_COLUMNS
    ]


def _not_vacuous(params: dict[str, Any]) -> list[gxe.Expectation]:
    """The row count, and the fill rate of every column a bounded rule reads."""
    bounds = params["validate"]
    return [
        gxe.ExpectTableRowCountToBeBetween(
            min_value=bounds["min_rows"],
            notes="Every column rule holds over no rows, so an empty frame would pass them all.",
            meta={KIND: RULE},
        ),
        *(
            gxe.ExpectColumnValuesToNotBeNull(
                column=column,
                mostly=bounds["filled_mostly"],
                notes=(
                    "A bounded rule skips missing values, so it would pass a column that "
                    "has emptied out; this keeps the column it reads filled."
                ),
                meta={KIND: RULE},
            )
            for column in BOUNDED_NULLABLE_COLUMNS
        ),
    ]


def raw_suite(params: dict[str, Any]) -> gx.ExpectationSuite:
    """The raw frame: the published file's contract, plus the tolerant rules."""
    mostly = params["validate"]["raw_mostly"]
    return gx.ExpectationSuite(
        name=RAW,
        expectations=[
            *contract_expectations(RAW_AS_VALIDATED),
            *_not_vacuous(params),
            *_required_filled(params, where="On the raw frame it reports a gap in the scrape."),
            _registered_by_the_reference_date(params, mostly=mostly),
            _mileage_in_range(params),
        ],
    )


def interim_suite(params: dict[str, Any]) -> gx.ExpectationSuite:
    """The cleaned frame: its contract, the fill rates, and the hard rules."""
    scope = params["preprocess"]
    return gx.ExpectationSuite(
        name=INTERIM,
        expectations=[
            *contract_expectations(INTERIM_SCHEMA),
            *_not_vacuous(params),
            *_required_filled(
                params, where="On the cleaned frame it reports preprocessing emptying a column."
            ),
            _registered_by_the_reference_date(params, mostly=1.0),
            gxe.ExpectColumnValuesToBeBetween(
                column="price",
                min_value=scope["price_min_eur"],
                max_value=scope["price_max_eur"],
                notes="The training price range, which preprocessing applies as a scope filter.",
                meta={KIND: RULE},
            ),
            _mileage_in_range(params),
        ],
    )


#: Each frame's suite, by the name the frame goes by everywhere else.
SUITES = {RAW: raw_suite, INTERIM: interim_suite}


# --------------------------------------------------------------------------
# The context
# --------------------------------------------------------------------------


def _store_path(target: Path, context_dir: Path) -> str:
    """`target` relative to the context, which is how the store config names it.

    Relative so that the cached `gx/` works in every clone it is pulled into.
    An absolute path would be the demo's machine-specific config again.
    """
    return os.path.relpath(target, context_dir) + "/"


def _remove_previous_context(context_dir: Path) -> None:
    """Delete an earlier build, and refuse to delete anything that is not one."""
    if not context_dir.exists():
        return
    if not (context_dir / "great_expectations.yml").exists():
        raise FileExistsError(
            f"{context_dir} exists but holds no great_expectations.yml, so it is not a "
            f"context this stage built; refusing to delete it"
        )
    shutil.rmtree(context_dir)


@app.command()
def main(
    context_dir: Path = GX_DIR,
    results_dir: Path = VALIDATION_RESULTS_DIR,
    data_docs_dir: Path = DATA_DOCS_DIR,
    params_path: Path = PARAMS_FILE,
) -> None:
    params = load_params(params_path)
    _remove_previous_context(context_dir)
    context = gx.get_context(mode="file", context_root_dir=context_dir)

    # A progress bar per checkpoint run is noise in a DVC log.
    context.variables.progress_bars = ProgressBarsConfig(globally=False)
    stores = context.variables.config.stores
    stores["validation_results_store"]["store_backend"]["base_directory"] = _store_path(
        results_dir, context_dir
    )
    context.variables.save()
    context.update_data_docs_site(
        "local_site",
        {
            "class_name": "SiteBuilder",
            "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
            "store_backend": {
                "class_name": "TupleFilesystemStoreBackend",
                "base_directory": _store_path(data_docs_dir, context_dir),
            },
        },
    )

    source = context.data_sources.add_pandas(DATASOURCE)
    for frame in FRAMES:
        batch = source.add_dataframe_asset(frame).add_batch_definition_whole_dataframe(frame)
        suite = context.suites.add(SUITES[frame](params))
        definition = context.validation_definitions.add(
            gx.ValidationDefinition(name=frame, data=batch, suite=suite)
        )
        context.checkpoints.add(
            gx.Checkpoint(
                name=frame,
                validation_definitions=[definition],
                result_format="SUMMARY",
            )
        )
        logger.info(f"Suite {frame!r}: {len(suite.expectations)} expectations.")
    logger.success(f"Built the Great Expectations context in {context_dir}.")


if __name__ == "__main__":
    app()
