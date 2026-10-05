"""Probe Great Expectations 1.23.2 against this project's pandas-3 frames (EDN-68).

Resolving next to pandas 3.0.6 is not the same as working on its frames, so every
expectation kind the suites use is run twice: on a frame that satisfies it, and on
one that breaks it. For the type expectation the breaking frames are the dtypes a
contract has to tell apart: another datetime unit, and the nullable, extension and
`object` counterparts of `bool`, `float64` and `str`. A check that passes both is broken, and a check that fails
both cannot be evaluated. The second half measures what a file context writes,
which decides whether `gx/` can be committed configuration.

Run from the repository root:

    uv run python reports/analysis/gx_probe.py > reports/analysis/gx_probe_results.txt
"""

import hashlib
from pathlib import Path
import tempfile

import great_expectations as gx
from great_expectations.data_context.types.base import ProgressBarsConfig
import great_expectations.expectations as gxe
import pandas as pd

print("great_expectations", gx.__version__, "pandas", pd.__version__)

context = gx.get_context(mode="ephemeral")
context.variables.progress_bars = ProgressBarsConfig(globally=False)
batch = (
    context.data_sources.add_pandas("probe")
    .add_dataframe_asset("frame")
    .add_batch_definition_whole_dataframe("whole")
)


def run(frame, expectation):
    result = batch.get_batch(batch_parameters={"dataframe": frame}).validate(expectation)
    raised = [
        info["exception_message"]
        for info in (
            result.exception_info.values()
            if "raised_exception" not in result.exception_info
            else [result.exception_info]
        )
        if isinstance(info, dict) and info.get("raised_exception")
    ]
    return result.success, result.result.get("observed_value"), raised[:1]


def probe(label, good, bad, expectation):
    on_good, on_bad = run(good, expectation), run(bad, expectation)
    if on_good[0] and not on_bad[0]:
        verdict = "works"
    elif on_good[0]:
        verdict = "BROKEN: passes a frame that breaks it"
    else:
        verdict = "BROKEN: fails a frame that satisfies it"
    print(f"\n{label}: {verdict}\n  satisfying: {on_good}\n  breaking:   {on_bad}")


text = pd.Series(["BMW", "Audi", None], dtype="str")
dates = pd.Series(["2020-01-01", None, "2025-11-08"], dtype="str")
parsed = pd.to_datetime(dates).astype("datetime64[ns]")
frame = pd.DataFrame(
    {
        "text": text,
        "number": [1.0, None, 2.0],
        "flag": [True, False, True],
        "date_text": dates,
        "date": parsed,
    }
)
print("\ndtypes:", {name: str(dtype) for name, dtype in frame.dtypes.items()})

probe(
    "OfType str on pandas 3 `str`",
    frame,
    frame.assign(text=[1, 2, 3]),
    gxe.ExpectColumnValuesToBeOfType(column="text", type_="str"),
)
probe(
    "OfType float64",
    frame,
    frame.assign(number=[1, 2, 3]),
    gxe.ExpectColumnValuesToBeOfType(column="number", type_="float64"),
)
probe(
    "OfType bool",
    frame,
    frame.assign(flag=frame["flag"].astype("str")),
    gxe.ExpectColumnValuesToBeOfType(column="flag", type_="bool"),
)
probe(
    "OfType datetime64[ns] against a datetime64[us] column (the unit)",
    frame,
    frame.assign(date=frame["date"].astype("datetime64[us]")),
    gxe.ExpectColumnValuesToBeOfType(column="date", type_="datetime64[ns]"),
)
probe(
    "OfType bool against a nullable `boolean` column holding a null",
    frame,
    frame.assign(flag=pd.Series([True, None, False], dtype="boolean")),
    gxe.ExpectColumnValuesToBeOfType(column="flag", type_="bool"),
)
probe(
    "OfType str against an `object` text column",
    frame,
    frame.assign(text=frame["text"].astype("object")),
    gxe.ExpectColumnValuesToBeOfType(column="text", type_="str"),
)
probe(
    "OfType str against a `string[python]` column",
    frame,
    frame.assign(text=frame["text"].astype("string[python]")),
    gxe.ExpectColumnValuesToBeOfType(column="text", type_="str"),
)
probe(
    "OfType float64 against a nullable `Float64` column",
    frame,
    frame.assign(number=frame["number"].astype("Float64")),
    gxe.ExpectColumnValuesToBeOfType(column="number", type_="float64"),
)
probe(
    "NotBeNull on `str` (missing is NaN)",
    frame.assign(text=["a", "b", "c"]),
    frame,
    gxe.ExpectColumnValuesToNotBeNull(column="text"),
)
probe(
    "TableColumnsToMatchOrderedList, a column added",
    frame,
    frame.assign(vin="x"),
    gxe.ExpectTableColumnsToMatchOrderedList(column_list=list(frame.columns)),
)
probe(
    "Between on float64 with NaN",
    frame,
    frame.assign(number=[1.0, None, 2e6]),
    gxe.ExpectColumnValuesToBeBetween(column="number", min_value=0, max_value=1e6),
)
reference = pd.Timestamp("2025-11-08")
probe(
    "Between on datetime64[ns], datetime bound",
    frame,
    frame.assign(date=pd.to_datetime(pd.Series(["2026-01-01", None, None]))),
    gxe.ExpectColumnValuesToBeBetween(column="date", max_value=reference.to_pydatetime()),
)
coerced = gxe.ExpectColumnValuesToBeBetween(column="date", max_value="2025-11-08")
print("\na text bound is stored as", repr(coerced.max_value))
probe(
    "Between on datetime64[ns], text bound",
    frame,
    frame.assign(date=pd.to_datetime(pd.Series(["2026-01-01", None, None]))),
    coerced,
)
probe(
    "Between on the raw text dates, text bound",
    frame,
    frame.assign(date_text=["2026-01-01", None, None]),
    gxe.ExpectColumnValuesToBeBetween(column="date_text", max_value="2025-11-08"),
)


# --------------------------------------------------------------------------
# What a file context writes, and what changes between two builds
# --------------------------------------------------------------------------


def build(root: Path, *, pinned: bool) -> None:
    store = gx.get_context(mode="file", context_root_dir=root / "gx")
    source = store.data_sources.add_or_update_pandas("listings")
    names = {asset.name for asset in source.assets}
    asset = source.get_asset("raw") if "raw" in names else source.add_dataframe_asset("raw")
    definition = (
        asset.get_batch_definition("raw")
        if asset.batch_definitions
        else asset.add_batch_definition_whole_dataframe("raw")
    )
    pin = {"id": "00000000-0000-0000-0000-000000000001"} if pinned else {}
    suite = store.suites.add_or_update(
        gx.ExpectationSuite(
            "raw", expectations=[gxe.ExpectColumnValuesToNotBeNull(column="a")], **pin
        )
    )
    validation = store.validation_definitions.add_or_update(
        gx.ValidationDefinition(name="raw", data=definition, suite=suite, **pin)
    )
    store.checkpoints.add_or_update(
        gx.Checkpoint(name="raw", validation_definitions=[validation], **pin)
    )


def digest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.md5(path.read_bytes()).hexdigest()
        for path in sorted((root / "gx").rglob("*"))
        if path.is_file()
    }


with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
    first, second = Path(first), Path(second)
    build(first, pinned=False)
    once = digest(first)
    build(first, pinned=False)
    twice = digest(first)
    build(second, pinned=True)
    elsewhere = digest(second)

    print(f"\nfiles a file context writes: {len(once)}")
    for name in once:
        print(f"  {name}")
    print("\nchanged by a second add_or_update build in the same store:")
    for name in once:
        if once[name] != twice[name]:
            print(f"  {name}")
    print("\ndiffering between two fresh builds of the same code:")
    for name in once:
        if once[name] != elsewhere.get(name):
            print(f"  {name}")
    pinned_suite = (second / "gx" / "expectations" / "raw.json").read_text()
    print(
        "\nan id passed to the suite, validation definition and checkpoint survives the save:",
        "00000000-0000-0000-0000-000000000001" in pinned_suite,
    )
