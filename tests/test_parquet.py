"""Parquet outputs carry no pandas metadata and lose no dtype without it.

`write_parquet` drops the `pandas` key that `DataFrame.to_parquet` stores,
because it records the pandas version and so changed the hash of every DVC
output on a pandas upgrade. These tests check the two things that change has
to keep true: the bytes no longer depend on the pandas version, and every frame
the pipeline writes reads back through the project's own reader exactly as it
did when the pandas metadata was there to restore the dtypes.
"""

from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
import pyarrow.pandas_compat
import pyarrow.parquet as pq
import pytest
from tests.conftest import feature_sets

from recommenditos.data.build_comparables import COMPARABLES_SCHEMA
from recommenditos.data.split_data import SPLIT_NAMES
from recommenditos.modeling.model import load_model
from recommenditos.pipeline import read_frame, write_frame, write_parquet
from recommenditos.schema import (
    INTERIM_SCHEMA,
    PROCESSED_SCHEMA,
    RAW_SCHEMA,
    Column,
    Schema,
    load_feature_schema,
)

#: One column per dtype a contract can declare, nullable where the dtype has a
#: missing value, so every way back from Arrow is exercised.
ALL_DTYPES = Schema(
    name="all-dtypes",
    columns=(
        Column("text", "str", True, "pandas 3's default string dtype."),
        Column("number", "float64", True, "A float with NaN."),
        Column("flag", "bool", False, "numpy bool, which has no missing value."),
        Column("nullable_flag", "boolean", True, "pandas' nullable boolean, with NA."),
        Column("complete_flag", "boolean", False, "Nullable boolean without a missing value."),
        Column("count", "Int64", True, "pandas' nullable integer."),
        Column("ratio", "Float64", True, "pandas' nullable float."),
        Column("when", "datetime64[ns]", True, "Nanosecond timestamps with NaT."),
        Column("level", "category", True, "A categorical with an unused level.", ("a", "b", "c")),
    ),
)


def _all_dtypes_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": pd.array(["x", None, "z"], dtype="str"),
            "number": [1.5, float("nan"), -2.0],
            "flag": [True, False, True],
            "nullable_flag": pd.array([True, None, False], dtype="boolean"),
            "complete_flag": pd.array([True, False, False], dtype="boolean"),
            "count": pd.array([1, None, 3], dtype="Int64"),
            "ratio": pd.array([0.5, None, 1.0], dtype="Float64"),
            "when": pd.to_datetime(
                ["2024-01-01 12:00:00.000000001", None, "2020-02-29 00:00:00.000000000"]
            ),
            "level": pd.Categorical(["a", None, "a"], categories=["a", "b", "c"]),
        },
        # An index the writer must not keep: the contracts have none.
        index=[7, 8, 9],
    )


def _as_the_old_writer_and_reader_gave_it(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """`frame` written by `DataFrame.to_parquet` and read back by `read_parquet`.

    What every stage read before `write_parquet`: the pandas metadata in the
    file restored each dtype.
    """
    frame.to_parquet(path, index=False)
    return pd.read_parquet(path)


def _assert_same_frame(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    assert_frame_equal(actual, expected, check_exact=True, check_dtype=True)
    for name in expected.columns:
        if isinstance(expected[name].dtype, pd.CategoricalDtype):
            # `assert_frame_equal` compares the values and the levels, not the
            # dtype the levels are held in.
            assert actual[name].dtype == expected[name].dtype, name
            assert actual[name].cat.categories.dtype == expected[name].cat.categories.dtype, name


def _pandas_metadata(path: Path) -> bytes | None:
    return (pq.read_schema(path).metadata or {}).get(b"pandas")


def test_every_dtype_a_contract_declares_survives_the_round_trip(tmp_path):
    written = write_frame(_all_dtypes_frame(), tmp_path / "new.parquet", ALL_DTYPES)

    read = read_frame(tmp_path / "new.parquet", ALL_DTYPES)

    _assert_same_frame(read, written.reset_index(drop=True))
    _assert_same_frame(
        read, _as_the_old_writer_and_reader_gave_it(written, tmp_path / "old.parquet")
    )
    assert _pandas_metadata(tmp_path / "new.parquet") is None


def test_the_reader_reports_a_column_it_cannot_restore_instead_of_raising(tmp_path):
    path = tmp_path / "broken.parquet"
    write_parquet(pd.DataFrame({"flag": ["yes", "no"]}), path)
    schema = Schema(name="flags", columns=(Column("flag", "boolean", False, "A flag."),))

    with pytest.raises(ValueError, match="expected dtype boolean, got str"):
        read_frame(path, schema)


@pytest.mark.parametrize(
    ("values", "stored"),
    [
        pytest.param([1.0, 0.0, float("nan")], "double", id="float"),
        pytest.param([1, 0, 1], "int64", id="int"),
    ],
)
def test_a_column_stored_as_another_arrow_type_is_not_cast_to_boolean(tmp_path, values, stored):
    # `astype("boolean")` accepts these, so a cast keyed on the declared dtype
    # alone would let a wrongly stored column through that `validate` rejects.
    path = tmp_path / "wrong.parquet"
    write_parquet(pd.DataFrame({"flag": values}), path)
    assert str(pq.read_schema(path).field("flag").type) == stored
    schema = Schema(name="flags", columns=(Column("flag", "boolean", True, "A flag."),))

    with pytest.raises(ValueError, match="expected dtype boolean, got"):
        read_frame(path, schema)


@pytest.mark.parametrize("values", [[True, False, True], [True, None, False]])
def test_a_column_stored_as_arrow_boolean_is_restored_to_boolean(tmp_path, values):
    path = tmp_path / "flags.parquet"
    write_parquet(pd.DataFrame({"flag": pd.array(values, dtype="boolean")}), path)
    schema = Schema(name="flags", columns=(Column("flag", "boolean", True, "A flag."),))

    read = read_frame(path, schema)

    assert str(read["flag"].dtype) == "boolean"
    assert read["flag"].tolist() == pd.array(values, dtype="boolean").tolist()


def test_a_nullable_integer_column_keeps_values_above_two_to_the_53(tmp_path):
    # pandas reads an int64 column with nulls as float64, which rounds
    # 2**53 + 1 to 2**53, so the column has to be rebuilt from Arrow.
    path = tmp_path / "ints.parquet"
    write_parquet(pd.DataFrame({"count": pd.array([2**53 + 1, None], dtype="Int64")}), path)
    schema = Schema(name="ints", columns=(Column("count", "Int64", True, "A count."),))

    read = read_frame(path, schema)

    assert str(read["count"].dtype) == "Int64"
    assert read["count"].tolist() == [2**53 + 1, pd.NA]


def _bytes(path: Path) -> bytes:
    return path.read_bytes()


class _PandasVersion:
    """pyarrow's view of pandas, with another version number.

    pyarrow reads `pd.__version__` once, when it first meets pandas, so
    patching `pd.__version__` itself would change nothing it writes.
    """

    def __init__(self, real, version: str):
        self._real, self.version = real, version

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.mark.req("NFR-06")
@pytest.mark.parametrize("version", ["0.0.1", "99.0.0"])
def test_the_bytes_written_do_not_depend_on_the_pandas_version(tmp_path, monkeypatch, version):
    frame = write_frame(_all_dtypes_frame(), tmp_path / "now.parquet", ALL_DTYPES)
    real = pyarrow.pandas_compat._pandas_api
    monkeypatch.setattr(pyarrow.pandas_compat, "_pandas_api", _PandasVersion(real, version))

    write_parquet(frame, tmp_path / "other.parquet")
    # The control: the writer it replaced does change with the version, so the
    # patch reaches what pyarrow records and the equality below means something.
    frame.to_parquet(tmp_path / "old-now.parquet", index=False)
    monkeypatch.setattr(pyarrow.pandas_compat, "_pandas_api", real)
    frame.to_parquet(tmp_path / "old-real.parquet", index=False)

    assert _bytes(tmp_path / "other.parquet") == _bytes(tmp_path / "now.parquet")
    assert _bytes(tmp_path / "old-now.parquet") != _bytes(tmp_path / "old-real.parquet")


# --------------------------------------------------------------------------
# Every frame the pipeline writes, from the synthetic fixture
# --------------------------------------------------------------------------


def _pipeline_outputs(matrices: dict, trained: dict) -> list[tuple[str, Path, Schema]]:
    processed, features = matrices["processed"], matrices["features"]
    outputs = [
        ("interim", matrices["root"] / "interim" / "listings.parquet", INTERIM_SCHEMA),
        *((name, processed / f"{name}.parquet", PROCESSED_SCHEMA) for name in SPLIT_NAMES),
        ("comparables", trained["comparables"], COMPARABLES_SCHEMA),
    ]
    for feature_set in feature_sets():
        schema = load_feature_schema(features / feature_set, name=feature_set)
        outputs += [
            (f"{feature_set}/{name}", features / feature_set / f"{name}.parquet", schema)
            for name in SPLIT_NAMES
        ]
    return outputs


def test_the_raw_listings_read_back_as_the_frame_written(raw_frame, tmp_path):
    path = tmp_path / "raw" / "listings.parquet"
    written = write_frame(raw_frame, path, RAW_SCHEMA)

    read = read_frame(path, RAW_SCHEMA)

    _assert_same_frame(read, written.reset_index(drop=True))
    _assert_same_frame(
        read, _as_the_old_writer_and_reader_gave_it(written, tmp_path / "old.parquet")
    )
    assert _pandas_metadata(path) is None


def test_every_stage_output_reads_back_as_the_old_writer_gave_it(matrices, trained, tmp_path):
    """Each stage output carries no pandas metadata and reads back unchanged by the old path.

    The fixtures hand over the files, not the frames that were written, so the
    oracle here is `read_frame`'s own `validate` (every dtype and null rule of
    the contract) plus the old writer's round trip of the frame read, which
    shows that the read frame loses nothing the old path would have kept.
    """
    outputs = _pipeline_outputs(matrices, trained)
    # The extended matrix is the one with nullable `boolean` columns, the dtype
    # Arrow alone cannot restore; without it this test would not cover them.
    assert any("boolean" in schema.dtypes.values() for _, _, schema in outputs)

    for name, path, schema in outputs:
        assert _pandas_metadata(path) is None, name
        read = read_frame(path, schema)
        old = tmp_path / f"{name.replace('/', '-')}.parquet"
        _assert_same_frame(read, _as_the_old_writer_and_reader_gave_it(read, old))


def test_the_baseline_lookup_reads_back_as_the_old_writer_gave_it(trained, tmp_path):
    path = trained["dir"] / "b0" / "lookup.parquet"
    assert _pandas_metadata(path) is None

    lookup = load_model(trained["dir"] / "b0").lookup
    _assert_same_frame(
        lookup, _as_the_old_writer_and_reader_gave_it(lookup, tmp_path / "old.parquet")
    )
