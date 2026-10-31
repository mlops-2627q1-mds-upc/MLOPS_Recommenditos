"""The plumbing every DVC stage shares: parameters in, contract-checked frames out.

Keeping the read and the write here means the contract discipline is applied in
one place instead of eight. `write_frame` conforms before it writes and
`read_frame` validates after it reads, so a stage that breaks the contract fails
in its own run rather than three stages later. `write_parquet` and `write_json`
are here for the same reason: every artefact is written the one way that hashes
the same whatever the platform and the pandas version.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from recommenditos.config import PARAMS_FILE
from recommenditos.schema import Schema


def load_params(path: Path = PARAMS_FILE) -> dict[str, Any]:
    """The whole of params.yaml.

    Stages take the sub-tree they need. What each one may read is declared
    under `params:` in dvc.yaml, which is what makes `dvc repro` rerun exactly
    the stages a parameter change affects.
    """
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_frame(path: Path, schema: Schema) -> pd.DataFrame:
    """Read a Parquet artefact and check it against its contract."""
    frame = _restore_nullable_dtypes(pd.read_parquet(path), schema)
    schema.validate(frame)
    logger.info(f"Read {len(frame):,} rows from {path} as {schema.name!r}.")
    return frame


def _restore_nullable_dtypes(frame: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """`frame` with the pandas-only nullable dtypes its contract declares restored.

    `write_parquet` keeps no pandas metadata, and the Arrow schema in the file
    cannot tell a nullable pandas dtype from its default one: an Arrow boolean
    column reads back as numpy `bool`, or `object` when it holds nulls, whether
    it was written from `bool` or `boolean`. Those dtypes are the ones whose
    missing value is `pd.NA`, the equipment flags among them. A column that will
    not cast is left alone, so `validate` reports it.
    """
    for column in schema.columns:
        declared = pd.api.types.pandas_dtype(column.dtype)
        if column.name not in frame or getattr(declared, "na_value", None) is not pd.NA:
            continue
        if str(frame[column.name].dtype) != column.dtype:
            try:
                frame[column.name] = frame[column.name].astype(declared)
            except (TypeError, ValueError):
                continue
    return frame


def write_frame(frame: pd.DataFrame, path: Path, schema: Schema) -> pd.DataFrame:
    """Conform `frame` to its contract and write it as Parquet.

    Returns the conformed frame, because that - not the input - is what the
    next stage will read back.
    """
    conformed = schema.conform(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(conformed, path)
    logger.success(f"Wrote {len(conformed):,} rows to {path} as {schema.name!r}.")
    return conformed


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write `frame` as Parquet without the index and without pandas' metadata.

    `DataFrame.to_parquet` stores a `pandas` key in the file's schema metadata
    that records, among the dtypes, the pandas version that wrote it. DVC hashes
    an output byte for byte, so a pandas upgrade changed the hash of every
    Parquet output although no value had changed. Nothing needs that key: the
    Arrow schema Parquet keeps on its own restores `str`, `float64`, `bool`,
    `datetime64[ns]` and `category` with its levels, `read_frame` casts the
    nullable dtypes it cannot tell apart back to the contract's, and
    `tests/test_parquet.py` checks both against what the metadata restored.
    """
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(None)
    pq.write_table(table, path)


def write_json(document: Any, path: Path) -> None:
    """Write `document` as indented JSON, with LF line endings on every platform.

    DVC hashes an output byte for byte, and `Path.write_text` turns "\\n" into
    the platform's line separator unless it is told otherwise, so without
    `newline` a run on Windows writes CRLF and a hash no Linux run produces (#84).
    """
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
