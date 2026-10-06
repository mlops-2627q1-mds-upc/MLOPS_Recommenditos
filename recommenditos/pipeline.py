"""The plumbing every DVC stage shares: parameters in, contract-checked frames out.

Keeping the read and the write here means the contract discipline is applied in
one place instead of eight. `write_frame` conforms before it writes and
`read_frame` validates after it reads, so a stage that breaks the contract fails
in its own run rather than three stages later. `write_json` is here for the same
reason: every JSON artefact is written the one way that hashes the same on every
platform.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger
import pandas as pd
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
    frame = pd.read_parquet(path)
    schema.validate(frame)
    logger.info(f"Read {len(frame):,} rows from {path} as {schema.name!r}.")
    return frame


def write_frame(frame: pd.DataFrame, path: Path, schema: Schema) -> pd.DataFrame:
    """Conform `frame` to its contract and write it as Parquet.

    Returns the conformed frame, because that - not the input - is what the
    next stage will read back.
    """
    conformed = schema.conform(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    conformed.to_parquet(path, index=False)
    logger.success(f"Wrote {len(conformed):,} rows to {path} as {schema.name!r}.")
    return conformed


def write_json(document: Any, path: Path) -> None:
    """Write `document` as indented JSON, with LF line endings on every platform.

    DVC hashes an output byte for byte, and `Path.write_text` turns "\\n" into
    the platform's line separator unless it is told otherwise, so without
    `newline` a run on Windows writes CRLF and a hash no Linux run produces (#84).
    """
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
