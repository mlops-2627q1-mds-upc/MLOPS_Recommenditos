"""`download` stage: make the raw snapshot available as a typed Parquet file.

STUB. While `download.source` is `synthetic` this generates a stand-in rather
than fetching anything, so `dvc repro` is green on a clean clone with no
credentials and no 548 MB download - which is what lets the other stage tickets
be built in parallel.

Issue #33 implements `source: zenodo`: fetch the file behind the pinned DOI,
assert the MD5 in params.yaml so a silently changed upstream file fails loudly,
and convert it to Parquet. Everything downstream reads this stage's output, so
that swap changes no other module.

Because the source is a *parameter*, the choice is recorded in `dvc.lock` and
shows up in `dvc params diff`; a pipeline still running on synthetic data
cannot pass unnoticed.
"""

from pathlib import Path

from loguru import logger
import typer

from recommenditos.config import PARAMS_FILE, RAW_DATA_DIR
from recommenditos.data.synthetic import generate_raw_listings
from recommenditos.pipeline import load_params, write_frame
from recommenditos.schema import RAW_SCHEMA

SYNTHETIC = "synthetic"
ZENODO = "zenodo"

app = typer.Typer()


@app.command()
def main(
    output_path: Path = RAW_DATA_DIR / "listings.parquet",
    params_path: Path = PARAMS_FILE,
):
    params = load_params(params_path)
    source = params["download"]["source"]
    seed = params["seed"]

    if source != SYNTHETIC:
        raise NotImplementedError(
            f"download.source={source!r} is not implemented yet. The skeleton only "
            f"supports {SYNTHETIC!r}; issue #33 adds {ZENODO!r}."
        )

    rows = params["download"]["rows"]
    logger.warning(
        f"STUB: generating {rows:,} synthetic listings instead of downloading the real "
        f"dataset. Set download.source to {ZENODO!r} once issue #33 has landed."
    )
    frame = generate_raw_listings(rows, seed=seed)
    write_frame(frame, output_path, RAW_SCHEMA)


if __name__ == "__main__":
    app()
