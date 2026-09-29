"""Write the synthetic fixture to disk, for the cases that need a file path.

The test suite does not use this: `tests/conftest.py` builds the frame in
memory, so `pytest` writes nothing and nothing can go stale. Run this when
something needs the fixture as an actual Parquet file - a Great Expectations
asset points at a path, and a notebook is easier to poke at with one.

    make fixture
    uv run python tests/fixtures/generate_fixture.py --rows 5000

The output lands under `data/`, which `.gitignore` excludes, so it is never
committed: the generator is the artefact, the Parquet is a convenience. Both
come from the same seeded function, so they cannot disagree.
"""

from pathlib import Path

from loguru import logger
import typer

from recommenditos.config import DATA_DIR
from recommenditos.data.synthetic import generate_raw_listings

DEFAULT_OUTPUT = DATA_DIR / "fixture" / "listings_fixture.parquet"

app = typer.Typer()


@app.command()
def main(
    output_path: Path = DEFAULT_OUTPUT,
    rows: int = 2000,
    seed: int = 20251108,
):
    frame = generate_raw_listings(rows, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    logger.success(f"Wrote {len(frame):,} synthetic listings to {output_path}.")


if __name__ == "__main__":
    app()
