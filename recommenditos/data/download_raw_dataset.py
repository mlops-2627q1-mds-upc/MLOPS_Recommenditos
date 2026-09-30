"""`download` stage: make the raw snapshot available as a typed Parquet file.

`download.source` chooses where the snapshot comes from. `zenodo` fetches the
file behind the pinned DOI and converts it; `synthetic` generates a stand-in, so
the test suite and a clean clone still run with no network and no credentials.

Because the source is a *parameter*, the choice is recorded in `dvc.lock`, so a
pipeline running on synthetic data cannot pass unnoticed: `dvc status` reports a
workspace whose `download.source` disagrees with the lock the artefacts were
built under. (Not `dvc params diff`, which compares the params files of two Git
revisions rather than the lock against the workspace.)

The downloaded CSV is not a stage output. DVC deletes a stage's outputs before it
runs the stage, so declaring 548 MB as an `out` would re-download the file on
every `dvc repro download`; the shapes that keep it, a `dep` or a `persist: true,
cache: false` out, hash those 548 MB on every `dvc status` instead. EDN-35 weighs
all three. It is kept under `data/external/` as a local cache
instead - gitignored like everything in `data/` - and `download.md5` is what
pins it: the stage refuses to read bytes that hash to anything else. So a
changed upstream file fails the stage rather than quietly retraining the model,
and because that MD5 is a parameter, changing the pinned file is a change DVC
sees and reruns on.
"""

import hashlib
from pathlib import Path
import time

from loguru import logger
import pandas as pd
import requests
import typer

from recommenditos.config import EXTERNAL_DATA_DIR, PARAMS_FILE, RAW_DATA_DIR
from recommenditos.data.synthetic import generate_raw_listings
from recommenditos.pipeline import load_params, write_frame
from recommenditos.schema import RAW_SCHEMA

SYNTHETIC = "synthetic"
ZENODO = "zenodo"

#: Every source this stage implements, so a run that names another one can be
#: told what it could have picked instead.
SOURCES = (SYNTHETIC, ZENODO)

#: Zenodo serves a record's files under a stable path, which is what makes the
#: DOI reproducible: the record id pins the version, the name pins the file.
ZENODO_FILE_URL = "https://zenodo.org/records/{record}/files/{filename}"

# Big enough that hashing and writing are not the bottleneck on a 548 MB file,
# small enough that the progress log below still says something useful.
_CHUNK_BYTES = 8 * 1024 * 1024

# MB as the dataset card and the Zenodo record count them, so a size logged here
# can be compared with the documented 548.6 MB without converting anything.
_BYTES_PER_MB = 1_000_000

# One progress line per 100 MB. A silent ten-minute stage looks like a hang.
_PROGRESS_STEP_BYTES = 100 * _BYTES_PER_MB

# Applies per socket read, not to the whole transfer, so a stalled connection
# fails instead of holding the pipeline open forever.
_TIMEOUT_SECONDS = 60

#: The dtype of every column the contract does not declare as `bool`, passed
#: straight to `read_csv`, so the frame's types come from the contract rather
#: than from whatever 548 MB of CSV happens to look like. Inference agrees on
#: this snapshot, but for a reason that is not guaranteed: `zip` holds '8801 PN'
#: as well as '81476', and a scrape of numeric postcodes only would arrive as
#: int64 and lose the leading zero the 7,065 rows that have one depend on.
#:
#: The `bool` columns are excluded on purpose. `dtype="bool"` refuses an empty
#: field with "Bool column has NA values in column 0", which names neither the
#: column nor the rule; inferred, a gap makes the column `object` and the
#: contract reports it as the null in a non-nullable column that it is.
_BOOL_DTYPE = "bool"
CSV_DTYPES = {
    column.name: column.dtype for column in RAW_SCHEMA.columns if column.dtype != _BOOL_DTYPE
}

app = typer.Typer()


class SourceChangedError(RuntimeError):
    """The raw file is not the one this pipeline is pinned to.

    Its own class rather than a bare `ValueError`, because the caller's next
    step differs from every other failure here: check the Zenodo record and
    decide whether to re-pin, rather than fix our code.
    """


def raw_csv_path(cache_dir: Path, *, record: str, filename: str, expected_md5: str) -> Path:
    """The pinned CSV on local disk, fetched once and reused afterwards.

    The MD5 is checked on every run, not only after a download, so the one rule
    that holds is that the bytes this stage reads hash to `download.md5` -
    whether they came from Zenodo a minute ago or from the cache last month.
    """
    path = cache_dir / filename
    if path.exists():
        logger.info(f"Reusing the cached raw file at {path}.")
    else:
        _download(ZENODO_FILE_URL.format(record=record, filename=filename), path)

    digest = _md5(path)
    if digest != expected_md5:
        raise SourceChangedError(
            f"{path} has MD5 {digest}, but download.md5 pins {expected_md5}. Either the "
            f"published file changed, in which case check the Zenodo record and re-pin "
            f"deliberately rather than following it, or this local copy is damaged, in "
            f"which case delete it and let the stage fetch it again."
        )
    logger.info(f"MD5 {digest} matches download.md5.")
    return path


def read_raw_csv(path: Path) -> pd.DataFrame:
    """The published CSV as a frame `RAW_SCHEMA` can conform."""
    started = time.monotonic()
    frame = pd.read_csv(path, dtype=CSV_DTYPES)
    logger.info(f"Read {len(frame):,} rows of {path.name} in {time.monotonic() - started:,.0f} s.")

    # `Schema.conform` *selects* the contract's columns, so a column added
    # upstream would be dropped without a word and a renamed one would surface
    # three stages later as a missing feature. The raw contract is a
    # description of the published file, so either is a change to report here.
    present = set(frame.columns)
    unexpected = sorted(present - set(RAW_SCHEMA.names))
    missing = [name for name in RAW_SCHEMA.names if name not in present]
    if unexpected or missing:
        raise SourceChangedError(
            f"{path} has columns the raw contract does not describe. "
            f"Missing: {missing or 'none'}. Unexpected: {unexpected or 'none'}."
        )
    return frame


def _download(url: str, destination: Path) -> None:
    """Stream `url` to `destination`, atomically.

    The bytes land in a `.part` file and are renamed only once the response has
    been read to the end, so an interrupted download cannot be mistaken for a
    cached copy on the next run - which would otherwise fail the MD5 check with
    a message blaming Zenodo for our own half-written file. A transfer that
    fails takes its `.part` file with it, because nothing here resumes one and
    half of 548 MB is only disk somebody has to go and find.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    started = time.monotonic()

    try:
        with requests.get(url, stream=True, timeout=_TIMEOUT_SECONDS) as response:
            response.raise_for_status()
            # Zenodo serves this file chunked and sends no `Content-Length`,
            # which is also the reason EDN-25 rejected `dvc import-url`: its
            # change detection wants a header this source does not send.
            announced = response.headers.get("Content-Length")
            size = (
                f"{int(announced) / _BYTES_PER_MB:,.1f} MB" if announced else "size not announced"
            )
            logger.info(f"Downloading {url} ({size}) to {destination}.")

            written = 0
            milestone = _PROGRESS_STEP_BYTES
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=_CHUNK_BYTES):
                    written += handle.write(chunk)
                    if written >= milestone:
                        logger.info(f"  {written / _BYTES_PER_MB:,.0f} MB so far.")
                        milestone += _PROGRESS_STEP_BYTES
    except BaseException:
        # `BaseException`, so that a Ctrl-C ten minutes into a download cleans up
        # after itself as well as a dropped connection does.
        partial.unlink(missing_ok=True)
        raise

    partial.replace(destination)
    elapsed = time.monotonic() - started
    megabytes = written / _BYTES_PER_MB
    rate = megabytes / elapsed if elapsed else 0.0
    logger.success(f"Downloaded {megabytes:,.1f} MB in {elapsed:,.0f} s ({rate:,.1f} MB/s).")


def _md5(path: Path) -> str:
    # `usedforsecurity=False` because this MD5 is the checksum Zenodo publishes,
    # used to tell the pinned bytes from any other bytes. Without the flag the
    # call raises on a FIPS-mode Python, which would make the stage unrunnable
    # there for no reason.
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


@app.command()
def main(
    output_path: Path = RAW_DATA_DIR / "listings.parquet",
    params_path: Path = PARAMS_FILE,
    cache_dir: Path = EXTERNAL_DATA_DIR,
):
    params = load_params(params_path)
    download = params["download"]
    source = download["source"]

    if source == SYNTHETIC:
        logger.info(f"Generating {download['rows']:,} synthetic listings (download.source).")
        frame = generate_raw_listings(download["rows"], seed=params["seed"])
    elif source == ZENODO:
        frame = read_raw_csv(
            raw_csv_path(
                cache_dir,
                record=download["zenodo_record"],
                filename=download["filename"],
                expected_md5=download["md5"],
            )
        )
    else:
        raise NotImplementedError(
            f"download.source={source!r} is not a source this stage implements. "
            f"Pick one of: {', '.join(SOURCES)}."
        )

    write_frame(frame, output_path, RAW_SCHEMA)


if __name__ == "__main__":
    app()
