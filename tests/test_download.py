"""The `download` stage's real source: the pinned Zenodo file.

Nothing here touches the network, and the autouse guard below makes that
structural rather than a convention. The stage takes its cache directory as an
argument, so a test can put a small, contract-shaped CSV there and exercise the
whole `zenodo` path - the MD5 check, the column check and the conversion - on a
file it wrote itself. The two tests that have to cover the fetch install a fake
downloader, which is also how the URL the stage would really request is pinned.

The synthetic source is tested in `test_pipeline.py`, next to the stage wiring.
"""

import hashlib
from pathlib import Path

import pandas as pd
import pytest
import yaml

from recommenditos.data import download_raw_dataset
from recommenditos.data.download_raw_dataset import SourceChangedError
from recommenditos.schema import RAW_SCHEMA

#: The `download.md5` a test uses when it wants the check to fail. A plausible
#: digest rather than a word, so the failure is the one a changed upstream file
#: would produce rather than a length or alphabet mismatch.
WRONG_MD5 = "0" * 32

#: What `download.filename` pins, which is what the cached file is called.
CSV_NAME = "autoscout24_dataset_20251108.csv"


@pytest.fixture(autouse=True)
def no_download(monkeypatch):
    """Refuse to download anything, for every test in this module.

    A wrong turn in the cache logic would otherwise pull 548 MB from Zenodo in
    the middle of a unit test run. A test that needs a fetch installs a fake of
    its own over this one.
    """

    def _refuse(url: str, destination: Path) -> None:
        raise AssertionError(f"a test tried to download {url}")

    monkeypatch.setattr(download_raw_dataset, "_download", _refuse)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Where the stage keeps the downloaded CSV between runs."""
    path = tmp_path / "external"
    path.mkdir()
    return path


@pytest.fixture
def pinned_csv(cache_dir: Path, raw_frame: pd.DataFrame) -> Path:
    """A CSV in the cache, shaped like the published one.

    The synthetic frame is written out and read back through the stage, so the
    columns the contract keeps as text - `mileage_km`, `ratings_average`,
    `weight_kg`, `registration_date` - make the round trip the real file makes.
    """
    path = cache_dir / CSV_NAME
    raw_frame.to_csv(path, index=False)
    return path


def test_a_cached_csv_is_converted_without_being_downloaded_again(
    tmp_path, params, cache_dir, pinned_csv, raw_frame
):
    # 548 MB per `dvc repro` is what the cache exists to avoid; `no_download`
    # above is what fails this test if the stage fetches it anyway.
    output = tmp_path / "listings.parquet"

    download_raw_dataset.main(output, _params(tmp_path, params, md5=_md5(pinned_csv)), cache_dir)

    written = pd.read_parquet(output)
    RAW_SCHEMA.validate(written)
    assert len(written) == len(raw_frame)


def test_a_changed_upstream_file_fails_the_stage(
    tmp_path, params, cache_dir, raw_frame, monkeypatch
):
    # The scenario the MD5 exists for: Zenodo serves something other than the
    # file this pipeline is pinned to. Retraining on it in silence would be the
    # worst outcome, so nothing is written.
    def _serve_something_else(url: str, destination: Path) -> None:
        raw_frame.head(10).to_csv(destination, index=False)

    monkeypatch.setattr(download_raw_dataset, "_download", _serve_something_else)
    output = tmp_path / "listings.parquet"

    with pytest.raises(SourceChangedError, match=WRONG_MD5):
        download_raw_dataset.main(output, _params(tmp_path, params, md5=WRONG_MD5), cache_dir)

    assert not output.exists()


def test_a_cached_file_that_does_not_match_the_pin_is_not_read(
    tmp_path, params, cache_dir, pinned_csv
):
    # The same check one run later: a cached copy is re-hashed rather than
    # trusted, so a truncated or hand-edited local file is caught as well.
    with pytest.raises(SourceChangedError, match=CSV_NAME):
        download_raw_dataset.main(
            tmp_path / "listings.parquet", _params(tmp_path, params, md5=WRONG_MD5), cache_dir
        )


def test_a_missing_file_is_fetched_from_the_url_the_record_pins(
    tmp_path, params, cache_dir, raw_frame, monkeypatch
):
    served = tmp_path / "served.csv"
    raw_frame.to_csv(served, index=False)
    requested: list[str] = []

    def _fake_download(url: str, destination: Path) -> None:
        requested.append(url)
        destination.write_bytes(served.read_bytes())

    monkeypatch.setattr(download_raw_dataset, "_download", _fake_download)

    download_raw_dataset.main(
        tmp_path / "listings.parquet", _params(tmp_path, params, md5=_md5(served)), cache_dir
    )

    download = params["download"]
    assert requested == [
        f"https://zenodo.org/records/{download['zenodo_record']}/files/{download['filename']}"
    ]


#: An upstream column change, as (what it does to the frame, the name the
#: failure has to mention). One parameter rather than two, so the test still
#: takes the five arguments a Pylint rule of this project allows.
COLUMN_CHANGES = [
    pytest.param(
        (lambda frame: frame.rename(columns={"power_kw": "power_kilowatt"}), "power_kw"),
        id="renamed",
    ),
    pytest.param(
        (lambda frame: frame.assign(dealer_rating_v2=1.0), "dealer_rating_v2"), id="added"
    ),
]


@pytest.mark.parametrize("change", COLUMN_CHANGES)
def test_a_column_the_contract_does_not_describe_fails_the_stage(
    tmp_path, params, cache_dir, raw_frame, change
):
    # `Schema.conform` selects, so an added column would be dropped in silence
    # and a renamed one would only surface as a missing feature stages later.
    mutation, reported = change
    path = cache_dir / CSV_NAME
    mutation(raw_frame).to_csv(path, index=False)

    with pytest.raises(SourceChangedError, match=reported):
        download_raw_dataset.main(
            tmp_path / "listings.parquet", _params(tmp_path, params, md5=_md5(path)), cache_dir
        )


def _params(tmp_path: Path, params: dict, **download) -> Path:
    """A copy of params.yaml with keys of its `download` block changed."""
    changed = {**params, "download": {**params["download"], "source": "zenodo", **download}}
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    return path


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()
