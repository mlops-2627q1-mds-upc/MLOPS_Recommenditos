"""Fixtures every test shares.

Nothing here touches `data/`, the DagsHub remote or the network: the synthetic
generator builds what the tests need, so `pytest` runs on a clean clone with no
credentials. That is what lets several people build pipeline stages at the same
time without waiting for the 548 MB download, and it keeps the personal data of
the real listings out of the repository itself (NFR-08).
"""

from pathlib import Path

import pandas as pd
import pytest

from recommenditos.config import PARAMS_FILE
from recommenditos.data.synthetic import generate_raw_listings
from recommenditos.pipeline import load_params

#: Small enough to keep the suite fast, large enough that a seller-grouped
#: split and a per-make count still mean something.
FIXTURE_ROWS = 2000

#: The columns problem-spec section 4 excludes as PII. No processed artefact,
#: model or log may carry one (NFR-08).
PII_COLUMNS = (
    "vin",
    "street",
    "zip",
    "city",
    "latitude",
    "longitude",
    "seller_company_name",
)


@pytest.fixture(scope="session")
def params() -> dict:
    """The project's real params.yaml, so tests fail when it drifts."""
    return load_params(PARAMS_FILE)


@pytest.fixture(scope="session")
def _generated_frame() -> pd.DataFrame:
    return generate_raw_listings(FIXTURE_ROWS)


@pytest.fixture
def raw_frame(_generated_frame: pd.DataFrame) -> pd.DataFrame:
    """A RAW_SCHEMA-valid synthetic snapshot, with every edge case present.

    Generated once per session but handed out as a copy, so a test that adds a
    column cannot corrupt every test that runs after it.
    """
    return _generated_frame.copy()


@pytest.fixture
def raw_path(raw_frame: pd.DataFrame, tmp_path: Path) -> Path:
    """The fixture written where a `download` stage would have put it."""
    path = tmp_path / "raw" / "listings.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_frame.to_parquet(path, index=False)
    return path
