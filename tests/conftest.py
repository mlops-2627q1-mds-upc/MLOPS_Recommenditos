"""Fixtures every test shares.

Nothing here touches `data/`, the DagsHub remote or the network: the synthetic
generator builds what the tests need, so `pytest` runs on a clean clone with no
credentials. That is what lets several people build pipeline stages at the same
time without waiting for the 548 MB download, and it keeps the personal data of
the real listings out of the repository itself (NFR-08).
"""

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from tools.requirement_matrix import MARKER, REQUIREMENTS_DOC, unknown_marker_ids
import yaml

from recommenditos.config import METRICS_FILE, MODELS_DIR, PARAMS_FILE, PROJ_ROOT, REPORTS_DIR
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


#: The data-validation stages' outputs, which `recommenditos.data.gx_context_configuration`
#: defines. Repeated rather than imported, because importing that module imports
#: Great Expectations, and this file is loaded by every test run, including the
#: ones that never touch data validation. `tests/test_validate_data.py` holds the
#: two copies equal.
GX_DIR = PROJ_ROOT / "gx"
VALIDATION_DIR = REPORTS_DIR / "data-validation"
VALIDATION_SUMMARY_FILE = VALIDATION_DIR / "summary.json"

#: Paths no test may create. MLflow resolves an unset tracking URI to
#: `sqlite:///$PWD/mlflow.db` (or to `mlruns/` under MLFLOW_ALLOW_FILE_STORE), so
#: a test that logs without setting one leaves a database in the repository root
#: instead of failing. Both are gitignored, which means nothing else would ever
#: point it out.
FORBIDDEN_PATHS = (PROJ_ROOT / "mlflow.db", PROJ_ROOT / "mlruns")


@pytest.fixture(autouse=True, scope="session")
def _repo_artefacts_stay_untouched():
    """Fail the suite if a test writes over the pipeline's own outputs.

    Every stage's path arguments default into the repository, so a test that
    forgets to redirect one silently overwrites a committed artefact. That
    happened once with the validation summary, and the only symptom was a
    mysterious diff after the next `dvc repro`.
    """
    watched = [METRICS_FILE, VALIDATION_SUMMARY_FILE]
    before = {path: _fingerprint(path) for path in watched}
    # The Great Expectations context by content, because `configure_gx`
    # rebuilds it under the same file names: only the UUIDs inside change.
    context_before = _tree_digest(GX_DIR)
    # The other DVC-cached outputs by listing: a checkpoint run adds result and
    # Data Docs files named after its run time.
    listed = (MODELS_DIR, VALIDATION_DIR)
    listings_before = {directory: _listing(directory) for directory in listed}
    already_there = [path for path in FORBIDDEN_PATHS if path.exists()]

    yield

    changed = sorted(str(path) for path in watched if _fingerprint(path) != before[path])
    assert not changed, f"the test suite wrote to {', '.join(changed)}"

    # By listing rather than by content: `models/` is a directory of directories
    # and what matters is that a test did not train into it. Its bundles are
    # DVC-cached, so a stray one would not show up in `git status` either - the
    # symptom would be `dvc status` reporting a model nobody had pushed.
    for directory in listed:
        assert _listing(directory) == listings_before[directory], (
            f"the test suite wrote into {directory}"
        )
    assert _tree_digest(GX_DIR) == context_before, f"the test suite rebuilt {GX_DIR}"

    appeared = sorted(
        str(path) for path in FORBIDDEN_PATHS if path.exists() and path not in already_there
    )
    assert not appeared, f"the test suite left {', '.join(appeared)} in the repository"


def _fingerprint(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def _tree_digest(directory: Path) -> str | None:
    """One digest over every file under `directory`, its path and its bytes."""
    if not directory.exists():
        return None
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(directory)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _listing(directory: Path) -> list[str]:
    """Every file under `directory`, relative to it, sorted."""
    if not directory.exists():
        return []
    return sorted(str(path.relative_to(directory)) for path in directory.rglob("*"))


@pytest.fixture(autouse=True)
def _req_marker_names_a_real_requirement(request):
    """Fail a test whose `req` marker names an ID the requirements do not define.

    A typo in a marker is invisible otherwise: the test passes, while the matrix
    of NFR-07 counts the requirement as verified by nothing, so the coverage is
    lost exactly where the evidence is supposed to come from.

    The check sits on the test rather than on collection so that one typo fails
    one test instead of the whole suite, and so the failure names the ID it
    could not find. The matrix job in CI is the second net: it reads every
    marker, including the ones on tests that never run.
    """
    markers = list(request.node.iter_markers(name=MARKER))
    for marker in markers:
        if not marker.args:
            pytest.fail(f"@pytest.mark.{MARKER} needs at least one requirement ID")

    named = tuple(str(argument) for marker in markers for argument in marker.args)
    unknown = unknown_marker_ids(named)
    if unknown:
        pytest.fail(
            f"the {MARKER} marker names {', '.join(unknown)}, which "
            f"{REQUIREMENTS_DOC.name} does not define"
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


def params_override(tmp_path: Path, params: dict, **blocks) -> Path:
    """A copy of params.yaml with some keys of some blocks changed.

    Every stage takes its params file as an argument for exactly this reason: a
    test can vary a parameter without monkeypatching anything, which is what lets
    a stage test run the real `main` against a parameter set the project would
    never ship. Shared here rather than per module, because more than one stage
    needs it.
    """
    changed = {**params, **{key: {**params[key], **value} for key, value in blocks.items()}}
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    return path
