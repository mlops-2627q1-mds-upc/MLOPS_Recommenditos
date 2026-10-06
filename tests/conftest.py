"""Fixtures every test shares.

Nothing here touches `data/`, the DagsHub remote or the network: the synthetic
generator builds what the tests need, so `pytest` runs on a clean clone with no
credentials. That is what lets several people build pipeline stages at the same
time without waiting for the 548 MB download, and it keeps the personal data of
the real listings out of the repository itself (NFR-08).
"""

import hashlib
import os
from pathlib import Path
import subprocess
import sys

from dvc.repo import Repo
import pandas as pd
import pytest
from tools.requirement_matrix import MARKER, REQUIREMENTS_DOC, unknown_marker_ids
import yaml

from recommenditos.config import METRICS_FILE, MODELS_DIR, PARAMS_FILE, PROJ_ROOT, REPORTS_DIR
from recommenditos.data.synthetic import generate_raw_listings
from recommenditos.modeling.energy import EMISSIONS_DIR
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
    # By content, like the context: CodeCarbon *appends* to these, so a test
    # that trained without redirecting `energy_dir` leaves the file list as it
    # was and adds a row to a git-tracked record of the real ladder's energy.
    emissions_before = _tree_digest(EMISSIONS_DIR)
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
    assert _tree_digest(EMISSIONS_DIR) == emissions_before, (
        f"the test suite wrote an energy measurement into {EMISSIONS_DIR}"
    )

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


#: The stage command of the pipeline `dvc_pipeline` builds. It first writes down
#: the tags the provenance seam computes from inside the stage DVC is running,
#: which is the only place the question "what does a stage see mid-`dvc repro`"
#: can be asked, and then does the stage's work. `PROVENANCE_RECORDS` names the
#: directory the record goes to, outside the repository, so that writing it is
#: not itself a change to the tree.
_STAGE_SCRIPT = """\
import json
import os
from pathlib import Path
import sys

from recommenditos.provenance import resumed_run_tags, run_tags

# Named after the stage, not its address, which carries the path to `dvc.yaml`
# when `dvc repro` was started from elsewhere: `../dvc.yaml:prepare`.
stage = os.environ["DVC_STAGE"].rpartition(":")[2]
record = {"run": run_tags(Path.cwd()), "resumed": resumed_run_tags(Path.cwd())}
Path(os.environ["PROVENANCE_RECORDS"], stage + ".json").write_text(json.dumps(record))

if sys.argv[1] == "prepare":
    Path("prepared").mkdir()
    for listing in sorted(Path("raw").iterdir()):
        Path("prepared", listing.name).write_text(listing.read_text().upper())
    listings = [Path("prepared", name).read_text() for name in sorted(os.listdir("prepared"))]
    Path("summary.json").write_text(json.dumps({"listings": listings}))
else:
    summary = Path("summary.json").read_text()
    Path("model-" + sys.argv[2] + ".txt").write_text(sys.argv[2] + " " + summary)
"""

#: Shaped like the project's pipeline where provenance is concerned: a directory
#: dependency, a parameter, which `dvc.lock` records apart from the `deps`, an
#: output DVC caches, an output it does not and which git tracks instead (as
#: `metrics.json` and the validation summary are), a later stage that depends on
#: that uncached output, and a `foreach`, so a stage's address has an `@` in it.
_DVC_YAML = """\
stages:
  prepare:
    cmd: '{python} stage.py prepare'
    deps:
      - stage.py
      - raw
    params:
      - scale
    outs:
      - prepared
      - summary.json:
          cache: false
  train:
    foreach:
      - small
      - large
    do:
      cmd: '{python} stage.py train ${{item}}'
      deps:
        - stage.py
        - prepared
        - summary.json
      outs:
        - model-${{item}}.txt
"""


def git_in(root: Path, *args: str) -> str:
    """`git` in `root`, isolated from the developer's own configuration."""
    completed = subprocess.run(
        [
            "git",
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-C",
            str(root),
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def commit_all(root: Path, message: str) -> str:
    """Commit every change in `root`, and return the new commit."""
    git_in(root, "add", "--all")
    git_in(root, "commit", "--quiet", "--message", message)
    return git_in(root, "rev-parse", "HEAD")


def dvc_repro(root: Path, records: Path, *targets: str, cwd: Path | None = None) -> None:
    """`dvc repro` in `root`, the way a contributor runs it: as a separate process.

    `cwd` starts it from somewhere else in the repository, with `targets` naming
    the `dvc.yaml` relative to there, which changes the address DVC hands the
    stage in `DVC_STAGE`: `../dvc.yaml:prepare` rather than `prepare`.
    """
    records.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "dvc", "repro", "--quiet", *targets],
        cwd=cwd or root,
        env={**os.environ, "PROVENANCE_RECORDS": str(records)},
        check=True,
    )


def build_dvc_pipeline(root: Path) -> Path:
    """A git repository holding a small DVC pipeline under `root`, committed and clean.

    Not yet reproduced: there is no `dvc.lock`, so a test decides which states
    it commits. The repository's own configuration turns off DVC's analytics and
    update check, so nothing here touches the network.
    """
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "a.csv").write_text("audi,a4,2015\n", encoding="utf-8")
    (root / "raw" / "b.csv").write_text("bmw,320d,2018\n", encoding="utf-8")
    (root / "params.yaml").write_text("scale: 2\n", encoding="utf-8")
    (root / "stage.py").write_text(_STAGE_SCRIPT, encoding="utf-8")
    # Double quotes rather than `shlex.quote`: DVC runs a command through `$SHELL`
    # on POSIX and through `cmd.exe` on Windows, and only double quotes mean the
    # same to both. The YAML quotes the whole command in single quotes around it.
    python = f'"{sys.executable}"'
    (root / "dvc.yaml").write_text(_DVC_YAML.format(python=python), encoding="utf-8")
    git_in(root, "init", "--quiet")
    Repo.init(str(root)).close()
    (root / ".dvc" / "config").write_text(
        "[core]\n    analytics = false\n    check_update = false\n", encoding="utf-8"
    )
    commit_all(root, "a pipeline")
    return root


def isolate_dvc(monkeypatch: pytest.MonkeyPatch, cache: Path) -> None:
    """Keep DVC's per-machine state out of the shared `/var/tmp/dvc`, and run outside DVC."""
    monkeypatch.setenv("DVC_SITE_CACHE_DIR", str(cache))
    monkeypatch.delenv("DVC_STAGE", raising=False)


@pytest.fixture
def dvc_pipeline(tmp_path: Path, monkeypatch) -> Path:
    """`build_dvc_pipeline` in this test's temporary directory."""
    isolate_dvc(monkeypatch, tmp_path / "dvc-site-cache")
    return build_dvc_pipeline(tmp_path / "pipeline")
