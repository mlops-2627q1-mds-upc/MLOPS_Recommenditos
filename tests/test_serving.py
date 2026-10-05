"""The serving path, run from the runtime dependencies alone (EDN-47).

`pyproject.toml` keeps what the API needs in `[project] dependencies` and
everything else in dependency groups, and the API image is built from that list
alone, because NFR-04 caps the image at 1 GB. A module on the serving path that
imported a group's package would work in every contributor's `.venv`, pass the
whole suite, and fail for the first time when M5 builds the image. This file is
how that fails on the pull request instead: `make test-serving`, and CI's
`Serving runtime` job, install the runtime list plus the `test` group into an
environment of their own and run this file there, and nothing else.

In the full environment the same tests are an ordinary round trip, cheap and
largely redundant with `tests/test_model.py`; they are evidence only when they
run from the runtime set. `test_the_environment_is_the_runtime_set_and_nothing_more`
checks that this is where `make test-serving` runs them, so the job cannot
quietly turn into a second copy of the full suite.

So this module may import nothing outside the serving path and the test
runner, and it is not the place to test the model. What it checks is that the
path the API takes - load a bundle from disk, encode a listing against the
bundle's own feature space, price it - runs. The bundles are fitted here, from
the synthetic fixture, because CI has no credentials to `dvc pull` the real
ones. The fit runs in the same environment, and that is intended: `model.py`
is the module the API imports, so whatever is only needed to train belongs in
`train.py`, which serving never imports, and a fit that needs a group's
package is a finding here too.
"""

from importlib.metadata import PackageNotFoundError, distribution
import os
import re
import tomllib

import numpy as np
import pandas as pd
import pytest

from recommenditos.config import PARAMS_FILE, PROJ_ROOT
from recommenditos.data import build_features, preprocess, split_data
from recommenditos.data.build_features import FeatureSpace, read_supported_makes
from recommenditos.modeling.model import TrainingData, fit_variant, load_model
from recommenditos.pipeline import load_params, read_frame
from recommenditos.schema import PROCESSED_SCHEMA

#: Set by `make test-serving`, and only there: it is the one place that knows the
#: environment was built from the runtime list and the `test` group alone.
SERVING_RUNTIME_ENV_VAR = "RECOMMENDITOS_SERVING_RUNTIME"

#: The group the serving check installs on top of the runtime list, so the one
#: group whose packages are allowed to be present.
TEST_GROUP = "test"

#: How many listings each bundle is asked to price. A handful is enough: this is
#: about the path running, and `tests/test_model.py` owns what the answers are.
LISTINGS = 20


def _variants() -> dict:
    return load_params(PARAMS_FILE)["train"]["variants"]


#: Every variant, so a parametrised test says which one failed in its own name.
VARIANTS = tuple(_variants())


@pytest.fixture(scope="module")
def served(tmp_path_factory, _generated_frame: pd.DataFrame) -> dict:
    """One bundle per variant on disk, built with the serving path's own modules.

    `preprocess`, `split` and `features` run as the pipeline runs them; the fit
    goes through `fit_variant` and `Model.save` rather than `train.main`, because
    `train` opens an MLflow run and MLflow is a pipeline package. The rows are
    restricted to the supported makes the way `train` restricts them (EDN-48),
    so the bundles carry the same scope as a real one.
    """
    params = load_params(PARAMS_FILE)
    root = tmp_path_factory.mktemp("serving")
    raw = root / "raw" / "listings.parquet"
    raw.parent.mkdir(parents=True)
    _generated_frame.to_parquet(raw, index=False)

    processed = root / "processed"
    features = processed / "features"
    preprocess.main(raw, root / "interim" / "listings.parquet", PARAMS_FILE)
    split_data.main(root / "interim" / "listings.parquet", processed, PARAMS_FILE)
    for feature_set in params["features"]["sets"]:
        build_features.main(feature_set, processed, features, PARAMS_FILE)

    supported = read_supported_makes(processed)
    models = root / "models"
    for variant, settings in params["train"]["variants"].items():
        directory = features / settings["feature_set"]
        space = FeatureSpace.load(directory, name=f"features-{settings['feature_set']}")
        frames = {
            split: _supported_only(
                read_frame(directory / f"{split}.parquet", space.schema), supported
            )
            for split in ("train", "validation")
        }
        data = TrainingData(
            space=space,
            train=frames["train"],
            validation=frames["validation"],
            supported_makes=supported,
        )
        model = fit_variant(
            variant,
            settings,
            data,
            seed=params["seed"],
            num_threads=params["train"]["num_threads"],
        )
        model.save(models / variant)
    return {"params": params, "processed": processed, "features": features, "models": models}


def _supported_only(frame: pd.DataFrame, supported: tuple[str, ...]) -> pd.DataFrame:
    return frame[frame["make"].isin(supported)].reset_index(drop=True)


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_bundle_on_disk_prices_a_listing(variant: str, served: dict):
    """Load, encode, predict: the three steps a request takes, from disk.

    The listings are `processed`-shaped rows encoded through the feature space the
    bundle carries, which is how a request will reach the model, rather than rows
    of the matrix the `features` stage wrote. The matrix is the oracle: a listing
    encoded at serving time has to be priced exactly as its training-time row is.
    """
    model = load_model(served["models"] / variant)
    supported = tuple(model.metadata["training"]["supported_makes"])
    listings = _supported_only(
        read_frame(served["processed"] / "test.parquet", PROCESSED_SCHEMA), supported
    ).head(LISTINGS)
    stage_rows = _supported_only(
        pd.read_parquet(served["features"] / model.feature_set / "test.parquet"), supported
    ).head(LISTINGS)

    request = model.space.matrix(listings, reference_date=served["params"]["reference_date"])
    prices = model.predict_eur(request)

    assert len(prices) == len(listings) > 0
    assert np.isfinite(prices).all()
    assert (prices > 0).all()
    np.testing.assert_array_equal(prices.to_numpy(), model.predict_eur(stage_rows).to_numpy())


@pytest.mark.skipif(
    os.environ.get(SERVING_RUNTIME_ENV_VAR) != "1",
    reason=f"only meaningful in the runtime-only environment; run `make test-serving`, "
    f"which sets {SERVING_RUNTIME_ENV_VAR}=1",
)
def test_the_environment_is_the_runtime_set_and_nothing_more():
    """No package of a group other than `test` is installed where this runs.

    Without it, the serving check could pass vacuously: a `uv sync` that lost its
    `--no-default-groups`, or a `uv run` that re-synced the default groups into
    the environment, would install everything and every test above would still
    pass. The packages are read from `pyproject.toml` rather than listed here, so
    a dependency added to a group is covered the day it is added.
    """
    groups = tomllib.loads((PROJ_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "dependency-groups"
    ]
    installed = sorted(
        f"{name} ({group})"
        for group, requirements in groups.items()
        if group != TEST_GROUP
        for name in map(_distribution_name, requirements)
        if _is_installed(name)
    )
    assert not installed, (
        f"the serving check is running in an environment that has group packages "
        f"installed: {', '.join(installed)}. It has to be built with "
        f"`uv sync --no-default-groups --group {TEST_GROUP}`, as `make test-serving` does."
    )


def _distribution_name(requirement: str) -> str:
    """The name part of a PEP 508 requirement such as `mlflow>=3.16.1,<4`."""
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    assert match, f"{requirement!r} does not start with a distribution name"
    return match.group(0)


def _is_installed(name: str) -> bool:
    try:
        distribution(name)
    except PackageNotFoundError:
        return False
    return True


def test_the_serving_check_reads_the_groups_it_guards():
    """The guard above parses what it has to, in every environment.

    It is skipped everywhere but the serving job, so a change that broke its
    parsing would only show up there; this keeps the parsing honest in the full
    suite as well.
    """
    assert _distribution_name("mlflow>=3.16.1,<4") == "mlflow"
    assert _distribution_name("pytest") == "pytest"
    assert _is_installed("pytest")
    assert not _is_installed("a-distribution-that-does-not-exist")
