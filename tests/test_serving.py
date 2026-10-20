"""The serving path, run from the runtime dependencies alone (EDN-47).

`pyproject.toml` keeps what the API needs in `[project] dependencies` and
everything else in dependency groups, and the API image is built from that list
alone, because NFR-04 caps the image at 1 GB. A module on the serving path that
imported a group's package would work in every contributor's `.venv`, pass the
whole suite, and fail for the first time when M5 builds the image. This file is
how that fails on the pull request instead: `make test-serving`, and CI's
`Serving runtime` job, install the runtime list plus the `test` group into an
environment of their own and run this file there, and nothing else.

**What the check loads is this file and the modules it imports, nothing more.**
`make test-serving` runs pytest with `--noconftest`, so `tests/conftest.py`,
which imports whatever the rest of the suite needs, never constrains it, and
this file takes nothing from it: it generates its own fixture frame. The
`recommenditos` modules it imports are therefore exactly the modules the check
holds to the runtime set:

- the serving path: `modeling/model.py` and what it imports, which is
  `data/build_features.py`, `data/split_data.py`, `pipeline.py`, `schema.py`
  and `config.py`;
- `data/preprocess.py`, on purpose, because the API is to reuse its
  request-side functions (`hash_seller_group`, see docs/docs/pipeline.md);
- `data/synthetic.py`, because the bundles are built from its fixture;
- `api/`, the FastAPI application around the seam, which the image runs.

`modeling/evaluate.py` is deliberately not among them: it imports MLflow
through `tracking.py`, so a function serving code needs from it, such as
`point_metrics`, has to move to a module this file can import first.

Two properties close the gaps the environment alone leaves open.
`test_the_environment_is_the_runtime_set_and_the_test_group_and_nothing_more`
fails when anything beyond those two closures is installed, so a lost
`--no-default-groups` cannot turn the job into a second copy of the full suite.
And the environment still holds the test group's own packages (pytest and the
five it pulls in), which the image will not, so
`test_serving_imports_nothing_only_the_test_group_installs` runs the serving
path in a fresh interpreter and fails when it imported one of them: a stray
`import packaging` in `model.py` cannot pass here and fail in the image.

In the full environment the same tests are an ordinary round trip, cheap and
largely redundant with `tests/test_model.py`; the first property is skipped
there, because the full environment is the one it exists to tell apart.

The bundles are fitted here, from the synthetic fixture, because CI has no
credentials to `dvc pull` the real ones. The fit runs in the same environment,
and that is intended: `model.py` is the module the API imports, so whatever is
only needed to train belongs in `train.py`, which serving never imports, and a
fit that needs a group's package is a finding here too.
"""

from importlib.metadata import distributions, packages_distributions
import json
import os
import subprocess
import sys

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from tests.lockfile import (
    PROJECT,
    group_closure,
    group_dependencies,
    normalise,
    read_lock,
    runtime_closure,
)

from recommenditos.api.app import create_app
from recommenditos.api.serving import ServingError
from recommenditos.config import PARAMS_FILE
from recommenditos.data import build_features, preprocess, split_data
from recommenditos.data.build_features import FeatureSpace, read_supported_makes
from recommenditos.data.synthetic import generate_raw_listings
from recommenditos.modeling.model import (
    RequestContract,
    TrainingData,
    fit_quantile_estimators,
    fit_variant,
    interval_settings,
    load_model,
)
from recommenditos.pipeline import load_params, read_frame
from recommenditos.schema import PROCESSED_SCHEMA

#: Set by `make test-serving`, and only there: it is the one place that knows the
#: environment was built from the runtime list and the `test` group alone.
SERVING_RUNTIME_ENV_VAR = "RECOMMENDITOS_SERVING_RUNTIME"

#: The group the serving check installs on top of the runtime list.
TEST_GROUP = "test"

#: The size of the generated fixture, the same the rest of the suite uses: small
#: enough to fit four variants in seconds, large enough that every make clears
#: the support threshold `split` applies.
FIXTURE_ROWS = 2000

#: How many listings each bundle is asked to price. A handful is enough: this is
#: about the path running, and `tests/test_model.py` owns what the answers are.
LISTINGS = 20

#: The serving path as the API will walk it, run in a fresh interpreter: import
#: the seam, load a bundle from disk, encode listings through its own feature
#: space, price them, and report every top-level module that is now imported.
_SERVE_AND_LIST_MODULES = r"""
import json, sys
from pathlib import Path
from recommenditos.modeling.model import load_model
from recommenditos.pipeline import read_frame
from recommenditos.schema import PROCESSED_SCHEMA
bundle, listings, reference_date = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
model = load_model(bundle)
frame = read_frame(listings, PROCESSED_SCHEMA).head(5)
request = model.space.matrix(frame, reference_date=reference_date)
model.predict_eur(request)
if model.has_intervals:
    model.predict_interval_eur(request)
print(json.dumps(sorted({name.partition(".")[0] for name in sys.modules})))
"""


#: The same, through the API: build the app around the bundle and send it one
#: valuation as an ASGI request, without an HTTP client (the client libraries
#: are test-only). The listing takes its categoricals from the bundle itself.
_SERVE_THROUGH_THE_API = r"""
import asyncio, json, sys
from pathlib import Path
from recommenditos.api.app import create_app
app = create_app(Path(sys.argv[1]))
served = app.state.served
levels = served.model.input_schema.levels
listing = {
    "make": served.allowed["make"][0], "model": levels["model"][0],
    "registration_date": "2019-06", "mileage_km_raw": 45000, "power_kw": 110,
    "fuel_category": served.allowed["fuel_category"][0],
    "transmission": served.allowed["transmission"][0],
    "country_code": levels["country_code"][0],
    "body_type": served.allowed["body_type"][0],
    "seller_type": served.allowed["seller_type"][0],
}
body = json.dumps(listing).encode()
sent = []
async def receive():
    return {"type": "http.request", "body": body, "more_body": False}
async def send(message):
    sent.append(message)
scope = {
    "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
    "scheme": "http", "path": "/v1/predict", "raw_path": b"/v1/predict", "query_string": b"",
    "root_path": "", "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 80),
    "headers": [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
    ],
}
asyncio.run(app(scope, receive, send))
status = sent[0]["status"]
answer = json.loads(b"".join(m.get("body", b"") for m in sent[1:]))
assert status == 200 and answer["price_eur"] > 0, (status, answer)
print(json.dumps(sorted({name.partition(".")[0] for name in sys.modules})))
"""


def _variants() -> dict:
    return load_params(PARAMS_FILE)["train"]["variants"]


#: Every variant, so a parametrised test says which one failed in its own name.
VARIANTS = tuple(_variants())

#: The variants the API can serve: those with a SHAP export to explain a
#: valuation with (FR-08).
SERVABLE = tuple(name for name, block in _variants().items() if block["estimator"] == "lightgbm")


@pytest.fixture(scope="module")
def served(tmp_path_factory) -> dict:
    """One bundle per variant on disk, built with the serving path's own modules.

    `preprocess`, `split` and `features` run as the pipeline runs them; the fit
    goes through `fit_variant` and `Model.save` rather than `train.main`, because
    `train` opens an MLflow run and MLflow is a pipeline package. The rows are
    restricted to the supported makes the way `train` restricts them (EDN-05),
    so the bundles carry the same scope as a real one.
    """
    params = load_params(PARAMS_FILE)
    root = tmp_path_factory.mktemp("serving")
    raw = root / "raw" / "listings.parquet"
    raw.parent.mkdir(parents=True)
    generate_raw_listings(FIXTURE_ROWS).to_parquet(raw, index=False)

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
            request=RequestContract.from_params(params),
        )
        model = fit_variant(
            variant,
            settings,
            data,
            seed=params["seed"],
            num_threads=params["train"]["num_threads"],
        )
        chosen = interval_settings(variant, settings)
        if chosen is not None:
            # The price range's boosters, fitted with LightGBM alone. Their
            # calibration needs MAPIE, a pipeline package, so the offsets here are
            # stand-ins: this file checks what serving reads and imports, and
            # `tests/test_intervals.py` checks the served range against MAPIE's.
            estimators, record = fit_quantile_estimators(
                model,
                data,
                chosen,
                seed=params["seed"],
                num_threads=params["train"]["num_threads"],
            )
            record["offsets"] = {"lower": 0.05, "upper": 0.05}
            model.attach_intervals(
                {name: each.booster_ for name, each in estimators.items()}, record
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


INTERVAL_VARIANTS = tuple(
    name for name, settings in _variants().items() if interval_settings(name, settings) is not None
)


@pytest.mark.parametrize("variant", INTERVAL_VARIANTS)
def test_a_bundle_on_disk_prices_a_range(variant: str, served: dict):
    """UC2's range from disk, with the runtime set alone (EDN-85).

    A request carrying only `make` is encoded through the bundle's own feature
    space, as `/price-range` will encode it, and gets a finite, positive range
    that is wider than the range of the complete listing.
    """
    model = load_model(served["models"] / variant)
    supported = tuple(model.metadata["training"]["supported_makes"])
    listings = _supported_only(
        read_frame(served["processed"] / "test.parquet", PROCESSED_SCHEMA), supported
    ).head(LISTINGS)
    request = model.space.matrix(listings, reference_date=served["params"]["reference_date"])
    make_only = model.mask_absent(request, [name for name in model.features if name != "make"])

    full, partial = model.predict_interval_eur(request), model.predict_interval_eur(make_only)

    assert model.has_intervals
    for interval in (full, partial):
        assert len(interval) == len(listings) > 0
        assert np.isfinite(interval.to_numpy()).all()
        assert (interval["lower_eur"] > 0).all()
    assert (partial["upper_eur"] - partial["lower_eur"]).mean() > (
        full["upper_eur"] - full["lower_eur"]
    ).mean()


@pytest.mark.parametrize(
    ("variant", "path"),
    [(variant, "model") for variant in VARIANTS] + [(variant, "api") for variant in SERVABLE],
)
def test_serving_imports_nothing_only_the_test_group_installs(
    variant: str, path: str, served: dict
):
    """The serving path, and the API around it, do not lean on pytest's dependencies.

    The check's environment is the runtime set plus the `test` group, so a module
    the serving path imported from `packaging` or `pluggy`, or the API from the
    test client's `httpx2`, would import here and fail in the image. A fresh
    interpreter is the only place to ask which modules the path imported: this
    process has pytest loaded already.
    """
    arguments = (
        [
            _SERVE_AND_LIST_MODULES,
            str(served["models"] / variant),
            str(served["processed"] / "test.parquet"),
            served["params"]["reference_date"],
        ]
        if path == "model"
        else [_SERVE_THROUGH_THE_API, str(served["models"] / variant)]
    )
    result = subprocess.run(
        [sys.executable, "-c", *arguments],
        capture_output=True,
        text=True,
        check=False,
        # Tracking is not on the serving path; keep a developer's `.env` out of it.
        env={**os.environ, "MLFLOW_TRACKING_URI": ""},
    )
    assert result.returncode == 0, result.stderr
    imported = json.loads(result.stdout.strip().splitlines()[-1])

    lock = read_lock()
    test_only = group_closure(lock, TEST_GROUP) - runtime_closure(lock)
    owners = packages_distributions()
    leaned_on = sorted(
        f"{module} ({', '.join(sorted(test_only & dists))})"
        for module in imported
        if (dists := {normalise(name) for name in owners.get(module, ())}) & test_only
    )
    assert not leaned_on, (
        f"serving {variant!r} imported modules that only the {TEST_GROUP!r} group installs: "
        f"{', '.join(leaned_on)}. The API image will not have them; declare what serving "
        f"needs in `[project] dependencies`."
    )


@pytest.mark.req("FR-06", "FR-08", "FR-12")
@pytest.mark.parametrize("variant", SERVABLE)
def test_the_api_values_a_listing_with_a_bundle_on_disk(variant: str, served: dict):
    """Every LightGBM variant can be served: the API loads it and answers a valuation.

    `tests/api/` tests the API in depth on `lgbm-basic`; this runs it in the
    runtime-only environment, where the API image's dependencies are all there is.
    The explanation comes from the booster itself, so it is answered here too.
    """
    with TestClient(create_app(served["models"] / variant)) as client:
        app_model = client.app.state.served  # type: ignore[attr-defined]
        levels = app_model.model.input_schema.levels
        listing = {
            "make": app_model.allowed["make"][0],
            "model": levels["model"][0],
            "registration_date": "2019-06",
            "mileage_km_raw": 45000,
            "power_kw": 110,
            "fuel_category": app_model.allowed["fuel_category"][0],
            "transmission": app_model.allowed["transmission"][0],
            "country_code": levels["country_code"][0],
            "body_type": app_model.allowed["body_type"][0],
            "seller_type": app_model.allowed["seller_type"][0],
        }
        response = client.post("/v1/predict", json=listing)
        health = client.get("/health").json()

    assert response.status_code == 200, response.text
    assert response.json()["price_eur"] > 0
    assert len(response.json()["explanation"]["top_features"]) == 5
    assert health["model"]["variant"] == variant


@pytest.mark.req("FR-08")
@pytest.mark.parametrize("variant", sorted(set(VARIANTS) - set(SERVABLE)))
def test_the_api_refuses_a_bundle_it_cannot_explain_at_startup(variant: str, served: dict):
    with pytest.raises(ServingError, match="SHAP export"):
        create_app(served["models"] / variant)


@pytest.mark.skipif(
    os.environ.get(SERVING_RUNTIME_ENV_VAR) != "1",
    reason=f"only meaningful in the runtime-only environment; run `make test-serving`, "
    f"which sets {SERVING_RUNTIME_ENV_VAR}=1",
)
def test_the_environment_is_the_runtime_set_and_the_test_group_and_nothing_more():
    """Everything installed here is in the runtime's closure or the test group's.

    Without it, the serving check could pass vacuously: a `uv sync` that lost its
    `--no-default-groups`, or a `uv run` that re-synced the default groups into
    the environment, would install everything and every test above would still
    pass. Both closures are read from `uv.lock`, so the set is exactly what uv
    installs, and a dependency added to a group is covered the day it is added.
    """
    lock = read_lock()
    allowed = runtime_closure(lock) | group_closure(lock, TEST_GROUP)
    installed = {normalise(dist.metadata["Name"]) for dist in distributions()}
    extra = sorted(installed - allowed - {PROJECT})
    assert not extra, (
        f"the serving check is running in an environment with packages the runtime and "
        f"the {TEST_GROUP!r} group do not install: {', '.join(extra)}. It has to be built "
        f"with `uv sync --no-default-groups --group {TEST_GROUP}`, as `make test-serving` does."
    )


def test_the_closures_the_check_relies_on_read_the_real_lock():
    """The two closures above, computed from the committed `uv.lock`.

    The guard is skipped everywhere but the serving job, so a lock it could no
    longer read would only show up there; this pins what it reads in every run.
    It asserts what the two tests rely on rather than the exact sets, so an
    upgrade that adds or drops a transitive package does not turn it red: the
    runtime closure follows requirements and leaves the groups out, and every
    package the test group declares is one the runtime does not install, which
    is what makes the test-only set the one worth guarding.
    """
    lock = read_lock()
    runtime = runtime_closure(lock)
    test_only = group_closure(lock, TEST_GROUP) - runtime
    declared = {normalise(package["name"]) for package in group_dependencies(lock, TEST_GROUP)}

    # Followed transitively: lightgbm brings scipy and narwhals with it.
    assert {"lightgbm", "scipy", "narwhals"} <= runtime
    assert runtime.isdisjoint({"mlflow", "dvc", "requests", "tqdm", "pytest"})
    assert declared <= test_only
