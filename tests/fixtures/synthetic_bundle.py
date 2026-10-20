"""Model bundles fitted on the synthetic fixture, for the tests and for the CI image.

CI has no credentials to `dvc pull` the real bundles, so everything that needs
one fits it here from the seeded fixture: the `matrices` and `trained` fixtures
of `tests/conftest.py`, and the integration job, which builds the API image
around such a bundle (docs/docs/deployment.md):

    uv run python -m tests.fixtures.synthetic_bundle lgbm-basic models

writes `models/lgbm-basic` and the comparables index `models/comparables.parquet`
beside it, and prints the version `/health` will report for the bundle.
The pipeline's own stage functions do the work, with tracking off, so a bundle
from here has the layout and the metadata of a real one.
"""

import argparse
from collections.abc import Iterable
import os
from pathlib import Path
import tempfile

import pandas as pd

from recommenditos.config import PARAMS_FILE
from recommenditos.data import build_comparables, build_features, preprocess, split_data
from recommenditos.data.synthetic import generate_raw_listings
from recommenditos.pipeline import load_params

#: Small enough to keep the suite fast and fit every variant in seconds, large
#: enough that a seller-grouped split and a per-make count still mean something.
FIXTURE_ROWS = 2000


def build_matrices(root: Path, raw_frame: pd.DataFrame, params_path: Path = PARAMS_FILE) -> dict:
    """The feature matrices of every feature set, built from `raw_frame` under `root`.

    `validate-data` is skipped: it asserts the data's own expectations and writes
    a summary nothing here reads, and `tests/test_pipeline.py` already runs it in
    the chain.
    """
    raw = root / "raw" / "listings.parquet"
    raw.parent.mkdir(parents=True)
    raw_frame.to_parquet(raw, index=False)

    processed = root / "processed"
    features = processed / "features"
    preprocess.main(raw, root / "interim" / "listings.parquet", params_path)
    split_data.main(root / "interim" / "listings.parquet", processed, params_path)
    for feature_set in load_params(params_path)["features"]["sets"]:
        build_features.main(feature_set, processed, features, params_path)
    return {"root": root, "processed": processed, "features": features}


def train_bundles(
    variants: Iterable[str],
    features: Path,
    models: Path,
    energy: Path,
    params_path: Path = PARAMS_FILE,
) -> None:
    """Fit each variant with the `train` stage into `models/<variant>`.

    The caller clears the tracking variables first; `train` logs to MLflow when
    they are set.
    """
    # Imported here and not at the top: these pull in MLflow, LightGBM and
    # scikit-learn, and a test file that trains nothing should not pay for them.
    from recommenditos.modeling import train  # noqa: PLC0415

    for variant in variants:
        train.main(variant, features, models, params_path, energy)


def build_index(processed: Path, models: Path, params_path: Path = PARAMS_FILE) -> Path:
    """The comparables index of the `split` frames in `processed`, beside the bundles.

    Written where the pipeline writes it, `models/comparables.parquet`, which is
    where the API looks for it beside the bundle it serves (FR-09).
    """
    path = models / build_comparables.INDEX_FILE
    build_comparables.main(processed, path, params_path)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("variant", help="a key of train.variants in params.yaml")
    parser.add_argument("output_dir", type=Path, help="where models/<variant> goes, e.g. models")
    args = parser.parse_args(argv)

    from recommenditos.api.serving import bundle_version  # noqa: PLC0415
    from recommenditos.tracking import REQUIRED_ENV_VARS  # noqa: PLC0415

    for name in REQUIRED_ENV_VARS:
        os.environ.pop(name, None)
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        matrices = build_matrices(root, generate_raw_listings(FIXTURE_ROWS))
        train_bundles([args.variant], matrices["features"], args.output_dir, root / "emissions")
        build_index(matrices["processed"], args.output_dir)
    print(bundle_version(args.output_dir / args.variant))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
