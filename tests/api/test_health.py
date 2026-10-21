"""`GET /health`: the status and the loaded model version (FR-11, FR-12)."""

from pathlib import Path
import tempfile

from dvc.fs import LocalFileSystem
from dvc_data.hashfile.build import build
from dvc_data.hashfile.db import HashFileDB
from fastapi.testclient import TestClient
import pytest

from recommenditos.api.serving import ServedModel, bundle_version


@pytest.mark.req("FR-11")
def test_fr11_health_returns_the_status_and_the_loaded_model(
    client: TestClient, served: ServedModel, bundle: Path
):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": {
            "variant": "lgbm-basic",
            "version": bundle_version(bundle),
            "trained_at": served.model.metadata["trained_at"],
        },
    }


@pytest.mark.req("FR-12")
def test_fr12_the_model_version_is_the_hash_dvc_records_for_the_bundle(bundle: Path):
    """The version `/health` reports is the value `dvc.lock` holds for `models/<variant>`.

    Computed by DVC's own hashing of the same directory, so FR-12's check on the
    built image can compare the two strings.
    """
    with tempfile.TemporaryDirectory() as cache:
        fs = LocalFileSystem()
        _, _, tree = build(HashFileDB(fs, cache), str(bundle), fs, "md5")

    assert f"{bundle_version(bundle)}.dir" == tree.hash_info.value
