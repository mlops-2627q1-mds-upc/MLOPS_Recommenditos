"""`POST /v1/feedback` inside the Compose network, where the replay job calls it.

From outside the proxy refuses it (`test_deployed_proxy.py`); this is the other
half of that rule: with the key, a sibling container stores a price, and the
prediction log on the API's volume joins it to its answer (FR-10, FR-13, EDN-86).
"""

import json
import os

import httpx2
import pytest
from tests.integration.conftest import Container

from recommenditos.api.settings import API_KEY_ENV_VAR, PREDICTION_LOG_ENV_VAR

pytestmark = [pytest.mark.integration, pytest.mark.docker_host]

#: Reads the labelled answer from the log, in the API's container, where the file is.
_READ_LABEL = f"""
import json, os, sqlite3, sys
log = sqlite3.connect(os.environ["{PREDICTION_LOG_ENV_VAR}"])
print(json.dumps(log.execute(
    "SELECT json_extract(outputs, '$.price_eur'), observed_price_eur "
    "FROM labelled_predictions WHERE request_id = ?", (sys.argv[1],)).fetchall()))
"""


def _environment(container: Container) -> dict[str, str]:
    pairs = container.inspect()["Config"]["Env"]
    return dict(pair.split("=", 1) for pair in pairs)


def _api_key(api_container: Container) -> str:
    key = _environment(api_container).get(API_KEY_ENV_VAR, "")
    if not key:
        if os.environ.get(API_KEY_ENV_VAR):
            pytest.fail(f"{API_KEY_ENV_VAR} is set here but did not reach the api service")
        pytest.skip(f"the stack runs without {API_KEY_ENV_VAR}; start it with the key set")
    return key


@pytest.mark.req("FR-10", "FR-13")
def test_feedback_with_the_key_is_stored_and_joins_its_answer_in_the_log(
    client: httpx2.Client,
    listing: dict,
    api_container: Container,
    proxy_container: Container,
):
    key = _api_key(api_container)
    answer = client.post("/v1/predict", json=listing)
    assert answer.status_code == 200, answer.text
    request_id = answer.json()["request_id"]
    body = json.dumps({"request_id": request_id, "observed_price_eur": 20500.0})

    # From the proxy's container, a sibling of the API like the replay job will
    # be; busybox `wget` fails on any status but 2xx.
    receipt = proxy_container.exec(
        "wget", "-q", "-O", "-",
        "--header", "Content-Type: application/json",
        "--header", f"X-API-Key: {key}",
        "--post-data", body,
        "http://api:8000/v1/feedback",
    )  # fmt: skip

    assert json.loads(receipt)["request_id"] == request_id
    labelled = json.loads(api_container.exec("python", "-c", _READ_LABEL, request_id))
    assert labelled == [[answer.json()["price_eur"], 20500.0]]
