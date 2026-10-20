"""Assertions the API tests share."""

from httpx2 import Response

from recommenditos.api.errors import PROBLEM_MEDIA_TYPE
from recommenditos.api.middleware import REQUEST_ID_HEADER

#: The members every problem carries (RFC 9457), and the one a 422 adds.
PROBLEM_MEMBERS = {"type", "title", "status", "detail", "instance"}


def assert_problem(response: Response, status: int) -> dict:
    """`response` is an RFC 9457 problem with `status`; returns its body.

    The same check for every error path is what FR-16's "none differ in shape"
    means in a test: the media type, the members, the status repeated in the
    body, and the request ID in `instance`.
    """
    assert response.status_code == status, response.text
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    body = response.json()
    expected = PROBLEM_MEMBERS | ({"errors"} if status == 422 else set())
    assert set(body) == expected, body
    assert body["status"] == status
    assert body["instance"] == f"urn:uuid:{response.headers[REQUEST_ID_HEADER]}"
    assert body["title"] and body["detail"]
    return body


def pointers(problem: dict) -> list[str]:
    """The fields a 422 names, in its order."""
    return [error["pointer"] for error in problem["errors"]]
