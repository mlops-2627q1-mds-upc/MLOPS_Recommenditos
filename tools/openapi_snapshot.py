"""Write the API's OpenAPI document to docs/docs/openapi.json (`make openapi`).

The committed file is the API contract as others read it: the docs site renders
it as the API reference, so the contract can be read without running the
server, and a change to it shows in the pull request that makes it. A test in
tests/api/test_openapi.py fails when the app's document and the file differ.

The document is `openapi_contract()`, which needs no model bundle, so the file
is the same wherever it is written. It leaves out the values a checked listing
field accepts, which the served bundle supplies at runtime (docs/docs/api.md).
"""

import json
from typing import Any

from recommenditos.api.app import openapi_contract
from recommenditos.config import PROJ_ROOT

#: The committed copy of the OpenAPI document.
SNAPSHOT = PROJ_ROOT / "docs" / "docs" / "openapi.json"


def render(document: dict[str, Any]) -> str:
    """`document` as the file holds it: indented, in FastAPI's key order, with a final newline."""
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    # LF on every platform, as .gitattributes checks the file out.
    SNAPSHOT.write_text(render(openapi_contract()), encoding="utf-8", newline="\n")
    print(f"wrote {SNAPSHOT.relative_to(PROJ_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
