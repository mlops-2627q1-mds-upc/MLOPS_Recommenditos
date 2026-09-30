"""Build the requirement-to-test matrix of NFR-07, and gate the build on it.

The `FR-xx` and `NFR-xx` IDs of docs/docs/requirements.md are the unit of
traceability. Three sources say something about each of them, and this module is
what joins them into one table the report can cite instead of a claim:

* the requirements, which define the IDs and their priority;
* the specification, whose "Verified by" column says whether a test is supposed
  to exist at all or whether the evidence is a drill a person runs, and names
  that evidence;
* the test suite, where `@pytest.mark.req("FR-15", "NFR-01")` records what a
  test verifies.

Three findings exit non-zero, because letting any of them pass destroys the
evidence quietly rather than loudly:

* a marker naming an ID no requirement has - the coverage it was meant to record
  is lost, and the matrix would still look complete;
* the two documents disagreeing about which IDs exist - the specification
  promises that no entry of it exists without a requirement behind it;
* a requirement whose evidence is due (see tools/expected_coverage.yaml) and
  which nothing verifies.

Completeness beyond that is not enforced here: before M4 there is no API, so most
functional requirements cannot have evidence yet. They are reported as missing
without failing the build until the milestone that owes them is enforced.
"""

import contextlib
from dataclasses import dataclass
from functools import lru_cache
import io
import json
from pathlib import Path
import re

from loguru import logger
import pytest
import typer
import yaml

# Derived here rather than imported from `recommenditos.config`: the matrix is
# repository metadata and has to build even when the package itself cannot be
# imported, which is exactly the state a broken pull request is in.
PROJ_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_DOC = PROJ_ROOT / "docs" / "docs" / "requirements.md"
SPECIFICATION_DOC = PROJ_ROOT / "docs" / "docs" / "specification.md"
EXPECTED_COVERAGE_FILE = Path(__file__).resolve().parent / "expected_coverage.yaml"
TESTS_DIR = PROJ_ROOT / "tests"
MATRIX_MD = PROJ_ROOT / "reports" / "requirement-matrix.md"
MATRIX_JSON = PROJ_ROOT / "reports" / "requirement-matrix.json"

#: The marker a test uses to declare the requirements it verifies.
MARKER = "req"

#: How the specification's "Verified by" column tags an entry.
AUTOMATED = "automated"
MANUAL = "manual"

#: What the matrix says about a requirement. `MISSING` is the only one the gate
#: reacts to; `VERIFIED_BY_HAND` is a route to evidence, not a gap.
COVERED = "covered"
VERIFIED_BY_HAND = "by hand"
MISSING = "missing"

_ID_PATTERN = re.compile(r"(?:FR|NFR)-\d{2}")

#: The specification's tables are `ID | Realisation | Verified by`. The parser
#: reads the verification tag by position, so a row of another shape is refused
#: rather than misread as having no tag at all.
_SPECIFICATION_CELLS = 3


class TraceabilityError(Exception):
    """The three sources of the matrix do not agree, so the matrix would lie."""


@dataclass(frozen=True)
class Requirement:
    """One row of the requirements, as the three sources jointly describe it."""

    id: str
    priority: str
    verification: str
    evidence: str
    milestone: str


@dataclass(frozen=True)
class Entry:
    """A requirement together with what actually verifies it."""

    requirement: Requirement
    tests: tuple[str, ...]
    enforced: bool

    @property
    def status(self) -> str:
        if self.tests:
            return COVERED
        if self.requirement.verification == MANUAL and self.requirement.evidence:
            return VERIFIED_BY_HAND
        return MISSING


@dataclass(frozen=True)
class Matrix:
    """The whole table, plus where the gate currently stands."""

    entries: tuple[Entry, ...]
    milestone: str
    enforced_milestones: tuple[str, ...]

    def blocking(self) -> tuple[Entry, ...]:
        """The gaps that fail the build: due already, and verified by nothing."""
        return tuple(entry for entry in self.entries if entry.enforced and entry.status == MISSING)

    def counts(self) -> dict[str, int]:
        counts = dict.fromkeys((COVERED, VERIFIED_BY_HAND, MISSING), 0)
        for entry in self.entries:
            counts[entry.status] += 1
        return counts


# --------------------------------------------------------------------------
# Reading the three sources
# --------------------------------------------------------------------------


def _id_rows(text: str) -> list[tuple[str, list[str]]]:
    """Every Markdown table row whose first cell is exactly a requirement ID.

    Anchoring on the first cell is what keeps everything else out: the prose and
    the quality-model table of the requirements both mention IDs, and neither
    opens a row with one.
    """
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if _ID_PATTERN.fullmatch(cells[0]):
            rows.append((cells[0], cells))
    return rows


def parse_requirements(text: str) -> dict[str, str]:
    """Requirement ID to MoSCoW priority, from the requirement tables."""
    priorities: dict[str, str] = {}
    for req_id, cells in _id_rows(text):
        if req_id in priorities:
            raise TraceabilityError(f"{req_id} has two rows in the requirements")
        priority = cells[1] if len(cells) > 1 else ""
        if not priority:
            raise TraceabilityError(f"the row of {req_id} in the requirements names no priority")
        priorities[req_id] = priority
    if not priorities:
        raise TraceabilityError("the requirements contain no FR-xx or NFR-xx table row at all")
    return priorities


def parse_specification(text: str) -> dict[str, tuple[str, str]]:
    """Requirement ID to how it is verified and to the evidence that is named.

    The "Verified by" cell has to carry exactly one of the two tags. Both or
    neither would leave the matrix guessing whether a missing test is a gap or
    the documented answer, which is the one thing this table must not do.
    """
    entries: dict[str, tuple[str, str]] = {}
    for req_id, cells in _id_rows(text):
        if req_id in entries:
            raise TraceabilityError(f"{req_id} has two rows in the specification")
        if len(cells) != _SPECIFICATION_CELLS:
            raise TraceabilityError(
                f"the row of {req_id} in the specification has {len(cells)} cells, "
                "expected ID, realisation and 'Verified by'"
            )
        verified_by = cells[2]
        tagged = [tag for tag in (AUTOMATED, MANUAL) if f"**[{tag}]**" in verified_by]
        if len(tagged) != 1:
            raise TraceabilityError(
                f"the 'Verified by' cell of {req_id} names {len(tagged)} of "
                "**[automated]** and **[manual]**, expected exactly one"
            )
        evidence = verified_by.replace(f"**[{tagged[0]}]**", "").strip()
        entries[req_id] = (tagged[0], evidence)
    if not entries:
        raise TraceabilityError("the specification contains no FR-xx or NFR-xx table row at all")
    return entries


@dataclass(frozen=True)
class ExpectedCoverage:
    """When each requirement owes evidence, and which milestones CI enforces."""

    current: str
    order: tuple[str, ...]
    due: dict[str, str]

    @property
    def enforced_milestones(self) -> tuple[str, ...]:
        return self.order[: self.order.index(self.current) + 1]

    def enforces(self, req_id: str) -> bool:
        return self.due[req_id] in self.enforced_milestones


def parse_expected_coverage(text: str, known_ids: set[str]) -> ExpectedCoverage:
    """Read tools/expected_coverage.yaml and check it against the real IDs.

    Insisting that every documented requirement appears exactly once is what
    makes the gate tighten by itself: a requirement added to the docs fails the
    build here until someone decides which milestone owes its evidence.
    """
    document = yaml.safe_load(text)
    if not isinstance(document, dict) or "current" not in document or "milestones" not in document:
        raise TraceabilityError(
            "the expected-coverage set needs a `current` milestone and a `milestones` list"
        )

    order: list[str] = []
    due: dict[str, str] = {}
    for block in document["milestones"]:
        milestone = block["milestone"]
        if milestone in order:
            raise TraceabilityError(f"the expected-coverage set lists {milestone} twice")
        order.append(milestone)
        for req_id in block.get("requirements") or []:
            if req_id in due:
                raise TraceabilityError(
                    f"the expected-coverage set books {req_id} to both {due[req_id]} "
                    f"and {milestone}"
                )
            due[req_id] = milestone

    current = document["current"]
    if current not in order:
        raise TraceabilityError(
            f"`current` is {current}, which is not one of the milestones {', '.join(order)}"
        )

    unknown = sorted(set(due) - known_ids)
    if unknown:
        raise TraceabilityError(
            f"the expected-coverage set books {', '.join(unknown)}, which "
            f"{REQUIREMENTS_DOC.name} does not define"
        )
    unbooked = sorted(known_ids - set(due))
    if unbooked:
        raise TraceabilityError(
            f"{', '.join(unbooked)} is not booked to any milestone in the expected-coverage "
            "set; add it to the milestone whose work produces its evidence"
        )

    return ExpectedCoverage(current=current, order=tuple(order), due=due)


@lru_cache(maxsize=1)
def known_requirement_ids(path: Path = REQUIREMENTS_DOC) -> frozenset[str]:
    """The IDs a `req` marker may name. Cached: the test suite asks per test."""
    return frozenset(parse_requirements(path.read_text(encoding="utf-8")))


def unknown_marker_ids(marked_ids: tuple[str, ...]) -> list[str]:
    """Which of the IDs a marker names are not requirements at all.

    The test suite calls this per test so that a typo fails there and not only
    in CI, which is why it lives next to the parser instead of in conftest.
    """
    return sorted(set(marked_ids) - known_requirement_ids())


# --------------------------------------------------------------------------
# Reading the markers
# --------------------------------------------------------------------------


class _MarkerCollector:
    """Records the `req` markers pytest sees, without running a single test."""

    def __init__(self) -> None:
        self.marked: dict[str, tuple[str, ...]] = {}

    def pytest_collection_modifyitems(self, items) -> None:
        for item in items:
            ids = tuple(
                dict.fromkeys(
                    str(arg) for marker in item.iter_markers(name=MARKER) for arg in marker.args
                )
            )
            if not ids:
                continue
            # A parametrised test is one test in the matrix: twelve rows of the
            # same node ID with a different `[case]` suffix say nothing extra.
            self.marked.setdefault(item.nodeid.partition("[")[0], ids)


def collect_req_markers(tests_dir: Path = TESTS_DIR) -> dict[str, tuple[str, ...]]:
    """Every test's `req` marker, read from a real pytest collection.

    Collecting rather than parsing the source means the matrix sees exactly what
    pytest sees, so a marker on a class or a module counts like one written
    above a function, and a test that was renamed cannot linger in the matrix.
    """
    collector = _MarkerCollector()
    # The configuration and the root are passed explicitly, so the matrix builds
    # the same from any working directory. `--no-cov` keeps pytest-cov from
    # starting a second measurement on top of the one the test job reports, and
    # nothing is cached because collecting is not a test run.
    arguments = [
        "--collect-only",
        "-q",
        "--no-cov",
        "-p",
        "no:cacheprovider",
        "-c",
        str(PROJ_ROOT / "pyproject.toml"),
        "--rootdir",
        str(PROJ_ROOT),
        str(tests_dir),
    ]
    # Swallowed on purpose: one line per collected test would bury the matrix in
    # the CI log. It is handed back in the error when collection fails, which is
    # the only case where anyone wants to read it.
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        outcome = pytest.main(arguments, plugins=[collector])
    if outcome != pytest.ExitCode.OK:
        raise TraceabilityError(
            f"pytest could not collect the test suite (exit code {int(outcome)}):\n"
            f"{captured.getvalue()}"
        )
    return collector.marked


# --------------------------------------------------------------------------
# The matrix
# --------------------------------------------------------------------------


def _sort_key(req_id: str) -> tuple[int, int]:
    kind, _, number = req_id.partition("-")
    return (0 if kind == "FR" else 1, int(number))


def build_matrix(
    *,
    requirements_text: str,
    specification_text: str,
    coverage_text: str,
    markers: dict[str, tuple[str, ...]],
) -> Matrix:
    """Join the three sources and the markers into the matrix.

    Takes text rather than paths so that the unit tests can state a tiny,
    complete pair of documents instead of depending on the real ones.
    """
    priorities = parse_requirements(requirements_text)
    specified = parse_specification(specification_text)

    unspecified = sorted(set(priorities) - set(specified), key=_sort_key)
    if unspecified:
        raise TraceabilityError(
            f"{', '.join(unspecified)} has no entry in the specification, so nothing says "
            "how it is verified"
        )
    stray = sorted(set(specified) - set(priorities), key=_sort_key)
    if stray:
        raise TraceabilityError(
            f"the specification has an entry for {', '.join(stray)}, which is not a requirement"
        )

    coverage = parse_expected_coverage(coverage_text, set(priorities))

    covering: dict[str, list[str]] = {req_id: [] for req_id in priorities}
    unknown: dict[str, list[str]] = {}
    for test_id, marked_ids in markers.items():
        for req_id in marked_ids:
            if req_id in covering:
                covering[req_id].append(test_id)
            else:
                unknown.setdefault(req_id, []).append(test_id)
    if unknown:
        named = "; ".join(
            f"{req_id} in {', '.join(sorted(tests))}" for req_id, tests in sorted(unknown.items())
        )
        raise TraceabilityError(
            f"a req marker names an ID {REQUIREMENTS_DOC.name} does not define: {named}"
        )

    entries = tuple(
        Entry(
            requirement=Requirement(
                id=req_id,
                priority=priorities[req_id],
                verification=specified[req_id][0],
                evidence=specified[req_id][1],
                milestone=coverage.due[req_id],
            ),
            tests=tuple(sorted(covering[req_id])),
            enforced=coverage.enforces(req_id),
        )
        for req_id in sorted(priorities, key=_sort_key)
    )
    return Matrix(
        entries=entries,
        milestone=coverage.current,
        enforced_milestones=coverage.enforced_milestones,
    )


def _evidence_cell(entry: Entry) -> str:
    """What verifies the requirement, or what the specification still owes.

    A missing automated entry shows the test the specification promises rather
    than an empty cell, because that sentence is the ticket for whoever closes
    the gap.
    """
    requirement = entry.requirement
    by_hand = requirement.verification == MANUAL and requirement.evidence
    if entry.tests:
        tests = ", ".join(f"`{test}`" for test in entry.tests)
        return f"{tests}; by hand: {requirement.evidence}" if by_hand else tests
    if by_hand:
        return f"by hand: {requirement.evidence}"
    if requirement.evidence:
        return f"expected: {requirement.evidence}"
    return "nothing"


_PREAMBLE = """\
Requirement-to-test matrix
==========================

Generated by `tools/requirement_matrix.py`, which every run rewrites, so an edit here
is lost. Rebuild it with `uv run python -m tools.requirement_matrix`.

It joins the IDs and priorities of `docs/docs/requirements.md`, the verification route
and the named evidence of `docs/docs/specification.md`, the `req` markers of the test
suite, and the milestone each requirement owes its evidence in, from
`tools/expected_coverage.yaml`.

- covered by a test: {covered}
- verified by hand, with named evidence: {by_hand}
- verified by nothing: {missing}, of which {blocking} are already due

The gate stands at milestone {milestone}, so it enforces {enforced}. A requirement
booked to one of those milestones and verified by nothing fails the build. A
requirement of a later milestone is reported and fails nothing, because its evidence
cannot exist yet.
"""


def render_markdown(matrix: Matrix) -> str:
    """The matrix as the job summary and the delivery review read it."""
    counts = matrix.counts()
    blocking = matrix.blocking()
    sections = [
        _PREAMBLE.format(
            covered=counts[COVERED],
            by_hand=counts[VERIFIED_BY_HAND],
            missing=counts[MISSING],
            blocking=len(blocking),
            milestone=matrix.milestone,
            enforced=", ".join(matrix.enforced_milestones),
        )
    ]

    if blocking:
        # Above the table, because the job summary has to answer "why is this
        # red?" without anyone scrolling through 29 rows.
        sections.append(
            "\n".join(
                [
                    "## Verified by nothing, and already due",
                    "",
                    *(
                        f"- {entry.requirement.id}, due {entry.requirement.milestone}"
                        for entry in blocking
                    ),
                ]
            )
            + "\n"
        )

    rows = []
    for entry in matrix.entries:
        requirement = entry.requirement
        # Bold, because in the job summary this is the row that broke the build.
        status = f"**{entry.status}**" if entry in blocking else entry.status
        rows.append(
            f"| {requirement.id} | {requirement.priority} | {requirement.verification} "
            f"| {requirement.milestone} | {status} | {_evidence_cell(entry)} |"
        )
    sections.append(
        "\n".join(
            [
                "## The matrix",
                "",
                "| ID | Priority | Verified by | Due | Status | Evidence |",
                "|----|----------|-------------|-----|--------|----------|",
                *rows,
            ]
        )
        + "\n"
    )
    return "\n".join(sections)


def as_json(matrix: Matrix) -> dict:
    """The same matrix for whatever reads it next.

    No timestamp: two runs on the same commit produce the same bytes, so a
    change in this file is a change in the traceability and not in the clock.
    """
    return {
        "gate": {
            "milestone": matrix.milestone,
            "enforced_milestones": list(matrix.enforced_milestones),
            "blocking": [entry.requirement.id for entry in matrix.blocking()],
        },
        "counts": matrix.counts(),
        "requirements": [
            {
                "id": entry.requirement.id,
                "priority": entry.requirement.priority,
                "verification": entry.requirement.verification,
                "evidence": entry.requirement.evidence,
                "due": entry.requirement.milestone,
                "enforced": entry.enforced,
                "status": entry.status,
                "tests": list(entry.tests),
            }
            for entry in matrix.entries
        ],
    }


def write_matrix(matrix: Matrix, markdown_path: Path, json_path: Path) -> None:
    for path, content in (
        (markdown_path, render_markdown(matrix)),
        (json_path, json.dumps(as_json(matrix), indent=2) + "\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.info(f"Wrote {path}.")


app = typer.Typer()


@app.command()
def main(markdown_path: Path = MATRIX_MD, json_path: Path = MATRIX_JSON):
    try:
        matrix = build_matrix(
            requirements_text=REQUIREMENTS_DOC.read_text(encoding="utf-8"),
            specification_text=SPECIFICATION_DOC.read_text(encoding="utf-8"),
            coverage_text=EXPECTED_COVERAGE_FILE.read_text(encoding="utf-8"),
            markers=collect_req_markers(),
        )
    except TraceabilityError as error:
        # Logged rather than raised: a mismatch between the documents and the
        # markers is a finding to read, not a stack trace to debug.
        logger.error(str(error))
        raise typer.Exit(1) from error

    write_matrix(matrix, markdown_path, json_path)

    blocking = matrix.blocking()
    if blocking:
        logger.error(
            f"{len(blocking)} requirement(s) due by {matrix.milestone} are verified by "
            f"nothing: {', '.join(entry.requirement.id for entry in blocking)}"
        )
        raise typer.Exit(1)

    counts = matrix.counts()
    logger.success(
        f"{counts[COVERED]} requirement(s) covered by a test, "
        f"{counts[VERIFIED_BY_HAND]} verified by hand, {counts[MISSING]} by nothing, "
        f"none of them due by {matrix.milestone}."
    )


if __name__ == "__main__":
    app()
