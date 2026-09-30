"""The generator behind the requirement-to-test matrix of NFR-07.

The matrix is the traceability evidence the report cites, so the generator is
held to the same standard as the pipeline: every rule that could silently turn
a gap into a green build has a test naming it.

Nothing here reads the 548 MB dataset or needs credentials. The parsers take
text, so each test states the smallest pair of documents that shows the rule,
and the few tests that do read the repository's real documents read only the
Markdown that is committed next to them.
"""

import json

import pytest
import tools.requirement_matrix as matrix_tool
from tools.requirement_matrix import (
    COVERED,
    MISSING,
    VERIFIED_BY_HAND,
    TraceabilityError,
    build_matrix,
)

REQUIREMENTS = """\
Requirements
============

This prose mentions FR-01 and NFR-01 without opening a table row.

| ID | Priority | Requirement | Acceptance criterion |
|----|----------|-------------|----------------------|
| FR-01 | Must | **The component answers.** | An answer arrives. |

| ID | Priority | Quality | Requirement | Acceptance criterion |
|----|----------|---------|-------------|----------------------|
| NFR-01 | Should | Availability | **It stays up.** | A drill. |

| Quality (this page) | ISO/IEC 25010 |
|----------------------|---------------|
| Availability (NFR-01) | Reliability |
"""

SPECIFICATION = """\
Specification
=============

| ID | Realisation | Verified by |
|----|-------------|-------------|
| FR-01 | `POST /predict` answers. | **[automated]** API test |
| NFR-01 | Compose restarts it. | **[manual]** Chaos test, killing the container |
"""

COVERAGE = """\
current: M3
milestones:
  - milestone: M3
    requirements: [FR-01]
  - milestone: M4
    requirements: [NFR-01]
"""

#: The same set with FR-01's evidence deferred, which is the state every
#: functional requirement is in until the API exists.
NOT_DUE_YET = """\
current: M3
milestones:
  - milestone: M3
    requirements: []
  - milestone: M4
    requirements: [NFR-01, FR-01]
"""


def matrix(
    requirements: str = REQUIREMENTS,
    specification: str = SPECIFICATION,
    coverage: str = COVERAGE,
    markers: dict[str, tuple[str, ...]] | None = None,
):
    return build_matrix(
        requirements_text=requirements,
        specification_text=specification,
        coverage_text=coverage,
        markers={} if markers is None else markers,
    )


def entry(built, req_id: str):
    return next(item for item in built.entries if item.requirement.id == req_id)


# --------------------------------------------------------------------------
# Reading the documents
# --------------------------------------------------------------------------


def test_only_a_row_that_opens_with_an_id_is_a_requirement():
    """IDs appear in prose and in the quality-model table of the real document.

    A parser that matched an ID anywhere would invent NFR-01 twice and pick a
    priority out of the ISO mapping table.
    """
    assert matrix_tool.parse_requirements(REQUIREMENTS) == {"FR-01": "Must", "NFR-01": "Should"}


def test_the_named_evidence_survives_the_verification_tag():
    """The manual evidence is the whole point of a manual entry.

    Without it a reviewer preparing a delivery has nothing to check against, so
    the tag is stripped and the sentence behind it is kept.
    """
    assert matrix_tool.parse_specification(SPECIFICATION) == {
        "FR-01": ("automated", "API test"),
        "NFR-01": ("manual", "Chaos test, killing the container"),
    }


def test_an_entry_that_does_not_say_how_it_is_verified_is_refused():
    """A missing tag would make the matrix guess whether a gap is a gap."""
    untagged = SPECIFICATION.replace("**[automated]** API test", "API test")
    with pytest.raises(TraceabilityError, match="FR-01"):
        matrix(specification=untagged)


def test_an_entry_tagged_both_ways_is_refused():
    both = SPECIFICATION.replace("**[automated]** API test", "**[automated]** **[manual]** test")
    with pytest.raises(TraceabilityError, match="expected exactly one"):
        matrix(specification=both)


def test_a_requirement_with_no_specification_entry_is_refused():
    """Nothing then says how it is verified, so the matrix cannot place it."""
    with pytest.raises(TraceabilityError, match="NFR-01"):
        matrix(specification=SPECIFICATION.replace("| NFR-01 | Compose restarts it.", "| x |"))


def test_a_specification_entry_without_a_requirement_is_refused():
    """The specification promises no entry of it exists without a requirement."""
    invented = SPECIFICATION.replace(
        "| NFR-01 | Compose", "| FR-99 | Invented. | **[automated]** test |\n| NFR-01 | Compose"
    )
    with pytest.raises(TraceabilityError, match="FR-99"):
        matrix(specification=invented)


# --------------------------------------------------------------------------
# The expected-coverage set
# --------------------------------------------------------------------------


def test_a_requirement_booked_to_no_milestone_is_refused():
    """This is what makes the gate tighten as the documents grow.

    A new requirement has to be booked to the milestone that produces its
    evidence, so nobody can add one and leave the question of when it is
    verified unanswered.
    """
    with pytest.raises(TraceabilityError, match="NFR-01 is not booked"):
        matrix(coverage="current: M3\nmilestones:\n  - milestone: M3\n    requirements: [FR-01]\n")


def test_booking_a_requirement_that_does_not_exist_is_refused():
    with pytest.raises(TraceabilityError, match="FR-42"):
        matrix(coverage=COVERAGE.replace("[NFR-01]", "[NFR-01, FR-42]"))


def test_booking_a_requirement_to_two_milestones_is_refused():
    with pytest.raises(TraceabilityError, match="both M3 and M4"):
        matrix(coverage=COVERAGE.replace("[NFR-01]", "[NFR-01, FR-01]"))


def test_the_current_milestone_must_be_one_of_the_listed_ones():
    """A typo in `current` would otherwise silently disable or widen the gate."""
    with pytest.raises(TraceabilityError, match="not one of the milestones"):
        matrix(coverage=COVERAGE.replace("current: M3", "current: M3b"))


def test_the_gate_enforces_every_milestone_up_to_the_current_one():
    """Evidence due in an earlier milestone stays due once the gate moves on."""
    built = matrix(coverage=COVERAGE.replace("current: M3", "current: M4"))
    assert built.enforced_milestones == ("M3", "M4")
    assert entry(built, "NFR-01").enforced


# --------------------------------------------------------------------------
# What covers what
# --------------------------------------------------------------------------


def test_a_marker_naming_an_unknown_id_is_refused():
    """A typo drops the coverage of the requirement it was meant to record.

    The matrix would still look complete, which is the one failure mode that
    makes the whole table worthless.
    """
    with pytest.raises(TraceabilityError, match="FR-15 in tests/test_api.py::test_it"):
        matrix(markers={"tests/test_api.py::test_it": ("FR-01", "FR-15")})


def test_a_test_carrying_the_id_covers_the_requirement():
    built = matrix(markers={"tests/test_api.py::test_it": ("FR-01",)})
    covered = entry(built, "FR-01")
    assert covered.status == COVERED
    assert covered.tests == ("tests/test_api.py::test_it",)
    assert not built.blocking()


def test_an_automated_requirement_without_a_test_is_missing():
    """The specification says a test is supposed to exist, and none does."""
    assert entry(matrix(), "FR-01").status == MISSING


def test_a_manual_requirement_counts_as_verified_by_its_named_evidence():
    """It is a route to evidence, not a gap: no test can kill a container."""
    assert entry(matrix(), "NFR-01").status == VERIFIED_BY_HAND


def test_a_manual_requirement_with_no_named_evidence_is_missing():
    """A manual tag with nothing named behind it is a promise, not evidence."""
    nameless = SPECIFICATION.replace(
        "**[manual]** Chaos test, killing the container", "**[manual]**"
    )
    assert entry(matrix(specification=nameless), "NFR-01").status == MISSING


def test_the_gate_blocks_a_due_requirement_that_nothing_verifies():
    built = matrix()
    assert [item.requirement.id for item in built.blocking()] == ["FR-01"]


def test_the_gate_ignores_a_requirement_whose_milestone_has_not_come():
    """Before M4 there is no API, so its requirements cannot have evidence.

    A build that fails for that reason cannot be fixed, and a gate that is
    always red stops being read.
    """
    built = matrix(coverage=NOT_DUE_YET)
    assert entry(built, "FR-01").status == MISSING
    assert not built.blocking()


def test_requirements_are_ordered_as_the_documents_order_them():
    """FR-02 sorts before FR-10, which a plain string sort gets wrong."""
    numbered = REQUIREMENTS.replace(
        "| FR-01 | Must", "| FR-10 | Must | **Ten.** | Ten. |\n| FR-01 | Must"
    )
    built = matrix(
        requirements=numbered,
        specification=SPECIFICATION.replace(
            "| FR-01 |", "| FR-10 | Ten. | **[automated]** test |\n| FR-01 |"
        ),
        coverage=COVERAGE.replace("[FR-01]", "[FR-01, FR-10]"),
    )
    assert [item.requirement.id for item in built.entries] == ["FR-01", "FR-10", "NFR-01"]


# --------------------------------------------------------------------------
# The artefacts
# --------------------------------------------------------------------------


def test_the_markdown_names_every_requirement_its_status_and_its_evidence():
    rendered = matrix_tool.render_markdown(matrix(markers={"tests/t.py::test_a": ("FR-01",)}))
    assert "| FR-01 | Must | automated | M3 | covered | `tests/t.py::test_a` |" in rendered
    assert "by hand: Chaos test, killing the container" in rendered


def test_a_missing_automated_requirement_shows_what_the_specification_owes():
    """The promised test is the ticket for whoever closes the gap.

    It is also not evidence yet, so it must not read like the named manual
    evidence of a requirement that no test can ever cover.
    """
    rendered = matrix_tool.render_markdown(matrix())
    assert "| FR-01 | Must | automated | M3 | **missing** | expected: API test |" in rendered


def test_the_markdown_lists_the_blocking_gaps_before_the_table():
    """The job summary has to answer "why is this red?" without scrolling."""
    rendered = matrix_tool.render_markdown(matrix())
    assert rendered.index("Verified by nothing, and already due") < rendered.index("## The matrix")
    assert "- FR-01, due M3" in rendered


def test_a_gap_nobody_has_to_fix_yet_does_not_get_a_blocking_section():
    """FR-01 owes a test from M4 on, so the matrix reports it without alarm."""
    rendered = matrix_tool.render_markdown(matrix(coverage=NOT_DUE_YET))
    assert "## Verified by nothing, and already due" not in rendered
    assert "verified by nothing: 1, of which 0 are already due" in rendered


def test_the_json_repeats_the_gate_the_counts_and_every_requirement():
    payload = matrix_tool.as_json(matrix(markers={"tests/t.py::test_a": ("FR-01",)}))
    assert payload["gate"] == {
        "milestone": "M3",
        "enforced_milestones": ["M3"],
        "blocking": [],
    }
    assert payload["counts"] == {COVERED: 1, VERIFIED_BY_HAND: 1, MISSING: 0}
    assert [item["id"] for item in payload["requirements"]] == ["FR-01", "NFR-01"]


def test_two_runs_on_the_same_commit_write_the_same_bytes(tmp_path):
    """No timestamp, so a diff in the artefact is a diff in the traceability."""
    built = matrix()
    first, second = tmp_path / "first.md", tmp_path / "second.md"
    first_json, second_json = tmp_path / "first.json", tmp_path / "second.json"
    matrix_tool.write_matrix(built, first, first_json)
    matrix_tool.write_matrix(built, second, second_json)
    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first_json.read_text()) == json.loads(second_json.read_text())


# --------------------------------------------------------------------------
# The repository's own documents
# --------------------------------------------------------------------------


@pytest.mark.req("NFR-07")
def test_the_repositorys_documents_and_coverage_set_agree():
    """The matrix CI publishes must be buildable from this commit alone.

    Everything the generator refuses is refused here too: an ID in one document
    and not the other, a requirement booked to no milestone, a stray booking.
    Running it as a test means the mismatch is named in the test job as well as
    in the matrix job, wherever the author looks first.
    """
    built = build_matrix(
        requirements_text=matrix_tool.REQUIREMENTS_DOC.read_text(encoding="utf-8"),
        specification_text=matrix_tool.SPECIFICATION_DOC.read_text(encoding="utf-8"),
        coverage_text=matrix_tool.EXPECTED_COVERAGE_FILE.read_text(encoding="utf-8"),
        markers={},
    )
    assert len(built.entries) == len(matrix_tool.known_requirement_ids())
    assert built.milestone in built.enforced_milestones


@pytest.mark.req("NFR-07")
def test_an_id_no_requirement_defines_is_reported_as_unknown():
    """The check the test suite runs on every marker, against the real IDs."""
    assert matrix_tool.unknown_marker_ids(("NFR-07", "FR-99")) == ["FR-99"]
    assert matrix_tool.unknown_marker_ids(("NFR-07",)) == []
