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
import textwrap

import pytest
import tools.requirement_matrix as matrix_tool
from tools.requirement_matrix import (
    MISSING,
    NAMED_BY_A_TEST,
    VERIFIED_BY_HAND,
    VERIFIED_BY_TEST,
    TraceabilityError,
    build_matrix,
    collect_req_markers,
)
import typer

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
    assert built.enforced_milestones == ("M1", "M2", "M3", "M4")
    assert entry(built, "NFR-01").enforced


def test_a_milestone_name_nothing_knows_is_refused():
    """Free-text names let a typo become a bucket that is never enforced."""
    with pytest.raises(TraceabilityError, match="the milestone M3b, which is not one of"):
        matrix(coverage=COVERAGE.replace("- milestone: M3\n", "- milestone: M3b\n"))


def test_a_milestone_listed_out_of_order_is_refused():
    """The exploit this closes: append `- milestone: M2` after M3 and book a
    requirement to it, and the old positional gate exempted it while `current`
    was M3, with no line a reviewer could read as a loosening."""
    appended = COVERAGE + "  - milestone: M2\n    requirements: []\n"
    with pytest.raises(TraceabilityError, match="lists M2 after M4"):
        matrix(coverage=appended)


def test_the_gate_reads_the_order_of_the_milestones_and_not_the_file_order():
    """A requirement booked to a milestone before `current` is enforced wherever
    its block sits, so moving the block cannot move it out of the gate."""
    reordered = """\
current: M3
milestones:
  - milestone: M2
    requirements: [FR-01]
  - milestone: M3
    requirements: []
  - milestone: M4
    requirements: [NFR-01]
"""
    built = matrix(coverage=reordered)
    assert entry(built, "FR-01").enforced
    assert [item.requirement.id for item in built.blocking()] == ["FR-01"]


def test_a_milestone_block_that_names_no_milestone_is_refused():
    broken = "current: M3\nmilestones:\n  - requirements: [FR-01, NFR-01]\n"
    with pytest.raises(TraceabilityError, match="needs a `milestone` name"):
        matrix(coverage=broken)


def test_a_coverage_set_without_current_or_milestones_is_refused():
    with pytest.raises(TraceabilityError, match="needs a `current` milestone"):
        matrix(coverage="milestones: []\n")


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


def test_a_test_carrying_the_id_verifies_an_automated_requirement():
    built = matrix(markers={"tests/test_api.py::test_it": ("FR-01",)})
    covered = entry(built, "FR-01")
    assert covered.status == VERIFIED_BY_TEST
    assert covered.tests == ("tests/test_api.py::test_it",)
    assert covered.has_the_promised_evidence
    assert not built.blocking()


def test_a_test_on_a_manual_requirement_names_it_rather_than_verifying_it():
    """The distinction the matrix exists to make.

    NFR-06 was reported as covered because one test named it, and that test's
    whole body asserted that two files exist: it measured no determinism and
    never touched MLflow, which is what NFR-06 is about. The specification says a
    person is the evidence for a **[manual]** entry, so a marker there records
    that a test touches the requirement and is not the evidence promised.
    """
    built = matrix(markers={"tests/test_api.py::test_it": ("NFR-01",)})
    named = entry(built, "NFR-01")
    assert named.status == NAMED_BY_A_TEST
    assert named.tests == ("tests/test_api.py::test_it",)
    # Still satisfied, because the drill its cell names is the route and the
    # marker neither created nor removed it.
    assert named.has_the_promised_evidence


def test_a_marker_on_an_empty_manual_cell_does_not_satisfy_the_gate():
    """Otherwise the cheapest fake there is: tag the entry **[manual]**, leave the
    cell empty, and put the marker on any test at all."""
    nameless = SPECIFICATION.replace(
        "**[manual]** Chaos test, killing the container", "**[manual]**"
    )
    built = matrix(
        specification=nameless,
        coverage=COVERAGE.replace("current: M3", "current: M4"),
        markers={"tests/test_api.py::test_it": ("NFR-01",)},
    )
    named = entry(built, "NFR-01")
    assert named.status == NAMED_BY_A_TEST
    assert not named.has_the_promised_evidence
    assert [item.requirement.id for item in built.blocking()] == ["FR-01", "NFR-01"]


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
    assert (
        "| FR-01 | Must | automated | M3 | verified by a test | `tests/t.py::test_a` |" in rendered
    )
    assert "by hand: Chaos test, killing the container" in rendered


def test_the_evidence_column_holds_evidence_and_not_the_specifications_promise():
    """It used to render "expected: API test" for eleven of sixteen functional
    requirements, which is a category of test nobody has written presented in the
    column a reader takes for a record. The route and its due date are the
    "Verified by" and "Due" columns; the sentence stays in the specification."""
    rendered = matrix_tool.render_markdown(matrix())
    assert "| FR-01 | Must | automated | M3 | **verified by nothing** | nothing yet |" in rendered
    assert "expected:" not in rendered


def test_the_markdown_lists_the_blocking_gaps_before_the_table():
    """The job summary has to answer "why is this red?" without scrolling."""
    rendered = matrix_tool.render_markdown(matrix())
    assert rendered.index("Missing the promised evidence, and already due") < rendered.index(
        "## The matrix"
    )
    assert "- FR-01, due M3, verified by nothing" in rendered


def test_a_gap_nobody_has_to_fix_yet_does_not_get_a_blocking_section():
    """FR-01 owes a test from M4 on, so the matrix reports it without alarm."""
    rendered = matrix_tool.render_markdown(matrix(coverage=NOT_DUE_YET))
    assert "## Missing the promised evidence, and already due" not in rendered
    assert "verified by nothing: 1, of which 0 are already due" in rendered


def test_the_json_repeats_the_gate_the_counts_and_every_requirement():
    payload = matrix_tool.as_json(matrix(markers={"tests/t.py::test_a": ("FR-01",)}))
    assert payload["gate"] == {
        "milestone": "M3",
        "enforced_milestones": ["M1", "M2", "M3"],
        "blocking": [],
    }
    assert payload["counts"] == {
        VERIFIED_BY_TEST: 1,
        NAMED_BY_A_TEST: 0,
        VERIFIED_BY_HAND: 1,
        MISSING: 0,
    }
    assert [item["id"] for item in payload["requirements"]] == ["FR-01", "NFR-01"]


def test_the_json_says_per_requirement_whether_the_gate_is_satisfied():
    """So that whatever reads the artefact next does not have to re-derive which
    of the four statuses the gate accepts."""
    payload = matrix_tool.as_json(matrix(markers={"tests/t.py::test_a": ("NFR-01",)}))
    by_id = {item["id"]: item for item in payload["requirements"]}
    assert by_id["NFR-01"]["status"] == NAMED_BY_A_TEST
    assert by_id["NFR-01"]["has_the_promised_evidence"] is True
    assert by_id["FR-01"]["status"] == MISSING
    assert by_id["FR-01"]["has_the_promised_evidence"] is False


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


#: What the gate currently enforces: every requirement whose evidence is already
#: due, and the route by which the specification says it is verified. Both are
#: parameters of a CI gate, and both were loosenable by a change that reads like
#: tidying: move an ID from the `M3` list to the `M4` list, or rewrite a "Verified
#: by" cell from **[automated]** to **[manual]** and add a sentence of prose, and
#: the requirement left the blocking set with no test written and nothing else
#: changed. Pinning them here makes either move cost a visible test change, which
#: is where the argument for it belongs.
DUE_AT_M3 = {
    "NFR-01": "automated",
    "NFR-06": "manual",
    "NFR-07": "manual",
    "NFR-08": "automated",
}


@pytest.mark.req("NFR-07")
def test_the_gate_stands_where_this_commit_says_it_stands():
    coverage = matrix_tool.parse_expected_coverage(
        matrix_tool.EXPECTED_COVERAGE_FILE.read_text(encoding="utf-8"),
        set(matrix_tool.known_requirement_ids()),
    )
    assert coverage.current == "M3"

    specified = matrix_tool.parse_specification(
        matrix_tool.SPECIFICATION_DOC.read_text(encoding="utf-8")
    )
    due = {
        req_id: specified[req_id][0]
        for req_id, milestone in coverage.due.items()
        if coverage.enforces(req_id)
    }
    assert dict(sorted(due.items())) == DUE_AT_M3


# --------------------------------------------------------------------------
# The guards on the documents' own shape
# --------------------------------------------------------------------------


def test_a_requirement_with_two_rows_is_refused():
    """Two rows mean two priorities, and the matrix would silently keep one."""
    twice = REQUIREMENTS + "\n| FR-01 | Could | **Again.** | Again. |\n"
    with pytest.raises(TraceabilityError, match="FR-01 has two rows in the requirements"):
        matrix(requirements=twice)


def test_a_specification_entry_with_two_rows_is_refused():
    twice = SPECIFICATION + "| FR-01 | Again. | **[manual]** Again |\n"
    with pytest.raises(TraceabilityError, match="FR-01 has two rows in the specification"):
        matrix(specification=twice)


def test_a_requirement_row_without_a_priority_is_refused():
    """The priority is read by position, so an empty cell would become an empty
    column in the matrix rather than a finding."""
    nameless = REQUIREMENTS.replace("| FR-01 | Must |", "| FR-01 |  |")
    with pytest.raises(TraceabilityError, match="FR-01 in the requirements names no priority"):
        matrix(requirements=nameless)


def test_a_specification_row_of_another_shape_is_refused():
    """The verification tag is read by position too, so a four-cell row would be
    misread as having no tag at all."""
    widened = SPECIFICATION.replace(
        "| FR-01 | `POST /predict` answers. | **[automated]** API test |",
        "| FR-01 | `POST /predict` answers. | extra | **[automated]** API test |",
    )
    with pytest.raises(TraceabilityError, match="FR-01 in the specification has 4 cells"):
        matrix(specification=widened)


def test_a_document_with_no_requirement_row_at_all_is_refused():
    """A parser that silently returned nothing would make the matrix empty and
    green, which is the worst possible way for this to fail."""
    with pytest.raises(TraceabilityError, match="requirements contain no FR-xx"):
        matrix(requirements="Requirements\n============\n")
    with pytest.raises(TraceabilityError, match="specification contains no FR-xx"):
        matrix(specification="Specification\n=============\n")


def test_an_id_with_three_digits_is_a_requirement_like_any_other():
    """A pattern that insisted on exactly two digits made FR-100 invisible to
    every check here: it was not parsed, so it was not booked, not specified and
    not covered, and nothing complained."""
    requirements = REQUIREMENTS.replace(
        "| FR-01 | Must", "| FR-100 | Must | **Hundred.** | Hundred. |\n| FR-01 | Must"
    )
    specification = SPECIFICATION.replace(
        "| FR-01 |", "| FR-100 | Hundred. | **[automated]** test |\n| FR-01 |"
    )
    with pytest.raises(TraceabilityError, match="FR-100 is not booked"):
        matrix(requirements=requirements, specification=specification)

    built = matrix(
        requirements=requirements,
        specification=specification,
        coverage=COVERAGE.replace("[FR-01]", "[FR-01, FR-100]"),
        markers={"tests/t.py::test_a": ("FR-100",)},
    )
    assert entry(built, "FR-100").status == VERIFIED_BY_TEST
    # FR-100 sorts after FR-01, which a plain string sort gets wrong.
    assert [item.requirement.id for item in built.entries] == ["FR-01", "FR-100", "NFR-01"]


# --------------------------------------------------------------------------
# Reading the markers out of a real pytest collection
# --------------------------------------------------------------------------


def _collected(tmp_path, module: str, source: str):
    """A one-file test directory for `collect_req_markers` to collect.

    The basename is per test on purpose: pytest imports a test file by its module
    name, so two temporary directories both holding `test_sample.py` are a name
    clash and a collection error rather than two independent collections.
    """
    directory = tmp_path / "collected"
    directory.mkdir()
    (directory / f"{module}.py").write_text(textwrap.dedent(source), encoding="utf-8")
    return directory


MARKED_EVERY_WAY = """
import pytest

pytestmark = pytest.mark.req("NFR-07")


def test_on_the_module():
    pass


@pytest.mark.req("FR-01")
class TestGroup:
    def test_on_the_class(self):
        pass


@pytest.mark.req("FR-02", "FR-03")
@pytest.mark.parametrize("case", [1, 2, 3])
def test_parametrised(case):
    pass
"""


def test_the_collector_reads_a_marker_wherever_pytest_sees_one(tmp_path):
    """Collecting rather than parsing the source is what buys this.

    A marker on a module or a class counts like one written above a function, and
    a parametrised test collapses to the one row it is in the matrix.
    """
    markers = collect_req_markers(_collected(tmp_path, "test_marked_every_way", MARKED_EVERY_WAY))
    by_name = {test_id.rpartition("::")[2]: set(ids) for test_id, ids in markers.items()}
    assert by_name == {
        "test_on_the_module": {"NFR-07"},
        "test_on_the_class": {"FR-01", "NFR-07"},
        "test_parametrised": {"FR-02", "FR-03", "NFR-07"},
    }


EMPTY_MARKER_ON_A_SKIPPED_TEST = """
import pytest


@pytest.mark.req()
@pytest.mark.skip(reason="never runs")
def test_skipped():
    pass
"""


def test_a_marker_naming_nothing_is_refused_even_on_a_test_that_never_runs(tmp_path):
    """The hole this closes: `@pytest.mark.req()` on a skipped test was caught by
    neither net. The per-test check in conftest cannot run on a test that does not
    run, and the collector used to skip an entry with no IDs, so an empty marker
    read like a record of coverage while being caught by nothing.
    """
    with pytest.raises(TraceabilityError, match=r"names no requirement ID in .*test_skipped"):
        collect_req_markers(
            _collected(tmp_path, "test_empty_marker", EMPTY_MARKER_ON_A_SKIPPED_TEST)
        )


def test_a_suite_that_cannot_be_collected_is_a_finding_and_not_an_empty_matrix(tmp_path):
    """An empty matrix is green, so a broken collection must never look like one.
    The captured output is handed back, because that is where the reason is."""
    with pytest.raises(TraceabilityError, match="could not collect the test suite"):
        collect_req_markers(
            _collected(tmp_path, "test_broken_import", "import a_module_that_is_not_installed\n")
        )


def test_collecting_the_markers_leaves_the_test_runs_own_reports_alone(tmp_path):
    """Building the matrix used to rewrite `reports/junit.xml` as a `tests="0"`
    document, because the collect-only run inherited the project's `addopts`. A
    test run followed by a matrix build therefore destroyed the record of the run
    that had just passed, which is the only thing that record is for.
    """
    reports = matrix_tool.PROJ_ROOT / "reports"
    watched = (reports / "junit.xml", reports / "coverage.xml")
    before = {path: path.read_bytes() if path.exists() else None for path in watched}

    collect_req_markers(
        _collected(tmp_path, "test_nothing_at_all", "def test_nothing():\n    pass\n")
    )

    after = {path: path.read_bytes() if path.exists() else None for path in watched}
    assert after == before


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------


def _as_the_repository(tmp_path, monkeypatch, coverage=COVERAGE, markers=None):
    """Point the module's document constants at the fixture pair above.

    The CLI reads the real documents and runs a real collection, so a test of it
    would otherwise be a test of the repository's current state instead of a test
    of its exit paths.
    """
    for name, text in (
        ("REQUIREMENTS_DOC", REQUIREMENTS),
        ("SPECIFICATION_DOC", SPECIFICATION),
        ("EXPECTED_COVERAGE_FILE", coverage),
    ):
        path = tmp_path / name.lower()
        path.write_text(text, encoding="utf-8")
        monkeypatch.setattr(matrix_tool, name, path)
    monkeypatch.setattr(matrix_tool, "collect_req_markers", lambda: markers or {})
    return tmp_path / "matrix.md", tmp_path / "matrix.json"


def test_the_cli_writes_both_artefacts_and_returns_when_nothing_due_is_missing(
    tmp_path, monkeypatch
):
    markdown, payload = _as_the_repository(tmp_path, monkeypatch, coverage=NOT_DUE_YET)
    matrix_tool.main(markdown_path=markdown, json_path=payload)
    assert markdown.read_text(encoding="utf-8").startswith("Requirement-to-test matrix")
    assert json.loads(payload.read_text(encoding="utf-8"))["gate"]["blocking"] == []


def test_the_cli_writes_the_matrix_before_it_fails_on_a_due_gap(tmp_path, monkeypatch):
    """The CI job publishes the table from the file, so the file has to exist even
    when the build is red. Writing first and failing afterwards is what buys that,
    and nothing else in the suite held that order."""
    markdown, payload = _as_the_repository(tmp_path, monkeypatch)
    with pytest.raises(typer.Exit) as failure:
        matrix_tool.main(markdown_path=markdown, json_path=payload)
    assert failure.value.exit_code == 1
    assert json.loads(payload.read_text(encoding="utf-8"))["gate"]["blocking"] == ["FR-01"]
    assert "- FR-01, due M3, verified by nothing" in markdown.read_text(encoding="utf-8")


def test_the_cli_exits_one_and_writes_nothing_when_the_sources_disagree(tmp_path, monkeypatch):
    """A half-written matrix from a mismatch would be published as if it were the
    table, so nothing is written until the sources agree."""
    unbooked = "current: M3\nmilestones:\n  - milestone: M3\n    requirements: [FR-01]\n"
    markdown, payload = _as_the_repository(tmp_path, monkeypatch, coverage=unbooked)
    with pytest.raises(typer.Exit) as failure:
        matrix_tool.main(markdown_path=markdown, json_path=payload)
    assert failure.value.exit_code == 1
    assert not markdown.exists()
    assert not payload.exists()
