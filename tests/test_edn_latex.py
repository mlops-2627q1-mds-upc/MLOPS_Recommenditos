"""The transfer of the EDN from reports/edn.md into the LaTeX notebook.

The PDF this feeds is graded, and reports/edn.md stays the only place an entry
is written, so the transfer is held to two promises: nothing in an entry is
dropped or altered on the way, and a second run on the same notebook changes
nothing. Each test below names one way a converter could break the first
promise quietly, mostly by turning Markdown into LaTeX that compiles and says
something else, and the transfer tests hold it to the second.

Nothing here runs LaTeX: the report job of CI builds edn.tex from the committed
files, which is the check that the output compiles. The tests read small
notebooks written for the rule they show, and the one test that reads the
repository's own notebook checks that the committed LaTeX is what it generates.
"""

import textwrap

import pytest
from tools.edn_latex import (
    DO_NOT_EDIT,
    EDN_MD,
    ENTRY_LIST_NAME,
    GENERATED_MARKER,
    LATEX_EDN_DIR,
    REPO_URL,
    Context,
    EdnError,
    check,
    main,
    parse_notebook,
    render_entry,
    render_inline,
    render_markdown,
    transfer,
)

CONTEXT = Context(known_ids=frozenset({"EDN-01", "EDN-14"}), entry_id="EDN-01")


def notebook(*entries: str, after: str = "") -> str:
    """A notebook with the given entries, laid out as reports/edn.md is."""
    body = "\n".join(textwrap.dedent(entry).strip() + "\n" for entry in entries)
    return f"# Engineering Decision Notebook\n\nPreamble.\n\n## Entries\n\n{body}{after}"


def entry_md(number: int = 1, **overrides: str) -> str:
    """One complete entry; a field given as None is left out."""
    fields = {
        "Date": "2026-09-22",
        "Milestone": "M1: Project Inception",
        "Activity / Topic": "Dataset Selection",
        "Participants": "All team members",
        "Decision": "Use the dataset.",
        "Alternatives considered": (
            "\n  - **Option A (chosen): the dataset.**\n    Pros: large.\n    Cons: skewed."
        ),
        "Rationale": "It is the only one large enough.",
        "AI involvement": "Information seeking",
        "Response to AI": "Accepted",
        "Assessment of the AI contribution": "AI profiled the file.",
        "AI interaction evidence": "A session on 2026-09-22.",
        "Other evidence": "[PR #13](https://github.com/x/y/pull/13).",
        "In LaTeX": "no",
    }
    fields.update(overrides)
    lines = [f"### EDN-{number:02d}: Entry number {number}", ""]
    for label, value in fields.items():
        if value is None:
            continue
        separator = "" if value.startswith("\n") else " "
        lines.append(f"- **{label}:**{separator}{value}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Reading the notebook
# --------------------------------------------------------------------------


def test_an_entry_is_read_into_its_fields_with_continuations_dedented():
    (entry,) = parse_notebook(notebook(entry_md()))
    assert (entry.id, entry.number, entry.title) == ("EDN-01", 1, "Entry number 1")
    assert entry.fields["decision"] == ["Use the dataset."]
    assert entry.fields["alternatives"] == [
        "- **Option A (chosen): the dataset.**",
        "  Pros: large.",
        "  Cons: skewed.",
    ]


def test_the_template_after_the_entries_is_not_an_entry():
    """The template's heading has the shape of an entry and must not become one."""
    text = notebook(entry_md(), after="\n## Template\n\n### EDN-NN: Short name\n- **Date:** x\n")
    assert [entry.id for entry in parse_notebook(text)] == ["EDN-01"]


def test_a_missing_required_field_names_the_entry_and_the_field():
    """edn.tex would refuse it too, but only with a TeX error about a key."""
    with pytest.raises(EdnError, match=r"EDN-01 .*missing the required field\(s\) rationale"):
        parse_notebook(notebook(entry_md(Rationale=None)))


def test_a_required_field_that_says_it_does_not_apply_is_still_missing():
    with pytest.raises(EdnError, match="decision"):
        parse_notebook(notebook(entry_md(Decision="-")))


def test_a_line_that_belongs_to_no_field_is_refused_rather_than_dropped():
    """A line at the margin after the fields would vanish from the PDF otherwise."""
    text = notebook(entry_md() + "\nA stray sentence at the margin.\n")
    with pytest.raises(EdnError, match="belongs to no field"):
        parse_notebook(text)


def test_a_field_given_twice_is_refused():
    text = notebook(entry_md() + "- **Decision:** Something else.\n")
    with pytest.raises(EdnError, match="appears twice"):
        parse_notebook(text)


def test_a_section_heading_between_entries_is_refused_rather_than_ending_the_section():
    """It would end the section there and drop every entry below it."""
    text = notebook(entry_md(1), "## Notes\n", entry_md(2))
    with pytest.raises(EdnError, match=r"line \d+: the heading '## Notes'.*EDN-02"):
        parse_notebook(text)


def test_the_template_in_a_code_block_after_the_entries_is_not_an_entry():
    """The real template is a fenced block whose example heading has a number."""
    template = "\n## Template\n\n```markdown\n### EDN-99: An example\n```\n"
    assert [entry.id for entry in parse_notebook(notebook(entry_md(), after=template))] == [
        "EDN-01"
    ]


def test_a_fenced_code_block_in_an_entry_is_refused():
    """The renderer has no verbatim block, so it would typeset the code as prose."""
    fenced = entry_md(Rationale="Because:\n\n  ```python\n  x = 1\n  ```")
    with pytest.raises(EdnError, match=r"EDN-01, reports/edn.md line \d+: a fenced code block"):
        parse_notebook(notebook(fenced))


def test_two_entries_with_one_id_are_refused():
    """The report cites entries by ID, so an ID must name one decision."""
    with pytest.raises(EdnError, match="EDN-01 has two entries"):
        parse_notebook(notebook(entry_md(1), entry_md(1)))


# --------------------------------------------------------------------------
# Rendering prose
# --------------------------------------------------------------------------


def test_the_characters_latex_reserves_are_escaped_in_prose():
    rendered = render_inline(r"50 % of #3 & $x_y {a} ~ ^ \ ", CONTEXT)
    assert rendered == (
        r"50\,\% of \#3 \& \$x\_y \{a\} \textasciitilde{} \textasciicircum{} \textbackslash{} "
    )


def test_a_code_span_keeps_its_text_and_may_break_after_a_separator():
    """Paths run into the margin unless the line may break inside them."""
    assert render_inline("see `reports/edn.md`", CONTEXT) == (
        r"see \texttt{reports/\allowbreak{}edn.\allowbreak{}md}"
    )
    assert render_inline("`it's 5% \\d`", CONTEXT) == (
        r"\texttt{it\textquotesingle{}s 5\% \textbackslash{}d}"
    )


def test_markdown_inside_a_code_span_is_left_alone():
    assert render_inline("`**not bold** [no](link)`", CONTEXT) == (
        r"\texttt{**not bold** [no](link)}"
    )


def test_straight_quotes_open_and_close_even_around_a_code_span():
    """A quote right after a code span closes; a naive scan would open it again."""
    assert render_inline('NFR-01\'s word is "every", run "`dvc repro`"', CONTEXT) == (
        r"NFR-01's word is ``every'', run ``\texttt{dvc repro}''"
    )


def test_a_spaced_hyphen_is_printed_as_a_dash():
    assert render_inline("one - two", CONTEXT) == "one -- two"
    assert render_inline("well-known", CONTEXT) == "well-known"


def test_bold_italics_and_a_code_span_inside_bold():
    assert render_inline("**Option A: `model`** and *is*", CONTEXT) == (
        r"\textbf{Option A: \texttt{model}} and \emph{is}"
    )


def test_a_lone_asterisk_is_text():
    assert render_inline("2 * 3", CONTEXT) == "2 * 3"


def test_a_relative_link_becomes_its_url_on_github():
    """The PDF is read outside the checkout, where a relative path leads nowhere."""
    assert render_inline("[spec](../docs/docs/specification.md#nfr-11)", CONTEXT) == (
        rf"\href{{{REPO_URL}/blob/main/docs/docs/specification.md\#nfr-11}}{{spec}}"
    )
    assert render_inline("[`split_gate.py`](analysis/split_gate.py)", CONTEXT) == (
        rf"\href{{{REPO_URL}/blob/main/reports/analysis/split_gate.py}}"
        r"{\texttt{split\_\allowbreak{}gate.\allowbreak{}py}}"
    )


def test_an_absolute_link_and_an_autolink_keep_their_url():
    assert render_inline("[PR #19](https://github.com/x/y/pull/19)", CONTEXT) == (
        r"\href{https://github.com/x/y/pull/19}{PR \#19}"
    )
    assert render_inline("<https://dagshub.com/a/b>", CONTEXT) == (
        r"\href{https://dagshub.com/a/b}{\texttt{https:/\allowbreak{}/\allowbreak{}"
        r"dagshub.\allowbreak{}com/\allowbreak{}a/\allowbreak{}b}}"
    )


def test_a_link_to_an_entry_stays_inside_the_pdf():
    assert render_inline("[EDN-14](#edn-14-nfr-11s-drift-control)", CONTEXT) == (
        r"\hyperref[EDN-14]{EDN-14}"
    )


def test_a_link_to_an_entry_the_pdf_does_not_hold_is_plain_text():
    """A dead internal link would be a broken reference; the label still names it."""
    assert render_inline("[EDN-69](#edn-69-energy)", CONTEXT) == "EDN-69"


def test_a_link_out_of_the_repository_is_refused():
    with pytest.raises(EdnError, match="outside the repository"):
        render_inline("[x](../../elsewhere.md)", CONTEXT)


def test_a_character_pdflatex_cannot_typeset_is_refused_with_the_entry_named():
    """The alternative is a TeX error naming a line of a generated file."""
    assert render_inline("§3 ± 1 ≤ 2, Häußler", CONTEXT) == (
        r"\S{}3 \ensuremath{\pm} 1 \ensuremath{\leq} 2, Häußler"
    )
    with pytest.raises(EdnError, match=r"EDN-01: pdfLaTeX cannot typeset '≠'"):
        render_inline("a ≠ b", CONTEXT)


# --------------------------------------------------------------------------
# Rendering blocks and entries
# --------------------------------------------------------------------------


def test_a_nested_list_and_a_table_keep_their_structure():
    lines = textwrap.dedent(
        """\
        Measured:

        | Window | Flagged |
        |---|---|
        | 100 | `5.4` |
        | **1,000** | 0.7 |

        - outer
          - inner one
          - inner two
        - second outer
        """
    ).splitlines()
    rendered = render_markdown(lines, CONTEXT)
    assert rendered.count(r"\begin{tabularx}") == 1
    assert r"100 & \texttt{5.\allowbreak{}4} \\" in rendered
    assert r"\textbf{1,000} & 0.7 \\" in rendered
    assert rendered.count(r"\begin{itemize}") == 2
    assert rendered.count(r"\item ") == 4
    # The inner list sits inside the first outer item, before the second one.
    assert (
        rendered.index("inner two")
        < rendered.index(r"\end{itemize}")
        < rendered.index("second outer")
    )


def test_a_pipe_inside_a_code_span_does_not_split_a_table_cell():
    lines = ["| a | b |", "|---|---|", "| `x | y` | z |"]
    assert r"\texttt{x | y} & z \\" in render_markdown(lines, CONTEXT)


def test_pros_and_cons_start_their_own_line():
    rendered = render_markdown(["**Option A**", "Pros: fast.", "Cons: none."], CONTEXT)
    assert rendered == "\\textbf{Option A}\n\\newline Pros: fast.\n\\newline Cons: none."


def _render(text: str) -> str:
    entries = parse_notebook(text)
    return render_entry(entries[0], frozenset(entry.id for entry in entries))


def test_the_alternatives_go_into_the_rationale_ahead_of_it():
    """The course's fields have no slot for alternatives (reports/latex/README.md)."""
    rendered = _render(notebook(entry_md()))
    rationale = rendered[rendered.index("rationale = {") :]
    assert rationale.index("Alternatives considered") < rationale.index("Option A (chosen)")
    assert rationale.index("Option A (chosen)") < rationale.index("the only one large enough")
    assert "In LaTeX" not in rendered


def test_an_amendment_note_is_printed_before_the_fields():
    """A reader of the PDF should know an entry was revised before reading it."""
    amended = entry_md().replace(
        "\n\n", "\n\n> **Amended by EDN-14** (2026-09-29): see there.\n\n", 1
    )
    rendered = _render(notebook(amended))
    assert rendered.index("note = {") < rendered.index("date = {")
    assert r"\textbf{Amended by EDN-14} (2026-09-29): see there." in rendered


def test_a_field_the_template_does_not_define_is_kept_under_the_field_above():
    """EDN-59 has one; dropping it would drop part of the decision."""
    text = notebook(entry_md()).replace(
        "- **Alternatives considered:**",
        "- **Scope, after review:** the empty list.\n- **Alternatives considered:**",
    )
    rendered = _render(text)
    decision = rendered[rendered.index("decision = {") : rendered.index("rationale = {")]
    assert r"\textbf{Scope, after review:} the empty list." in decision


def test_an_optional_field_that_does_not_apply_is_left_out():
    rendered = _render(notebook(entry_md(**{"Response to AI": "-"})))
    assert "response = " not in rendered
    assert "assessment = {AI profiled the file.}" in rendered


def test_an_error_deep_in_a_field_names_the_entry():
    """A broken table row knows nothing of its entry; the message still names it."""
    broken = entry_md(Rationale="Measured:\n\n  | a | b |\n  |---|---|\n  | 1 |")
    with pytest.raises(EdnError, match=r"^EDN-01: a table row has 1 cells, its header 2$"):
        _render(notebook(broken))


def test_a_generated_file_says_where_its_source_is():
    rendered = _render(notebook(entry_md()))
    assert rendered.splitlines()[:3] == [GENERATED_MARKER, DO_NOT_EDIT, r"\ednentry{"]


# --------------------------------------------------------------------------
# The transfer
# --------------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path):
    edn_md = tmp_path / "edn.md"
    latex_dir = tmp_path / "edn"
    latex_dir.mkdir()
    (latex_dir / "hand-written.tex").write_text("% Written by hand, not generated.\n")
    return edn_md, latex_dir


def test_a_transfer_writes_one_file_per_entry_and_the_list_of_them(workspace):
    edn_md, latex_dir = workspace
    edn_md.write_text(notebook(entry_md(1), entry_md(2)), encoding="utf-8")

    result = transfer(edn_md, latex_dir)

    assert result.transferred == ["EDN-01", "EDN-02"]
    assert sorted(path.name for path in latex_dir.glob("*.tex")) == [
        "01-entry-number-1.tex",
        "02-entry-number-2.tex",
        ENTRY_LIST_NAME,
        "hand-written.tex",
    ]
    assert (latex_dir / ENTRY_LIST_NAME).read_text().splitlines()[-2:] == [
        r"\input{edn/01-entry-number-1}",
        r"\input{edn/02-entry-number-2}",
    ]


def test_in_latex_is_set_and_added_and_nothing_else_in_the_notebook_changes(workspace):
    edn_md, latex_dir = workspace
    original = notebook(entry_md(1), entry_md(2, **{"In LaTeX": None}))
    edn_md.write_text(original, encoding="utf-8")

    transfer(edn_md, latex_dir)

    updated = edn_md.read_text(encoding="utf-8")
    entries = parse_notebook(updated)
    assert [entry.fields["in_latex"] for entry in entries] == [["yes"], ["yes"]]

    def without_the_field(text: str) -> list[str]:
        return [line for line in text.splitlines() if not line.startswith("- **In LaTeX:**")]

    assert without_the_field(updated) == without_the_field(original)


def test_a_second_transfer_changes_nothing(workspace):
    """The final delivery reruns the transfer the first one made."""
    edn_md, latex_dir = workspace
    edn_md.write_text(notebook(entry_md(1), entry_md(2, **{"In LaTeX": None})), "utf-8")
    transfer(edn_md, latex_dir)
    before = {path.name: path.read_text() for path in latex_dir.iterdir()}
    notebook_before = edn_md.read_text()

    result = transfer(edn_md, latex_dir)

    assert (result.written, result.removed, result.notebook_changed) == ([], [], False)
    assert {path.name: path.read_text() for path in latex_dir.iterdir()} == before
    assert edn_md.read_text() == notebook_before


def test_an_edit_rewrites_only_the_entry_it_touches(workspace):
    edn_md, latex_dir = workspace
    edn_md.write_text(notebook(entry_md(1), entry_md(2)), "utf-8")
    transfer(edn_md, latex_dir)
    edn_md.write_text(edn_md.read_text().replace("Use the dataset.", "Use it.", 1), "utf-8")

    result = transfer(edn_md, latex_dir)

    assert [path.name for path in result.written] == ["01-entry-number-1.tex"]


def test_a_stale_generated_file_goes_and_a_hand_written_one_stays(workspace):
    """A changed title changes the slug, and the old file must not be built twice."""
    edn_md, latex_dir = workspace
    edn_md.write_text(notebook(entry_md(1)), "utf-8")
    transfer(edn_md, latex_dir)
    edn_md.write_text(edn_md.read_text().replace("Entry number 1", "Renamed entry"), "utf-8")

    result = transfer(edn_md, latex_dir)

    assert [path.name for path in result.removed] == ["01-entry-number-1.tex"]
    assert (latex_dir / "01-renamed-entry.tex").exists()
    assert (latex_dir / "hand-written.tex").exists()


def test_the_check_changes_nothing_and_names_what_a_transfer_would(workspace):
    """CI's report job runs it, so an edit without `make edn` cannot be merged."""
    edn_md, latex_dir = workspace
    edn_md.write_text(notebook(entry_md(1)), "utf-8")
    transfer(edn_md, latex_dir)
    assert check(edn_md, latex_dir) == []

    edn_md.write_text(edn_md.read_text().replace("Use the dataset.", "Use it."), "utf-8")
    edn_md.write_text(edn_md.read_text() + "\n" + entry_md(2), "utf-8")
    before = {path.name: path.read_text() for path in latex_dir.iterdir()}
    notebook_before = edn_md.read_text()

    assert check(edn_md, latex_dir) == [
        "01-entry-number-1.tex would be written",
        "02-entry-number-2.tex would be written",
        f"{ENTRY_LIST_NAME} would be written",
        "edn.md would get In LaTeX: yes on an entry",
    ]
    assert {path.name: path.read_text() for path in latex_dir.iterdir()} == before
    assert edn_md.read_text() == notebook_before


def test_the_check_exits_non_zero_only_when_out_of_sync(workspace, capsys):
    edn_md, latex_dir = workspace
    edn_md.write_text(notebook(entry_md(1)), "utf-8")
    arguments = ["--edn-md", str(edn_md), "--latex-dir", str(latex_dir)]

    assert main([*arguments, "--check"]) == 1
    assert "run `make edn`" in capsys.readouterr().out
    assert main(arguments) == 0
    assert main([*arguments, "--check"]) == 0


def test_a_notebook_that_cannot_be_transferred_exits_non_zero_and_says_why(workspace, capsys):
    edn_md, latex_dir = workspace
    edn_md.write_text(notebook(entry_md(Rationale=None)), "utf-8")

    assert main(["--edn-md", str(edn_md), "--latex-dir", str(latex_dir)]) == 1
    assert "EDN-01" in capsys.readouterr().err
    assert list(latex_dir.glob("0*.tex")) == []


def test_a_reserved_id_is_skipped_until_its_entry_is_written(workspace):
    """IDs are reserved on main as a bare heading before the branch is opened."""
    edn_md, latex_dir = workspace
    edn_md.write_text(notebook(entry_md(1), "### EDN-02: Reserved for issue #99\n"), "utf-8")

    result = transfer(edn_md, latex_dir)

    assert (result.transferred, result.skipped) == (["EDN-01"], ["EDN-02"])
    assert "Reserved for issue #99\n" in edn_md.read_text()
    assert edn_md.read_text().count("In LaTeX") == 1


# --------------------------------------------------------------------------
# The repository's own notebook
# --------------------------------------------------------------------------


def test_the_committed_latex_edn_is_what_the_notebook_generates():
    """`In LaTeX: yes` has to mean that the entry is in the PDF as the notebook states it.

    The same check runs in CI's report job, which also runs when only
    reports/edn.md changed; this test names the problem in the test job too.
    """
    entries = parse_notebook(EDN_MD.read_text(encoding="utf-8"))
    numbers = [entry.number for entry in entries]

    assert numbers == sorted(numbers), "entries are kept oldest first"
    assert check(EDN_MD, LATEX_EDN_DIR) == [], "run `make edn` and commit what it writes"
