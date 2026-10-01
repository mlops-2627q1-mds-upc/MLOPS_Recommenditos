"""The gate that runs Pynblint over every notebook, against the documented policy.

docs/docs/notebooks.md says why Pynblint cannot gate a build on its own. Each of
those gaps would turn a finding into a green build without anyone noticing, so
each has a test here.

None of these tests needs Pynblint itself, which lives in its own environment
(tools/pynblint-env). `main` takes the process runner as an argument, and
`FakeTools` stands in for git and Pynblint with answers recorded from Pynblint
0.1.6, so what is tested is what the gate does with an answer.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import subprocess
import sys

import pytest
from tools import notebook_lint
from tools.notebook_lint import (
    KEEP_OUTPUT_RULE,
    PROJECT_RULE,
    PolicyError,
    check_policy,
    clean_environment,
    follows_project_rule,
    parse_policy,
    pynblint_command,
)

POLICY = """\
Notebooks
=========

### Pynblint rules

<!-- The notebook lint gate reads this table. -->

| Rule | Decision | Reason |
|------|----------|--------|
| `empty-cells` | enforced | No leftover cells. |
| `missing-closing-MD-text` | enforced | The last cells say what the notebook found. |
| `cell-too-long` | enforced | Computation belongs in the package. |
| `non-executed-notebook` | excluded | nbstripout clears every execution count. |

Prose after the table is not part of it.
"""

#: What the fake registry reports as installed, slug to level: exactly the rules
#: POLICY classifies, plus nothing else.
INSTALLED = {
    "missing-closing-MD-text": "notebook",
    "non-executed-notebook": "notebook",
    "empty-cells": "cell",
    "cell-too-long": "cell",
}

NOTEBOOK = "notebooks/1.0-lh-x.ipynb"

#: 31 lines, one more than `cell-too-long` allows.
LONG_CELL = "rows = [\n" + "".join(f"    {number},\n" for number in range(29)) + "]"


def notebook_json(*sources: str) -> str:
    """A notebook whose first cell is a Markdown title and the rest are code."""
    cells = [{"cell_type": "markdown", "metadata": {}, "source": "# Title\n\nWhat it is for."}]
    cells += [
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s}
        for s in sources
    ]
    return json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5})


#: The notebook Pynblint 0.1.6 was run on to record FINDINGS below.
NOTEBOOK_WITH_FINDINGS = notebook_json("import re\n\nPATTERN = re.compile('x')", LONG_CELL, "")

#: Recorded from `python -m pynblint <file> --exclude [...] --output x.json` on
#: NOTEBOOK_WITH_FINDINGS: one notebook-level finding, which names no cells, and
#: two cell-level ones, which do. The lints are verbatim; the statistics, which
#: the gate does not read, are cut short.
FINDINGS = {
    "notebook_metadata": {"notebook_name": "1.0-lh-x.ipynb"},
    "notebook_stats": {"number_of_cells": 4, "number_of_MD_cells": 1, "number_of_code_cells": 3},
    "lints": [
        {
            "slug": "missing-closing-MD-text",
            "description": "The final notebook cells (i.e., the last 3 cells in the notebook) "
            "contain no Markdown text.",
            "recommendation": "Conclude your notebook by describing what you have accomplished "
            "in one or more concluding Markdown cells.",
        },
        {
            "slug": "empty-cells",
            "description": "Empty cells are present in the notebook.",
            "recommendation": "Keep your notebook clean by deleting unused cells.",
            "cells": [{"index": 3, "type": "CellType.CODE", "execution_count": None}],
        },
        {
            "slug": "cell-too-long",
            "description": "One or more code cells in this notebook are too long (i.e., they "
            "exceed the fixed threshold of 30 lines).",
            "recommendation": "Consider consolidating your code outside the notebook by moving "
            "utility functions to a structured and tested codebase.\n"
            "Use notebooks to display results, not to compute them.",
            "cells": [{"index": 2, "type": "CellType.CODE", "execution_count": None}],
        },
    ],
}

CLEAN = {**FINDINGS, "lints": []}


def completed(command, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)


@dataclass
class FakeTools:
    """Plays git and Pynblint for `main`, and records how it was called."""

    notebooks: tuple[str, ...] = (NOTEBOOK,)
    rules: dict = field(default_factory=lambda: INSTALLED)
    report: dict | None = field(default_factory=lambda: CLEAN)
    returncode: int = 0
    writes_report: bool = True
    git_missing: bool = False
    git_returncode: int = 0
    registry_returncode: int = 0
    registry_stdout: str | None = None
    calls: list = field(default_factory=list)

    def __call__(self, command, **options):
        self.calls.append((command, options))
        if command[0] == "git":
            if self.git_missing:
                raise FileNotFoundError(2, "No such file or directory", "git")
            listed = "".join(f"{path}\0" for path in self.notebooks)
            return completed(command, self.git_returncode, listed, "fatal: not a git repository")
        if command[1] == "-c":
            registry = json.dumps({"version": "0.1.6", "rules": self.rules})
            stdout = registry if self.registry_stdout is None else self.registry_stdout
            return completed(command, self.registry_returncode, stdout, "ImportError: boom")
        if self.writes_report:
            # Like Pynblint, report nothing for a rule the command excludes.
            excluded = set(json.loads(command[command.index("--exclude") + 1]))
            report = self.report
            if isinstance(report, dict) and "lints" in report:
                lints = [lint for lint in report["lints"] if lint["slug"] not in excluded]
                report = {**report, "lints": lints}
            output = Path(command[command.index("--output") + 1])
            output.write_text(json.dumps(report), encoding="utf-8")
        return completed(
            command,
            returncode=self.returncode,
            stderr="Traceback (most recent call last):\nValueError: boom",
        )

    def pynblint_calls(self):
        return [(command, options) for command, options in self.calls if command[0] != "git"]


def repository(tmp_path, notebooks=None) -> Path:
    """A checkout with the policy page and the given notebooks on disk."""
    files = {NOTEBOOK: notebook_json("x = 1")} if notebooks is None else notebooks
    files = {"docs/docs/notebooks.md": POLICY, **files}
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return tmp_path


def reports(root: Path) -> tuple[str, dict]:
    markdown = (root / "reports" / "notebook-lint.md").read_text(encoding="utf-8")
    payload = json.loads((root / "reports" / "notebook-lint.json").read_text(encoding="utf-8"))
    return markdown, payload


# --------------------------------------------------------------------------
# The policy table
# --------------------------------------------------------------------------


def test_the_policy_table_says_which_rules_are_enforced_and_why_the_others_are_not():
    policy = parse_policy(POLICY)
    # In the table's order, which is how the page groups them.
    assert policy.enforced == ["empty-cells", "missing-closing-MD-text", "cell-too-long"]
    assert policy.excluded == {"non-executed-notebook": "nbstripout clears every execution count."}


ROW = "| `empty-cells` | enforced | No leftover cells. |"


@pytest.mark.parametrize(
    ("table", "match"),
    [
        # A rule excluded without a reason is a rule nobody can review.
        (POLICY.replace("| nbstripout clears every execution count. |", "|  |"), "no reason"),
        # Two rows would give one rule two decisions, and the gate would keep one.
        (POLICY.replace(ROW, f"{ROW}\n| `empty-cells` | excluded | Again. |"), "twice"),
        # Anything but the two decisions would leave the gate guessing.
        (POLICY.replace("| enforced | No leftover", "| ignored | No leftover"), "ignored"),
        # A slug outside backticks is how a heading or a prose cell gets misread.
        (POLICY.replace("| `empty-cells` |", "| empty-cells |"), "backticks"),
        # No table at all must not read as "no rule is enforced".
        ("Notebooks\n=========\n\nNo table here.\n", "found no rule table"),
        # With two tables the gate would read one and the page would show both.
        (POLICY + "\n| Rule | Decision | Reason |\n|---|---|---|\n" + ROW + "\n", "2 rule tables"),
        # A cell lost or added shifts the decision into the reason, or the reverse.
        (POLICY.replace(ROW, "| `empty-cells` | enforced |"), "2 cells"),
        # A header with no rows is a table that decides nothing.
        ("| Rule | Decision | Reason |\n|---|---|---|\n\nText.\n", "no rows"),
        # A table that excludes every rule passes every notebook, and says it checked them.
        (POLICY.replace("| enforced |", "| excluded |"), "enforces no rule"),
    ],
    ids=[
        "missing reason",
        "duplicate rule",
        "unknown decision",
        "bare slug",
        "no table",
        "second table",
        "missing cell",
        "empty table",
        "nothing enforced",
    ],
)
def test_a_policy_table_the_gate_could_misread_is_refused(table, match):
    with pytest.raises(PolicyError, match=match):
        parse_policy(table)


@pytest.mark.req("NFR-07")
def test_the_repositorys_policy_table_enforces_rules_and_gives_every_exclusion_a_reason():
    """NFR-07 points at this table for the rules we do not enforce and why.

    The CI gate refuses to run on a table it cannot read, so this catches the
    same mistake in the test job, wherever the author looks first.
    """
    policy = parse_policy(
        (notebook_lint.PROJ_ROOT / notebook_lint.POLICY_DOC).read_text(encoding="utf-8")
    )
    assert policy.enforced
    assert policy.excluded


# --------------------------------------------------------------------------
# The policy against the installed Pynblint
# --------------------------------------------------------------------------


def test_a_policy_that_classifies_exactly_the_installed_rules_is_accepted():
    check_policy(INSTALLED, parse_policy(POLICY))


def test_a_rule_the_installed_pynblint_does_not_have_is_refused():
    """The likely typo: Pynblint spells this one slug with underscores.

    Passing a misspelt slug to `--exclude` would be silently ignored by
    Pynblint, and the rule meant to be excluded would run anyway; in the other
    direction, a misspelt enforced rule would never be checked by anything.
    """
    policy = parse_policy(
        POLICY.replace("| `cell-too-long` |", "| `long-multiline-python-comment` |")
    )
    installed = {**INSTALLED, "long_multiline_python_comment": "cell"}
    del installed["cell-too-long"]
    with pytest.raises(PolicyError) as refused:
        check_policy(installed, policy)
    assert "long-multiline-python-comment" in str(refused.value)
    assert "long_multiline_python_comment" in str(refused.value)


def test_a_rule_the_policy_does_not_classify_is_refused():
    """A new Pynblint brings new rules, and none of them may run undecided."""
    with pytest.raises(PolicyError, match="notebook-has-no-tests"):
        check_policy({**INSTALLED, "notebook-has-no-tests": "notebook"}, parse_policy(POLICY))


def test_an_enforced_repository_rule_is_refused_because_it_never_runs():
    """Pynblint runs its repository rules only on a directory, and the gate gives it
    one notebook file at a time, so an enforced repository rule would be reported
    as checked while nothing checks it."""
    policy = parse_policy(
        POLICY.replace(ROW, f"{ROW}\n| `duplicate-notebook-filename` | enforced | Unique. |")
    )
    with pytest.raises(PolicyError, match="duplicate-notebook-filename is a repository rule"):
        check_policy({**INSTALLED, "duplicate-notebook-filename": "path"}, policy)


# --------------------------------------------------------------------------
# How Pynblint is called
# --------------------------------------------------------------------------


def test_the_clean_environment_keeps_nothing_pynblint_would_read_as_a_setting():
    """Pynblint reads every setting from an environment variable of the same name,
    case-insensitively and with no prefix, so `INCLUDE=/usr/include` from a C
    toolchain crashes it, and a stray `exclude` would quietly switch rules off."""
    environment = clean_environment(
        {
            "PATH": "/usr/bin",
            "HOME": "/home/me",
            "INCLUDE": "/usr/include",
            "PLUGINS": "foo",
            "exclude": '["empty-cells"]',
            "PYTHONPATH": "/somewhere/else",
        }
    )
    assert environment == {"PATH": "/usr/bin", "HOME": "/home/me", "PYTHONUTF8": "1"}


def test_the_exclusions_reach_pynblint_as_a_json_array_and_it_runs_unattended(tmp_path):
    """Pynblint parses `--exclude` with `json.loads`: the comma-separated form its
    own help suggests crashes it. Without `--yes` an existing output file stops
    the run at a prompt, and the output format is chosen by the `.json` suffix."""
    output = tmp_path / "report.json"
    command = pynblint_command(
        Path("/repo/notebooks/1.0-lh-x.ipynb"), ["non-linear-execution", "empty-cells"], output
    )
    assert command[:4] == [sys.executable, "-m", "pynblint", "/repo/notebooks/1.0-lh-x.ipynb"]
    options = command[4:]
    assert json.loads(options[options.index("--exclude") + 1]) == [
        "empty-cells",
        "non-linear-execution",
    ]
    assert options[options.index("--output") + 1] == str(output)
    assert "--yes" in options
    assert "--quiet" in options


def test_pynblint_never_sees_the_callers_environment_or_working_directory(tmp_path):
    """Also a `.pynblint` file in the working directory configures it, so it runs
    in an empty temporary directory, and with the cleaned environment, every time."""
    root = repository(tmp_path)
    tools = FakeTools()
    notebook_lint.main(root, run=tools, environ={"PATH": "/usr/bin", "INCLUDE": "/usr/include"})

    calls = tools.pynblint_calls()
    assert len(calls) == 2  # the registry, then the one notebook
    for _, options in calls:
        assert options["env"] == {"PATH": "/usr/bin", "PYTHONUTF8": "1"}
        assert not Path(options["cwd"]).resolve().is_relative_to(root.resolve())


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


def test_findings_fail_the_gate_with_the_rule_the_cells_and_the_notebooks_path(tmp_path, capsys):
    """Pynblint itself exits 0 here, which is the reason this wrapper exists.

    The reports are written before the gate fails, because CI publishes them
    from the files, and a red job has to say why without anyone reading a log.
    """
    root = repository(tmp_path, {NOTEBOOK: NOTEBOOK_WITH_FINDINGS})
    exit_code = notebook_lint.main(root, run=FakeTools(report=FINDINGS), environ={})

    assert exit_code == 1
    markdown, payload = reports(root)
    assert payload["passed"] is False
    assert [(f["notebook"], f["rule"], f["cells"]) for f in payload["findings"]] == [
        (NOTEBOOK, "missing-closing-MD-text", []),
        (NOTEBOOK, "empty-cells", [{"index": 3, "first_line": ""}]),
        (NOTEBOOK, "cell-too-long", [{"index": 2, "first_line": "rows = ["}]),
    ]
    assert "| `notebooks/1.0-lh-x.ipynb` | `cell-too-long` | 2 |" in markdown

    printed = capsys.readouterr().out
    assert "cell 2: rows = [" in printed
    assert "cell 3: (empty)" in printed
    assert "Use notebooks to display results, not to compute them." in printed
    assert printed.rstrip().endswith("3 findings")


def test_a_clean_run_passes_and_says_what_it_checked(tmp_path, capsys):
    root = repository(tmp_path)
    assert notebook_lint.main(root, run=FakeTools(), environ={}) == 0

    summary = (
        "1 notebook checked with Pynblint 0.1.6: 3 rules and 2 project rules enforced, "
        "1 rule excluded (see docs/docs/notebooks.md), 0 findings"
    )
    assert summary in capsys.readouterr().out
    markdown, payload = reports(root)
    assert summary in markdown
    assert "nbstripout clears every execution count." in markdown
    assert payload == {
        "pynblint": "0.1.6",
        "notebooks": [NOTEBOOK],
        "enforced": ["empty-cells", "missing-closing-MD-text", "cell-too-long"],
        "project_rules": [PROJECT_RULE, KEEP_OUTPUT_RULE],
        "excluded": {"non-executed-notebook": "nbstripout clears every execution count."},
        "findings": [],
        "passed": True,
    }


def test_the_policys_exclusions_reach_pynblint_and_switch_those_rules_off(tmp_path):
    """The fake drops what `--exclude` names, as Pynblint does, so a gate that
    passed no exclusions, or the wrong ones, would fail this notebook."""
    excluded_only = {
        **FINDINGS,
        "lints": [{**FINDINGS["lints"][0], "slug": "non-executed-notebook"}],
    }
    root = repository(tmp_path)
    tools = FakeTools(report=excluded_only)
    assert notebook_lint.main(root, run=tools, environ={}) == 0
    command, _ = tools.pynblint_calls()[-1]
    assert json.loads(command[command.index("--exclude") + 1]) == ["non-executed-notebook"]


def test_a_finding_on_a_notebook_whose_cells_cannot_be_read_is_still_reported(tmp_path):
    """Pynblint reads nbformat 3 notebooks, whose cells sit under `worksheets`; the
    gate reads cells only to quote their first lines, so it quotes nothing then
    rather than crashing on the notebook Pynblint has just linted."""
    version_3 = json.dumps({"worksheets": [{"cells": []}], "metadata": {}, "nbformat": 3})
    root = repository(tmp_path, {NOTEBOOK: version_3})
    assert notebook_lint.main(root, run=FakeTools(report=FINDINGS), environ={}) == 1
    cells = [finding["cells"] for finding in reports(root)[1]["findings"]]
    assert cells == [[], [{"index": 3, "first_line": ""}], [{"index": 2, "first_line": ""}]]


def test_a_notebook_whose_suffix_is_not_lowercase_fails_the_project_rule_only(tmp_path):
    """Pynblint treats a file that does not end in `.ipynb` as a zip archive and
    crashes on it, so such a notebook gets the project rule's finding and no run."""
    upper = "notebooks/1.0-lh-x.IPYNB"
    root = repository(tmp_path, {upper: notebook_json("x = 1")})
    tools = FakeTools(notebooks=(upper,))
    assert notebook_lint.main(root, run=tools, environ={}) == 1
    assert [(f["notebook"], f["rule"]) for f in reports(root)[1]["findings"]] == [
        (upper, PROJECT_RULE)
    ]
    assert len(tools.pynblint_calls()) == 1  # the registry only


def keep_output_notebook(*, notebook=None, cell=None, tags=()) -> str:
    """A notebook with one code cell, flagged the ways nbstripout reads as "keep"."""
    content = json.loads(notebook_json("x = 1"))
    content["metadata"].update(notebook or {})
    content["cells"][1]["metadata"].update(cell or {})
    if tags:
        content["cells"][1]["metadata"]["tags"] = list(tags)
    return json.dumps(content)


@pytest.mark.parametrize(
    ("content", "cells"),
    [
        (keep_output_notebook(notebook={"keep_output": True}), []),
        (keep_output_notebook(cell={"keep_output": True}), [{"index": 1, "first_line": "x = 1"}]),
        (keep_output_notebook(cell={"init_cell": True}), [{"index": 1, "first_line": "x = 1"}]),
        (keep_output_notebook(tags=["keep_output"]), [{"index": 1, "first_line": "x = 1"}]),
    ],
    ids=["notebook metadata", "cell metadata", "init cell", "cell tag"],
)
def test_a_notebook_that_tells_nbstripout_to_keep_outputs_fails(tmp_path, content, cells):
    """nbstripout commits the outputs of whatever carries these flags, and an output
    of this dataset can hold personal data, so the gate refuses them all."""
    root = repository(tmp_path, {NOTEBOOK: content})
    assert notebook_lint.main(root, run=FakeTools(), environ={}) == 1
    findings = reports(root)[1]["findings"]
    assert [(f["rule"], f["cells"]) for f in findings] == [(KEEP_OUTPUT_RULE, cells)]


def test_flags_that_keep_nothing_are_no_finding(tmp_path):
    root = repository(tmp_path, {NOTEBOOK: keep_output_notebook(cell={"keep_output": False})})
    assert notebook_lint.main(root, run=FakeTools(), environ={}) == 0


def test_a_repository_without_notebooks_passes_and_says_so(tmp_path, capsys):
    """Zero notebooks is a pass, and the summary must not read like a check of one."""
    root = repository(tmp_path, notebooks={})
    assert notebook_lint.main(root, run=FakeTools(notebooks=()), environ={}) == 0
    assert "0 notebooks checked" in capsys.readouterr().out
    assert reports(root)[1]["notebooks"] == []


def test_a_notebook_outside_notebooks_fails_the_project_rule(tmp_path):
    root = repository(tmp_path, {"reports/1.0-lh-x.ipynb": notebook_json("x = 1")})
    tools = FakeTools(notebooks=("reports/1.0-lh-x.ipynb",))
    assert notebook_lint.main(root, run=tools, environ={}) == 1
    findings = reports(root)[1]["findings"]
    assert [(f["notebook"], f["rule"]) for f in findings] == [
        ("reports/1.0-lh-x.ipynb", PROJECT_RULE)
    ]


@pytest.mark.parametrize(
    ("tools", "reason"),
    [
        (FakeTools(returncode=1), "exit code 1"),
        (FakeTools(writes_report=False), "wrote no report"),
        (FakeTools(git_missing=True), "git"),
        (FakeTools(git_returncode=128), "not a git repository"),
        (FakeTools(registry_returncode=1), "ImportError"),
        (FakeTools(registry_stdout="Warning: something\n{}"), "JSONDecodeError"),
        (FakeTools(report={"notebook_metadata": {}}), "KeyError"),
        (
            FakeTools(rules={**INSTALLED, "notebook-has-no-tests": "notebook"}),
            "notebook-has-no-tests",
        ),
    ],
    ids=[
        "pynblint crashed",
        "pynblint wrote no report",
        "git is missing",
        "git failed",
        "registry failed",
        "registry answered garbage",
        "report of another shape",
        "policy out of date",
    ],
)
def test_a_run_that_cannot_be_trusted_exits_two_and_is_never_a_pass(tmp_path, tools, reason):
    """A crash with an empty findings list would read exactly like a clean run.

    Exit 2 keeps it apart from findings, and the report says why, so the CI
    summary does not show a stale or missing table.
    """
    root = repository(tmp_path)
    assert notebook_lint.main(root, run=tools, environ={}) == 2
    markdown, payload = reports(root)
    assert payload["passed"] is False
    assert reason in payload["error"]
    assert "could not run" in markdown
    assert reason in markdown


# --------------------------------------------------------------------------
# Which notebooks, and where they live
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "follows"),
    [
        ("notebooks/1.0-lh-dataset-card-profiling.ipynb", True),
        ("notebooks/12.3-ka-a1-b2.ipynb", True),
        ("notebooks/Untitled.ipynb", False),
        ("notebooks/sub/1.0-lh-x.ipynb", False),
        ("reports/1.0-lh-x.ipynb", False),
        ("1.0-lh-x.ipynb", False),
        ("notebooks/eda.ipynb", False),
        ("notebooks/1.0-LH-x.ipynb", False),
        ("notebooks/1-lh-x.ipynb", False),
        ("notebooks/1.0-lh-x-.ipynb", False),
    ],
)
def test_every_notebook_lives_directly_in_notebooks_and_follows_the_naming_convention(
    path, follows
):
    assert follows_project_rule(path) is follows


def git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True)


def test_the_gate_lints_exactly_the_notebooks_git_would_commit(tmp_path):
    """Tracked and untracked notebooks are linted, wherever they are; a gitignored
    one is not part of the repository, and a deleted one is not there to lint.
    Runs the real git, because which files it lists is the behaviour under test."""
    git(tmp_path, "init", "--quiet")
    files = {
        ".gitignore": "scratch/\n",
        "notebooks/1.0-lh-untracked.ipynb": "{}",
        "notebooks/2.0-lh-upper.IPYNB": "{}",
        "reports/1.0-lh-staged.ipynb": "{}",
        "notebooks/1.0-lh-deleted.ipynb": "{}",
        "scratch/1.0-lh-ignored.ipynb": "{}",
        "notebooks/notes.md": "",
    }
    repository(tmp_path, files)
    git(tmp_path, "add", "reports/1.0-lh-staged.ipynb", "notebooks/1.0-lh-deleted.ipynb")
    (tmp_path / "notebooks" / "1.0-lh-deleted.ipynb").unlink()

    assert notebook_lint.candidate_notebooks(tmp_path, subprocess.run) == [
        "notebooks/1.0-lh-untracked.ipynb",
        "notebooks/2.0-lh-upper.IPYNB",
        "reports/1.0-lh-staged.ipynb",
    ]
