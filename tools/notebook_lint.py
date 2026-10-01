"""Lint every notebook with Pynblint, against the rule policy in docs/docs/notebooks.md.

Pynblint on its own cannot gate a build, and this wrapper closes the three gaps
that keep it from doing so:

* it exits 0 whatever it finds, so a CI step that only runs it is green with
  findings;
* its configuration leaks in from outside: every setting is also read from an
  environment variable of the same name, case-insensitively and with no prefix,
  and from a `.pynblint` file in the working directory, so `INCLUDE=/usr/include`
  from a C toolchain crashes it and a stray `EXCLUDE` quietly switches rules off;
* which rules we enforce, and why we exclude the others, is a decision the team
  reviews, so it lives in the docs as a table. The gate reads that table rather
  than a copy of it that could drift, and refuses to run unless the table
  classifies exactly the rules the installed Pynblint has.

It adds two rules of its own: every notebook lives directly in `notebooks/` and
is named after the convention the README gives, and no notebook carries the flags
that make nbstripout keep, and so commit, its outputs.

The exit code is 0 when nothing was found, 1 for findings, and 2 when the run
cannot be trusted: the table is malformed or out of date, git failed, Pynblint
crashed or wrote no report, or anything else went wrong that the gate did not
expect. Both reports are written before it exits, so CI publishes them whatever
the outcome, and a report from an earlier run never stands in for this one.

It runs in the environment of tools/pynblint-env, which holds Pynblint and its
pins and none of the project's dependencies, so it uses the standard library
only and runs as a script, which is what `make notebook-lint` does:

    uv run --project tools/pynblint-env --locked python tools/notebook_lint.py
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import traceback

# Derived from this file's location, so the gate lints the checkout it belongs to
# from any working directory. The other paths are relative to that root, so that
# a test can point `main` at a checkout of its own.
PROJ_ROOT = Path(__file__).resolve().parents[1]
POLICY_DOC = Path("docs/docs/notebooks.md")
REPORT_MD = Path("reports/notebook-lint.md")
REPORT_JSON = Path("reports/notebook-lint.json")

ENFORCED = "enforced"
EXCLUDED = "excluded"

#: The project's own rule. Pynblint checks a notebook's name only for a default
#: title and for non-portable characters, and never where the notebook lives.
PROJECT_RULE = "notebook-location-or-name"
# In backticks, because the job summary is Markdown and would drop `<number>`
# and the rest as unknown HTML tags.
_CONVENTION = "`<number>.<version>-<initials>-<description>.ipynb`"
PROJECT_RULE_TEXT = f"every notebook lives directly in `notebooks/` and is named {_CONVENTION}"
PROJECT_RULE_ADVICE = (
    f"Move the notebook directly into `notebooks/` and name it {_CONVENTION} in lowercase, "
    "e.g. `1.0-lh-dataset-card-profiling.ipynb`."
)
NOTEBOOKS_DIR = "notebooks"
NOTEBOOK_SUFFIX = ".ipynb"

#: The second of the project's rules. nbstripout keeps every output of a notebook
#: whose metadata sets `keep_output`, and a cell's when the cell's metadata sets
#: `keep_output` or `init_cell` or its tags hold `keep_output`; it then commits
#: them, and an output of this dataset can hold personal data.
KEEP_OUTPUT_RULE = "notebook-keeps-outputs"
KEEP_OUTPUT_TEXT = "no notebook or cell sets the `keep_output` or `init_cell` flags of nbstripout"
KEEP_OUTPUT_ADVICE = (
    "Remove `keep_output` from the notebook's metadata, and `keep_output` and `init_cell` "
    "from the cells' metadata and tags: nbstripout commits the outputs of whatever "
    "carries them."
)
PROJECT_RULES = {PROJECT_RULE: PROJECT_RULE_TEXT, KEEP_OUTPUT_RULE: KEEP_OUTPUT_TEXT}
NOTEBOOK_NAME = re.compile(r"[0-9]+\.[0-9]+-[a-z]+-[a-z0-9]+(?:-[a-z0-9]+)*\.ipynb")

#: What Python, git and Pynblint need to start and to read text, on Linux, macOS
#: and Windows. Everything else is dropped, because any variable named like a
#: Pynblint setting configures it.
KEPT_VARIABLES = frozenset(
    {
        "PATH",
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMDATA",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "LANG",
        "LC_ALL",
    }
)

#: Run by the tool environment's interpreter: the installed version, and every
#: rule Pynblint registers when nothing is excluded, with the level it runs at.
#: Read from the registry rather than from a list kept here, so that a rule a new
#: release adds reaches the policy check instead of running unclassified.
REGISTRY_SCRIPT = """\
import json
from importlib.metadata import version
from pynblint import lint_register, loader
loader.load_core_modules()
levels = {
    "notebook": lint_register.enabled_notebook_level_lints,
    "cell": lint_register.enabled_cell_level_lints,
    "project": lint_register.enabled_project_level_lints,
    "path": lint_register.enabled_path_level_lints,
}
rules = {rule.slug: level for level, lints in levels.items() for rule in lints}
print(json.dumps({"version": version("pynblint"), "rules": rules}))
"""

#: The levels Pynblint runs only when it is given a whole directory. The gate
#: gives it one notebook file at a time, so a rule at these levels never runs.
REPOSITORY_LEVELS = frozenset({"project", "path"})

#: Generous for a linter that takes about a second per notebook, and short of the
#: six hours a hung CI job would otherwise hold a runner for.
TIMEOUT_SECONDS = 300

#: How long the first line of a cell may get in a report before it is cut.
FIRST_LINE_LENGTH = 80

_TABLE_HEADER = re.compile(r"\|\s*Rule\s*\|\s*Decision\s*\|\s*Reason\s*\|")
_SEPARATOR_ROW = re.compile(r"\|[\s:|-]+\|")
_SLUG_CELL = re.compile(r"`([A-Za-z0-9_-]+)`")
_POLICY_CELLS = 3

Run = Callable[..., subprocess.CompletedProcess]

#: How every subprocess runs: output captured as text, no stdin to wait on, and
#: a deadline. The exit code is checked by the caller, which knows what it means.
_QUIET = {
    "capture_output": True,
    "encoding": "utf-8",
    "stdin": subprocess.DEVNULL,
    "timeout": TIMEOUT_SECONDS,
    "check": False,
}


class PolicyError(Exception):
    """The rule table cannot be read, or does not match the installed Pynblint."""


class ToolError(Exception):
    """git or Pynblint did not give an answer the gate can trust."""


@dataclass(frozen=True)
class Policy:
    """The rule table: each rule's decision and the reason for it.

    Kept in the table's order, so the reports list the rules the way the page
    groups them.
    """

    rules: dict[str, tuple[str, str]]

    @property
    def enforced(self) -> list[str]:
        return [slug for slug, (decision, _) in self.rules.items() if decision == ENFORCED]

    @property
    def excluded(self) -> dict[str, str]:
        return {
            slug: reason for slug, (decision, reason) in self.rules.items() if decision == EXCLUDED
        }


@dataclass(frozen=True)
class Finding:
    notebook: str
    rule: str
    recommendation: str
    #: Each affected cell's 0-based index and first line. The line is there
    #: because Jupyter does not number cells, so an index alone is hard to find.
    cells: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class Result:
    pynblint: str
    notebooks: list[str]
    policy: Policy
    findings: list[Finding]

    @property
    def passed(self) -> bool:
        return not self.findings


# --------------------------------------------------------------------------
# The policy
# --------------------------------------------------------------------------


def parse_policy(text: str) -> Policy:
    """Read the one table whose header is `| Rule | Decision | Reason |`.

    Strict, because every way of misreading it loosens the gate without a
    trace: a skipped row leaves its rule to Pynblint's default, and a table
    that is not found enforces nothing.
    """
    lines = [line.strip() for line in text.splitlines()]
    headers = [index for index, line in enumerate(lines) if _TABLE_HEADER.fullmatch(line)]
    if not headers:
        raise PolicyError("found no rule table, i.e. no table headed | Rule | Decision | Reason |")
    if len(headers) > 1:
        raise PolicyError(f"found {len(headers)} rule tables, expected one")

    rules: dict[str, tuple[str, str]] = {}
    for line in lines[headers[0] + 1 :]:
        if not line.startswith("|"):
            break
        if _SEPARATOR_ROW.fullmatch(line):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != _POLICY_CELLS:
            raise PolicyError(f"the row {line} has {len(cells)} cells, expected 3")
        rule, decision, reason = cells
        slug = _SLUG_CELL.fullmatch(rule)
        if not slug:
            raise PolicyError(f"the rule {rule!r} is not a slug in backticks")
        if slug[1] in rules:
            raise PolicyError(f"{slug[1]} is listed twice")
        if decision not in (ENFORCED, EXCLUDED):
            raise PolicyError(f"{slug[1]} is {decision!r}, expected {ENFORCED} or {EXCLUDED}")
        if not reason:
            raise PolicyError(f"{slug[1]} gives no reason")
        rules[slug[1]] = (decision, reason)
    if not rules:
        raise PolicyError("the rule table has no rows")
    policy = Policy(rules)
    if not policy.enforced:
        raise PolicyError("the rule table enforces no rule, so the gate would pass anything")
    return policy


def check_policy(installed: Mapping[str, str], policy: Policy) -> None:
    """Refuse a table that does not classify exactly the installed rules.

    `installed` maps each rule's slug to the level it runs at. A slug Pynblint
    does not have is a typo, or a rule a release removed: as an exclusion
    Pynblint ignores it silently, so the rule meant to be off runs, and as an
    enforced rule nothing checks it. A rule the table does not name is new in the
    installed release, and would run without anyone having decided to enforce it.
    An enforced repository rule would be reported as checked and never run.
    """
    unknown = sorted(set(policy.rules) - set(installed))
    unclassified = sorted(set(installed) - set(policy.rules))
    never_run = [slug for slug in policy.enforced if installed.get(slug) in REPOSITORY_LEVELS]
    problems = []
    if unknown:
        problems.append(f"it lists {', '.join(unknown)}, which Pynblint does not have")
    if unclassified:
        problems.append(f"it does not classify {', '.join(unclassified)}; add a row for each")
    problems += [
        f"{slug} is a repository rule, which never runs on a single notebook; exclude it"
        for slug in never_run
    ]
    if problems:
        raise PolicyError(
            f"the rule table does not match the installed Pynblint: {'; '.join(problems)}"
        )


# --------------------------------------------------------------------------
# Running git and Pynblint
# --------------------------------------------------------------------------


def clean_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """The environment Pynblint runs in: only what it needs to start."""
    kept = {name: value for name, value in environ.items() if name in KEPT_VARIABLES}
    # Pynblint opens a notebook with the platform's default encoding, which is
    # not UTF-8 on Windows, and our notebooks contain names like "Škoda".
    return {**kept, "PYTHONUTF8": "1"}


def installed_rules(run: Run, env: dict[str, str]) -> tuple[str, dict[str, str]]:
    """The installed Pynblint's version, and the level of every rule it has.

    In an empty working directory, because importing Pynblint already reads a
    `.pynblint` file there, whose exclusions would hide rules from the registry.
    """
    with tempfile.TemporaryDirectory() as scratch:
        answer = run([sys.executable, "-c", REGISTRY_SCRIPT], cwd=scratch, env=env, **_QUIET)
    if answer.returncode != 0:
        raise ToolError(
            f"could not read the installed Pynblint's rules (exit code {answer.returncode}):\n"
            f"{answer.stderr}"
        )
    registry = json.loads(answer.stdout)
    return registry["version"], registry["rules"]


def candidate_notebooks(root: Path, run: Run) -> list[str]:
    """Every notebook git would commit, relative to the root.

    Untracked ones too, so a new notebook is linted before its first commit;
    not gitignored ones, which are not part of the repository, and not a tracked
    file that was deleted from the disk. The suffix is matched in any case, so a
    `.IPYNB` reaches the project rule instead of being skipped.
    """
    command = ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others"]
    command += ["--exclude-standard", "--", ":(icase)*.ipynb"]
    answer = run(command, **_QUIET)
    if answer.returncode != 0:
        raise ToolError(
            f"git could not list the notebooks (exit code {answer.returncode}):\n{answer.stderr}"
        )
    listed = {path for path in answer.stdout.split("\0") if path}
    return sorted(path for path in listed if (root / path).is_file())


def follows_project_rule(notebook: str) -> bool:
    """Whether a notebook lives directly in `notebooks/` and is named by the convention."""
    folder, _, name = notebook.rpartition("/")
    return folder == NOTEBOOKS_DIR and NOTEBOOK_NAME.fullmatch(name) is not None


def pynblint_command(notebook: Path, excluded: Iterable[str], output: Path) -> list[str]:
    """Pynblint on one notebook file, which runs its notebook- and cell-level rules.

    One file at a time rather than the repository, because given a directory
    Pynblint walks every `.ipynb` on the disk, gitignored ones included. Pynblint
    parses `--exclude` with `json.loads`, so the comma-separated form its own
    help shows crashes it; `--yes` answers the prompt an existing output file
    raises, and `--quiet` drops the terminal rendering this wrapper replaces.
    """
    return [
        sys.executable,
        "-m",
        "pynblint",
        str(notebook),
        "--exclude",
        json.dumps(sorted(excluded)),
        "--output",
        str(output),
        "--yes",
        "--quiet",
    ]


def lint_one(
    root: Path, notebook: str, excluded: Iterable[str], run: Run, env: dict[str, str]
) -> list[Finding]:
    """Pynblint's findings on one notebook.

    A non-zero exit code or a missing report raises rather than returning no
    findings, because a run that crashed and a clean one would otherwise look
    the same.
    """
    with tempfile.TemporaryDirectory() as scratch:
        output = Path(scratch) / "pynblint.json"
        command = pynblint_command(root / notebook, excluded, output)
        answer = run(command, cwd=scratch, env=env, **_QUIET)
        if answer.returncode != 0:
            raise ToolError(
                f"Pynblint failed on {notebook} (exit code {answer.returncode}):\n{answer.stderr}"
            )
        if not output.is_file():
            raise ToolError(f"Pynblint exited 0 on {notebook} but wrote no report")
        report = json.loads(output.read_text(encoding="utf-8"))

    sources = cell_sources(read_notebook(root / notebook))
    return [
        Finding(
            notebook=notebook,
            rule=lint["slug"],
            recommendation=lint["recommendation"],
            cells=tuple(
                (cell["index"], _first_line(sources, cell["index"]))
                for cell in lint.get("cells", [])
            ),
        )
        for lint in report["lints"]
    ]


def read_notebook(path: Path) -> dict:
    """The notebook's JSON, for the project's rules and for quoting cells.

    Best effort: Pynblint lints the file itself, and a notebook it reads but this
    does not, such as one in the nbformat 3 layout, gets Pynblint's findings
    without the quotes rather than failing the run.
    """
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return content if isinstance(content, dict) else {}


def cell_sources(content: dict) -> list[str]:
    return ["".join(cell.get("source", "")) for cell in content.get("cells", [])]


def keeps_outputs(notebook: str, content: dict) -> Finding | None:
    """A finding when nbstripout would keep any of the notebook's outputs."""
    sources = cell_sources(content)
    cells = tuple(
        (index, _first_line(sources, index))
        for index, cell in enumerate(content.get("cells", []))
        if _keeps_output(cell.get("metadata", {}))
    )
    if content.get("metadata", {}).get("keep_output") or cells:
        return Finding(notebook, KEEP_OUTPUT_RULE, KEEP_OUTPUT_ADVICE, cells)
    return None


def _keeps_output(metadata: dict) -> bool:
    tags = metadata.get("tags", [])
    return bool(metadata.get("keep_output") or metadata.get("init_cell") or "keep_output" in tags)


def _first_line(sources: list[str], index: int) -> str:
    source = sources[index] if 0 <= index < len(sources) else ""
    line = next((line.strip() for line in source.splitlines() if line.strip()), "")
    return line if len(line) <= FIRST_LINE_LENGTH else f"{line[: FIRST_LINE_LENGTH - 3]}..."


# --------------------------------------------------------------------------
# The reports
# --------------------------------------------------------------------------


def _counted(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def summary(result: Result) -> str:
    return (
        f"{_counted(len(result.notebooks), 'notebook')} checked with Pynblint "
        f"{result.pynblint}: {_counted(len(result.policy.enforced), 'rule')} and "
        f"{_counted(len(PROJECT_RULES), 'project rule')} enforced, "
        f"{_counted(len(result.policy.excluded), 'rule')} excluded "
        f"(see {POLICY_DOC.as_posix()}), {_counted(len(result.findings), 'finding')}"
    )


def render_text(result: Result) -> str:
    """The terminal and job-log view: every finding with the cells it names."""
    lines = []
    for notebook in result.notebooks:
        findings = [finding for finding in result.findings if finding.notebook == notebook]
        lines.append(f"{notebook}: {'' if findings else 'no findings'}".rstrip())
        for finding in findings:
            lines.append(f"  {finding.rule}")
            lines += [f"    cell {index}: {line or '(empty)'}" for index, line in finding.cells]
            lines += [f"    {line}" for line in finding.recommendation.splitlines()]
    return "\n".join([*lines, "", summary(result)] if lines else [summary(result)])


def render_markdown(result: Result) -> str:
    """The CI job summary: the verdict, the findings, and the policy behind them."""
    lines = [f"## Notebook lint {'passed' if result.passed else 'failed'}", ""]
    lines += [f"{summary(result)}.", ""]
    if result.findings:
        lines += [
            "### Findings",
            "",
            "Cells count from 0, Markdown cells included; the job log shows their first lines.",
            "",
            "| Notebook | Rule | Cells | Recommendation |",
            "|----------|------|-------|----------------|",
            *(
                f"| `{finding.notebook}` | `{finding.rule}` "
                f"| {', '.join(str(index) for index, _ in finding.cells) or '-'} "
                f"| {' '.join(finding.recommendation.split())} |"
                for finding in result.findings
            ),
            "",
        ]
    if result.notebooks:
        lines += ["### Notebooks", "", *(f"- `{path}`" for path in result.notebooks), ""]
    lines += [
        "### Enforced",
        "",
        ", ".join(f"`{slug}`" for slug in result.policy.enforced) + ", and the project's rules:",
        "",
        *(f"- `{slug}`: {text}." for slug, text in PROJECT_RULES.items()),
        "",
        "<details><summary>Excluded rules and why</summary>",
        "",
        "| Rule | Reason |",
        "|------|--------|",
        *(f"| `{slug}` | {reason} |" for slug, reason in result.policy.excluded.items()),
        "",
        "</details>",
    ]
    return "\n".join(lines) + "\n"


def as_json(result: Result) -> dict:
    return {
        "pynblint": result.pynblint,
        "notebooks": result.notebooks,
        "enforced": result.policy.enforced,
        "project_rules": list(PROJECT_RULES),
        "excluded": result.policy.excluded,
        "findings": [
            {
                "notebook": finding.notebook,
                "rule": finding.rule,
                "cells": [{"index": index, "first_line": line} for index, line in finding.cells],
                "recommendation": finding.recommendation,
            }
            for finding in result.findings
        ],
        "passed": result.passed,
    }


def _write_reports(root: Path, markdown: str, payload: dict) -> None:
    for path, content in (
        (root / REPORT_MD, markdown),
        (root / REPORT_JSON, json.dumps(payload, indent=2) + "\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _cannot_run(root: Path, reason: str) -> int:
    """Report a run whose answer cannot be trusted, and exit 2.

    The report is still written, so that CI's job summary says why instead of
    showing nothing, or a stale table from an earlier run.
    """
    markdown = f"## Notebook lint could not run\n\n```text\n{reason}\n```\n"
    _write_reports(root, markdown, {"passed": False, "error": reason})
    print(f"Notebook lint could not run: {reason}", file=sys.stderr)
    return 2


def main(
    root: Path = PROJ_ROOT, run: Run = subprocess.run, environ: Mapping[str, str] = os.environ
) -> int:
    env = clean_environment(environ)
    try:
        policy = parse_policy((root / POLICY_DOC).read_text(encoding="utf-8"))
        version, installed = installed_rules(run, env)
        check_policy(installed, policy)
        notebooks = candidate_notebooks(root, run)
        findings = []
        for notebook in notebooks:
            if not follows_project_rule(notebook):
                findings.append(Finding(notebook, PROJECT_RULE, PROJECT_RULE_ADVICE))
            if keeping := keeps_outputs(notebook, read_notebook(root / notebook)):
                findings.append(keeping)
            # Pynblint reads any other suffix as a zip archive and crashes on it.
            if notebook.endswith(NOTEBOOK_SUFFIX):
                findings += lint_one(root, notebook, policy.excluded, run, env)
    except PolicyError as error:
        return _cannot_run(root, f"{POLICY_DOC.as_posix()}: {error}")
    # An OSError names its own file: the policy page, a notebook, or git itself.
    except (OSError, ToolError) as error:
        return _cannot_run(root, str(error))
    # Anything else is an answer the gate did not expect, such as a report of
    # another shape from a new Pynblint. Exit 1 would read as "findings", so it is
    # an untrustworthy run like the others, with the traceback for whoever fixes it.
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        return _cannot_run(root, f"unexpected {type(error).__name__}: {error}")

    result = Result(version, notebooks, policy, findings)
    _write_reports(root, render_markdown(result), as_json(result))
    print(render_text(result))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
