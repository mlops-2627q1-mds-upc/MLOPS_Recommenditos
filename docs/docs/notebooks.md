Notebooks
=========

Notebooks are where we explore the data and show results.
Anything the pipeline relies on is computed in `recommenditos/`, where it is tested, and a notebook imports it from there.
This page says where notebooks live, how to run them, how their outputs stay out of git, and which checks they pass before a merge.

## Where they live and how they are named

Every notebook lives directly in `notebooks/`, with no subfolders, and is named `<number>.<version>-<initials>-<description>.ipynb`, the convention of the [README](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/README.md#project-organization):

- `<number>` orders the notebooks, and `<version>` counts rewrites of the same analysis;
- `<initials>` are the author's, in lowercase;
- `<description>` is a few lowercase words joined by hyphens.

`1.0-lh-dataset-card-profiling.ipynb` is the profiling behind the [dataset card](dataset-card.md).
`make notebook-lint` enforces both the folder and the name (see [the project's own rules](#the-projects-own-rules)).

## Running one

Notebooks run in the project environment, so they import `recommenditos` like any other code:

```bash
uv run jupyter lab
```

To check that every notebook still runs top to bottom on a fresh kernel:

```bash
make notebook-run
```

It runs each notebook in `notebooks/` with `jupyter execute`, stops at the first cell that raises, and writes nothing back, so no executed copy with outputs lands on the disk.
The notebooks read the real data, so it needs the 548 MB source CSV in `data/external/`.
The `download` stage caches the file there; when it is missing, the profiling notebook downloads it itself and checks it against the MD5 pinned in `params.yaml` before reading it.
That is also why it does not run in CI: every run would download 548 MB and hold the whole snapshot in memory, which for the profiling notebook takes about a minute and 2.8 GB of RAM.
It proves that a notebook runs, and for the profiling notebook also that every dataset card figure it checks reproduces, because its summary cell fails when one does not.
`jupyter execute` prints no outputs, so read the results themselves in Jupyter.
Run it before a pull request that changes a notebook or the code a notebook imports.

## Outputs never reach git

The `nbstripout` pre-commit hook strips every output, every execution count and the editor's cell metadata from a notebook before it is committed.
CI's `Lint & format` job runs the same hook over every file, so a notebook committed with outputs, for example past a hook that was never installed, fails there.

There are two reasons.
Outputs change on every run, which buries the change a reviewer is meant to read under hundreds of lines of rendered tables.
And the dataset carries personal data, among it VINs, street addresses and seller company names (see the [dataset card](dataset-card.md)), so one `raw.head()` left in an output would put rows of it into the git history for good.

nbstripout keeps the outputs of a notebook whose metadata sets `keep_output`, and of a cell whose metadata sets `keep_output` or `init_cell` or whose tags hold `keep_output`.
We do not use those flags, and `make notebook-lint` fails a notebook that carries one (see [the project's own rules](#the-projects-own-rules)).

## `make notebook-lint`

```bash
make notebook-lint
```

This runs [Pynblint](https://github.com/collab-uniba/pynblint) over every notebook git would commit: tracked or not yet, wherever it lies, and not the gitignored ones.
It also checks [the project's own rules](#the-projects-own-rules).
It fails on any finding, and also when the run cannot be trusted; make shows the first as `Error 1` and the second as `Error 2`.
Either way it writes `reports/notebook-lint.md` and `reports/notebook-lint.json`, which are generated and gitignored.
CI's `Notebook lint (Pynblint)` job runs exactly this target on every pull request into `main` and every push to `main`, and publishes the Markdown as its job summary.

### Why it has its own environment

Pynblint 0.1.6, its last release (August 2024), pins `typer<0.13` and `ipython<9`, which our `uv.lock` cannot satisfy, so it is not a project dependency.
It runs from `tools/pynblint-env/`, an environment with its own `pyproject.toml` and `uv.lock`.
The first `make notebook-lint` builds it in a few seconds, and `uv run --locked` refuses to run when the lock no longer matches its `pyproject.toml`.
A bare `uvx pynblint` would resolve the newest versions on every run instead, and today that includes a `click` on which typer 0.12 crashes at start-up.

Pynblint on its own cannot gate a build, so `make notebook-lint` runs it through [`tools/notebook_lint.py`](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/tools/notebook_lint.py):

- Pynblint exits 0 whatever it finds, so the wrapper reads its JSON report and sets the exit code itself, and a crash or a missing report is an error rather than a clean run.
- Pynblint reads every setting from an environment variable of the same name, case-insensitively and with no prefix, and from a `.pynblint` file in the working directory.
  `INCLUDE=/usr/include`, which C toolchains set, crashes it, and a stray `EXCLUDE` would switch rules off without a trace.
  The wrapper runs it with only the variables it needs to start, in an empty temporary directory.
- The rules we enforce are the table below, and the wrapper reads them from this page rather than from a copy that could drift.
  It refuses to run unless the table classifies exactly the rules the installed Pynblint has, so a typo in a slug or a rule a new release adds stops the gate instead of slipping past it.
  It also refuses a table that enforces no rule, and one that enforces a repository rule, which would be reported as checked and never run (see below).

To upgrade Pynblint, run `uv lock --upgrade --project tools/pynblint-env` and then `make notebook-lint`, and give every new rule a row here.

### Pynblint rules

The wrapper runs Pynblint on one notebook file at a time, which runs its 17 notebook- and cell-level rules.
The 5 repository rules only run when Pynblint is given a whole directory, which would also make it walk every gitignored notebook on the disk; their rows say why none of them would add anything.
Every threshold is Pynblint's default.

<!--
The notebook lint gate (tools/notebook_lint.py) reads this table: one row per
Pynblint rule, the slug in backticks, the decision `enforced` or `excluded`, and
a reason. It refuses to run unless the rows classify exactly the rules the
installed Pynblint has, enforce at least one of them and no repository rule, and
it refuses a second table with this header.
-->

| Rule | Decision | Reason |
|------|----------|--------|
| `notebook-too-long` | enforced | More than 50 cells is two analyses in one notebook; Pynblint itself advises splitting it. |
| `cell-too-long` | enforced | A code cell of more than 30 lines, blank lines included, is computation, which belongs in `recommenditos/` where it is tested; a notebook displays results. |
| `imports-beyond-first-cell` | enforced | Dependencies are declared once, at the top. It catches less than its name says: it sees only `import x`, not `from x import y`, never checks the last code cell, and ignores imports nested in a function. |
| `missing-h1-MD-heading` | enforced | An H1 title in the first 3 cells says what the notebook is. |
| `missing-opening-MD-text` | enforced | Prose in the first 3 cells says what the notebook is for, before any code. |
| `missing-closing-MD-text` | enforced | Prose in the last 3 cells says what the notebook found. |
| `too-few-MD-cells` | enforced | At least 0.3 Markdown cells per code cell, so the reasoning is written down rather than left to be read out of the code. |
| `long_multiline_python_comment` | enforced | A code cell that opens with 4 or more comment lines holds an explanation, which belongs in Markdown. Its pattern also counts a line of code with an inline comment as a comment line. |
| `empty-cells` | enforced | No leftover empty code cells. |
| `invalid-python-syntax` | enforced | Code that does not parse. ruff reports it too, and the overlap costs nothing. |
| `untitled-notebook` | enforced | No `Untitled*.ipynb` left with Jupyter's default name. |
| `duplicate-notebook-not-renamed` | enforced | No `*-Copy<n>.ipynb` left over from Jupyter's "Duplicate". |
| `non-portable-chars-in-nb-name` | enforced | File names only in `[A-Za-z0-9_.-]`, so every operating system can check them out. |
| `non-executed-notebook` | excluded | nbstripout clears every execution count on commit, so every committed notebook is non-executed by construction and the rule would fire on all of them. That a notebook runs is shown by `make notebook-run` instead. |
| `non-executed-cells` | excluded | It switches itself off when no code cell has an execution count, which nbstripout guarantees. On a stripped notebook it fires only when there is an empty code cell, and then flags every other code cell; `empty-cells` already names that problem. |
| `non-linear-execution` | excluded | It reads execution counts, which nbstripout removes, so it can never fire, and enforcing it would claim a check that does not happen. Counts in order would not prove a fresh top-to-bottom run anyway; `make notebook-run` does. |
| `notebook-name-too-long` | excluded | Off in Pynblint 0.1.6 itself: its threshold is 0, which disables it, and only an environment variable or a `.pynblint` file could set another, both of which the gate shuts out on purpose. The naming convention keeps names descriptive, and any limit we chose would be arbitrary. |
| `repository-not-versioned` | excluded | A repository rule, so it does not run. It looks for a `.git` directory, which a git worktree does not have, so it would report a false finding there; the repository is on GitHub, with branch protection on `main`. |
| `dependencies-unmanaged` | excluded | A repository rule, so it does not run. It checks that a file like `pyproject.toml` exists; we have `pyproject.toml` and `uv.lock`, and CI's `uv sync --locked` fails when the two drift apart, which is a stronger check. |
| `test-coverage-data-not-available` | excluded | A repository rule, so it does not run. It checks for a `.coverage` file at the root, which is gitignored, so it says whether the tests ran in this checkout, not whether the code is tested. Every pytest run measures coverage, and CI's `Tests` job publishes it (NFR-07). |
| `duplicate-notebook-filename` | excluded | A repository rule, so it does not run. Every notebook lives directly in `notebooks/` (the project's location rule), so no two can share a name. |
| `large-data-file-not-versioned` | excluded | A repository rule, so it does not run. It fires only when the repository has no `.dvc/` directory, which ours has, and never checks whether a large file is tracked by DVC. Large files are kept out of git by the `.gitignore` rule that ignores everything under `data/` except DVC's pointer files, by the `.gitignore` files DVC writes next to its outputs, and by the `check-added-large-files` pre-commit hook, which refuses any file over 1000 KB, locally on commit and in CI over every tracked file. |

### The project's own rules

The wrapper adds two rules of its own, which run on every notebook git would commit, whatever its name.

`notebook-location-or-name`: the notebook lives directly in `notebooks/`, and its name matches `<number>.<version>-<initials>-<description>.ipynb`, that is `^[0-9]+\.[0-9]+-[a-z]+-[a-z0-9]+(-[a-z0-9]+)*\.ipynb$`.
Pynblint checks a name only for Jupyter's default titles and for non-portable characters, and never where a notebook lives.
A notebook outside `notebooks/` is easy to miss in a review, the convention keeps notebooks ordered and attributable, and one flat folder is what makes `duplicate-notebook-filename` unnecessary.
A file whose suffix is not exactly `.ipynb`, such as `.IPYNB`, fails this rule and is not given to Pynblint, which would read it as a zip archive.

`notebook-keeps-outputs`: neither the notebook nor any of its cells carries a flag that makes nbstripout keep outputs, which are the notebook's `keep_output` metadata, and a cell's `keep_output` or `init_cell` metadata or `keep_output` tag.
nbstripout is what keeps outputs out of git, and these flags are the one way past it that no other check sees.

### How a finding is handled

A finding fails `make notebook-lint` and the CI job.
Fix the notebook: split it, move the computation into `recommenditos/`, write the Markdown, delete the empty cell, rename the file.

There is no waiver per notebook or per cell.
If a rule is wrong for us in general, change its row to `excluded` with a reason in the same pull request, where the change is reviewed like any other.
Never change a threshold to make a notebook pass.

`Error 2` is not a finding but a run whose answer cannot be trusted: the table is malformed, enforces nothing or a repository rule, or does not match the installed Pynblint, git failed, Pynblint crashed, or the gate got an answer it did not expect, such as a report of another shape from a new Pynblint.
The message names the rule or shows the error: fix the table, the environment, or a notebook file Pynblint cannot read.

## Who covers what

| Check | What it covers | Where it runs |
|-------|----------------|---------------|
| ruff check and ruff format | The code in the code cells, with the same rules as every other Python file | pre-commit, `make lint`, CI's `Lint & format` job |
| Pynblint | The structure and narrative of a notebook: its length, its title, its opening and closing prose, Markdown per code cell, empty cells, its file name | `make notebook-lint`, CI's `Notebook lint (Pynblint)` job |
| The project's own rules | Where a notebook lives, how it is named, and that it carries no flag that keeps outputs | `make notebook-lint`, CI's `Notebook lint (Pynblint)` job |
| nbstripout | Outputs, execution counts and editor metadata stay out of git | pre-commit, CI's `Lint & format` job |
| `make notebook-run` | Every notebook runs top to bottom on a fresh kernel against the pinned data | locally, before a pull request that touches a notebook or the code it imports; not in CI |
| The notebook's own check tables | Whether every dataset card figure it checks matches the data; a mismatch is listed in the summary table and then fails the cell | inside the profiling notebook, on every run, so also in `make notebook-run` |
