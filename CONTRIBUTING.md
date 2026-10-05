# Contributing

This applies to every contributor to this repo, human or AI agent.

## Process

We work in Scrum, tracked as GitHub Issues on the [project board](https://github.com/orgs/mlops-2627q1-mds-upc/projects/1) (Backlog → To Do → In Progress → In Review → Done).
Team communication happens on Discord.
Full process details (roles, ceremonies, Definition of Done, working agreements) live in [docs/docs/scrum/](docs/docs/scrum/) - read that before picking up work.

## Data versioning

We use DVC for data and models, backed by DagsHub Storage.
Conventions for tracking granularity, the remote, and pipeline ownership live in
[docs/docs/data-versioning.md](docs/docs/data-versioning.md) - read that before running `dvc add`.

## Workflow

1. Pick an issue, assign yourself, move it to **In Progress** on the board.
2. Branch off `main`: `feature/<short-description>` or `fix/<short-description>`.
3. Commit using [Conventional Commits](https://www.conventionalcommits.org/) (enforced by pre-commit) - e.g. `feat: add price model training script`.
4. Open a pull request into `main` with a title that also follows Conventional Commits (checked in CI) - it becomes the squash commit message. Move the issue to **In Review**.
5. Passing checks required before merging. No direct pushes to `main`.
   Formal approval isn't enforced by GitHub: PRs from an AI agent run under a teammate's own account, and GitHub blocks self-approval, so requiring it would make those PRs permanently unmergeable. Get a teammate to look it over when you can, but it won't block the merge.
6. **Always squash merge** - it's the only merge method enabled on the repo, so `main` gets exactly one commit per PR and stays readable. Merging deletes the branch automatically.
7. After merge, move the issue to **Done**.

### Freeze window before a presentation

Every merge to `main` deploys to the VM ([requirements](docs/docs/requirements.md) NFR-13), and NFR-05 commits to zero unplanned downtime during a presentation.
So in the 48 hours before a presentation, merge only what fixes something broken, and check `/health` afterwards.
Everything else waits until the presentation is over.

## Local setup

```bash
uv sync
uv run pre-commit install
uv run pre-commit install-hooks
```

`uv sync` installs every dependency, in every group, so that is all a contributor ever needs.
`pre-commit install` puts the hooks in place, and `install-hooks` builds their environments now rather than during your first commit.
One of them is gitleaks, which pre-commit builds from source, downloading Go first if you have none: that takes one to six minutes and a few hundred MB of disk, once per machine, and well under a second per commit afterwards.

## Dependencies

`pyproject.toml` splits the dependencies in two, because the API image is built from one half only and NFR-04 caps that image at 1 GB ([EDN-47](reports/edn.md)):

- `[project] dependencies` is the **serving runtime**: what it takes to load a bundle from `models/<variant>/` and price a listing.
  FastAPI, uvicorn and pydantic join it with the API in M4.
- `[dependency-groups]` is everything else, by who needs it: `pipeline` (the DVC stages and experiment tracking, so also data validation and energy measurement), `notebook`, `docs`, `test` and `dev`.

`[tool.uv] default-groups = "all"` is why a plain `uv sync` and `uv run` still install both halves.
Only the image, and `make test-serving` below, opt out with `--no-default-groups`.

Adding one:

```bash
uv add --group pipeline <package>   # or notebook, docs, test, dev
uv add <package>                    # the serving runtime, only by the rule below
```

A package goes into the runtime only when a module the API imports needs it, at import time or to predict.
Anything only training needs - MLflow, CodeCarbon, Great Expectations, `shap` (EDN-11) - goes in a group, and is imported from a module serving never imports, such as `modeling/train.py`.
`pyproject.toml` keeps a comment beside each dependency saying why it is there; keep it with the dependency when you move one.

These are the modules held to the runtime, all under `recommenditos/`, because they are the ones `tests/test_serving.py` imports and `make test-serving` checks:

| Module | Why it is held to the runtime |
|--------|-------------------------------|
| `modeling/model.py` | The serving seam: `load_model` and `predict_eur` |
| `data/build_features.py`, `data/split_data.py`, `pipeline.py`, `schema.py`, `config.py` | What `model.py` imports |
| `data/preprocess.py` | The API is to reuse its request-side functions, such as `hash_seller_group` |
| `data/synthetic.py` | The check builds its bundles from this fixture |

`modeling/evaluate.py` is not one of them: it imports MLflow through `tracking.py`.
So `point_metrics`, which [the pipeline docs](docs/docs/pipeline.md) list among the functions the API will import, has to move to one of the modules above before serving code can use it.
A module the API comes to import goes into this table and into `tests/test_serving.py` in the same pull request.

## Secrets

No secret is ever committed, in any commit of any branch ([requirements](docs/docs/requirements.md) NFR-15).
Your DagsHub token lives only in the gitignored `.env` and `.dvc/config.local`, and [Getting started](docs/docs/getting-started.md) shows how it gets there.

Two checks enforce that, with the same rules from `.gitleaks.toml`:

- The **gitleaks pre-commit hook** scans what you stage and refuses the commit when it finds a secret.
  It redacts what it found, so its output is safe to paste into an issue.
- CI's **`Secret scan (gitleaks)`** job scans every commit of every branch, tag and pull request on every pull request and every push to `main`, which catches a commit made with `--no-verify` or from a clone without the hook.

The hook prevents a leak; the CI job only detects one.
The repository is public, so by the time CI reports a secret, the push has already published it, and GitHub keeps every commit a pull request ever pointed at, so rewriting the branch does not take it back.
When the scan finds one:

1. **Revoke it first**, on DagsHub under **Your Settings** → **Tokens**, and put a new token into `.env` and `.dvc/config.local`.
2. Remove it from the branch, by rewriting the commits that carry it and force-pushing the feature branch.
   `main` cannot be rewritten, so a secret that reached it stays in its history.
3. A revoked secret that has to stay in the history is acknowledged by adding the fingerprint the job log prints to `.gitleaksignore`, in a pull request that says it was revoked.
   That file is never a way to let a live secret pass.

Never commit past a finding with `--no-verify`.
If the hook fires on something that is not a secret, the fix is a narrow allowlist entry in `.gitleaks.toml`, by path and pattern, in a reviewed pull request; EDN-71 has why the rules look the way they do.

## Before opening a PR

```bash
make format         # ruff format + fix
make lint           # ruff format --check + ruff check
make test           # pytest, which also reports the coverage of recommenditos/
make test-serving   # the serving path from the runtime dependencies alone
make notebook-lint  # Pynblint over every notebook, in its own environment
```

`make lint` runs the same ruff rules as the pre-commit hook and CI, over the same files, so a green `make lint` means a green CI lint job.

`make notebook-lint` is exactly what CI's `Notebook lint (Pynblint)` job runs, and the rules it enforces are the table in [docs/docs/notebooks.md](docs/docs/notebooks.md#pynblint-rules).

When a pull request changes a notebook, or code a notebook imports, also run `make notebook-run`.
It runs every notebook top to bottom on the real data, which CI cannot do because it needs the 548 MB source CSV, and the profiling notebook fails it when a dataset card figure no longer reproduces.

`make test` prints a coverage table; CI puts the same table in the summary of its test job.
There is no coverage gate on a PR, so a number below 80 % does not fail anything.
NFR-07 of the requirements asks for 80 % on a delivery commit, and that is when we read the number and act on it.

`make test-serving` builds `.venv-serving/` from `uv.lock` with the runtime dependencies and the `test` group, and runs `tests/test_serving.py` from it.
It loads that file and the modules in the table above, and nothing else: pytest runs with `--noconftest`, so what `tests/conftest.py` imports for the rest of the suite cannot fail the check.
It fails when a serving module imports a group's package, when the serving path imports a module only the `test` group installs, such as `packaging`, and when the environment holds anything beyond the runtime and the `test` group.
CI's `Serving runtime` job runs the same target and is a required check, so such a module fails the PR rather than the M5 image build.
It only matters when you change a dependency or what a serving module imports, but it is cheap enough to run every time.

## Requirement traceability

Every `FR-xx` and `NFR-xx` of the [requirements](docs/docs/requirements.md) is either verified by a test or verified by hand, and NFR-07 asks us to prove which.
A test says what it verifies with the `req` marker:

```python
@pytest.mark.req("FR-15", "NFR-01")
def test_a_candidate_that_misses_a_criterion_is_not_promoted(): ...
```

- The IDs have to exist in the [requirements](docs/docs/requirements.md).
  A test whose marker names an unknown ID **fails**, so a typo cannot quietly drop the coverage it was meant to record.
- A marker naming nothing at all fails too.
  The per-test check fails the test, and because it cannot run on a test that never runs, the matrix build fails on an empty marker as well and names the test.
- One marker may name several IDs, and several tests may name the same ID.
- Mark the test that actually verifies the requirement, not every test that happens to touch the code on the way.

There are two nets for a typo on purpose, and they cover different ground.
The per-test check fails one test and names the ID, which is what you want locally, but a `skip`, a `skipif` or an `xfail` keeps it from ever running.
The matrix build reads every marker pytest can collect, whether or not the test runs, so that is the net that catches those.

### What the matrix does and does not say

A marker is a claim, and the matrix reports it as one.
Which claim depends on what the [specification](docs/docs/specification.md) says the evidence for that entry is, because that document is what decides, not the marker:

| Status | What it means |
|--------|---------------|
| verified by a test | The entry is **[automated]**, so a test is the promised evidence, and at least one test carries the ID. |
| named by a test | A test carries the ID, but the entry is **[manual]**, so the evidence it promises is the drill its cell names and not that test. |
| verified by hand | The entry is **[manual]** and its cell names the evidence. No test carries the ID. |
| verified by nothing | Neither. |

The distinction between the first two is the whole point of the table.
Before it existed, the reproducibility requirement - which then promised both that `dvc repro` on a clean clone reproduces the same splits and metrics and that every MLflow run records its commit, data version and parameters, today NFR-06 and NFR-14 - was reported as covered because one test named it, and that test's entire body asserted that two files exist.
So: a marker on a **[manual]** entry is recorded, because it is worth knowing a test touches the requirement, and it does not stand in for the drill.

The gate accepts *verified by a test*, *verified by hand*, and *named by a test* where the **[manual]** cell also names its evidence.
It refuses *named by a test* on its own, that is a **[manual]** cell left empty with a marker put on some test, because letting the marker alone pass would reopen the cheapest way to fake coverage there is.
None of the four statuses says a person agreed the evidence is enough; a tool cannot read a test body and judge that.
That judgement is the review NFR-07 asks for before each delivery, and the matrix is what it reads.

Build the matrix locally with

```bash
uv run python -m tools.requirement_matrix
```

It writes `reports/requirement-matrix.md` and `reports/requirement-matrix.json` - both generated, both gitignored - and exits non-zero when a requirement that owes evidence has none.
CI runs the same command, puts the Markdown into the job summary and uploads both files as the `requirement-matrix` artefact.

Which requirements owe evidence *now* is [tools/expected_coverage.yaml](tools/expected_coverage.yaml): every requirement is booked to the milestone whose work produces its evidence, and the gate enforces every milestone at or before `current`.
Before M4 there is no API, so an `FR-xx` without a test is reported without failing the build.
Four things to know when working there:

- A requirement added to the documents has to be booked to a milestone, or the generator fails.
  That is deliberate: it forces the question of when the requirement gets its evidence.
- Book it to the milestone whose work produces the *last* piece of its evidence, not the one its topic belongs to.
  A requirement that is half M3 work and half M4 work is booked M4, because that is when it can first be complete.
- Milestone names are checked against M1 to M6 and have to be listed in that order, and the gate reads the order off that list rather than off the file.
  Neither a name nothing knows nor a block in the wrong place can quietly become a bucket that is never enforced.
- Bumping `current` is how the gate tightens.
  Do it when the milestone's work is merged, and expect it to turn reported gaps into a red build.

The gate's own parameters are pinned by a test: which requirements are already due, and by which route the specification says each is verified.
Moving an ID to a later milestone and rewriting an **[automated]** cell as **[manual]** are both one-line diffs that would otherwise take a requirement out of the blocking set with no test written and nothing else changed.
Either is a fine thing to do with a reason; the test is there so the reason gets written down.

A requirement no test can reach - a load test, a deployment, a drill - is tagged **[manual]** in the [specification](docs/docs/specification.md), and the cell names the evidence.
The matrix then shows it as verified by hand rather than as a gap, which is why that cell must never be left empty.
