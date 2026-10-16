# Report and Engineering Decision Notebook

LaTeX sources for our two course deliverables:

- `report.tex` - the team report, following the course template ([references/mlops_report_template.md](../../references/mlops_report_template.md)).
  One source for both deliveries: the initial report shows Milestones 1-3, the final report adds Milestones 4-6 and the retrospective.
- `edn.tex` - the Engineering Decision Notebook ([references/Instruction_EDN_MLOps_v2026.md](../../references/Instruction_EDN_MLOps_v2026.md)), one file per entry in `edn/`, transferred from the working copy [reports/edn.md](../edn.md).

## Layout

| Path | What goes there |
|------|-----------------|
| `config.tex` | Team name, members, links and contact. The only place to edit cover page facts. |
| `sections/` | One file per report section; `2-N-*.tex` is Milestone N. |
| `edn/` | One EDN entry per file and their list, `entries.tex`, all generated from [reports/edn.md](../edn.md) by `make edn`. |
| `figures/` | Figures for the report (`../figures/` is searched too, but the Docker and CI builds mount only this folder, so a figure the report uses must lie here). `data-composition.pdf`, `energy-vs-error.pdf` and `error-by-segment.pdf` are copies of what the `profile-data`, `compare-energy` and `compare-segments` stages write to `reports/figures/`; the test suite fails when a copy differs from the stage's output. `pipeline.pdf` is drawn from `dvc.yaml` by `figures/pipeline.sh`, which needs the project environment, Graphviz and the DejaVu Sans font; the PDF is committed with the font embedded, so a build needs none of them. |
| `references.bib` | Bibliography (biblatex + biber, IEEE style). |
| `preamble.tex` | Shared packages, build flags and the `\guidance` / `\tbd` helpers. |

## Writing

- Blue **Guidance** boxes say what a section must contain: the course template's prompt, the rubric question and points, and what the repo already has.
  They disappear in submission builds, so there is no need to delete them.
- `\tbd{...}` marks a placeholder.
  It renders in red in every build, and `make report-submit` counts the ones left.
- Refer to EDN entries by ID in the report, e.g. "(EDN-03)".
- A figure that shows data is drawn by a pipeline stage, not by hand: the stage writes a PDF to `reports/figures/`, and its copy in `figures/` is what the report includes.
  `recommenditos/plots.py` holds the style every such figure shares, and its docstring says how colour and size are used; `recommenditos/modeling/compare_segments.py` is the stage to copy from.
  Draw a figure at the size it is printed at, and check it on the built page and in a greyscale print before it goes in.
- EDN entries are written in [reports/edn.md](../edn.md), and only there, and transferred here with `make edn` (`tools/edn_latex.py`).
  It writes one `\ednentry` file per entry into `edn/` and the list of them into `edn/entries.tex`, deletes the file of an entry whose title changed, and sets each entry's **In LaTeX** to yes in the Markdown.
  It is idempotent, and only the entries edited since the last run change; never edit the generated files.
- **Rerun `make edn` after every change to reports/edn.md, in the same pull request**, and commit what it writes.
  The required Tests job runs `test_the_committed_latex_edn_is_what_the_notebook_generates` on every change to reports/edn.md or `edn/`, which fails, and blocks the merge, when a transfer would change a file; the report job runs the same check as `python3 tools/edn_latex.py --check`.
  So **In LaTeX: yes** always means the entry is in the PDF exactly as the Markdown states it, and that holds for a new entry too.
  The official fields have no separate slot for alternatives, so they go into `rationale`, ahead of the reasoning that refers to them, and the block quote above an entry's fields, where the notebook records that a later entry amended it, is printed first as `note`.
  A missing required field stops the transfer with an error naming the entry, and the build with an error naming the field.

## Before a submission

A delivery is cut from one commit, the delivery commit, and everything below is run on it; the delivery's issue (#93 for the first one) tracks the cut.
The whole list was run once as a dry run on `804ea06` (2026-10-12), so every command in it is known to work.

The cut:

- [ ] The delivery commit is chosen: after the last merge that changes the report, and before the freeze window of [CONTRIBUTING.md](../../CONTRIBUTING.md#freeze-window-before-a-presentation), which starts 48 hours before the presentation.
- [ ] The commit is tagged once every line below holds, and the tag is pushed.

Several figures in the report describe the commit it is built from, and go stale as `main` moves.
Re-measure them on the delivery commit:

- [ ] Test count, skips and coverage: `make test`, in 2.3.3. The report quotes CI's counts, and CI has no data, so a machine that ran `dvc pull` skips fewer tests: 4 against CI's 10 on `804ea06`.
- [ ] Files `ruff format --check` reports, and that `ruff check` finds nothing: `make lint`, in 2.3.2.
- [ ] Notebook lint findings: `make notebook-lint`, in 2.3.2.
- [ ] Tests carrying a `req` marker, and the matrix's counts: `uv run python -m tools.requirement_matrix` and `reports/requirement-matrix.json`, in 2.3.3.
- [ ] Number of EDN entries (written ones, not reserved headings) and the pull request counts of the coding-agent paragraph, in 2.1.4, and `make edn`, whose `--check` must pass.
- [ ] Gate result, metrics and fit times: `metrics.json`, `reports/metrics/` and `reports/figures/energy-vs-error.csv`, in the success-criteria table of 1.1, in 2.1.3, in the ladder table and its text in 2.2.4, and in 2.3.1.
- [ ] The split-noise figures of 1.1 (EDN-32), if the split or the candidate changed: `reports/analysis/split_strategy_results.txt`.
- [ ] Split and validation figures, if a data stage ran: the dataset card and `reports/data-validation/summary.json`, in 2.1.2 and 2.3.4.
- [ ] The pipeline figure, if `dvc.yaml` changed: `reports/latex/figures/pipeline.sh`.
- [ ] The three figures the pipeline draws, if a stage redrew one: `cp reports/figures/*.pdf reports/latex/figures/`. `tests/test_committed_outputs.py` fails while a copy is stale.
- [ ] The energy table of 2.3.1, if the ladder was re-run: the rows, the total and the ratios in the text from `reports/figures/energy-vs-error.csv`.
- [ ] The captions of the segment figure in 1.1 and of the composition figure in 2.1.2, if the candidate or the data changed: the numbers they state are in `reports/metrics/segments.csv` and `reports/figures/data-composition.csv`.

The commit itself, after `dvc pull`:

- [ ] The checks CI runs, locally: `make lint`, `make test`, `make test-serving`, `make notebook-lint`, and `uv run mkdocs build --strict` in `docs/`.
- [ ] CI ran what it shows as green. The `Tests`, `Tests (Windows)`, `Serving runtime` and `Build docs` jobs pass with every step skipped when the commit changes no file of their paths filter: `gh run view <run> --json jobs` lists the steps, and where they were skipped the counts come from the pull request run of the last change to the code.
- [ ] The pipeline is up to date: plain `uv run dvc status` reports nothing changed. `dvc status -c` is not this check: it only compares the cache with the remote, and it passed while six stages were stale.
- [ ] Every artefact the lock names is on the remote: `uv run dvc status -c` reports cache and remote in sync.
- [ ] The MLflow runs behind the committed models name the committed state (NFR-14): `uv run python reports/analysis/run_provenance_check.py` ends with `OK` for every variant. It only reads from the tracking server.
- [ ] NFR-06: the clean-clone drill of [The DVC pipeline](../../docs/docs/pipeline.md#running-it), repeated in a fresh `git clone` of the delivery commit without a `.env`, and its record there replaced by this run. The commit that records it changes that page alone, comes right after the commit the drill ran on, and is the one that gets the tag.
- [ ] NFR-07 and NFR-15: one person reviews the requirement matrix and the secret scan of the delivery commit's CI run and records it, as described under [The delivery review](#the-delivery-review).

The PDF and the hand-in:

- [ ] The page count against the limit, and no placeholder left: `make report-submit DELIVERABLE=initial` (or `final`), which prints both and the space left on the last page.
- [ ] The size against Atenea's 20 MB: `ls -l reports/latex/build/MLOps_Recommenditos_*_report.pdf`. The same build writes `MLOps_Recommenditos_EDN.pdf`.
- [ ] Every page break: no heading orphaned at the bottom of a page (such as 2.3 with no text under it) and no figure or table pushed away from its text.
- [ ] Every link of the cover and the text opens in a browser that is not logged in, the repository above all, which the lecturer has to be able to open: `pdfinfo -url <pdf>` lists them.
- [ ] Who submits is named in the delivery's issue, and the PDF uploaded to Atenea is the one built from the tagged commit.

### The delivery review

NFR-07 and NFR-15 each ask for a person's review before a delivery, and no tool can stand in for it.
Both read the CI run of the delivery commit, and neither job has a paths filter, so both run on every commit:

- **The requirement matrix (NFR-07):** the summary of the `Requirement matrix` job, or its `requirement-matrix` artefact (`gh run download <run> -n requirement-matrix`).
  Check that the four counts are the ones 2.3.3 of the report states and that none of the requirements with nothing is already due.
  Then judge what the job cannot: for every requirement due by the current milestone, whether the tests the matrix names really verify it, and whether the drill a **[manual]** cell names was really run on this commit and is recorded where the cell says.
- **The secret scan (NFR-15):** the summary of the `Secret scan (gitleaks)` job.
  Check that the commit it names is the delivery commit, that the number of commits scanned is plausible for the whole history and not for a shallow checkout (404 on `804ea06`), and that it says no leaks were found.
  In the job's log, the canary step has to say that it found the planted DagsHub token in every place it was planted, which is what shows that a clean scan is not a blind one.

The review is recorded as one comment on the delivery's issue: the delivery commit, the links to the CI run and to both jobs, the reviewer's name, the date, and what was found, even when that is nothing.
The reviewer writes that comment; a prepared summary of the two artefacts is not a review.

## Building with Docker (recommended)

Works the same on Linux, macOS (Intel and Apple Silicon) and Windows; the only requirement is [Docker](https://docs.docker.com/get-docker/).
Copy-paste from the **repository root**:

```bash
docker build -t mlops-latex reports/latex && docker run --rm -v "$PWD/reports/latex:/report" mlops-latex
```

On Windows, run it in PowerShell like this instead (or use the command above in WSL):

```powershell
docker build -t mlops-latex reports/latex; docker run --rm -v "${PWD}/reports/latex:/report" mlops-latex
```

The PDFs land in `reports/latex/build/`:

| File | Contents |
|------|----------|
| `report.pdf` | Initial report draft (Milestones 1-3) with guidance boxes |
| `report-final.pdf` | Final report draft (Milestones 1-6) with guidance boxes |
| `edn.pdf` | Engineering Decision Notebook draft |

For the PDFs we hand in (no guidance, named as the course requires), append `initial` or `final` to the command:

```bash
docker build -t mlops-latex reports/latex && docker run --rm -v "$PWD/reports/latex:/report" mlops-latex initial
```

This produces `MLOps_Recommenditos_Initial_report.pdf` (or `..._Final_report.pdf`) and `MLOps_Recommenditos_EDN.pdf`, and prints each PDF's page count against the limit (15 pages initial, 30 final), the number of unfilled placeholders and, for the report, the space left on its last page, which is how much the sections can still grow.

The first run downloads and installs TeX Live and takes a few minutes; after that Docker reuses the image and a build takes seconds.
The image (~240 MB) contains only a minimal, pinned TeX Live 2025, so everyone gets identical PDFs.
If a new LaTeX package is needed, add it to the `tlmgr install` list in the `Dockerfile`.

CI does not build the image on every run.
It pulls `ghcr.io/mlops-2627q1-mds-upc/mlops_recommenditos/latex:<arch>-<hash of the Dockerfile>`, and builds and pushes that tag only when it does not exist yet, which is the first CI run after the `Dockerfile` changes.
So a change to the `Dockerfile` makes the first run of its pull request slow (TeX Live comes from a single historic mirror, so the download is retried), and every run after it, on any branch, fast.

## Building without Docker

With a local TeX Live installation that has `latexmk` and `biber`, run from the repository root:

```bash
make report                                 # drafts with guidance
make report-submit                          # initial delivery
make report-submit DELIVERABLE=final        # final delivery
```

All ways of building (Docker, `make`, CI) run the same `build.sh`.
Switching between Docker and a local TeX Live is safe: `build.sh` clears `build/` when the toolchain changes.
Editors that run `latexmk` in this folder (e.g. VS Code LaTeX Workshop) pick up `.latexmkrc` and build the initial draft.

## CI

CI builds the Docker image on x86 and ARM and both submission variants on every PR that touches this folder, and attaches the PDFs as the `report-pdfs` artifact.
