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
| `figures/` | Figures for the report (`../figures/` is searched too). `pipeline.pdf` is drawn from `dvc.yaml` by `figures/pipeline.sh`, which needs the project environment, Graphviz and the DejaVu Sans font; the PDF is committed with the font embedded, so a build needs none of them. |
| `references.bib` | Bibliography (biblatex + biber, IEEE style). |
| `preamble.tex` | Shared packages, build flags and the `\guidance` / `\tbd` helpers. |

## Writing

- Blue **Guidance** boxes say what a section must contain: the course template's prompt, the rubric question and points, and what the repo already has.
  They disappear in submission builds, so there is no need to delete them.
- `\tbd{...}` marks a placeholder.
  It renders in red in every build, and `make report-submit` counts the ones left.
- Refer to EDN entries by ID in the report, e.g. "(EDN-03)".
- EDN entries are written in [reports/edn.md](../edn.md), and only there, and transferred here with `make edn` (`tools/edn_latex.py`).
  It writes one `\ednentry` file per entry into `edn/` and the list of them into `edn/entries.tex`, deletes the file of an entry whose title changed, and sets each entry's **In LaTeX** to yes in the Markdown.
  It is idempotent, and only the entries edited since the last run change; never edit the generated files.
- **Rerun `make edn` after every change to reports/edn.md, in the same pull request**, and commit what it writes.
  The required Tests job runs `test_the_committed_latex_edn_is_what_the_notebook_generates` on every change to reports/edn.md or `edn/`, which fails, and blocks the merge, when a transfer would change a file; the report job runs the same check as `python3 tools/edn_latex.py --check`.
  So **In LaTeX: yes** always means the entry is in the PDF exactly as the Markdown states it, and that holds for a new entry too.
  The official fields have no separate slot for alternatives, so they go into `rationale`, ahead of the reasoning that refers to them, and the block quote above an entry's fields, where the notebook records that a later entry amended it, is printed first as `note`.
  A missing required field stops the transfer with an error naming the entry, and the build with an error naming the field.

## Before a submission

Several figures in the report describe the commit it is built from, and go stale as `main` moves.
Re-measure them on the delivery commit, after the last merge that goes into it:

- [ ] Test count, skips and coverage: `make test`, in 2.3.3.
- [ ] Files `ruff format --check` reports, and that `ruff check` finds nothing: `make lint`, in 2.3.2.
- [ ] Notebook lint findings: `make notebook-lint`, in 2.3.2.
- [ ] Tests carrying a `req` marker, and the matrix's counts: `uv run python -m tools.requirement_matrix` and `reports/requirement-matrix.json`, in 2.1.1 and 2.3.3.
- [ ] Number of EDN entries, in 2.1.4, and `make edn`, whose `--check` must pass.
- [ ] Gate result, metrics and fit times: `metrics.json`, `reports/metrics/` and the MLflow runs they name, in 1.1, 2.1.3, Table 1 of 2.2.4 and 2.3.1, and the commit those runs carry in 2.2.4.
- [ ] Split and validation figures, if a data stage ran: the dataset card and `reports/data-validation/summary.json`, in 2.1.2 and 2.3.4.
- [ ] The pipeline figure, if `dvc.yaml` changed: `reports/latex/figures/pipeline.sh`.
- [ ] The page count against the limit, and no placeholder left: `make report-submit`, which prints both and the space left on the last page.
- [ ] Once the DRAFT boxes are gone, every page break: no heading orphaned at the bottom of a page (such as 2.3 with no text under it) and no figure or table pushed away from its text.

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
