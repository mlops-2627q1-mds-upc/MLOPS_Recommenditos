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
| `edn/` | One EDN entry per file, copied from `edn/_template.tex`. |
| `figures/` | Figures for the report (`../figures/` is searched too). |
| `references.bib` | Bibliography (biblatex + biber, IEEE style). |
| `preamble.tex` | Shared packages, build flags and the `\guidance` / `\tbd` helpers. |

## Writing

- Blue **Guidance** boxes say what a section must contain: the course template's prompt, the rubric question and points, and what the repo already has.
  They disappear in submission builds, so there is no need to delete them.
- `\tbd{...}` marks a placeholder.
  It renders in red in every build, and `make report-submit` counts the ones left.
- Refer to EDN entries by ID in the report, e.g. "(EDN-03)".
- EDN entries are written first in [reports/edn.md](../edn.md) and only transferred here before a delivery.
  To transfer one, copy `edn/_template.tex` to `edn/NN-short-slug.tex`, fill it in from the Markdown entry and add `\input{edn/NN-short-slug}` at the end of `edn.tex`.
  The official fields have no separate slot for alternatives, so summarise them with their pros and cons in `rationale`.
  Missing required fields stop the build with an error naming the field.

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

This produces `MLOps_Recommenditos_Initial_report.pdf` (or `..._Final_report.pdf`) and `MLOps_Recommenditos_EDN.pdf`, and prints each PDF's page count against the limit (15 pages initial, 30 final) and the number of unfilled placeholders.

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
