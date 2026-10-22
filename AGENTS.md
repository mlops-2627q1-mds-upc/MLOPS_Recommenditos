# Agent Instructions

## Language

The course language is English, so everything we produce is in English, always.
That covers commit messages, branch names, PR titles, descriptions and comments, issues and issue comments, code, identifiers, code comments, docstrings, documentation, the report and the EDN.
Chat with the team may happen in any language, but nothing written into the repo or onto GitHub does.

The report and the EDN use the Oxford comma in every list of three or more items ("make, model, and age band"), as the course's report template does; see the Writing section of [reports/latex/README.md](reports/latex/README.md).

## Git workflow

Full workflow lives in [CONTRIBUTING.md](CONTRIBUTING.md) - read it before opening a PR.
Key point for agents: branch protection blocks direct pushes to `main`, and squash merge is the only merge method enabled on GitHub - never merge commit or rebase merge.

## Dependencies

`[project] dependencies` in `pyproject.toml` is the serving runtime the API image is built from, not a list of everything the project uses.
Before adding or moving a dependency, read the Dependencies section of [CONTRIBUTING.md](CONTRIBUTING.md): almost every new package belongs in a group (`uv add --group pipeline <package>`), and `make test-serving` checks that serving code imports nothing from one.

## Type checking

mypy checks `recommenditos/` and `tools/` in pre-commit and in the required `Lint & format` CI job (`make typecheck`, or `make lint`).
Fix a finding in the code with an explicit check or an `assert`, not with `cast` or `# type: ignore`; an unavoidable ignore names its error code and says why.
An untyped third-party import gets a per-module override in `pyproject.toml`, never a global `ignore_missing_imports`, and `pandas-stubs` stays out.
mypy cannot see pandas, so it says nothing on column names or dtypes.
Details: the Type checking section of [CONTRIBUTING.md](CONTRIBUTING.md).

## Pipeline

A change to any file a DVC stage depends on makes `dvc.lock` stale until `dvc repro` re-runs that stage, even in a pull request that is not about the pipeline: `recommenditos/config.py` and `recommenditos/schema.py` are dependencies of every stage.
What that pull request has to do, and where a new stage puts its outputs: [Working on a stage](docs/docs/pipeline.md#working-on-a-stage).

## Project brief

[docs/docs/project-brief.md](docs/docs/project-brief.md) holds the project goal, the verified data facts, the modelling and architecture plan and the open decisions.
Read it before working on data, pipeline, model or API code, and keep it up to date when any of these change.

## Keeping instructions up to date

These instructions, `CONTRIBUTING.md`, and other documented best practices are not fixed.
Extend them whenever something important comes up that they don't already cover, for example a new convention the team agrees on or a lesson learned the hard way.
Be careful not to bloat them with anything trivial or one-off, every addition should earn its place, but don't skip a genuinely useful update just to keep the file short.

## Course material

Read these before working on a milestone, the report, or the EDN:

- [references/MLOps-lab.md](references/MLOps-lab.md) - lab rules, milestones, schedule, grading rubric and points per practice.
- [references/mlops_report_template.md](references/mlops_report_template.md) - the official report structure.
- [references/Instruction_EDN_MLOps_v2026.md](references/Instruction_EDN_MLOps_v2026.md) - the Engineering Decision Notebook (EDN): which decisions to record and the fields of an entry.
- [references/course-demos.md](references/course-demos.md) - takeaways from the teachers' demo repo: expected tools and setup per milestone.

The source PDFs and DOCX files are gitignored; the Markdown files above are their conversions.

The report and the EDN are written in LaTeX in [reports/latex/](reports/latex/) (see its README), not in the DOCX template.

## Engineering decisions and the EDN

The course grades how we make engineering decisions and how we use AI in them, so every agent follows this process:

1. **Flag it.** When a task involves a decision that belongs in the EDN, say so explicitly before acting on it.
   A decision belongs in the EDN when it meaningfully affects the design, quality, development, deployment or operation of the ML system and could reasonably have gone differently.
   Using a tool the course prescribes is not a decision; how we use it can be (see the EDN instructions for examples).
2. **Lay out the options.** Present the realistic alternatives, each with its pros and cons, and give your recommendation with the reason.
   Do not pick one silently, even when one option is clearly better.
3. **Ask.** Ask the user which option they want, and whether the decision should be recorded in the EDN.
   Only continue with the chosen option.
4. **Record it.** Once decided, and if the user wants it recorded, add an entry to [reports/edn.md](reports/edn.md) following the template there, which also sets the shape and the word limit of an entry.
   An entry is short: the decision, the alternatives that were really weighed with the reason each lost, the deciding argument, who decided, and what AI contributed and what the team checked, changed or rejected.
   Link the evidence (issue, PR, commit, experiment) and do not retell it.
   When an entry has to be cut to fit the word limit, keep what it says about AI's contribution and what the team checked, and compare the cut entry with the previous one before you commit it.
   When a decision is revised, update its entry so that it states the current decision, and do not add a second entry for it.

[reports/edn.md](reports/edn.md) is the working source of the EDN.
The LaTeX EDN ([reports/latex/edn/](reports/latex/edn/)) is generated from `reports/edn.md` by `make edn`, so never write or edit an entry directly in LaTeX.
Run `make edn` in the same pull request as any change to `reports/edn.md`, new entries included; the required Tests job fails, and blocks the merge, when the LaTeX is behind.

## PDF Conversion

When converting a PDF to a readable Markdown file for agents, use the `docling` tool.
