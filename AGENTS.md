# Agent Instructions

## Git workflow

Full workflow lives in [CONTRIBUTING.md](CONTRIBUTING.md) - read it before opening a PR.
Key point for agents: branch protection blocks direct pushes to `main`, and squash merge is the only merge method enabled on GitHub - never merge commit or rebase merge.

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
4. **Record it.** Once decided, and if the user wants it recorded, add an entry to [reports/edn.md](reports/edn.md) following the template there.
   Capture everything needed to justify the decision later: the alternatives with their pros and cons, why the chosen one won, who decided, and exactly how AI was involved and how the team responded to it.
   Link the evidence (issue, PR, commit, experiment).

[reports/edn.md](reports/edn.md) is the working source of the EDN.
Entries are only transferred to the LaTeX EDN ([reports/latex/edn/](reports/latex/edn/)) before a delivery, so never write new entries directly in LaTeX.

## PDF Conversion

When converting a PDF to a readable Markdown file for agents, use the `docling` tool.
