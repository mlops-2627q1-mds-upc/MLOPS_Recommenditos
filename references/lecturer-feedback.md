# Lecturer feedback

What the lecturers have told us about our work, kept so that whoever writes the final report knows what they look for.
Read it before writing any section of the report.
The comments come first as they were written, then what follows from them.

## What the final report is graded on

After the feedback on the first report, the lecturer told the team that improving the chapters of Milestones 1 to 3 earns nothing, because the final report is graded on its new part only.
So the effort and the pages of the final report go to Milestones 4 to 6 and the retrospective.
The sections of Milestones 1 to 3 stay as submitted, except where a fact they state has become wrong.

## Comments on the first report (2026-11-02)

Santiago del Rey annotated the PDF of the first report (Milestones 1 to 3, submitted 2026-10-19).
The comments, verbatim, by the section they sit next to:

| Section | Comment |
|---------|---------|
| 1 Introduction | "Section 1 should start on a new page." |
| 2.1.1 Selection of problem and requirements | "The goal is clear and the requirements list is far more detailed than I expected, with priorities, acceptance criteria and the requirement-to-test matrix that runs in CI. I liked that you say which requirements still have no evidence." |
| 2.1.2 Dataset card | "Excellent dataset card, with machine-readable metadata, limitations, and figures that match the pipeline outputs (118,382 rows, 75 columns, 105,405 after cleaning)." |
| 2.1.3 Model card | "Very complete card with bias, risks and fairness sections." |
| 2.1.4 Project coordination and communication | "Excellent use of GitHub Issues with milestones, labels and pull requests that close their issues." |
| 2.1.4 Project coordination and communication | "Discord is named in the report and in the README. I found no evidence of how the team used it." |
| 2.2.1 Project structure | "Good adaptation of the template with a descriptive README. The the README still calls LightGBM the planned model." |
| 2.2.2 Code versioning | "Very good application of the GitHub Flow with added extras like the pre-commit hooks and CI." |
| 2.2.3 Data versioning | "Excellent. I appreciate the simplification of the DVC pipeline in Fig. 3 which makes it easy to understand compared to DagsHub's DAG. The synthetic-data option for a clean clone is a nice touch." |
| 2.2.4 Experiment tracking | "Excellent experimemtation setup. I like your comparison of the four variants and how you argure the choice of lgbm-basic with SC-06 and the energy cost, which is what I wanted to see." |
| 2.3.1 Energy efficiency awareness | "Good job with the measurement configuration. I missed a comparison with common activities that puts these grams in a wider context." |

Sections 2.3.2 to 2.3.4 (static analysis, model testing, data testing), the teammates' evaluation, and the EDN appendix got no comment.
Three comments name a gap, on coordination, project structure, and energy, and each is the only criticism of its practice.

## What follows for the final report

Each lesson is in the guidance box of the sections it applies to, so the writer of a section sees it there.

- **Show a tool in use, not only its name.**
  Discord was named in the report and the README but never shown in use.
  Every tool the final report names (Prometheus, Grafana, Better Uptime, Alibi Detect, GitHub Actions) needs evidence the reader can see or open: a screenshot of the dashboard on the VM, an alert that fired, the uptime monitor's record, a link to a CI or deploy run, the drift job's output on the Spanish replay.
- **Put every number into a context the reader knows.**
  The energy section measured carefully, but left its grams without a comparison.
  State a measured value against its budget, and for energy and emissions also against an everyday activity.
  For example, the 0.191 Wh of the whole experiment ladder is what a 10 W LED bulb uses in about 70 seconds.
  The serving energy per answer that 2.3.1 of the first report promised for Milestone 4 is the first place for it; latency, memory, and image size are read against their NFR budgets.
- **Keep every document the lecturer opens in line with the code.**
  The lecturer reads the README and the documentation site when grading, and a stale sentence in the README was the one criticism of a project structure that was otherwise praised.
  Before a delivery, read the README against the repository.
  `tests/test_readme.py` checks that its project tree names every top-level directory, every package, and every pipeline stage's module; the prose it cannot check.
- **Start Section 1 on a new page.**
  `report.tex` now does that.

## What the lecturer praised

Keep doing these in the new sections:

- Requirements with priorities, acceptance criteria, and a requirement-to-test matrix that CI enforces, and saying openly which requirements still have no evidence.
- Figures and numbers that match the pipeline's outputs.
- Bias, risks, and fairness in the model card.
- GitHub Issues with milestones and labels, and pull requests that close their issues.
- Extras beyond the course demo: pre-commit hooks and CI around GitHub Flow, a synthetic dataset so a clean clone runs.
- A diagram drawn for the reader instead of a tool's generated one (Fig. 3 of the first report against DagsHub's DAG).
- A choice argued from measured criteria and their cost: lgbm-basic over lgbm-extended because of SC-06 and the energy, "which is what I wanted to see".

## Earlier feedback

Feedback that came before the first report, and where it is applied now:

- **2026-09-29, Milestone 1:** our requirements read as a specification, because they named endpoints, status codes, and container mechanics.
  We split them into requirements and a specification (EDN-27).
- **2026-09-22, Milestone 1:** the supervisor asked whether the DataMarket sample is feasible as the new-market scenario.
  It is not, and the Spanish listings are held out instead (EDN-03).
- **2026-10-14, before the first delivery:** section 2.1.5, the selection of the cloud provider, is a leftover of the old course template and belongs to Milestone 4.
  It moved to 2.4.1 (#122), and `references/mlops_report_template.md` still lists it under Milestone 1.
- **2026-10-27:** the AI interaction evidence of an EDN entry quotes the prompt that led to the decision, word for word or its core marked "(paraphrased)".
  The rule is in `AGENTS.md` and in the template of `reports/edn.md`; entries recorded before that date need not be adapted.
