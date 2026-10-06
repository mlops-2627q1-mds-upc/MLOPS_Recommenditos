Sprint log
==========

Record of each sprint's goal and outcome.

| Sprint | Dates | Goal | Outcome |
| ------ | ----- | ---- | ------- |
| 1      | 2026-09-22 - 2026-09-29 | M1 Inception setup: stand up team coordination (Discord), lock the dataset and modelling direction, and get DVC, requirements, and the model/dataset cards underway. | Met. 16 pull requests of sprint 1 scope merged, 7 issues closed. Dataset, modelling direction and drift scenario locked; DVC and the DagsHub remote in place with the raw file tracked; dataset card, model card, problem specification and requirements written. The model card (#5, PR #30) landed on the last day, minutes after the review was written, so nothing was carried into sprint 2 but the Discord ticket (#15), which is set up and still open. |
| 2      | 2026-09-29 - 2026-10-06 | From documents to a running pipeline: land the contract every stage builds on, then fill the DVC stages in parallel, so the first report has measured numbers to cite. | In progress |

## Sprint 1 planning notes (2026-09-22)

**Attendees:** @lukas2510, @kadameit, @W11W11W11, @michudud04, @ulasawczuk

**Decisions made:**

- Main dataset: AutoScout24 Car Listings Dataset (2025 snapshot). See [project brief §3.1](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/docs/docs/project-brief.md#31-main-dataset-decided) and [EDN-01](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md).
- Model family: gradient boosting, with LightGBM as the main model. See [project brief §4](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/docs/docs/project-brief.md#4-modelling-plan) and [EDN-02](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md).

**Backlog assigned this sprint:**

| Issue | Title | Assignee |
| ----- | ----- | -------- |
| [#2](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/2) | Define ML problem specification | @lukas2510 |
| [#3](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/3) | Select and acquire dataset | @W11W11W11 |
| [#4](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/4) | Write Dataset Card | @michudud04 |
| [#5](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/5) | Write Model Card (initial draft) | @ulasawczuk |
| [#10](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/10) | Set up DVC for data versioning | @W11W11W11 |
| [#14](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/14) | Define functional and non-functional requirements | @lukas2510, @kadameit |
| [#15](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/15) | Set up Discord server as team collaboration space | @kadameit |

**PRs opened today:**

- [#11](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/11) docs: add DVC data versioning conventions (merged)
- [#12](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/12) docs: add LaTeX report and EDN templates with Docker build (merged)
- [#13](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/13) docs: add project brief with verified data facts and open decisions (open)

## Sprint 1 review (2026-09-29)

**Done and merged:** project structure, CI, branch protection and the squash-merge policy; the DVC
setup with the DagsHub remote (#26) and the raw AutoScout24 file tracked (#27); the dataset card
(#22); the problem specification (#17); the project brief (#13); the LaTeX report and EDN templates
(#12); ruff's Pylint rules and coverage (#23); the data-fact corrections found while re-profiling
(#24); the DagsHub repository decisions (#28); English as the repository language (#29); and the
requirements with their separate specification (#19), merged on the last day of the sprint.

**Carried into sprint 2:** only the model card (#5, PR #30), still in review at the end of the
sprint. The Discord server (#15) is set up but its issue is still open.

**Decisions recorded:** 27 EDN entries on `main`, each with its alternatives, its rationale and a
record of how AI was involved.

## Sprint 2 planning notes (2026-09-29)

**Attendees:** @lukas2510, @kadameit, @W11W11W11, @michudud04, @ulasawczuk

**Sprint goal:** a `dvc repro` that runs end to end, with the first experiments tracked in MLflow
and the quality gate measurable. The first report covering milestones 1 to 3 is due 2026-10-19 and
needs numbers that only a running pipeline produces.

**How the work is cut.** The pipeline is sequential, so five people can only work on it in parallel
once the contracts between the stages exist. #32 lands those first: every DVC stage defined and
running as a stub, `params.yaml`, a `schema.py` that fixes the processed columns, and a synthetic
fixture. After that everyone codes against the schema and the fixture instead of against each
other's branches. One owner per file, dependencies stated on contracts rather than on people.

**Backlog:**

| Issue | Title | Assignee |
| ----- | ----- | -------- |
| [#31](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/31) | Clear the open PR backlog and land the M1 documents | @lukas2510 |
| [#32](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/32) | Pipeline skeleton and the processed-data contract | @lukas2510 |
| [#33](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/33) | Download stage: import the raw dataset from Zenodo | @W11W11W11 |
| [#34](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/34) | Preprocess stage: scope, deduplicate, remove PII, build the target | @W11W11W11 |
| [#25](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/25) | Great Expectations on raw and processed data | @michudud04 |
| [#35](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/35) | Split stage: hold out Spain, then split grouped by seller | @kadameit |
| [#36](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/36) | Features stage: basic and extended feature sets | @kadameit |
| [#37](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/37) | Train stage: the experiment ladder with MLflow on DagsHub | @ulasawczuk |
| [#38](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/38) | CodeCarbon: emissions per run, energy against error | @ulasawczuk |
| [#39](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/39) | Evaluate stage: metrics, segments and the SC gate | @lukas2510 |
| [#40](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/40) | Requirement markers and the traceability matrix in CI | @lukas2510 |
| [#41](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/41) | Pynblint and notebook quality in CI | @lukas2510 |
| [#42](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/42) | DagsHub access for the whole team | @lukas2510 |
| [#43](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/43) | Sprint process: board, labels, milestones, agent rule | @lukas2510 |
| [#44](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/44) | First report (M1-M3): one section per owner | everyone |

#40, #41 and #42 depend on nothing and can start immediately. Everything else on the pipeline waits
for #32 only. #44 sits in the backlog column: writing starts 2026-10-08 at the latest, in sprint 3.

**Decisions taken in planning:**

- **The project structure follows the teachers' demo:** one module per DVC stage
  (`recommenditos/data/{download_raw_dataset,preprocess,validate_data,split_data,build_features}.py`
  and `recommenditos/modeling/{train,evaluate}.py`) rather than the flat Cookiecutter layout. It is
  the structure the course demonstrates, it makes the `deps` in `dvc.yaml` precise, and it gives
  every ticket its own file. The EDN entry for this deviation is still to be written.
- **Four deliberate additions beyond the demo**, each to be defended in the report: `params.yaml`
  instead of configuration in `config.py`; an `evaluate` stage writing `metrics.json` instead of a
  threshold hidden in a test; a `features` stage, which the demo's text model does not need; and a
  synthetic fixture instead of a data sample, because the raw file carries PII we may not re-host.
- **NFR-11 restated** at a window of 1,000 requests with `model` excluded from the drift comparison,
  after measuring that neither change works on its own (EDN-26).
- **A working rule for coding agents**, written into the [working agreements](working-agreements.md).
  Its EDN entry follows before the first delivery.

**Known risks carried into the sprint:**

- Nobody has access to the FIB Virtech VM yet. If that is still true at the M4a lab, we raise it
  with the teachers instead of planning further on it.
- The committed analysis outputs in `reports/analysis/` no longer reproduce now that #24 dropped the
  164 listings registered after the snapshot. They are cited as EDN evidence, so both scripts need
  one re-run against the new scope.
- The first delivery on 2026-10-19 falls after sprint 2. Nothing in sprint 2 may slip past
  2026-10-08 without the report losing the numbers it is meant to cite.

## Sprint 2 progress (2026-09-30)

Recorded on day 2 of the sprint from the merged pull requests and the closed issues, so the sprint 2 review has a trail to read rather than a week to reconstruct.

**Landed since planning:**

- The model card ([#5](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/5), [PR #30](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/30)) merged on 2026-09-29, minutes after the sprint 1 review was written.
  That emptied the pull request backlog and closed [#31](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/31), so sprint 1 carried nothing into sprint 2.
- The project structure deviation is recorded as EDN-28 ([PR #46](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/46)), which the planning notes above still list as outstanding.
- The committed analysis outputs in `reports/analysis/` reproduce again and NFR-11's false-alarm bound is corrected as EDN-29 ([PR #47](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/47)).
  That closes the second risk the sprint carried in.
- The pipeline skeleton and the processed-data contract ([#32](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/32), [PR #48](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/48)) merged on 2026-09-30: all eight DVC stages defined and running as stubs, `params.yaml`, `recommenditos/schema.py`, the synthetic fixture, and EDN-30 to EDN-34 for the decisions behind them.

**What that unblocks.** [#33](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/33) to [#39](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/39) and [#25](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/25) each waited on #32 and on nothing else, so every pipeline ticket can now be built in parallel against the schema and the fixture instead of against another branch.
The board and its labels are in place as of this sprint, so the work can be filtered per milestone and per area.

**Still open from planning.** Nobody has access to the FIB Virtech VM; if that holds at the M4a session it is raised with the teachers.
The 2026-10-08 cut-off for pipeline work stands, because the first report has to cite measured numbers.

**Sprint boundary.** The sprint runs to the laboratory session on Tuesday 2026-10-06, so that Planning, Review and Retrospective happen where the whole team is present, and sprint 3 starts at that session, as the [working agreements](working-agreements.md) describe.

## Retrospective notes

### Sprint 1

Drafted from the sprint's record on 2026-09-29 and extended on 2026-09-30, to be confirmed by the team in the laboratory session.

- **What went well:**
    - The scope was locked before any code was written: dataset, modelling family, supported makes
      and the drift scenario are all decided and justified.
    - Claims were measured rather than assumed. The fill rates, the drift feasibility and the
      dependency resolution under Python 3.12 were each checked against the real data or the real
      package set before they were written down.
    - Decisions were recorded as they were made, with their alternatives and rationale, instead of
      being reconstructed for the report afterwards.
    - The engineering guard rails were in place in week one: branch protection, squash-merge only,
      conventional commits, pre-commit hooks and CI.
- **What to improve:**
    - Seven pull requests were open at the same time and blocked each other, two of them with merge
      conflicts. Nothing could be built on the data contract while its pull request was open.
    - EDN numbers were handed out on parallel branches without reserving them first, so EDN-19 was
      assigned twice and the requirements split had to be renumbered to EDN-27 while merging.
    - Requirements were written ahead of the code they describe. Two of them referenced a pointer
      file that was never created, and one prescribed a mechanism the team later reversed.
    - Committed analysis outputs were treated as settled, but a later scope decision changed the
      data under them. They are cited as EDN evidence, so they stopped reproducing.
    - The work was very unevenly distributed: almost every commit on `main` came from one person.
      The laboratory grade carries an individual factor based on contributions, so this matters
      beyond fairness.
    - The board had no milestone labels, no area labels and no GitHub milestones until the last day
      of the sprint, and six tickets were still closed without any of them. The trace the course
      grades had to be assembled from the git history instead of being read off the board.
- **Action items:**
    - Cut sprint 2 so that each ticket owns its own files, and land the shared contracts first
      (#32).
    - Reserve EDN numbers in `reports/edn.md` on `main` before opening a branch that needs one.
    - Re-run the committed analysis outputs whenever a decision changes the data scope.
    - Every teammate owns at least one pipeline stage and writes the report section for it.
    - Give every ticket its milestone label, area label, GitHub milestone and owner when it is
      written, not when the sprint is reviewed.
