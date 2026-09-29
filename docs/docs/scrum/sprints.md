Sprint log
==========

Record of each sprint's goal and outcome.

| Sprint | Dates | Goal | Outcome |
| ------ | ----- | ---- | ------- |
| 1      | 2026-09-22 - 2026-09-29 | M1 Inception setup: stand up team coordination (Discord), lock the dataset and modelling direction, and get DVC, requirements, and the model/dataset cards underway. | Met, except the model card. 14 pull requests merged, 6 issues closed. Dataset, modelling direction and drift scenario locked; DVC and the DagsHub remote in place with the raw file tracked; dataset card, problem specification and requirements written. Model card (#30) and requirements (#19) carried into sprint 2. |
| 2      | 2026-09-29 - 2026-10-06 | From documents to a running pipeline: land the contract every stage builds on, then fill the DVC stages in parallel, so the first report has measured numbers to cite. | In progress |

## Sprint 1 planning notes (2026-09-22)

**Attendees:** @lukas2510, @kadameit, @W11W11W11, @michudud04, @ulasawczuk

**Decisions made:**

- Main dataset: AutoScout24 Car Listings Dataset (2025 snapshot). See [project brief §3.1](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/docs/docs/project-brief.md#31-main-dataset-decided) and [EDN-01](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md).
- Model family: gradient boosting, with LightGBM as the main model. See [project brief §4](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/docs/docs/project-brief.md#4-modelling-plan-planned) and [EDN-02](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md).

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
(#24); the DagsHub repository decisions (#28); English as the repository language (#29).

**Carried into sprint 2:** the model card (#5, PR #30) and the requirements and specification
(#14, PR #19), both still in review at the end of the sprint. The Discord server (#15) is set up
but its issue is still open.

**Decisions recorded:** 13 EDN entries on `main`, 14 more waiting in PR #19, each with its
alternatives, its rationale and a record of how AI was involved.

## Sprint 2 planning notes (2026-09-29)

**Attendees:** @lukas2510, @kadameit, @W11W11W11, @michudud04, @ulasawczuk

**Sprint goal:** a `dvc repro` that runs end to end, with the first experiments tracked in MLflow
and the quality gate measurable. The first report covering milestones 1 to 3 is due 2026-10-13 and
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
| [#41](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/41) | Pynblint and notebook quality in CI | @michudud04 |
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
  after measuring that neither change works on its own (EDN-26, in PR #19).
- **A working rule for coding agents**, written into the [working agreements](working-agreements.md).
  Its EDN entry follows before the first delivery.

**Known risks carried into the sprint:**

- Nobody has access to the FIB Virtech VM yet. If that is still true at the M4a lab on 2026-10-21,
  we raise it with the teachers instead of planning further on it.
- EDN-19 was assigned twice on parallel branches, to the requirements split and to the DagsHub
  repository decision. One of them has to be renumbered before both land.
- The committed analysis outputs in `reports/analysis/` no longer reproduce now that #24 dropped the
  164 listings registered after the snapshot. They are cited as EDN evidence, so both scripts need
  one re-run against the new scope.
- The first delivery on 2026-10-13 falls into sprint 3. Nothing in sprint 2 may slip past
  2026-10-08 without the report losing the numbers it is meant to cite.

## Retrospective notes

### Sprint 1

Drafted from the sprint's record on 2026-09-29, to be confirmed and extended by the team.

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
    - EDN numbers were handed out on parallel branches without reserving them first, so EDN-19
      exists twice.
    - Requirements were written ahead of the code they describe. Two of them referenced a pointer
      file that was never created, and one prescribed a mechanism the team later reversed.
    - Committed analysis outputs were treated as settled, but a later scope decision changed the
      data under them. They are cited as EDN evidence, so they stopped reproducing.
    - The work was very unevenly distributed: almost every commit on `main` came from one person.
      The laboratory grade carries an individual factor based on contributions, so this matters
      beyond fairness.
- **Action items:**
    - Cut sprint 2 so that each ticket owns its own files, and land the shared contracts first
      (#32).
    - Reserve EDN numbers in `reports/edn.md` on `main` before opening a branch that needs one.
    - Re-run the committed analysis outputs whenever a decision changes the data scope.
    - Every teammate owns at least one pipeline stage and writes the report section for it.
