Sprint log
==========

Record of each sprint's goal and outcome.

| Sprint | Dates | Goal | Outcome |
| ------ | ----- | ---- | ------- |
| 1      | 2026-09-22 - 2026-09-29 | M1 Inception setup: stand up team coordination (Discord), lock the dataset and modelling direction, and get DVC, requirements, and the model/dataset cards underway. | Met. 16 pull requests of sprint 1 scope merged, 7 issues closed. Dataset, modelling direction and drift scenario locked; DVC and the DagsHub remote in place with the raw file tracked; dataset card, model card, problem specification and requirements written. The model card (#5, PR #30) landed on the last day, minutes after the review was written, so nothing was carried into sprint 2 but the Discord ticket (#15), which is set up and still open. |
| 2      | 2026-09-29 - 2026-10-06 | From documents to a running pipeline: land the contract every stage builds on, then fill the DVC stages in parallel, so the first report has measured numbers to cite. | Met. The pipeline runs end to end on the real Zenodo snapshot with every stage implemented, tracked in MLflow on DagsHub and gated by SC-01 to SC-06, and CodeCarbon measures every training fit. 29 pull requests merged and 26 issues closed after planning. Hyperparameter tuning (#88) and the split strategy (#89) were not started and move into sprint 3. |
| 3 | 2026-10-06 - 2026-10-13 | Fine-tune the model and measure its carbon emissions, so the numbers the first report quotes are final before the delivery on 2026-10-19. | Met. 16 pull requests merged and 7 issues closed after planning. The hyperparameter search (#88) and the split review (#89) kept the committed values, and `lgbm-basic` became the candidate after training on blanked copies (#116, EDN-82). The carbon emissions are measured for every training fit and for the whole search, and NFR-10 now separates a search from the final fit (EDN-73); that work landed inside #88, because the planned ticket of its own was never opened. The candidate changed on the last day, so the report's numbers were final only at the end of the sprint. |
| 4 | 2026-10-13 - 2026-10-20 | Finalise and submit the first report and prepare the M1-M3 presentation. | In progress |

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

> **Dates corrected on 2026-10-06** ([PR #97](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/97)). These notes were written against the course PDF's dates read as 2026 dates: the first report due 2026-10-13 and the M4a session on 2026-10-21, a Wednesday.
> The first report is due 2026-10-19 at 23:59, the laboratory meets on Tuesdays, and the M4a session's date is not known yet.
> Sprint 2 ends at the laboratory session on 2026-10-06.
> The 2026-10-08 cut-off was set against 2026-10-13; the sprint 3 planning in [#94](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/94) sets the new one.

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
  every ticket its own file. The deviation is recorded in EDN-28.
- **Four deliberate additions beyond the demo**, each to be defended in the report: `params.yaml`
  instead of configuration in `config.py`; an `evaluate` stage writing `metrics.json` instead of a
  threshold hidden in a test; a `features` stage, which the demo's text model does not need; and a
  synthetic fixture instead of a data sample, because the raw file carries PII we may not re-host.
- **NFR-11 restated** at a window of 1,000 requests with `model` excluded from the drift comparison,
  after measuring that neither change works on its own (EDN-14).
- **A working rule for coding agents**, written into the [working agreements](working-agreements.md).
  The decision and how the rule has been practised are recorded in [EDN-81](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md).

**Known risks carried into the sprint:**

- Nobody has access to the FIB Virtech VM yet. If that is still true at the M4a lab on 2026-10-21,
  we raise it with the teachers instead of planning further on it.
- The committed analysis outputs in `reports/analysis/` no longer reproduce now that #24 dropped the
  164 listings registered after the snapshot. They are cited as EDN evidence, so both scripts need
  one re-run against the new scope.
- The first delivery on 2026-10-13 falls into sprint 3. Nothing in sprint 2 may slip past
  2026-10-08 without the report losing the numbers it is meant to cite.

## Sprint 2 progress (2026-09-30)

> **Dates corrected on 2026-10-06** ([PR #97](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/97)). This entry was written against the course PDF's dates read as 2026 dates: the first report due 2026-10-13 and the M4a session on 2026-10-21, a Wednesday.
> The first report is due 2026-10-19 at 23:59, and the M4a session's date is not known yet.
> The laboratory meets on Tuesdays, so the sprint boundary below does not move: sprint 2 ends at the laboratory session on 2026-10-06, and sprints stay Tuesday to Tuesday.
> The 2026-10-08 cut-off was set against 2026-10-13; the sprint 3 planning in [#94](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/94) sets the new one.

Recorded on day 2 of the sprint from the merged pull requests and the closed issues, so the sprint 2 review has a trail to read rather than a week to reconstruct.

**Landed since planning:**

- The model card ([#5](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/5), [PR #30](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/30)) merged on 2026-09-29, minutes after the sprint 1 review was written.
  That emptied the pull request backlog and closed [#31](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/31), so sprint 1 carried nothing into sprint 2.
- The project structure deviation is recorded as EDN-28 ([PR #46](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/46)), which the planning notes above still list as outstanding.
- The committed analysis outputs in `reports/analysis/` reproduce again and NFR-11's false-alarm bound is corrected as EDN-14 ([PR #47](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/47)).
  That closes the second risk the sprint carried in.
- The pipeline skeleton and the processed-data contract ([#32](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/32), [PR #48](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/48)) merged on 2026-09-30: all eight DVC stages defined and running as stubs, `params.yaml`, `recommenditos/schema.py`, the synthetic fixture, and EDN-30 to EDN-20 for the decisions behind them.

**What that unblocks.** [#33](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/33) to [#39](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/39) and [#25](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/25) each waited on #32 and on nothing else, so every pipeline ticket can now be built in parallel against the schema and the fixture instead of against another branch.
The board and its labels are in place as of this sprint, so the work can be filtered per milestone and per area.

**Still open from planning.** Nobody has access to the FIB Virtech VM; if that holds at the M4a session on 2026-10-21 it is raised with the teachers.
The 2026-10-08 cut-off for pipeline work stands, because the first report is due 2026-10-13 and has to cite measured numbers.

**Sprint boundary.** The sprint runs to the laboratory session on 2026-10-07 rather than to Tuesday 2026-10-06, so that Planning, Review and Retrospective happen where the whole team is present, and sprint 3 starts at that session.
This is the change the [working agreements](working-agreements.md) now describe, and it is **[proposed]** until the team confirms it in the laboratory.

## Sprint 2 review (2026-10-06)

Written from the merged pull requests and the closed issues, confirmed by the team at the laboratory session of 2026-10-13.

**Done and merged:**

- The pipeline skeleton and the processed-data contract ([#32](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/32)), then every stage on top of it: download ([#33](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/33)), preprocess ([#34](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/34)), split ([#35](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/35)), features ([#36](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/36)), train ([#37](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/37)) and evaluate ([#39](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/39)).
- Great Expectations suites on the raw and the cleaned frame that fail the pipeline on a broken rule ([#25](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/25)).
- The pipeline on the real Zenodo snapshot, with the artefacts pushed to DagsHub ([#57](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/57), [#59](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/59)).
- CodeCarbon around every training fit, with the model ladder compared by energy against error ([#38](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/38)).
- CI: requirement markers and the traceability matrix ([#40](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/40)), Pynblint ([#41](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/41)), gitleaks on every commit, grouped dependencies so the API image does not carry the training stack ([#61](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/61)), and the test suite on Windows ([#84](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/84), EDN-75).
- Early stopping now decides the LightGBM tree count under a 5,000-tree ceiling ([#64](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/64)), and every run's provenance tags identify a committed state ([#79](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/79)).
- DagsHub access, the board, its labels and milestones, and the coding-agent rule ([#42](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/42), [#43](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/43)).
- The report's TeX Live image pulled from GHCR with a retry on the mirror ([PR #90](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/90)).
- The course dates and the Tuesday lab corrected across the documents ([PR #97](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/97)).

**Not done, moved into sprint 3:**

- [#88](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/88): tune the LightGBM hyperparameters with a small recorded grid search.
  [#64](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/64) closed with its "then tune the rest" part open, so the model card still calls the hyperparameters untuned.
- [#89](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/89): reconsider the split strategy.
- [#44](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/44): the first report, whose writing was planned for sprint 3 from the start.
- Open pull requests that carry over: [#86](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/86) (SC-04's accepted limitation) and [#98](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/98) (the interim suite's filled columns).

**Risks from planning.**
The committed analysis outputs reproduce again (EDN-14).
Nobody having access to the FIB Virtech VM was not resolved, and the M4a session date is still to be confirmed.

## Sprint 3 planning notes (2026-10-06)

**Sprint goal:** fine-tune the model and measure its carbon emissions, so that the numbers the first report quotes are final before the delivery.
The first report is due on 2026-10-19 at 23:59 and the M1-M3 presentation is on 2026-10-20.

**Focus:**

- **Fine-tuning.**
  The grid search of [#88](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/88), with the method decision recorded in the EDN before the search runs, and the split strategy of [#89](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/89).
  Both move every number the report quotes, so they land first.
- **Carbon emissions.**
  Build on the CodeCarbon measurement of [#38](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/38).
  The ticket for this scope is still to be written, with an owner and the oracle that says when it is done.

**Backlog:**

| Issue | Title | Assignee |
| ----- | ----- | -------- |
| [#88](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/88) | Tune the LightGBM hyperparameters with a small recorded grid search | @kadameit |
| [#89](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/89) | Reconsider the split strategy and choose one or more with reasons | unassigned |
| [#96](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/96) | Test every column the interim suite keeps filled, not only make | @kadameit |
| [#87](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/87) | The report build rebuilds TeX Live from a slow mirror whenever its layer cache is evicted | unassigned |
| [#95](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/95) | Record the two EDN entries promised for the first delivery | @lukas2510 |
| [#94](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/94) | Close sprint 2 and plan sprint 3 around the new delivery dates | @lukas2510 |
| [#44](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/44) | First report (M1-M3): write the sections, one owner per file | @lukas2510 |
| [#93](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/93) | Cut and submit the first delivery by 2026-10-19, 23:59 | @lukas2510 |
| [#92](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/92) | Prepare the M1-M3 presentation for 2026-10-20 | @lukas2510 |

**Dates:**

- The sprint runs Tuesday to Tuesday, from 2026-10-06 to 2026-10-13.
- **[proposed]** Pipeline work stops on 2026-10-13, the end of the sprint, so the report is finalised on numbers that no longer change.
  The earlier cut-off of 2026-10-08 no longer applies, because it was set for a delivery on 2026-10-13.
- The freeze window of [CONTRIBUTING.md](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/CONTRIBUTING.md) starts on 2026-10-18, 48 hours before the presentation.

## Sprint 3 review (2026-10-13)

Written on 2026-10-17 from the merged pull requests and the closed issues, and confirmed by the team at the laboratory session of 2026-10-13.

**Done and merged:**

- Hyperparameter tuning: a tracked grid search with its energy measured kept the committed `learning_rate`, `num_leaves` and `min_child_samples` ([#88](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/88), [PR #101](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/101), EDN-73), after its EDN number was reserved on `main` ([PR #100](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/100)).
- The split strategy: the seller-grouped hash split stays, and its figures are reported with their spread over 20 other splits ([#89](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/89), [PR #107](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/107), EDN-32).
- Behavioural tests of the trained models: direction and invariance ([#103](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/103), [PR #106](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/106)), then sorted into the three test types with the missing cases added ([PR #120](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/120)).
- `lgbm-basic` is trained on blanked copies of its listings and is the candidate ([#116](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/116), [PR #121](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/121), EDN-82), and SC-04's miss on cars over 20 years is recorded as an accepted limitation ([PR #86](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/86), EDN-62).
- The EDN: the project choice and the coding-agent rule ([#95](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/95), [PR #105](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/105), EDN-80, EDN-81) with their IDs reserved first ([PR #102](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/102)), the trim to a selected set of short entries with the rest archived ([#110](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/110), [PR #112](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/112)), and the team's confirmation of EDN-19, EDN-20 and EDN-25 ([PR #115](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/115)).
- The first report as a complete draft with measured numbers ([PR #104](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/104)), updated after the pipeline rerun and the behavioural tests ([PR #111](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/111)).
- The DVC pipeline back in sync with its code ([PR #109](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/109)), and a dry run of the delivery cut on `804ea06` ([PR #114](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/114)).
- Every column the interim suite keeps filled is tested, not only `make` ([#96](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/96), [PR #98](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/98)).

**Not done, moved into sprint 4:**

- [#44](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/44), [#92](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/92) and [#93](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/93): the first report, the presentation and the delivery cut, which fall after the sprint by plan.
- [#94](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/94) stays open with the checkboxes it has left.
- The carbon-emissions half of the sprint goal is done, but without the ticket of its own that the planning notes said was still to be written.
  Its work landed inside #88 ([PR #101](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/101)): the energy of every fit of the search, its total reported apart from the final fit's, and NFR-10 changed to match (EDN-73).
  It builds on the CodeCarbon measurement of every training fit from sprint 2 (#38).
- Follow-ups opened during the sprint for the second delivery: [#108](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/108), [#113](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/113) and [#117](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/117).

**Risks from planning.**
The proposed cut-off for pipeline work on 2026-10-13 held for the numbers: #121 changed the candidate on that day, and nothing merged since changes a metric.
Access to the FIB Virtech VM is still not recorded as resolved.

**Questions from [PR #97](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/97), answered by the team:**

- The M4a session is on 2026-10-27.
- The sprint 1 and 2 plannings and the sprint 1 review were held in the laboratory session itself.

## Sprint 4 planning notes (2026-10-13)

**Sprint goal:** submit the first report on 2026-10-19 and give the M1-M3 presentation on 2026-10-20.

**Backlog:**

| Issue | Title | Assignee |
| ----- | ----- | -------- |
| [#123](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/123) | Finalize the first report: close the open TODOs, proofread and refactor | @kadameit |
| [#126](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/126) | Close sprint 3 and plan sprint 4 around the first delivery | @kadameit |
| [#92](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/92) | Prepare the M1-M3 presentation for 2026-10-20 | @lukas2510 |
| [#93](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/93) | Cut and submit the first delivery by 2026-10-19, 23:59 | @lukas2510 |
| [#44](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/44) | First report (M1-M3) | @lukas2510 |
| [#94](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/94) | Close sprint 2 and plan sprint 3 around the new delivery dates (its remaining checkboxes) | @lukas2510 |

**Dates:**

- The sprint runs Tuesday to Tuesday, from 2026-10-13 to the presentation on 2026-10-20.
- The freeze window of [CONTRIBUTING.md](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/CONTRIBUTING.md) starts on 2026-10-18, so the report is merged by then.
- The first report is due on 2026-10-19 at 23:59.

**The presentation (#92).**
15 minutes in nine parts, and every member presents (the individual factor counts the discussions in the laboratory):

| Part | Minutes | Presenter |
| ---- | ------- | --------- |
| 1. Problem and use cases | 1 | Michal |
| 2. Requirements and success criteria | 1.5 | Michal |
| 3. Data | 2 | Mark |
| 4. Way of working | 1 | Mark |
| 5. Structure and pipeline | 3 | Kevin |
| 6. Experiments and model choice | 2.5 | Lukas |
| 7. Energy | 1 | Lukas |
| 8. Quality assurance | 2.5 | Ula |
| 9. Summary, lessons learned and outlook | 0.5 | Ula |

Every member finishes the slides of their parts by Sunday, 2026-10-18, so that the slides can be checked for formatting and content issues before the presentation.
Every member also reads through the whole report before it is submitted.

## Sprint 4 progress (2026-10-17)

Recorded on day 5 of the sprint from the merged and open pull requests.

**Landed since planning:**

- The report's figures are drawn by the pipeline, the segment and data-composition figures are added, and the existing two read in greyscale ([PR #119](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/119)).
- The DRAFT owner markers are removed, and the cloud provider choice moved from Milestone 1 to Milestone 4 ([PR #122](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/122)).

**In progress:**

- The report is being finalised in [PR #124](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/124) (#123): every number and every cited EDN entry checked against its source, each section proofread, the Oxford comma used throughout the report and the EDN, and the branch naming in `CONTRIBUTING.md` aligned with what the team does.
- The behavioural tests were checked against the trained models and found correct. Nothing needed to change, so no pull request records the check.
- The presentation is being prepared (#92).

**Still open for the delivery:** the slides link and Michal's and Mark's mail domains on the report's cover, and the project board link, which returns 404 to a reader who is not logged in (#123).

## Sprint 5 meeting (2026-10-27)

Lukas could not attend, so Kevin stood in as scrum master.

**The FIB Virtech VM (#134).**
The team read through the FIB Virtech setup guide together, and Kevin posted the findings on [#134](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/134#issuecomment-6439525384) with a summary of each setup step to test.
Mark, Michal, and Kevin are available to set up the VM and now own #134, and Mark does the first setup steps on 2026-10-28.
Everyone else helps whenever needed.

**Sprint 5 review.**
The team did not review what sprint 5 delivered, because the course required nothing before the next laboratory: FastAPI and deployment have not been covered in class yet.

**Sprint 5 retrospective.**
The presentation slides went well.
The retrospective produced no new tickets, because none was needed.

**Sprint 6 goals.**
The sprint runs from 2026-10-28 to 2026-11-03.

- Milestone 4: set up the VM ([#134](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/134)), deploy the API to it behind the reverse proxy ([#142](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/142)), load test it ([#143](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/143)), and write report section 2.4 ([#145](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/145)).
- Milestone 5: decide how CI reaches the VM, since a GitHub-hosted runner cannot reach it over SSH (an EDN decision), and deploy from CI with a rollback ([#198](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/198)).
- Scrum records: close sprint 4 ([#146](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/146)) and sprint 5 ([#202](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/202)).

Still open: who owns each issue besides #134 and #202, and which option CD uses.

**Feedback from the professor.**
The EDN has to document the AI interaction evidence in more detail: the literal prompt, or the core prompt, paraphrased where the literal one is not available.
The guidance in [reports/edn.md](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md) and in `AGENTS.md` now asks for that.
It applies to new entries only, and the entries recorded before 2026-10-27 do not have to be adapted.

## Retrospective notes

### Sprint 1

Drafted from the sprint's record on 2026-09-29 and extended on 2026-09-30, confirmed by the team at the laboratory session of 2026-10-13.

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
    - The working agreements named a ceremony day that the lab schedule contradicts: they put
      Planning in a Tuesday laboratory session while every session in the schedule is a Wednesday.
      A page describing our own process was never checked against the document that fixes it.
      *Withdrawn on 2026-10-06 ([PR #97](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/97)): the finding was wrong. The laboratory is on Tuesdays,
      as the working agreements said; the Wednesday came from reading the course PDF's session
      dates as 2026 dates.*
- **Action items:**
    - Cut sprint 2 so that each ticket owns its own files, and land the shared contracts first
      (#32).
    - Reserve EDN numbers in `reports/edn.md` on `main` before opening a branch that needs one.
    - Re-run the committed analysis outputs whenever a decision changes the data scope.
    - Every teammate owns at least one pipeline stage and writes the report section for it.
    - Give every ticket its milestone label, area label, GitHub milestone and owner when it is
      written, not when the sprint is reviewed.
    - Hold the ceremonies in the Wednesday laboratory session and run the sprint from Wednesday to
      Wednesday, so the plan is made where the whole team is present.
      *Withdrawn on 2026-10-06 with the finding above; the sprint stays Tuesday to Tuesday.*

### Sprint 2

Drafted from the sprint's record on 2026-10-06, confirmed by the team at the laboratory session of 2026-10-13.

- **What went well:**
    - The contract-first cut worked: once #32 landed, every stage was built against the schema and the fixture, and the pipeline ran on the real snapshot the next day.
    - The Windows problems a teammate hit (CRLF in `dvc.lock`, a POSIX path in a test, MLflow's emoji on a cp1252 console) were each fixed, and the suite now runs on Windows in CI so they cannot return unnoticed.
    - Claims were again checked against the data before they were written down, for example the raw fill rates that moved into the Great Expectations suites.
- **What to improve:**
    - The work is still unevenly distributed: 27 of the 29 pull requests merged since planning were authored under @lukas2510's account, 2 under @kadameit's, and 23 of the 26 closed issues were assigned to @lukas2510.
      The laboratory grade carries an individual factor based on contributions, and the account that opens an agent's pull request does not show who did the thinking.
    - EDN numbers were again taken without reserving them on `main` first (EDN-69, since archived, and EDN-75), after the sprint 1 retrospective made reserving them an action item.
    - Two pull requests moved the lab from Tuesday to Wednesday because the course PDF's session dates were read as 2026 dates and never checked against the day the lab actually meets.
      #97 corrected it.
    - Tuning and the split strategy were left until after the pipeline worked, so they now sit in front of a delivery date.
- **Action items:**
    - Every teammate owns a ticket of sprint 3, and the owner opens the pull request from their own account.
    - Reserve the EDN number on `main` in its own small pull request before the branch that needs it, and say so in the ticket.
    - Check a date from a course document against the calendar before it is written into a plan.
    - Land the changes that move the report's numbers (#88, #89) before 2026-10-13.

### Sprint 3

Drafted from the sprint's record on 2026-10-17, and confirmed by the team at the laboratory session of 2026-10-13.

- **What went well:**
    - The changes that move the report's numbers landed before the cut-off, as the sprint 2 retrospective asked: the search and the split review merged on 2026-10-12.
    - EDN numbers were reserved on `main` in their own pull requests before the branches that used them (PR #100, PR #102), the sprint 2 action item.
    - Decisions were again made by rules fixed before the measurement: the simultaneous intervals of the search (EDN-73) and the qualifying rule of the blanked copies (EDN-82), which turned down `lgbm-extended` although it was better on most requests.
    - The delivery cut was rehearsed a week before the delivery (#114).
- **What to improve:**
    - The work is still concentrated: 13 of the 16 pull requests merged since planning were opened from @lukas2510's account and 3 from @kadameit's, and 6 of the 7 closed issues were assigned to those two accounts.
      The sprint 2 action item that every teammate owns a ticket of sprint 3 was not met.
    - The candidate changed on the last day of the sprint (#121), after the report draft, so the draft had to be updated again.
    - The carbon-emissions half of the sprint goal was done but had no ticket of its own, because the ticket the planning promised was never written, so the board does not show that work apart from #88.
      The team agrees that the tickets have to be cut so that this does not happen again.
    - #117, opened during the sprint, has no milestone and no owner, which the sprint 1 action item rules out.
- **Action items:**
    - From sprint 5 on, every teammate owns an M4 ticket and opens its pull request from their own account.
    - Give #117 a milestone and an owner.
    - At sprint planning, every part of the sprint goal gets a ticket of its own, with an owner and the criterion that says when it is done, before the sprint starts; work found during the sprint gets its ticket when it is found.
    - Set the cut-off for changes that move reported numbers one day before the sprint ends, not on its last day.
