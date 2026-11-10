Sprint log
==========

Record of each sprint's goal and outcome.

| Sprint | Dates | Goal | Outcome |
| ------ | ----- | ---- | ------- |
| 1      | 2026-09-22 - 2026-09-29 | M1 Inception setup: stand up team coordination (Discord), lock the dataset and modelling direction, and get DVC, requirements, and the model/dataset cards underway. | Met. 16 pull requests of sprint 1 scope merged, 7 issues closed. Dataset, modelling direction and drift scenario locked; DVC and the DagsHub remote in place with the raw file tracked; dataset card, model card, problem specification and requirements written. The model card (#5, PR #30) landed on the last day, minutes after the review was written, so nothing was carried into sprint 2 but the Discord ticket (#15), which is set up and still open. |
| 2      | 2026-09-29 - 2026-10-06 | From documents to a running pipeline: land the contract every stage builds on, then fill the DVC stages in parallel, so the first report has measured numbers to cite. | Met. The pipeline runs end to end on the real Zenodo snapshot with every stage implemented, tracked in MLflow on DagsHub and gated by SC-01 to SC-06, and CodeCarbon measures every training fit. 29 pull requests merged and 26 issues closed after planning. Hyperparameter tuning (#88) and the split strategy (#89) were not started and move into sprint 3. |
| 3 | 2026-10-06 - 2026-10-13 | Fine-tune the model and measure its carbon emissions, so the numbers the first report quotes are final before the delivery on 2026-10-19. | Met. 16 pull requests merged and 7 issues closed after planning. The hyperparameter search (#88) and the split review (#89) kept the committed values, and `lgbm-basic` became the candidate after training on blanked copies (#116, EDN-82). The carbon emissions are measured for every training fit and for the whole search, and NFR-10 now separates a search from the final fit (EDN-73); that work landed inside #88, because the planned ticket of its own was never opened. The candidate changed on the last day, so the report's numbers were final only at the end of the sprint. |
| 4 | 2026-10-13 - 2026-10-20 | Finalise and submit the first report and prepare the M1-M3 presentation. | Met. The first report was finalised with no placeholder left (#123, PR #124) and submitted on Atenea (#93), and the M1-M3 presentation was given on 2026-10-20 (#92). 4 pull requests merged and 3 issues closed after planning. Access to the FIB Virtech VM was still missing, so getting it became a ticket of sprint 5 (#134). |
| 5 | 2026-10-20 - 2026-10-27 | Milestone 4: serve the model through a documented and tested API in a Compose stack behind a reverse proxy, and deploy and load test it on the FIB Virtech VM. | Partly met. The API serves valuations, price ranges, explanations, and comparables, with its OpenAPI document and its contract, integration, and load tests, and runs as a Compose stack behind nginx. The CI half of Milestone 5 was pulled in: the image is published to GHCR, scanned, and hardened, and a workflow retrains on request. 52 pull requests merged and 41 issues closed after planning. The VM was not available yet, so its setup (#134), the deployment on it (#142), its load test (#143), and CD (#198) moved into sprint 6, together with report section 2.4 (#145). |
| 6 | 2026-10-27 - 2026-11-03 | Milestone 4 on the VM: set it up, deploy the API behind the reverse proxy, load test it, and write report section 2.4; Milestone 5: deploy from CI with a rollback. | Partly met. The VM is set up (#134), the API runs on it behind nginx (#142), its load test settled NFR-02 to NFR-04 and NFR-10 (#143, EDN-88), and CD deploys every release through a runner on the VM that may only run `deploy.sh`, and puts the previous tag back when the check from outside fails (#198, EDN-94). 29 pull requests merged and 6 issues closed after planning. Report section 2.4 (#145) and the records of sprints 4 and 5 (#146, #202) were not done and move into sprint 7. |
| 7 | 2026-11-03 - 2026-11-10 | Milestone 6: monitor the deployed stack and the model, rehearse the retraining chain, and write report section 2.6; finish report section 2.4 and the sprint records that sprint 6 left. | Mostly met. Milestone 6 is done: Prometheus, cAdvisor, node-exporter, Grafana with alerts to Discord, and the Better Stack uptime check monitor the stack (#224 to #226, EDN-99), the replay, the drift job, and the error and coverage per window monitor the model (#227 to #230, EDN-100, EDN-101), the retraining chain was rehearsed on the VM from the finding to the rollback (#231, EDN-102), and report section 2.6 is written (#232). The team confirmed the retrospectives of sprints 4 to 6 (#146, #202). 51 pull requests merged and 28 issues closed after planning. Report section 2.4 (#145) was not done and moves into sprint 8. |
| 8 | 2026-11-10 - 2026-11-17 | The second report: start first drafts of its sections, with report sections 2.4 and 2.5.2 and the monitoring slides first. | In progress |

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

## Sprint 4 review (2026-10-20)

From the merged pull requests and the closed issues.

**Done and merged:**

- The first report is final: no placeholder left, every number and every cited EDN entry checked against its source, each section proofread, and the cover complete with the slides link ([#123](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/123), [PR #124](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/124)).
- The report's figures are drawn by the pipeline and the DRAFT owner markers are gone ([PR #119](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/119), [PR #122](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/122)), as the progress entry above records.
- The first delivery is submitted on Atenea ([#93](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/93)), and the M1-M3 presentation was given on 2026-10-20 ([#92](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/92)).
- The sprint 3 review and retrospective and the sprint 4 plan are recorded ([PR #127](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/127)).

**Closed after the sprint:**

- [#92](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/92) and [#93](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/93) were done by 2026-10-20 and closed on 2026-10-22.
- [#126](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/126) closed on 2026-10-21 with the team's confirmation of the sprint 3 records and its answers to the two questions from [PR #97](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/97) ([PR #131](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/131)).

**Dates from planning.**
The freeze window started on 2026-10-18, and the report's last pull request (#124) merged on 2026-10-20.

**Risks.**
Access to the FIB Virtech VM is still not resolved, and Milestone 4 deploys on it, so getting it is a ticket of sprint 5 ([#134](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/134)).

## Sprint 5 planning notes (2026-10-20)

From the issues written for Milestone 4 on 2026-10-20 and the issues still open then.

**Sprint goal:** Milestone 4.
Serve the model through an API with its documentation and its tests, run it as a Compose stack behind a reverse proxy, and deploy and load test it on the FIB Virtech VM.

**How the work is cut.**
Only the lecturers can request the VM, so getting it has the longest lead time of all M4 work, and the deployment ([#142](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/142)) and the load test ([#143](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/143)) wait for it ([#134](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/134)).
The endpoints, their tests, and the Compose stack do not, so they are built and checked locally first.

**EDN IDs.**
EDN-83 to EDN-88 are reserved on `main` for the Milestone 4 decisions before any branch needs one ([PR #148](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/148)), as the sprint 2 retrospective asked.

**Backlog:**

| Issue | Title | Assignee |
| ----- | ----- | -------- |
| [#133](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/133) | Type-check the package and tools with mypy before the M4 API code | @lukas2510 |
| [#135](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/135) | API skeleton: valuations at `/v1/predict` and `/v1/health`, with input checks from the loaded model and one error shape | @lukas2510 |
| [#136](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/136) | Explain each valuation by the five features with the largest effect on its price (FR-08) | @lukas2510 |
| [#137](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/137) | Build the UC2 price intervals with conformalized quantile regression, and add SC-05 to the gate | @lukas2510 |
| [#108](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/108) | Check the UC2 interval coverage per make when the conformal intervals are built | @lukas2510 |
| [#139](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/139) | Serve price ranges at `/v1/price-range` (FR-02, FR-07) | @lukas2510 |
| [#138](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/138) | Build the comparables index in the pipeline and serve it at `/v1/comparables` (FR-09) | @lukas2510 |
| [#140](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/140) | Record every answer in a prediction log and accept observed prices at `/v1/feedback` (FR-13, FR-10) | @lukas2510 |
| [#141](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/141) | Expose request, latency and error metrics at `/metrics` in Prometheus format (FR-11) | @lukas2510 |
| [#154](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/154) | Log every API request as one JSON line with its request ID | @lukas2510 |
| [#153](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/153) | Commit the OpenAPI schema, check it for drift and render it in the docs | @lukas2510 |
| [#144](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/144) | Generate API contract tests from the OpenAPI schema with Schemathesis | @lukas2510 |
| [#163](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/163) | Integration tests against the running Compose stack, in CI and after a deployment | @lukas2510 |
| [#152](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/152) | Document the architecture in the docs: C4 container and deployment views and the patterns we use | @lukas2510 |
| [#155](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/155) | Enable ruff's security rules (flake8-bandit, `S`) | @lukas2510 |
| [#134](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/134) | Get the FIB Virtech VM and find out how it can be reached from outside | @lukas2510; from 2026-10-27 @kadameit, @W11W11W11, @michudud04 |
| [#142](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/142) | Deploy the API to the FIB VM behind a reverse proxy that refuses `/feedback` from outside | @lukas2510 |
| [#143](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/143) | Load test the API on the VM and settle the latency, throughput and resource targets (NFR-02, NFR-03, NFR-10) | @lukas2510 |
| [#145](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/145) | Write report section 2.4 and close Milestone 4: architecture, API design, API testing and the M4 gate | unassigned |
| [#117](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/117) | Try L2 regularisation (reg_lambda) for the LightGBM variants with a recorded search | @kadameit |
| [#128](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/128) | The evaluate stage writes its result files and the MLflow records in one loop, so a failed MLflow call leaves them half-written | @kadameit |
| [#129](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/129) | Block the merge of a promotion pull request whose committed gate verdict fails | @lukas2510 |
| [#113](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/113) | Explain why the fits use more than one CPU second per second with train.num_threads at 1 | @lukas2510 |
| [#130](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/130) | The EDN generator writes CRLF line endings on Windows | @kadameit |
| [#125](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/125) | Update code comments that cite merged or retired EDN IDs | @kadameit |
| [#156](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/156) | Update the course demo notes to the demo repo as of 2026-10-20 | @lukas2510 |
| [#126](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/126) | Close sprint 3 and plan sprint 4 around the first delivery | @kadameit |
| [#146](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/146) | Close sprint 4 and plan sprint 5 around Milestone 4 | unassigned |

**Dates:**

- The sprint runs Tuesday to Tuesday, from 2026-10-20 to the M4a laboratory on 2026-10-27.
- The M4-M6 presentation is on 2026-12-15, and the due date of the second report is not announced yet.

## Sprint 5 meeting (2026-10-27)

Lukas could not attend, so Kevin stood in as scrum master.

**The FIB Virtech VM (#134).**
The team read through the FIB Virtech setup guide together, and Kevin posted the findings on [#134](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/134#issuecomment-6439525384) with a summary of each setup step to test.
Mark, Michal, and Kevin are available to set up the VM and now own #134, and Mark does the first setup steps on 2026-10-28.
Everyone else helps whenever needed.

**Sprint 5 review.**
The team did not review what sprint 5 delivered, because the course required nothing before the next laboratory: FastAPI and deployment have not been covered in class yet.

What the sprint delivered, from the merged pull requests and the closed issues (52 pull requests merged and 41 issues closed after planning):

- **The API.**
  Valuations at `/v1/predict` and the service status at `/health`, with input checks from the loaded model and RFC 9457 errors ([#135](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/135), EDN-84).
  Price ranges at `/v1/price-range` from conformalized quantile regression, with SC-05 measured and the coverage reported per make ([#137](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/137), [#139](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/139), [#108](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/108), EDN-85, EDN-89).
  The five fields with the largest effect on each price ([#136](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/136)), comparable listings at `/v1/comparables` ([#138](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/138)), and a SQLite prediction log with observed prices accepted at `/v1/feedback` ([#140](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/140), EDN-86).
  Request, latency, error, and model metrics at `/metrics`, and one JSON log line per request ([#141](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/141), [#154](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/154)).
- **Its documentation and tests.**
  The OpenAPI document committed with a drift check and rendered as the API reference ([#153](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/153)), contract tests generated from it with Schemathesis ([#144](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/144)), an integration suite against the Compose stack through nginx ([#163](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/163)), a repeatable Locust load test with a dry run on a laptop ([PR #176](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/176)), and the architecture page with the C4 container and deployment views ([#152](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/152)).
- **The stack.**
  The API runs behind nginx in a Docker Compose stack, with a VM bootstrap and a smoke check ([PR #162](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/162), EDN-87), and scores each request on one thread ([#179](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/179)).
- **The code and the model.**
  mypy and ruff's security rules run in CI ([#133](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/133), [#155](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/155), EDN-83), and a promotion pull request whose gate verdict fails cannot merge ([#129](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/129)).
  @kadameit ran the L2 regularisation search, which kept 0 ([#117](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/117)), made the evaluate stage write its result files before the MLflow records ([#128](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/128)), and fixed the EDN generator's line endings and the code comments' EDN IDs ([#130](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/130), [#125](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/125)).
- **Milestone 5, pulled in** once the API was done, with its tickets written on 2026-10-22.
  The image is built after a green CI run on `main` and pushed to GHCR ([#197](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/197), EDN-91), scanned with trivy and watched by Dependabot ([#199](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/199)), and run read-only, without capabilities, and with memory limits ([#200](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/200), EDN-95).
  A hung API is restarted ([#185](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/185), EDN-93), a retrain can be started from GitHub Actions and prepares the promotion pull request ([#201](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/201), EDN-92), and CI warns when `dvc.lock` is stale ([#193](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/193), EDN-90).
  The deploy workflow with its rollback is merged ([PR #214](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/214)) and needs the VM to run.

**Not done, moved into sprint 6:** the VM ([#134](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/134)), the deployment and the load test on it ([#142](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/142), [#143](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/143)), CD from CI ([#198](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/198)), report section 2.4 ([#145](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/145)), and the sprint 4 records ([#146](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/146)).

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

## Sprint 6 review (2026-11-03)

From the merged pull requests and the closed issues.

**The backlog and its owners.**
The sprint 5 meeting left open who owns each issue besides #134 and #202.
The owners on GitHub and how each issue ended:

| Issue | Title | Assignee | Outcome |
| ----- | ----- | -------- | ------- |
| [#134](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/134) | Get the FIB Virtech VM and find out how it can be reached from outside | @kadameit, @W11W11W11, @michudud04 | Done |
| [#142](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/142) | Deploy the API to the FIB VM behind a reverse proxy that refuses `/feedback` from outside | @lukas2510 | Done |
| [#143](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/143) | Load test the API on the VM and settle the latency, throughput and resource targets (NFR-02, NFR-03, NFR-10) | @lukas2510 | Done |
| [#198](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/198) | Deploy the pushed image to the VM from CI and put the previous tag back when the smoke check fails (NFR-13) | @lukas2510 | Done |
| [#145](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/145) | Write report section 2.4 and close Milestone 4: architecture, API design, API testing and the M4 gate | unassigned | Moved into sprint 7 |
| [#146](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/146) | Close sprint 4 and plan sprint 5 around Milestone 4 | unassigned | Moved into sprint 7 |
| [#202](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/202) | Close sprint 5 and plan sprint 6 around Milestone 5 | @kadameit | Moved into sprint 7 |

**How CD reaches the VM**, the other question the meeting left open.
SSH to the VM works only from the UPC network, so a GitHub-hosted runner cannot deploy over it ([#134](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/134)).
CD deploys through a self-hosted runner on the VM that may only run `deploy.sh` ([PR #237](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/237), EDN-94).

**Done and merged:**

- The VM: @W11W11W11 created it from the sense VPN template and ran phases 1 to 4 of the runbook on 2026-10-30, and the bootstrap and the first deployment by hand followed the same day ([#134](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/134)).
  Its verified facts and measurements are in the deployment docs and the specification ([PR #241](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/241), [PR #243](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/243)).
- The API on the VM behind nginx, which refuses `/feedback` and `/metrics` from outside; the deploy workflow checks that from the public internet on every deploy ([#142](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/142)).
- The load test on the VM, which settles NFR-02 to NFR-04 and NFR-10 ([#143](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/143), [PR #244](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/244), EDN-88).
- CD to the VM through the runner, switched on ([#198](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/198), [PR #237](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/237), [PR #247](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/247)).
- The requirement matrix enforces every requirement booked to M4 and to M5 ([PR #246](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/246), [PR #261](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/261)).
- The class, sequence, and use-case views beside the C4 views, with the ML components marked ([PR #249](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/249)), which report section 2.4 needs.
- The image scan and the container hardening recorded as EDN-95 ([PR #250](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/250)).
- Two retrains from GitHub Actions promoted ([PR #255](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/255), [PR #258](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/258)), with the reproducibility fixes they called for ([#215](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/215), [PR #256](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/256), [PR #257](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/257), [PR #260](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/260)).
- A public web page for the price API, added during the sprint ([#269](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/269), [PR #270](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/270), [PR #275](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/275), EDN-96).
- The sprint 5 meeting recorded by @kadameit, together with the professor's request for the prompt as AI interaction evidence in the EDN ([PR #236](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/236)), which [PR #276](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/276) turns into the question and the chosen option.

**Not done, moved into sprint 7:** report section 2.4 ([#145](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/145)) and the records of sprints 4 and 5 ([#146](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/146), [#202](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/202)).

**Risks.**
Access to the FIB Virtech VM, a risk since the sprint 2 planning, is resolved ([#134](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/134)).

## Sprint 7 planning notes (2026-11-03)

From the issues open at the start of the sprint.

**Sprint goal:** Milestone 6.
Monitor the deployed stack and the model, rehearse the retraining chain that a drift finding starts, and write report section 2.6.
Finish what sprint 6 left: report section 2.4 and the sprint records.
The Milestone 6 tickets were written on 2026-10-23.

**Backlog:**

| Issue | Title | Assignee |
| ----- | ----- | -------- |
| [#224](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/224) | Run Prometheus, cAdvisor and node-exporter in the Compose stack and scrape the API and the VM (FR-11, NFR-04) | unassigned |
| [#225](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/225) | Provision the Grafana dashboard for resources and service health, with alerts against the NFR-02 to NFR-05 targets | unassigned |
| [#226](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/226) | Set up the Better Uptime check on /health and report the monitored uptime (NFR-05) | unassigned |
| [#227](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/227) | Replay the held-out ES market against the API and report the observed prices (FR-10, NFR-11) | unassigned |
| [#228](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/228) | Build the Alibi Detect drift job in its own scheduled container and verify NFR-11 (FR-14) | unassigned |
| [#229](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/229) | Report the real error and the interval coverage per window by joining feedback to the prediction log (FR-14) | unassigned |
| [#230](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/230) | Show drift, error and coverage in Grafana and alert on them (FR-14, NFR-11) | unassigned |
| [#231](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/231) | Rehearse the full retraining chain on the ES replay: finding, retrain, gate, promotion, rollback (FR-15) | unassigned |
| [#232](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/232) | Write report section 2.6 and close Milestone 6: monitoring, model performance and the feedback loop | unassigned |
| [#191](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/191) | Log only the fields a client sent, or mark defaults, in the prediction log | @lukas2510 |
| [#259](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/259) | Make preprocess's log_price bit-identical across CPUs (numpy SIMD on AVX-512) | unassigned |
| [#272](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/272) | Let the integration tests read the Compose project name from the environment | @kadameit |
| [#273](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/273) | Serve an HTML error page when the API is down and a browser loads / | @kadameit |
| [#274](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/274) | Bring the VM's nginx to the slim image (rerun the bootstrap) | unassigned |
| [#145](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/145) | Write report section 2.4 and close Milestone 4: architecture, API design, API testing and the M4 gate | unassigned |
| [#146](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/146) | Close sprint 4 and plan sprint 5 around Milestone 4 | unassigned |
| [#202](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/202) | Close sprint 5 and plan sprint 6 around Milestone 5 | @kadameit |

**Dates:**

- The sprint runs Tuesday to Tuesday, from 2026-11-03 to 2026-11-10.
- The M4-M6 presentation is on 2026-12-15, and the due date of the second report is not announced yet.

## Sprint 7 progress (2026-11-07)

Recorded on day 5 of the sprint from the merged and open pull requests: 27 pull requests merged and 14 issues closed since planning.

**Landed since planning:**

- Resource and service monitoring: Prometheus, cAdvisor, and node-exporter in every stack ([#224](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/224)) and the Grafana dashboard with alerts on NFR-02 to NFR-05 through Alertmanager to Discord ([#225](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/225)), both decided as EDN-99, and the Better Stack check of `/health` from outside, which reports the uptime ([#226](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/226)).
- Model monitoring: the held-out listings replayed against the API ([#227](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/227), EDN-100) from versioned monitoring samples ([PR #283](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/283)), each window of the prediction log tested for input drift in a scheduled container ([#228](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/228), EDN-101), each window's real error and range coverage from its feedback ([#229](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/229)), and all three in Grafana with alerts ([#230](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/230)).
- The prediction log stores only the fields a client sent, marked per row ([#191](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/191), EDN-98).
- The retraining chain: FR-15's retrain trains on 60 % of the `ES` sellers and holds out the rest ([PR #300](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/300), EDN-102), the chain was rehearsed on the VM from the finding to the rollback ([#231](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/231), [PR #308](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/308)), and the promoted model's numbers are quoted in the docs and the report ([PR #303](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/303), [PR #309](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/309)).
- Deployment: the stack files ship with every release ([#279](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/279)), the VM's nginx runs the slim image ([#274](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/274)), and @kadameit made the integration tests read the Compose project from the environment ([#272](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/272)).
- Preprocess's `log_price` is bit-identical across CPUs ([#259](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/259)), and the load test was re-run on the VM with the full monitoring stack ([PR #296](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/296)).
- Report section 2.6, with the requirement matrix enforcing M6 ([#232](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/232), [PR #302](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/302), [PR #310](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/310)).

**In progress:** the lecturer's feedback on the first report, recorded so that the final report learns from it ([#304](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/304), [PR #305](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/305)).

**Still open:** report sections 2.4 and 2.5.2 ([#145](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/145), [#316](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/316)), the monitoring slides ([#298](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/298)), the HTML error page ([#273](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/273)), the team's confirmation of the retrospectives of sprints 4 and 5 ([#146](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/146), [#202](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/202)) and of sprint 6, and the follow-ups opened during the sprint ([#286](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/286), [#289](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/289), [#301](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/301), [#306](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/306), [#307](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/307), [#311](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/311) to [#315](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/315)).

## Sprint 7 review (2026-11-10)

From the merged pull requests and the closed issues.
@kadameit led the meeting as scrum master, because @lukas2510 could not attend.

**The backlog and how each issue ended:**

| Issue | Title | Assignee | Outcome |
| ----- | ----- | -------- | ------- |
| [#224](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/224) | Run Prometheus, cAdvisor and node-exporter in the Compose stack and scrape the API and the VM (FR-11, NFR-04) | unassigned | Done |
| [#225](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/225) | Provision the Grafana dashboard for resources and service health, with alerts against the NFR-02 to NFR-05 targets | unassigned | Done |
| [#226](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/226) | Set up the Better Uptime check on /health and report the monitored uptime (NFR-05) | unassigned | Done |
| [#227](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/227) | Replay the held-out ES market against the API and report the observed prices (FR-10, NFR-11) | unassigned | Done |
| [#228](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/228) | Build the Alibi Detect drift job in its own scheduled container and verify NFR-11 (FR-14) | unassigned | Done |
| [#229](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/229) | Report the real error and the interval coverage per window by joining feedback to the prediction log (FR-14) | unassigned | Done |
| [#230](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/230) | Show drift, error and coverage in Grafana and alert on them (FR-14, NFR-11) | unassigned | Done |
| [#231](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/231) | Rehearse the full retraining chain on the ES replay: finding, retrain, gate, promotion, rollback (FR-15) | unassigned | Done |
| [#232](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/232) | Write report section 2.6 and close Milestone 6: monitoring, model performance and the feedback loop | unassigned | Done |
| [#191](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/191) | Log only the fields a client sent, or mark defaults, in the prediction log | @lukas2510 | Done |
| [#259](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/259) | Make preprocess's log_price bit-identical across CPUs (numpy SIMD on AVX-512) | unassigned | Done |
| [#272](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/272) | Let the integration tests read the Compose project name from the environment | @kadameit | Done |
| [#273](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/273) | Serve an HTML error page when the API is down and a browser loads / | @lukas2510 | Done |
| [#274](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/274) | Bring the VM's nginx to the slim image (rerun the bootstrap) | unassigned | Done |
| [#145](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/145) | Write report section 2.4 and close Milestone 4: architecture, API design, API testing and the M4 gate | unassigned | Moved into sprint 8 |
| [#146](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/146) | Close sprint 4 and plan sprint 5 around Milestone 4 | unassigned | Done, retrospective confirmed at this meeting |
| [#202](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/202) | Close sprint 5 and plan sprint 6 around Milestone 5 | @kadameit | Done, retrospective confirmed at this meeting |

**Done and merged since the progress of 2026-11-07,** in addition to the list above:

- The lecturer's feedback on the first report, with its lessons carried into the final report's guidance ([#304](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/304), [PR #305](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/305)).
- Each window is scored against the test values of the model that answered it, and judged against the test error of its countries ([#306](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/306), [#307](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/307)).
- The HTML page a browser gets while the API restarts ([#273](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/273), [PR #338](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/338)).
- Deployment fixes: the containers' logs survive a deploy ([#313](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/313)), `deploy.sh` names the variable it rejects ([#312](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/312)), and CD rebuilds its cached apt layers when Debian publishes ([#314](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/314)).
- The prediction log is pruned without being emptied when the limit is small ([#311](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/311)), and validate-data's outputs are byte-stable ([#289](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/289)).
- The image findings: the expiring ignores resolved and renewed to 2027-01-31 ([#286](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/286), [#301](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/301)).
- The split strategy and split gate analyses re-run on the current pool ([#315](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/315), [PR #344](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/344)), and the Data Docs fixed ([#335](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/335), [#347](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/347)).

**Not done, moved into sprint 8:** report sections 2.4 and 2.5.2 ([#145](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/145), [#316](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/316)) and the monitoring slides ([#298](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/298)).

**Retrospectives.**
The team confirmed the retrospectives of sprints 4, 5, and 6 as drafted.

## Sprint 8 planning notes (2026-11-10)

From the issues open at the start of the sprint and the decisions of the meeting.

**Sprint goal:** the second report.
The technical milestones M4 to M6 are done, so the sprint starts first drafts of the report's sections, with report sections 2.4 and 2.5.2 and the monitoring slides first.

**Backlog:**

| Issue | Title | Assignee |
| ----- | ----- | -------- |
| [#145](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/145) | Write report section 2.4 and close Milestone 4: architecture, API design, API testing and the M4 gate | to be agreed |
| [#316](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/316) | Write report section 2.5.2 on CI/CD | to be agreed |
| [#298](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/298) | Prepare the monitoring slides for the final presentation | to be agreed |
| [#348](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/348) | Rebuild dvc.lock under pyarrow 26 | unassigned |
| [#346](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/346) | Keep the images' environment layer when only a dev dependency changes | unassigned |
| [#345](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/345) | Survive an outage of GitHub's release downloads in CI's binary downloads | unassigned |
| [#334](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/334) | Drop the zlib1g upgrade step once the base image ships zlib1g +deb13u1 | unassigned |

**Owners.**
The team is busy with exams this week, and the second report is not due yet, so the meeting did not split the report sections and the slides.
The team agrees on who takes #145, #316, and #298 during the week.

**Dates:**

- The sprint runs Tuesday to Tuesday, from 2026-11-10 to 2026-11-17.
- The M4-M6 presentation is on 2026-12-15, and the due date of the second report is not announced yet.

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

### Sprint 4

Drafted from the sprint's record, confirmed by the team at the sprint meeting of 2026-11-10.

- **What went well:**
    - The sprint goal was met: the first report was submitted and the M1-M3 presentation was given (#92, #93).
    - @kadameit owned and finished the report's finalisation (#123, PR #124) and recorded the sprint 3 review (PR #127), so 2 of the sprint's 4 pull requests came from an account other than @lukas2510's, the first sprint where half of them did.
    - The delivery cut had been rehearsed in sprint 3 (PR #114), and the final build had no placeholder left.
    - Every part of the presentation had a named presenter from the sprint planning on.
- **What to improve:**
    - The report's last pull request (#124) merged on 2026-10-20, two days into the freeze window that started on 2026-10-18.
    - #92 and #93 were closed on 2026-10-22, two days after the presentation, so the board showed the delivery as open in between.
    - Access to the FIB Virtech VM, a risk since the sprint 2 planning, was still missing, and it had no ticket until the M4 tickets were written on 2026-10-20 (#134).
- **Action items:**
    - Make getting the VM a sprint 5 ticket with an owner (#134).
    - Merge a report's last change before its freeze window starts, and close a delivery ticket on the day the delivery is done.

### Sprint 5

Drafted from the sprint's record, in addition to what the team said at the meeting of 2026-10-27, confirmed by the team at the sprint meeting of 2026-11-10.

- **What went well:**
    - The API was built and tested before the VM existed: contract tests from the OpenAPI document, an integration suite against the Compose stack, and a load test dry run on a laptop (#144, #163, PR #176), so the deployment in sprint 6 had its smoke check and its load test ready.
    - EDN-83 to EDN-89 were reserved on `main` before the branches that needed them (PR #148, PR #170), and the sprint's decisions were recorded as they were taken (EDN-83 to EDN-87 and EDN-89 to EDN-93).
    - The CI half of Milestone 5 was pulled in once the API was done (#197, #199, #200, #201, PR #214).
    - The sprint 3 action item to give #117 a milestone and an owner was met, and @kadameit closed it (PR #161).
- **What to improve:**
    - The work is still concentrated: 43 of the 52 pull requests were opened from @lukas2510's account, 5 from @kadameit's, and 4 by Dependabot.
      The sprint 3 action item that every teammate owns an M4 ticket and opens its pull request from their own account was not met.
    - Getting the VM had the longest lead time of all M4 work and was not available during the sprint, so the deployment, the load test on the VM, and CD moved into sprint 6.
      It had one owner until the meeting of 2026-10-27.
    - Report section 2.4 (#145) and the sprint 4 records (#146) were written as tickets without an owner, which the sprint 1 action item rules out.
    - The meeting held no review.
- **Action items:**
    - The VM work is shared by three teammates from 2026-10-27 (#134), as the meeting decided.
    - Give #145 and #146 an owner.

### Sprint 6

Drafted from the sprint's record, confirmed by the team at the sprint meeting of 2026-11-10.

- **What went well:**
    - @W11W11W11, one of the three owners the sprint 5 meeting named, created the VM and ran the first four phases of the runbook, and the API ran on it the same day (#134).
    - The CD question the meeting left open was decided on the VM's verified network facts, with the options laid out (EDN-94).
    - Claims about the VM were measured on it before they went into the specification: its network, its recovery after a reboot, and its load test (PR #241, PR #243, PR #244).
    - The requirement matrix enforces M4 and M5 (PR #246, PR #261).
- **What to improve:**
    - 26 of the 29 pull requests were opened from @lukas2510's account, 1 from @kadameit's, and 2 by Dependabot.
      The VM work of @W11W11W11 is recorded in the comments of #134 and does not show in these numbers.
    - Report section 2.4 (#145) had no owner at planning and was not started, so the sprint 3 action item that every part of the sprint goal gets a ticket with an owner before the sprint starts was not met.
    - The first retrain on a GitHub runner put the Ridge baseline outside NFR-06's tolerance against the laptop's fit, because its result depended on the machine's thread count and CPU.
      Pinning both took two pull requests (PR #253, PR #256), and a third dependence on the CPU became a follow-up (#259).
- **Action items:**
    - Give report section 2.4 (#145) an owner, and plan it with the other report sections of the second delivery.

### Sprint 7

Drafted from the sprint's record, for the team to confirm.

- **What went well:**
    - Every Milestone 6 ticket was done inside the sprint, most of them by day 5, and each decision was recorded as it was taken (EDN-98 to EDN-102).
    - The retraining chain was rehearsed on the real VM, from the drift finding to the rollback, instead of only on a laptop (#231, PR #308).
    - The lecturer's feedback on the first report was recorded with the lessons for the final report while the sprint was running (#304).
    - The team confirmed the retrospectives of sprints 4 to 6, which had been open since sprint 5.
- **What to improve:**
    - 44 of the 51 pull requests were opened from @lukas2510's account, 1 from @kadameit's, and 6 by Dependabot.
    - The Milestone 6 tickets had no owner at planning, so the sprint 3 action item that every part of the sprint goal gets an owner before the sprint starts was again not met.
    - Report section 2.4 (#145) moved into the next sprint for the second time.
- **Action items:**
    - Agree the owners of report sections 2.4 and 2.5.2 and of the monitoring slides (#145, #316, #298) during the week of 2026-11-10.
    - Start the first drafts of the second report's sections in sprint 8, so the drafts are ready before the due date is announced.
