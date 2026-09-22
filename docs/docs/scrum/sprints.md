Sprint log
==========

Record of each sprint's goal and outcome.

| Sprint | Dates | Goal | Outcome |
| ------ | ----- | ---- | ------- |
| 1      | 2026-09-22 - 2026-09-29 | M1 Inception setup: stand up team coordination (Discord), lock the dataset and modelling direction, and get DVC, requirements, and the model/dataset cards underway. | In progress |

## Sprint 1 planning notes (2026-09-22)

**Attendees:** @lukas2510, @kadameit, @W11W11W11, @michudud04, @ulasawczuk

**Decisions made:**

- Main dataset: AutoScout24 Car Listings Dataset (2025 snapshot). See [project brief §3.1](../project-brief.md#31-main-dataset-decided) and [EDN-01](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md).
- Model family: gradient boosting, with LightGBM as the main model. See [project brief §4](../project-brief.md#4-modelling-plan-planned) and [EDN-02](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md).

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

## Retrospective notes

### Sprint 1

- **What went well:**
- **What to improve:**
- **Action items:**
