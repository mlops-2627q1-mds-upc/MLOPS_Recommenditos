Working agreements
===================

Team conventions for how we work together on this project.

## Sprint

- Sprint length: 1 week, Tuesday to Tuesday.
- Sprint Planning, Review and Retrospective all happen in the Tuesday laboratory session, the only slot in the week where the whole team is together.
- The first report is due on 2026-10-19 at 23:59, the day before its presentation on 2026-10-20, so the 48-hour [freeze window](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/CONTRIBUTING.md#freeze-window-before-a-presentation) starts on 2026-10-18.
  The second presentation is on 2026-12-15, and the due date of its report is not announced yet.
  The course changed these dates from the ones in the [lab schedule](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/references/MLOps-lab.md).

## Git and PRs

- Branch naming: `feature/<short-description>` or `fix/<short-description>`, always branched off `main`.
- Every change goes through a pull request into `main`; there are no direct pushes.
- Passing checks gate the merge; a formal approval does not.
  GitHub blocks self-approval and a pull request written with an agent runs under its author's account, so requiring one would leave those pull requests unmergeable.
  A teammate's review is welcome and does not gate the merge: 2 of the 56 pull requests merged by 2026-10-12 carry one ([EDN-81](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
  The workflow this follows from is in [CONTRIBUTING.md](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/CONTRIBUTING.md).
- Reserve the next EDN ID in [reports/edn.md](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md) on `main` before opening a branch that needs one.
  IDs handed out on parallel branches collide, and an ID never changes once it is published.

## Communication

- No daily standup: the team is only together in person on Tuesdays, so that's our sync point.
- Outside of the laboratory session, we work async and post progress updates in Discord as they happen.
- Blockers are raised in Discord as soon as they come up, not saved for the next ceremony.

## Coding agents

We use AI coding agents on this project, and the course grades how we do it.

- An agent works against a ticket that has a **local oracle**: tests or `dvc repro` that tell it
  whether it is done. A ticket without one is done by a person. Writing the oracle into the ticket
  is part of writing the ticket.
- **[proposed]** Agent-written code goes through the same pull request as anything else.
  An independent agent with a fresh context reviews every substantive change by reproducing its claims and trying to break it, and the required checks gate the merge.
  A person decides the questions that belong in the EDN and is accountable for the change.
  This wording was proposed by @lukas2510 on 2026-10-12 and is to be confirmed by the team at the lab on 2026-10-13.
- The author of the pull request is responsible for its content, whoever or whatever typed it.
- When an agent's claim decides something, it is checked before it is written down. Agents have
  reported numbers on this project that turned out to be wrong, and a wrong number in a
  requirement is worse than no number.

[EDN-81](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md) records this decision, the earlier wording of the second rule and why it changed, and where the rule has fallen short.

_Extend this list as the team agrees on new conventions._
