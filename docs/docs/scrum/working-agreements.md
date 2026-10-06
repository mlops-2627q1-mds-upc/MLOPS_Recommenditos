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
  A teammate still looks the change over whenever one is free.
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
- Agent-written code goes through the same pull request and review as anything else. The reviewer
  is a person, and reviewing means having understood the tests, not having watched them pass.
- The author of the pull request is responsible for its content, whoever or whatever typed it.
- When an agent's claim decides something, it is checked before it is written down. Agents have
  reported numbers on this project that turned out to be wrong, and a wrong number in a
  requirement is worse than no number.

The EDN entry recording this decision follows before the first delivery.

_Extend this list as the team agrees on new conventions._
