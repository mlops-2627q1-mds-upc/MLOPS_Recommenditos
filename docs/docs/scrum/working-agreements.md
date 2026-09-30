Working agreements
===================

Team conventions for how we work together on this project.

## Sprint

- Sprint length: 1 week, Wednesday to Wednesday.
- Sprint Planning, Review and Retrospective all happen in the Wednesday laboratory session, the only slot in the week where the whole team is together.
  The [lab schedule](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/references/MLOps-lab.md) fixes those sessions, and every one of them falls on a Wednesday, from 2026-09-09 to 2026-12-16.
- A delivery is due the day before the session it belongs to, so a freeze always lands on a Tuesday: the first report is due 2026-10-13 for the presentation on 2026-10-14.
- Sprints 1 and 2 were run Tuesday to Tuesday and planned the day before the laboratory.
  The boundary moves onto the laboratory session with sprint 3, which starts on 2026-10-07 (**[proposed]**, a marker the [project brief](../project-brief.md) defines).

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

- No daily standup: the team is only together in person on Wednesdays, so that's our sync point.
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
