Working agreements
===================

Team conventions for how we work together on this project.

## Sprint

- Sprint length: 1 week, Tuesday to Tuesday.
- Sprint Planning, Review and Retrospective all happen on Tuesday, our fixed sync slot.
- The laboratory session itself is on Wednesday (see [the lab schedule](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/references/MLOps-lab.md)), so the sprint is planned the day before we present it.

## Git and PRs

- Branch naming: _e.g. `feature/<short-description>`, `fix/<short-description>`_
- Every change goes through a pull request; no direct pushes to `main`.
- At least one approval required before merging.

## Communication

- No daily standup: the team is only together in person on Tuesdays, so that's our sync point.
- Outside of Tuesdays, we work async and post progress updates in Discord as they happen.
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
