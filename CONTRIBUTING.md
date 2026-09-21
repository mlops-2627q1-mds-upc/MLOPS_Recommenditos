# Contributing

This applies to every contributor to this repo, human or AI agent.

## Process

We work in Scrum, tracked as GitHub Issues on the [project board](https://github.com/orgs/mlops-2627q1-mds-upc/projects/1) (Backlog → To Do → In Progress → In Review → Done).
Team communication happens on [Discord](https://discord.gg/uG2eACGng).
Full process details (roles, ceremonies, Definition of Done, working agreements) live in [docs/docs/scrum/](docs/docs/scrum/) - read that before picking up work.

## Workflow

1. Pick an issue, assign yourself, move it to **In Progress** on the board.
2. Branch off `main`: `feature/<short-description>` or `fix/<short-description>`.
3. Commit using [Conventional Commits](https://www.conventionalcommits.org/) (enforced by pre-commit) - e.g. `feat: add price model training script`.
4. Open a pull request into `main` with a title that also follows Conventional Commits (checked in CI) - it becomes the squash commit message. Move the issue to **In Review**.
5. At least one approval and passing checks required before merging. No direct pushes to `main`.
6. **Always squash merge** - it's the only merge method enabled on the repo, so `main` gets exactly one commit per PR and stays readable. Merging deletes the branch automatically.
7. After merge, move the issue to **Done**.

## Local setup

```bash
uv sync
pre-commit install
```

## Before opening a PR

```bash
make format   # ruff format + fix
make lint     # ruff check
make test     # pytest
```
