"""On main, only the Deploy workflow targets the self-hosted runner on the VM.

A job on the VM's runner runs on the production machine. Only deploy.yml may
target it, and it starts only on a finished CD run of main or by hand, which
only people with write access can do (EDN-94,
docs/docs/deployment.md#turn-continuous-deployment-on).
This test keeps main's workflows that way. It cannot stop a pull request from a
fork, which runs the workflow files of its own merge commit, so the job it adds
runs beside the CI run in which this test fails: against that, the repository
requires a team member to approve every fork pull request's workflows, after
reading its changes to .github/.
"""

from collections.abc import Iterable
from pathlib import Path
import re
from typing import Any

import pytest
import yaml

from recommenditos.config import PROJ_ROOT

WORKFLOWS = PROJ_ROOT / ".github" / "workflows"
DEPLOY_WORKFLOW = "deploy.yml"
#: The triggers that no pull request can fire: workflow_run (limited to main
#: below) runs main's copy of the workflow, and workflow_dispatch the copy of
#: the ref a person with write access chooses.
ALLOWED_TRIGGERS = {"workflow_run", "workflow_dispatch"}
#: GitHub-hosted runner labels, such as ubuntu-24.04, ubuntu-24.04-arm or
#: windows-2025-vs2026. Anything else counts as self-hosted, an expression
#: that cannot be resolved too.
HOSTED_LABEL = re.compile(r"^(ubuntu|windows|macos)-[0-9A-Za-z.-]+$")
MATRIX_EXPRESSION = re.compile(r"^\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}$")


def triggers(workflow: dict[Any, Any]) -> dict[str, Any]:
    """The workflow's events with their filters; YAML 1.1 reads the key `on` as True."""
    on = workflow.get("on", workflow.get(True))
    if isinstance(on, str):
        return {on: None}
    if isinstance(on, list):
        return {event: None for event in on}
    return dict(on or {})


def matrix_values(job: dict[str, Any], key: str) -> list[Any]:
    """Every value `matrix.<key>` takes, from the matrix and its `include` entries."""
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    if not isinstance(matrix, dict):
        return []
    values = matrix.get(key, [])
    values = list(values) if isinstance(values, list) else [values]
    values += [entry[key] for entry in matrix.get("include", []) if key in entry]
    return values


def runner_labels(job: dict[str, Any]) -> list[list[str]]:
    """The label sets the job may ask for, one per resolved matrix value."""
    runs_on = job.get("runs-on")
    if runs_on is None:
        return []
    if isinstance(runs_on, dict):
        # `group:` picks a runner group, which only self-hosted runners have.
        labels = runs_on.get("labels", [])
        labels = [labels] if isinstance(labels, str) else list(labels)
        return [[*labels, *(["group"] if "group" in runs_on else [])]]
    if isinstance(runs_on, str):
        found = MATRIX_EXPRESSION.match(runs_on)
        if found:
            values = matrix_values(job, found.group(1))
            return [[str(value)] for value in values] or [[runs_on]]
        return [[runs_on]]
    return [[str(label) for label in runs_on]]


def is_self_hosted(labels: Iterable[str]) -> bool:
    return not all(HOSTED_LABEL.match(label) for label in labels)


def violations(workflows: dict[str, dict[Any, Any]]) -> list[str]:
    """What breaks the rule in `workflows`, a map of file name to parsed workflow."""
    found = []
    for name, workflow in sorted(workflows.items()):
        self_hosted = [
            job_id
            for job_id, job in (workflow.get("jobs") or {}).items()
            if any(is_self_hosted(labels) for labels in runner_labels(job))
        ]
        if not self_hosted:
            continue
        if name != DEPLOY_WORKFLOW:
            found.append(f"{name}: jobs {self_hosted} target a self-hosted runner")
        events = triggers(workflow)
        for event in sorted(set(events) - ALLOWED_TRIGGERS):
            found.append(f"{name}: has self-hosted jobs and the trigger {event!r}")
        run_filter = events.get("workflow_run")
        if "workflow_run" in events and (run_filter or {}).get("branches") != ["main"]:
            found.append(f"{name}: has self-hosted jobs and a workflow_run not limited to main")
    return found


def load(directory: Path) -> dict[str, dict[Any, Any]]:
    paths = [*directory.glob("*.yml"), *directory.glob("*.yaml")]
    return {path.name: yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths}


@pytest.mark.req("NFR-13")
def test_only_the_deploy_workflow_runs_on_the_vm_and_only_from_main():
    workflows = load(WORKFLOWS)

    assert violations(workflows) == []


@pytest.mark.req("NFR-13")
def test_the_deploy_workflow_is_seen_to_use_the_vm_runner():
    """Without this, a parser that finds no self-hosted job anywhere would pass the test above."""
    deploy = load(WORKFLOWS)[DEPLOY_WORKFLOW]

    on_the_vm = {
        job_id
        for job_id, job in deploy["jobs"].items()
        if any(is_self_hosted(labels) for labels in runner_labels(job))
    }

    assert on_the_vm == {"deploy", "rollback"}


HOSTED_JOB = {"runs-on": "ubuntu-24.04", "steps": []}
VM_JOB = {"runs-on": ["self-hosted", "recommenditos-vm"], "steps": []}
FROM_MAIN = {"workflow_run": {"workflows": ["CD"], "branches": ["main"]}, "workflow_dispatch": {}}

BAD: dict[str, dict[str, dict[Any, Any]]] = {
    "a pull request's job on the VM": {
        "ci.yml": {True: {"pull_request": None}, "jobs": {"test": VM_JOB}}
    },
    "another workflow on the VM, even from main": {
        "other.yml": {True: FROM_MAIN, "jobs": {"job": VM_JOB}}
    },
    "a push trigger beside the deploy": {
        DEPLOY_WORKFLOW: {True: {**FROM_MAIN, "push": None}, "jobs": {"deploy": VM_JOB}}
    },
    "a workflow_run of any branch": {
        DEPLOY_WORKFLOW: {
            True: {"workflow_run": {"workflows": ["CD"]}},
            "jobs": {"deploy": VM_JOB},
        }
    },
    "a runner group": {"ci.yml": {True: "push", "jobs": {"test": {"runs-on": {"group": "vm"}}}}},
    "a label from the matrix": {
        "ci.yml": {
            True: ["pull_request"],
            "jobs": {
                "test": {
                    "runs-on": "${{ matrix.runner }}",
                    "strategy": {
                        "matrix": {
                            "include": [{"runner": "ubuntu-24.04"}, {"runner": "self-hosted"}]
                        }
                    },
                }
            },
        }
    },
    "an expression it cannot resolve": {
        "ci.yml": {True: "pull_request", "jobs": {"test": {"runs-on": "${{ inputs.runner }}"}}}
    },
}


@pytest.mark.req("NFR-13")
@pytest.mark.parametrize("case", BAD)
def test_a_workflow_that_could_put_other_code_on_the_vm_is_found(case: str):
    assert violations(BAD[case]) != []


@pytest.mark.req("NFR-13")
def test_hosted_jobs_and_the_deploy_workflow_as_it_should_be_pass():
    workflows = {
        "ci.yml": {
            True: {"pull_request": None, "push": None},
            "jobs": {
                "test": HOSTED_JOB,
                "report": {
                    "runs-on": "${{ matrix.runner }}",
                    "strategy": {"matrix": {"include": [{"runner": "ubuntu-24.04-arm"}]}},
                },
                "windows": {"runs-on": "windows-2025-vs2026"},
            },
        },
        DEPLOY_WORKFLOW: {True: FROM_MAIN, "jobs": {"prepare": HOSTED_JOB, "deploy": VM_JOB}},
    }

    assert violations(workflows) == []
