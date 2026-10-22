"""The CI check that reports stages whose `dvc.lock` entry is stale (#193, EDN-90).

A stale lock means the committed outputs were not produced by the committed code
and parameters, so the pipeline no longer reproduces what the repository shows
(NFR-06). The end-to-end tests build a small DVC project in a temporary git
repository: a `prepare` stage reading a data file that only `dvc pull` would
provide, a `train` stage expanded by `foreach`, and an `evaluate` stage that
depends on the directory the cached models live in, as the real `evaluate` does.
"""

import hashlib
from pathlib import Path
import subprocess
import sys

import pytest
from tools.stale_lock import checkable, collect, main, stale_stages, summary
import yaml

pytestmark = pytest.mark.req("NFR-06")

DVC_YAML = """\
stages:
  prepare:
    cmd: python src/prepare.py
    deps: [src/prepare.py, data/raw.csv]
    params: [prepare.seed]
    outs: [data/clean.csv]
  train:
    foreach: ${train.variants}
    do:
      cmd: python src/train.py ${item}
      deps: [src/train.py, data/clean.csv]
      params: [train.lr]
      outs:
        - models/${item}
  evaluate:
    cmd: python src/evaluate.py
    deps: [src/evaluate.py, models]
    outs:
      - report.json:
          cache: false
"""

PARAMS = {"prepare": {"seed": 1}, "train": {"variants": ["a", "b"], "lr": 0.1}}

FILES = {
    "src/prepare.py": "print('prepare')\n",
    "src/train.py": "print('train')\n",
    "src/evaluate.py": "print('evaluate')\n",
    "report.json": "{}\n",
    "models/.gitignore": "/a\n/b\n",
    "data/.gitignore": "/raw.csv\n/clean.csv\n",
}


def file_dep(root: Path, path: str) -> dict[str, object]:
    data = (root / path).read_bytes()
    md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
    return {"path": path, "hash": "md5", "md5": md5, "size": len(data)}


def pulled(path: str, *, directory: bool = False) -> dict[str, object]:
    """A lock entry for data or a model that a clean checkout does not have."""
    md5 = hashlib.md5(path.encode(), usedforsecurity=False).hexdigest()
    entry: dict[str, object] = {
        "path": path,
        "hash": "md5",
        "md5": md5 + (".dir" if directory else ""),
    }
    if directory:
        entry["nfiles"] = 2
    return entry


def write_lock(root: Path) -> None:
    """The `dvc.lock` a `dvc repro` of the current files would have written."""
    stages: dict[str, object] = {
        "prepare": {
            "cmd": "python src/prepare.py",
            "deps": [pulled("data/raw.csv"), file_dep(root, "src/prepare.py")],
            "params": {"params.yaml": {"prepare.seed": 1}},
            "outs": [pulled("data/clean.csv")],
        },
        "evaluate": {
            "cmd": "python src/evaluate.py",
            "deps": [pulled("models", directory=True), file_dep(root, "src/evaluate.py")],
            "outs": [file_dep(root, "report.json")],
        },
    }
    for item in ("a", "b"):
        stages[f"train@{item}"] = {
            "cmd": f"python src/train.py {item}",
            "deps": [pulled("data/clean.csv"), file_dep(root, "src/train.py")],
            "params": {"params.yaml": {"train.lr": 0.1}},
            "outs": [pulled(f"models/{item}", directory=True)],
        }
    lock = {"schema": "2.0", "stages": stages}
    (root / "dvc.lock").write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")


def run(root: Path, *args: str) -> None:
    subprocess.run(args, cwd=root, check=True, capture_output=True)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A DVC project whose lock is in sync, checked out without its data."""
    monkeypatch.setenv("DVC_NO_ANALYTICS", "1")
    run(tmp_path, "git", "init", "--quiet")
    run(tmp_path, sys.executable, "-m", "dvc", "init", "--quiet")
    for path, text in FILES.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text, encoding="utf-8")
    (tmp_path / "dvc.yaml").write_text(DVC_YAML, encoding="utf-8")
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(PARAMS), encoding="utf-8")
    write_lock(tmp_path)
    run(tmp_path, "git", "add", ".")
    return tmp_path


def test_an_in_sync_lock_reports_nothing_although_the_data_and_models_are_missing(project):
    assert collect(project) == {}


def test_a_comment_in_a_code_dependency_reports_every_stage_that_lists_it(project):
    with (project / "src/train.py").open("a", encoding="utf-8") as handle:
        handle.write("# A comment changes the hash too.\n")

    assert collect(project) == {
        "train@a": ["src/train.py (modified)"],
        "train@b": ["src/train.py (modified)"],
    }


def test_a_changed_parameter_reports_the_stages_that_read_it(project):
    params = {**PARAMS, "train": {**PARAMS["train"], "lr": 0.2}}
    (project / "params.yaml").write_text(yaml.safe_dump(params), encoding="utf-8")

    assert collect(project) == {
        "train@a": ["params.yaml: train.lr (modified)"],
        "train@b": ["params.yaml: train.lr (modified)"],
    }


def test_main_writes_the_job_summary_and_one_warning_and_still_succeeds(
    project, tmp_path_factory, monkeypatch, capsys
):
    (project / "src/evaluate.py").write_text("print('changed')\n", encoding="utf-8")
    summary_file = tmp_path_factory.mktemp("summary") / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

    assert main(["--root", str(project)]) == 0

    warnings = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("::warning")
    ]
    assert len(warnings) == 1
    assert "evaluate" in warnings[0]
    assert "| `evaluate` | `src/evaluate.py (modified)` |" in summary_file.read_text(
        encoding="utf-8"
    )


TRACKED = frozenset(
    {
        "src/train.py",
        "params.yaml",
        "models/.gitignore",
        "reports/metrics/a.json",
        "reports/metrics/b.json",
    }
)
OUTPUTS = ["data/clean.csv", "models/a", "reports/metrics"]


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/train.py", True),
        # Pulled from DVC: a clean checkout does not have it.
        ("data/clean.csv", False),
        # Never in the repository at all, like the source CSV `download` reads.
        ("data/external/source.csv", False),
        # Git tracks only its `.gitignore`; the cached model directories in it are pulled.
        ("models", False),
        # An output committed to git (`cache: false`), read by a later stage.
        ("reports/metrics", True),
    ],
)
def test_only_dependencies_a_clean_checkout_holds_are_checked(path, expected):
    assert checkable(path, TRACKED, OUTPUTS) is expected


def test_the_filter_keeps_code_parameters_and_commands_and_drops_missing_data():
    status = {
        "train@a": [
            {
                "changed deps": {
                    "data/clean.csv": "deleted",
                    "models": "modified",
                    "src/train.py": "modified",
                    "params.yaml": {"train.lr": "modified"},
                }
            },
            {"changed outs": {"models/a": "not in cache"}},
            "changed command",
        ],
        "prepare": [{"changed deps": {"data/clean.csv": "deleted"}}],
    }

    assert stale_stages(status, TRACKED, OUTPUTS) == {
        "train@a": [
            "src/train.py (modified)",
            "params.yaml: train.lr (modified)",
            "command changed",
        ],
    }


def test_the_summary_says_so_when_the_lock_is_in_sync():
    assert "matches the code and parameters of every stage" in summary({})


def test_a_check_that_cannot_run_says_so_instead_of_passing_for_a_clean_lock(
    project, tmp_path_factory, monkeypatch, capsys
):
    # A command that names a parameter params.yaml lacks: `dvc status` fails.
    broken = DVC_YAML.replace("python src/evaluate.py", "python src/evaluate.py ${missing.key}")
    (project / "dvc.yaml").write_text(broken, encoding="utf-8")
    summary_file = tmp_path_factory.mktemp("summary") / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

    assert main(["--root", str(project)]) == 1

    warnings = [
        line for line in capsys.readouterr().out.splitlines() if line.startswith("::warning")
    ]
    assert len(warnings) == 1
    assert warnings[0].startswith("::warning title=Stale dvc.lock check failed::")
    assert "missing.key" in warnings[0]
    text = summary_file.read_text(encoding="utf-8")
    assert "The check could not run" in text
    assert "matches the code" not in text
