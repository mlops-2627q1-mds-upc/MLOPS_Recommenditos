"""The serving runtime of EDN-47: what it installs, how big it is, and whether it serves.

Question: once `pyproject.toml` keeps only what serving imports in
`[project] dependencies` and everything else in `[dependency-groups]`, how much
of the environment does the API image still have to carry, and can that set on
its own load the committed model bundles and price a listing?

Both environments are built from `uv.lock` into a scratch directory, never into
`.venv`, with the same flags the image and a contributor use:

- `full`: `uv sync --locked`, which installs every group (`default-groups = "all"`).
- `runtime`: `uv sync --locked --no-default-groups`, the set the API image installs.

Each is measured twice: as installed, and after `compileall`, because an image
built with `UV_COMPILE_BYTECODE=1` carries the bytecode and a contributor's
`.venv` grows it on first import. The size per distribution is summed over the
files its `RECORD` lists, so `scipy` and `scipy.libs` count as the one package
they are, which `du` over top-level directories does not do.

The serving check loads every bundle under `models/` with `load_model` in each
environment, prices the first rows of its own test matrix and compares the two
answers, so "it imports" is not mistaken for "it predicts the same thing".

Run from the repository root after `dvc pull` (or `dvc checkout train
features`), so that `models/` and `data/processed/features/` exist:

    uv run python reports/analysis/runtime_footprint.py <scratch-dir>
"""

import json
import os
from pathlib import Path
import subprocess
import sys

ENVIRONMENTS = {
    "full": [],
    "runtime": ["--no-default-groups"],
}

ROWS = 5
TOP = 12

_SIZES = r"""
import json, sysconfig
from importlib.metadata import distributions
from pathlib import Path
site = Path(sysconfig.get_paths()["purelib"])
sizes = {}
for dist in distributions():
    total = 0
    for file in dist.files or ():
        path = Path(dist.locate_file(file))
        if path.is_file():
            total += path.stat().st_size
    sizes[f"{dist.metadata['Name']}=={dist.version}"] = total
everything = sum(p.stat().st_size for p in site.rglob("*") if p.is_file() and not p.is_symlink())
print(json.dumps({"site": str(site), "total": everything, "distributions": sizes}))
"""

_PREDICT = r"""
import json, sys
from pathlib import Path
import pandas as pd
from recommenditos.modeling.model import load_model
rows = int(sys.argv[1])
answers = {}
for bundle in sorted(Path("models").iterdir()):
    if not (bundle / "model.json").exists():
        continue
    model = load_model(bundle)
    frame = pd.read_parquet(
        Path("data/processed/features") / model.feature_set / "test.parquet"
    ).head(rows)
    answers[model.variant] = model.predict_eur(frame).tolist()
groups = ("mlflow", "dvc", "requests", "tqdm", "matplotlib", "jupyterlab", "pytest", "mkdocs")
imported = sorted(name for name in groups if name in sys.modules)
print(json.dumps({"answers": answers, "group_modules_imported": imported}))
"""


def _python(environment: Path) -> Path:
    return environment / "bin" / "python"


def _sync(environment: Path, flags: list[str]) -> None:
    subprocess.run(
        ["uv", "sync", "--locked", *flags],
        env={**os.environ, "UV_PROJECT_ENVIRONMENT": str(environment)},
        check=True,
        capture_output=True,
    )


def _measure(environment: Path) -> dict:
    result = subprocess.run(
        [str(_python(environment)), "-c", _SIZES], check=True, capture_output=True, text=True
    )
    return json.loads(result.stdout)


def _compile(environment: Path, site: str) -> None:
    subprocess.run(
        [str(_python(environment)), "-m", "compileall", "-q", "-j0", site],
        check=False,  # a handful of test fixtures inside wheels do not compile
        capture_output=True,
    )


def _predict(environment: Path) -> dict:
    result = subprocess.run(
        [str(_python(environment)), "-c", _PREDICT, str(ROWS)],
        check=True,
        capture_output=True,
        text=True,
        # Tracking is irrelevant to loading a bundle; keep `.env` out of it anyway.
        env={**os.environ, "MLFLOW_TRACKING_URI": ""},
    )
    return json.loads(result.stdout)


def _mb(size: int) -> str:
    return f"{size / 1_000_000:,.0f} MB"


def main(scratch: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    measured = {}
    for label, flags in ENVIRONMENTS.items():
        environment = scratch / label
        _sync(environment, flags)
        installed = _measure(environment)
        _compile(environment, installed["site"])
        compiled = _measure(environment)
        measured[label] = {
            "environment": environment,
            "installed": installed,
            "compiled": compiled,
            "predictions": _predict(environment),
        }

    full, runtime = measured["full"], measured["runtime"]
    print("Installed size of site-packages (decimal MB)")
    print(f"{'':10}{'as installed':>16}{'with bytecode':>16}{'packages':>10}")
    for label, record in measured.items():
        print(
            f"{label:10}{_mb(record['installed']['total']):>16}"
            f"{_mb(record['compiled']['total']):>16}"
            f"{len(record['installed']['distributions']):>10}"
        )
    share = runtime["compiled"]["total"] / full["compiled"]["total"]
    print(f"runtime / full, with bytecode: {share:.0%}")
    print()

    # Per distribution as installed: `RECORD` lists what the installer wrote, and
    # bytecode that `compileall` adds afterwards is in no distribution's `RECORD`.
    for label in ("runtime", "full"):
        sizes = measured[label]["installed"]["distributions"]
        print(f"Largest {TOP} distributions of {label}, as installed")
        for name, size in sorted(sizes.items(), key=lambda item: -item[1])[:TOP]:
            print(f"  {name:40}{_mb(size):>10}")
        print()

    print("The runtime environment, every distribution")
    print("  " + ", ".join(sorted(runtime["installed"]["distributions"], key=str.lower)))
    print()

    print(f"Serving check: every bundle under models/, first {ROWS} test rows, in euros")
    for variant, answer in runtime["predictions"]["answers"].items():
        same = answer == full["predictions"]["answers"][variant]
        rounded = ", ".join(f"{value:,.0f}" for value in answer)
        print(f"  {variant:15}{rounded}   identical to full: {same}")
    print(
        "  group packages imported by the serving path in full: "
        f"{full['predictions']['group_modules_imported'] or 'none'}"
    )
    print(
        "  group packages imported by the serving path in runtime: "
        f"{runtime['predictions']['group_modules_imported'] or 'none'}"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
