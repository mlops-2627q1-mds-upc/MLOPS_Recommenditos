"""Which packages the serving path imports, and who imports each one first (EDN-47).

Question: which distributions does the API need, as opposed to which ones the
project happens to have installed? Reading the `import` lines of the serving
modules answers half of it; the other half is what those packages import in
turn, some of it only when it can.

This runs the serving path the way the API will - `load_model` on every bundle
under `models/`, then `predict_eur` on the first rows of its own test matrix -
with a hook on `__import__` that records, for every top-level module that was
not imported yet, which module asked for it first. Each module is then mapped
to the distribution that installs it. An importer inside `recommenditos` makes
the package a direct need of the serving path; an importer inside another
package makes it that package's business, and whether it is a requirement or an
optional import is what running this in two environments shows.

Run it with the interpreter of the environment to trace, from the repository
root, after `dvc pull` (or `dvc checkout train features`):

    uv run python reports/analysis/serving_imports.py                 # full
    .venv-serving/bin/python reports/analysis/serving_imports.py      # runtime
"""

import builtins
from importlib.metadata import packages_distributions
from pathlib import Path
import sys

_first_importer: dict[str, str] = {}
_original_import = builtins.__import__


def _recording_import(name, globals=None, locals=None, fromlist=(), level=0):
    top = name.partition(".")[0]
    if level == 0 and top not in sys.modules and top not in _first_importer:
        _first_importer[top] = (globals or {}).get("__name__", "?")
    return _original_import(name, globals, locals, fromlist, level)


def main() -> None:
    builtins.__import__ = _recording_import
    # The seam first, so that every package it needs is recorded against the
    # serving module that asked for it rather than against this script.
    from recommenditos.modeling.model import load_model  # noqa: PLC0415

    import pandas as pd  # noqa: PLC0415

    for bundle in sorted(Path("models").iterdir()):
        if not (bundle / "model.json").exists():
            continue
        model = load_model(bundle)
        frame = pd.read_parquet(
            Path("data/processed/features") / model.feature_set / "test.parquet"
        ).head(5)
        model.predict_eur(frame)
    builtins.__import__ = _original_import

    owners = packages_distributions()
    rows = []
    for module, importer in _first_importer.items():
        if module in sys.stdlib_module_names or module not in owners:
            continue
        direct = importer.partition(".")[0] == "recommenditos"
        rows.append((not direct, ", ".join(sorted(set(owners[module]))), module, importer))

    print(f"Interpreter: {Path(sys.prefix).name}")
    print(f"{'distribution':22}{'module':22}first imported by")
    for transitive, distribution, module, importer in sorted(rows):
        kind = "  (another package)" if transitive else ""
        print(f"{distribution:22}{module:22}{importer}{kind}")


if __name__ == "__main__":
    main()
