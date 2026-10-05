"""Which packages the serving path imports, and who imports each one first (EDN-47).

Question: which distributions does the API need, as opposed to which ones the
project happens to have installed? Reading the `import` lines of the serving
modules answers half of it; the other half is what those packages import in
turn, some of it only when it can.

This runs the serving path the way the API will - `load_model` on every bundle
under `models/`, then `predict_eur` on the first rows of its own test matrix -
with a hook on `__import__` that records which module asked for which top-level
module. Each module is then mapped to the distribution that installs it. A
package some `recommenditos` module imports is a direct need of the serving
path, and the report names those modules; a package only other packages import
is their business, and whether it is a requirement of theirs or an optional
import is what running this in two environments shows.

Run it with the interpreter of the environment to trace, from the repository
root, after `dvc pull` (or `dvc checkout train features`):

    uv run python reports/analysis/serving_imports.py                 # full
    .venv-serving/bin/python reports/analysis/serving_imports.py      # runtime
"""

import builtins
from importlib.metadata import packages_distributions
from pathlib import Path
import sys

_importers: dict[str, list[str]] = {}
_original_import = builtins.__import__


def _recording_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0:
        importer = (globals or {}).get("__name__", "?")
        seen = _importers.setdefault(name.partition(".")[0], [])
        if importer not in seen:
            seen.append(importer)
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
    for module, importers in _importers.items():
        if module in sys.stdlib_module_names or module not in owners or module not in sys.modules:
            continue
        ours = sorted({name for name in importers if name.partition(".")[0] == "recommenditos"})
        by = ", ".join(ours) if ours else f"only other packages, first {importers[0]}"
        rows.append((not ours, ", ".join(sorted(set(owners[module]))), module, by))

    print(f"Interpreter: {Path(sys.prefix).name}")
    print(f"{'distribution':22}{'module':22}imported by")
    for _, distribution, module, by in sorted(rows):
        print(f"{distribution:22}{module:22}{by}")


if __name__ == "__main__":
    main()
