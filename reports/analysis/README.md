One-off analyses
================

Scripts behind EDN entries, kept so a decision can be re-checked later.
They are not part of the DVC pipeline, and `[tool.ruff]` in `pyproject.toml` excludes this directory, so they stay as they were run.

## NFR-11 drift feasibility (EDN-14)

`nfr11_check.py` measures whether an `ES` replay window is flagged as input drift, and whether a
control window is not.
`nfr11_diag.py` varies how the control is drawn (seller-grouped vs i.i.d.) and how large the
reference is, and compares the effect sizes of `ES` windows against control windows.
`nfr11_results.json` is the output of the run recorded in EDN-14 (2026-09-23).
`nfr11_model_excluded.py` re-measures both halves of NFR-11 in one configuration, because
EDN-14 took its `ES` half from `nfr11_check.py` and its control half from `nfr11_diag.py`,
whose settings differ; it also tests whether the control's false alarms come from `model`.
`nfr11_model_excluded_results.json` is its output (2026-09-29).

Both emulate alibi-detect 0.13 `TabularDrift` with scipy: KS (`mode="asymp"`) for numerical
features, chi-square contingency for categorical ones, Bonferroni at `p = 0.05 / n_features`.
alibi-detect itself is not installed, because it requires numpy<2 and pandas<3 (project brief
section 6); the tests and the aggregation rule are taken from its v0.13 source.
alibi-detect does no NaN handling at all, so a missing value would make a p-value NaN and read as
"no drift"; the scripts therefore add an explicit missing indicator per numerical feature.

## FR-01 fill rates (EDN-15)

`fillrates.py` measures how often each input field of the basic feature set is filled in the
training scope, to find the fields that FR-01 may not leave optional because the model would never
see them missing.
`fillrates_results.txt` is the output of the run recorded in EDN-15 (2026-09-29).

Run them against the raw dataset, which is not in the repo (NFR-08, EDN-07):

```bash
curl -L -o cars.csv "https://zenodo.org/records/17643343/files/autoscout24_dataset_20251108.csv?download=1"
python reports/analysis/nfr11_check.py cars.csv
python reports/analysis/nfr11_diag.py    # expects cars.csv in the working directory
python reports/analysis/fillrates.py cars.csv
PYTHONPATH=reports/analysis python reports/analysis/nfr11_model_excluded.py cars.csv
```
