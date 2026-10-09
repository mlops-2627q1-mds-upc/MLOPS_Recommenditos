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
`nfr11_alpha_sweep.py` measures the control clause across three independent i.i.d. splits and
several significance levels at once: p-values do not depend on the threshold, so each trial is
computed once and evaluated at each level. It exists because the Bonferroni threshold is
`P_VAL / n_features`, which makes the family-wise false-alarm rate 5 % by construction, exactly
the "at most 1 of 20 windows" NFR-11 promises, leaving the requirement no margin.
`nfr11_alpha_sweep_results.json` and `nfr11_alpha_sweep_results.txt` are its output (2026-09-29).

The level the drift job actually runs at is **0.005**, decided in EDN-29. `nfr11_check.py` keeps
`P_VAL = 0.05`, because its committed output is the evidence EDN-14 cites and re-running it at a
different level would silently change what that entry points at. Use `nfr11_alpha_sweep.py` to
re-check the decided level.

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

## The `split` stage's size gate (EDN-39, EDN-40)

`split_gate.py` chooses the four constants the gate of `recommenditos/data/split_data.py` is
calibrated with, and reports the realised split sizes, the group count, the largest dealer and
the supported-make list that the dataset card states as fact.
`DISPERSION_SIGMAS` is chosen from a stated false-alarm budget - at most 1 seed in 1,000 may
turn the stage red on a pool satisfying the concentration limit - measured by sweeping 2,000
seeds on three pools: the real snapshot and the two synthetic fixtures the tests use.
It exists because the first version of the constant was justified by a sweep that stopped at 200
seeds and was wrong by 2,000, in both directions at once: sound splits would have been failed and
a calibration set at 0.58 of its intended size would have passed.
`split_gate_results.txt` is the output of the run recorded in EDN-39 and EDN-40 (2026-09-30).

Unlike the other scripts here it imports the stage's own functions rather than re-deriving them,
so the justification cannot drift from the code it justifies; it therefore has to run from the
repository root.
The `split` block of `params.yaml` is copied into it rather than loaded, the way `nfr11_check.py`
keeps its own `P_VAL`, so the committed output stays attached to the values it was measured at.

## SC-04 coverage per make (model card)

`make_support.py` compares EDN-05's support threshold, which admits a make into scope at 300
listings, against SC-04's segment size, which only checks a segment once it holds 500 test rows.
It reports which supported makes therefore never reach the quality gate.
`make_support_results.txt` is the output of the run cited in the model card (2026-09-29).

## Extended feature set: `model_version` and `weight_kg` (EDN-41, model card)

`extended_features.py` answers the two questions the `features` stage had to decide and the
specification did not cover.
It measures how much the `model_version` normalisation merges - per make and model, and against
the two finer alternatives of prefixing the token with make and model or keeping two tokens after
an engine letter - and what range `weight_kg` covers once parsed out of its text form.
`extended_features_results.txt` is the output of the run cited in EDN-41, the model card and the
dataset card (2026-09-30).

## The feature space over the served makes (EDN-67, model card)

`served_vocabulary.py` builds the `features` stage's vocabulary twice from the same `split` output,
once over every training row and once over the training rows of the supported makes, and reports
the level counts, the equipment columns, the `model_version` levels and their coverage at both
floors `params.yaml` quotes, and the rows each frame keeps.
It is the before-and-after of moving the supported-make filter in front of the vocabulary (issue #63).
Like `split_gate.py` it imports the stage's own functions, so it runs from the repository root, and
it reads the `split` artefacts under `data/processed/` rather than the raw CSV, so `dvc pull` is all
it needs.
`served_vocabulary_results.txt` is the output of the run cited in EDN-67, the model card and
`params.yaml` (2026-10-05).

## Where early stopping lands, and what tuning could still buy (EDN-70, issue #64)

`early_stopping_ceiling.py` fits each LightGBM variant once with `n_estimators` at 20,000, every other setting as `params.yaml` has it, and reports the round early stopping chose, the validation curve at fixed tree counts, and what the trees cost: fit seconds, `booster.txt` size, and the single-row latency of a prediction and of the native SHAP export FR-08 serves.
It is the evidence behind the committed ceiling of 5,000.
`early_stopping_ceiling_results.txt` is the output of the run cited in EDN-70, the model card and `params.yaml` (2026-10-05).

`tuning_sweep.py` is the evidence behind EDN-73, not a tuning run: a small grid of `learning_rate` and `num_leaves` for the candidate `lgbm-basic`, each point fitted with early stopping deciding the trees, and a paired bootstrap by seller group of whether the validation split can tell any point from the committed one.
`tuning_sweep_results.txt` is its output (2026-10-05).
It was also the first decision step of the tuning protocol in `docs/docs/pipeline.md`, which adopts a point only when its simultaneous interval, from a max-statistic bootstrap over all the points, lies below zero.
Since issue #88 that step is `recommenditos/modeling/tune.py`, which runs it for any grid and variant and measures and tracks every fit (EDN-78); its results are kept in MLflow, in the experiment `recommenditos-price-tuning`, not here.
Its first sweep, `88-lgbm-basic-lr-leaves`, repeats this grid and reproduces `tuning_sweep_results.txt` exactly, apart from the fit times.
This script now imports the seller-group join, the paired bootstrap and the simultaneous intervals from that module, with the same draws from the same seed, so it still reproduces `tuning_sweep_results.txt`.

Both import the `train` stage's own `read_matrices` and `fit_variant`, so the fits are the stage's fits, and both read the `train` and `validation` matrices only: the test split is never opened, because a choice made on test rows would leak them into the model the gate then judges.
Run them from the repository root after `dvc pull`, with tracking off (`MLFLOW_TRACKING_URI= uv run python reports/analysis/<script>.py`).

## The preprocess row funnel (dataset card, issue #34)

`preprocess_funnel.py` reports how many listings each row rule of the `preprocess` stage removes
from the real snapshot, because the pipeline only ever logs the funnel for the data it was run on,
and a committed artefact is what the report can cite instead of a log line somebody has to have kept.
Unlike the scripts above it calls the stage's own steps - the group key and PII drop, the row rules
and the target - instead of re-implementing any of them, so its numbers cannot drift from what the
pipeline does.
`preprocess_funnel_results.txt` is the output of the run cited in the dataset card (2026-09-30).

## Secret scanning and the DagsHub token (EDN-71, NFR-15)

`secret_scan_dagshub_token.py` plants fake DagsHub tokens, 200 per place, in the thirteen places a real one has been or could plausibly be written in this project, and measures how often gitleaks' default rules, gitleaks with `.gitleaks.toml`, and detect-secrets' default plugins find them.
It also measures the Shannon entropy of such tokens against the 3.5 threshold of gitleaks' generic rule, and what each scanner flags on the committed tree and, for gitleaks, on the history of every branch.
It needs no data and no credentials: every token is generated and lives only in a temporary directory, and the shape it copies, 40 lowercase hex characters with no prefix, was read off a real token without printing it.
Run it from the repository root with the gitleaks binary the CI job pins, `python reports/analysis/secret_scan_dagshub_token.py path/to/gitleaks`; it fetches detect-secrets through `uvx`.
`secret_scan_dagshub_token_results.txt` is the output of the run recorded in EDN-71 (2026-10-05).

## What CodeCarbon measures on our machines (EDN-69, issue #38)

`codecarbon_validity.py` tests the assumption issue #38 was written on - that CodeCarbon *measures* the energy of a fit - before any figure is quoted.
It asks which CPU power method CodeCarbon 3.3.1 falls back to when RAPL is root-only, whether its figure follows the CPU work of four different workloads in `process` and in `machine` mode, whether the offline tracker makes any network call, how completely `measure_power_secs` of 1 s and 15 s integrate a fit, what the tracker costs on the test fixture, and what `on_csv_write="append"` does to a DVC stage output.
`codecarbon_validity_results.txt` is its output (2026-10-05), on a laptop other jobs were sharing, with the load average beside every row.
`codecarbon_github_runner.txt` is the same question asked of a GitHub-hosted runner, through a temporary CI step on pull request #68, with the step and its log.
Unlike the scripts below it needs no raw data: it uses the synthetic fixture and CodeCarbon itself.

```bash
uv run python reports/analysis/codecarbon_validity.py
```

`rapl_validation.py` is the one-off check EDN-69 chose instead of granting RAPL access permanently: it measures an idle baseline and three fits of each ladder variant with the RAPL counters themselves, with CodeCarbon's `cpu_load` estimate forced as the pipeline gets it, and with CodeCarbon's own RAPL figure, all around the same fit.
It refuses to run on a busy machine, because RAPL counts the whole package, and when the counters are not readable it prints the command that makes them readable.
`rapl_validation_results.txt` is its output.

```bash
sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj /sys/class/powercap/intel-rapl-mmio:*/energy_uj
MLFLOW_TRACKING_URI= uv run python reports/analysis/rapl_validation.py > reports/analysis/rapl_validation_results.txt
sudo chmod 0400 /sys/class/powercap/intel-rapl:*/energy_uj /sys/class/powercap/intel-rapl-mmio:*/energy_uj
```

Run them against the raw dataset, which is not in the repo (NFR-08; EDN-35 for where it does live).
Use the `download` stage's own cache rather than a second 548 MB copy: `uv run dvc repro download`
with `download.source: zenodo` puts the pinned file there, verified against `download.md5`, and
reuses it on every later run.

```bash
ROOT="$PWD"
CSV="$ROOT/data/external/autoscout24_dataset_20251108.csv"
python reports/analysis/nfr11_check.py "$CSV"
python reports/analysis/fillrates.py "$CSV"
python reports/analysis/make_support.py "$CSV"
python reports/analysis/extended_features.py "$CSV"
uv run python reports/analysis/preprocess_funnel.py "$CSV"
PYTHONPATH=reports/analysis python reports/analysis/nfr11_model_excluded.py "$CSV"
PYTHONPATH=reports/analysis python reports/analysis/nfr11_alpha_sweep.py "$CSV"

# nfr11_diag.py is kept exactly as it was run and hard-codes `cars.csv` in the working
# directory, so give the cache that name inside the gitignored cache directory.
ln -sf "$CSV" "$ROOT/data/external/cars.csv"
cd "$ROOT/data/external" && PYTHONPATH="$ROOT/reports/analysis" python "$ROOT/reports/analysis/nfr11_diag.py"
```

The scope of these scripts follows the pipeline: used cars only (EDN-04), listings registered
after the age reference date dropped (EDN-22), the training price range, deduplicated before any
split. Re-run them whenever a decision changes that scope, because their outputs are cited as
EDN evidence.

## The serving runtime's footprint (EDN-47)

`runtime_footprint.py` builds two environments from `uv.lock` into a scratch directory: the full one a plain `uv sync` installs, and the runtime one the API image installs with `--no-default-groups`.
It measures each one's site-packages as installed and after `compileall`, lists the largest distributions by the files their `RECORD` names, and then loads every committed bundle under `models/` in both environments and prices the first rows of its test matrix, so the runtime set is shown to serve rather than only to import.
`runtime_footprint_results.txt` is the output of the run recorded in EDN-47 (2026-10-05).
Its last two lines say which group packages the serving path imported: `tqdm` in the full environment, because `recommenditos/config.py` imports it when it can, and nothing in the runtime one, where it is absent and the import is skipped.

Unlike the scripts above it needs no raw dataset, only `models/` and `data/processed/features/`, so run it after `dvc pull`, from the repository root:

```bash
uv run python reports/analysis/runtime_footprint.py /tmp/footprint
```

`serving_imports.py` is the trace behind EDN-47's runtime list: it runs the same load-and-predict over `models/` with a hook on `__import__`, and reports for every package the serving path imported which `recommenditos` modules import it, or, when none does, which package imported it first.
`serving_imports_results.txt` holds its output in the full environment and in `make test-serving`'s `.venv-serving/` (2026-10-05).
The difference between the two is the optional imports: tqdm, psutil and charset_normalizer appear only in the full one.

```bash
uv run python reports/analysis/serving_imports.py
.venv-serving/bin/python reports/analysis/serving_imports.py
```

## Great Expectations on pandas 3 (EDN-68)

`gx_probe.py` runs every expectation kind the suites use against a small pandas-3 frame that satisfies it and one that breaks it, because a dependency resolving is not the same as it working on our dtypes.
It also builds the same file context twice and lists which files differ, which is what decided that `gx/` is a cached stage output rather than committed configuration.
It needs no data: `uv run python reports/analysis/gx_probe.py`.
`gx_probe_results.txt` is the output of the run cited in EDN-68 (2026-10-05).

`gx_fingerprint_cost.py` runs the real `validate-data` stage on the real snapshot with Great Expectations' batch fingerprint switched off, as the stage does, or on, against a context in a temporary directory, so the pipeline's outputs are not touched.
Run each mode under `/usr/bin/time -v`; `gx_fingerprint_cost_results.txt` holds the wall time and peak RSS of both runs cited in EDN-68 (2026-10-05).

`mileage_scope.py` measures the alternative EDN-72 rejected for the mileage rule: dropping the three cleaned listings above FR-03's 1,000,000 km in preprocessing.
It runs the real `split`, `features`, `train` and `evaluate` stages on the interim frame without them, in a temporary directory with tracking off, and compares the result with the committed `metrics.json`.
`mileage_scope_results.txt` is the output of the run cited in EDN-72 (2026-10-05), taken against the `metrics.json` of commit `abea241`.

## What a run's provenance tags identify (EDN-74, issue #79)

`run_provenance_check.py` takes the runs the committed pipeline points at - the run id in each `models/<variant>/model.json`, after checking that the bundle is the one the committed `dvc.lock` records - reads them back from the tracking server, and compares them with that lock.
For every run it checks that each `train.deps.<path>` tag equals the `md5` the lock records for that dependency of `train@<variant>`, that each `evaluate.deps.<path>` equals the one under `evaluate`, with no dependency missing and no tag extra, and that `git_dirty` and `evaluate.git_dirty` are `false`; for a run made before the fix it also says whether its `dvc_lock_md5` equals the committed lock's digest.
It needs the tracking credentials in `.env`, `models/` from `dvc pull`, and a clean tree, and runs from the repository root: `uv run python reports/analysis/run_provenance_check.py`.
`run_provenance_check_results.txt` holds two runs of it (2026-10-05): against the runs `main`'s models pointed at before the fix, and against the four runs of the `dvc repro` this pull request committed.
A third, appended on 2026-10-06, checks the runs of the next committed `dvc repro`, issue #64's, which pass the same way.
