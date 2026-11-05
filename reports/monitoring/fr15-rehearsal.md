# FR-15 rehearsal on the VM, 2026-11-05

The retraining chain of FR-15 ([#231](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/231)), run once end to end on the production VM: finding, retraining, quality gate, promotion, deployment, the same replay again, and a rollback drill.
Times are UTC.
The raw evidence is in [fr15/](fr15/).

| Step | Commit or run | Result |
|------|---------------|--------|
| Finding | replay `fr15-before`, windows 11 to 14 | `ES` windows flagged in 14 and 15 features, MdAPE 9.09 % and 9.16 % |
| Recovery criterion | [#231 comment](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/231#issuecomment-6655379311), posted before any run | three points, below |
| Split with `ES` | [#300](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/300), `ea2b384` | `split.holdout_seller_share` 0.4 (EDN-102) |
| Retraining | Retrain run [42028216667](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/actions/runs/42028216667) on `ea2b384` | gate passed, `lgbm-basic` `2a60912b` |
| Promotion | [#303](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/303), merged by Lukas as `ccb175e`, [review](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/303#issuecomment-6661992152) | `Promotion gate` check green |
| Release | CI [42056213509](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/actions/runs/42056213509), CD [42058543205](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/actions/runs/42058543205), Deploy [42060162456](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/actions/runs/42060162456) | NFR-11 on the new samples and the FR-12 smoke check passed |
| After | replay `fr15-after`, windows 15 to 18 | `ES` MdAPE 7.49 % and 7.82 %, drift still flagged |
| Rollback drill | Deploy [42065603905](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/actions/runs/42065603905) (`ea2b384`), then [42065848838](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/actions/runs/42065848838) (`ccb175e`) | about 5 s of 503 each, served version checked |

## The finding

The replay ran on the VM from 11:55:35 to 12:06:21 against model `afc3335eefa80dafe3d9ae6ab85c1ae3`.
The open window held 779 `ES` requests (not sellers) of an earlier replay, so the replay sent 1,221 control listings to close it and fill one control window, then 2,000 listings of the held-out `ES` sellers.
The samples were a copy of the image's `manifest.json` and `control.jsonl.gz` with the dry run's `es.jsonl.gz` (md5 `d34b801c`, 2,266 listings of the 779 `ES` sellers, not requests, that the share 0.4 keeps out of the split), mounted into the replay container, so that both replays send the same `ES` rows.
All 3,221 listings were priced (200) and their prices stored (201), at 5.00 listings/s ([before-replay.txt](fr15/before-replay.txt)).

| Window | `ES` share | Flagged | Drifted features | MdAPE | Test MdAPE | Coverage | Labelled ranges |
|--------|-----------|---------|------------------|-------|------------|----------|-----------------|
| 11 | 78 % | yes | 13 | 7.76 % | 6.73 % | 87.98 % | 499 |
| 12 | 0 % | no | 0 | 5.44 % | 6.73 % | 95.80 % | 500 |
| 13 | 100 % | yes | 14 | 9.09 % | 6.73 % | 85.00 % | 500 |
| 14 | 100 % | yes | 15 | 9.16 % | 6.73 % | 83.60 % | 500 |

The figures are `report --all --json` as saved at 12:12, before the promotion ([before-report-all.json](fr15/before-report-all.json), and `before-window-N.json` for each window).
A later `report --all` shows these windows against a test MdAPE of 6.80 %, the promoted model's, because a window that is not yet final is scored again on every pass against the `metrics.json` of the running release ([#307](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/307)); the figures above are the ones the finding was made on.

The silence that had held the model alerts since the previous replay was expired before the run.
At 12:12 DriftDetected, ModelErrorHigh and ModelCoverageLow were firing for window 14, the latest, and Alertmanager's Discord counter, which counts from Alertmanager's start, stood at 3, none failed ([before-alertmanager-discord.txt](fr15/before-alertmanager-discord.txt)).
After the replay of the promoted model it still stood at 3, since a silence held the alerts ([after-alertmanager-discord.txt](fr15/after-alertmanager-discord.txt)).
The finding of windows 13 and 14 became the Retrain run's `reason`.

## Retraining and promotion

Runner of run 42028216667: AMD EPYC 9V74, with AVX-512 (`avx512f`, `avx512bw`, `avx512dq`, `avx512vl` and eight more).

| Variant | Gate | Blocked by | MdAPE | SC-05 coverage |
|---------|------|------------|-------|----------------|
| `lgbm-basic` | passed | - | 6.80 % | 0.9180 (range 0.88 to 0.92) |
| `lgbm-extended` | blocked | sc05 unmeasured | 6.33 % | - |
| `b1` | blocked | sc01, sc02, sc03; sc05 unmeasured | 9.53 % | - |
| `b0` | blocked | sc01, sc02, sc03; sc05 unmeasured | 12.19 % | - |

SC-04 stays waived by EDN-62 (failed at 0.1764).
The `ES` test segment, 593 listings of `ES` sellers in the split, has an MdAPE of 7.61 % for `lgbm-basic`.

The payloads of the branch `retrain/run-5-1` equal the laptop dry run of #300 file by file:

| File | md5 |
|------|-----|
| `data/interim/listings.parquet` | `e2808f7f2f07b5bec7252f5171d5efa6` |
| `models/lgbm-basic/booster.txt` | `ea52514598d2c9038f2d4b0239514133` |
| `models/lgbm-basic/interval_lower.txt` | `4f8e7ec41b0d3fc639e8333faf2cdc09` |
| `models/lgbm-basic/interval_median.txt` | `2a76f36cd594ddd3682c5cc87112b625` |
| `models/lgbm-basic/interval_upper.txt` | `d3f0d8d8a5ad0d9698a35ce2c2deabd6` |
| `models/lgbm-basic/feature_space.json` | `3b3b3c2a1827ad07c2ff3239112141df` |
| `metrics.json` | `cc41e6ca1293d7deafe42ab972f4216b` |
| `data/monitoring/es.jsonl.gz` | `d34b801ca3d6918a7bac6e922b432228` |

The dry run ran on a laptop without AVX-512 (Intel i5-10210U), so the runner and the laptop produced the same interim data and the same booster on two instruction sets: the cross-CPU check that [#259](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/259) waited for.
Only `model.json` differs, since it records the training time and the run ids.

Lukas merged the promotion pull request #303 at 15:33, after the review posted on it.
CD ran NFR-11's test on the new samples with `RECOMMENDITOS_REQUIRE_MONITORING_SAMPLES=1`, and Deploy's smoke check held `/health` to the deployed commit's `dvc.lock`: the VM answered with `2a60912bd232083e6950f26fe72fe6c4` from 15:57.

## After the promotion

The replay ran from 15:59:53 to 16:13:15 against `2a60912b`, on the image's own samples.
The image's `es.jsonl.gz` is the dry run's file (md5 `d34b801c`, checked in the image on the VM), so the `ES` rows are the same as before: the 2,000 observed prices match in order ([paired-es-requests.csv](fr15/paired-es-requests.csv)).
Its control sample is drawn from the new training data, as the new drift reference is, so a control window is not flagged for the `ES` listings it holds (3.7 % of the control sample).
The new model cut the open window, so the replay sent 2,000 control listings, then the same 2,000 `ES` listings; all 4,000 were priced and stored ([after-replay.txt](fr15/after-replay.txt)).

| Window | `ES` share | Flagged | Drifted features | MdAPE | Test MdAPE | Coverage | Labelled ranges |
|--------|-----------|---------|------------------|-------|------------|----------|-----------------|
| 15 | 4 % | no | 0 | 5.76 % | 6.80 % | 96.80 % | 500 |
| 16 | 4 % | no | 0 | 5.39 % | 6.80 % | 96.00 % | 500 |
| 17 | 100 % | yes | 13 | 7.49 % | 6.80 % | 90.60 % | 500 |
| 18 | 100 % | yes | 14 | 7.82 % | 6.80 % | 86.40 % | 500 |

The figures are [after-report-all.json](fr15/after-report-all.json) and `after-window-N.json`.

## The recovery criterion

Over the two `ES` windows, pooled: the median of all 2,000 absolute percentage errors and the share of all 1,000 labelled ranges that hold the observed price, with the drift job's own label join and formulas.
[pooled.py](fr15/pooled.py), as it ran in the monitor container, wrote [before-pooled-13-14.json](fr15/before-pooled-13-14.json) and [after-pooled-17-18.json](fr15/after-pooled-17-18.json), and per window it reproduces the stored figures.
[pair_runs.py](fr15/pair_runs.py) turns the two into [pooled-es-windows.json](fr15/pooled-es-windows.json), [paired-es-requests.csv](fr15/paired-es-requests.csv) and [criterion-calculation.txt](fr15/criterion-calculation.txt).

| Point | Before (13, 14) | After (17, 18) | Holds |
|-------|-----------------|----------------|-------|
| 1. MdAPE lower after | 9.13 % | 7.68 % (-1.45 pp) | yes |
| 2. Offline, all 2,266 held-out `ES` rows, seller bootstrap | 9.16 % | 7.70 %: -1.46 pp, 95 % CI [-1.92, -1.03] | yes |
| 3a. Coverage moves toward 90 % | 84.3 % | 88.5 % | yes |
| 3b. ModelErrorHigh and ModelCoverageLow do not fire | fired | window 17: neither; window 18: both | no |

The coverage of point 3a is not a paired comparison: the replay alternates `/v1/predict` and `/v1/price-range`, and the `ES` phase started at an odd position before and an even one after, so 84.3 % and 88.5 % are the two halves of the same 2,000 rows.

The recovery criterion is not met in full: points 1 and 2 hold, point 3 holds for coverage and fails on its alert clause, on window 18 by a narrow margin.
Point 2 is the dry run of #300, reproduced by its reviewer; it applies to the promoted model because the booster and the interval files are the same files.
The offline figures per window also equal the live ones: 7.49 % and 7.82 % for the first and the second 1,000 rows.

Window 18's MdAPE is 1.149 times the test MdAPE, above the ceiling of 1.111 for 1,000 labelled answers, and 432 of its 500 ranges hold the price, 86.40 % against a floor of 86.44 %.
Prometheus's alert history shows both alerts absent while window 17 was the latest, from 16:12 to 16:28, and firing for window 18 from 16:28 ([after-prometheus-alerts-history.json](fr15/after-prometheus-alerts-history.json), in 1-minute steps; [after-prometheus-alerts.json](fr15/after-prometheus-alerts.json) has the exact 16:27:58); a silence kept them from Discord.
The cause is the baseline: the alerts compare a window with the overall test MdAPE of 6.80 %, while `ES` on the test set itself is at 7.61 %, 1.12 times that, so an `ES` window at the model's own `ES` error sits at the ceiling.
[#306](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/306) follows this up: judging a window against its market's baseline.

Drift stays flagged, as the criterion expected: the `ES` inputs still differ from a reference that is 3.8 % `ES`.

## Rollback drill

A probe on a laptop asked `GET /health` once a second through an SSH tunnel to the VM's nginx port, from 16:32:39 to 16:37:04, 265 answers ([drill-probe.txt](fr15/drill-probe.txt), [drill-analysis.txt](fr15/drill-analysis.txt) by [drill_analyze.py](fr15/drill_analyze.py)).

| | Rollback to `ea2b384` | Roll forward to `ccb175e` |
|-|-----------------------|---------------------------|
| Deploy run | [42065603905](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/actions/runs/42065603905) | [42065848838](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/actions/runs/42065848838) |
| Run created (GitHub) | 16:33:07 | 16:34:33 |
| 503 from nginx (probe) | 16:33:22.006 to 16:33:26.006 | 16:34:46.006 to 16:34:50.006 |
| 503 answers in a row | 5 answers of 503 in a row (1 s probe), about 5 s | 5 answers of 503 in a row (1 s probe), about 5 s |
| First answer of the other model | 16:33:27.005, `afc3335e` | 16:34:51.006, `2a60912b` |
| Smoke check (NFR-12, FR-12) | passed | passed |
| Run completed (GitHub) | 16:34:05 | 16:35:36 |

Every 200 answer named the old or the new model, and no answer of the previous model came after a switch.
`deploy.sh status` afterwards: running `ccb175e`, previous tag `ea2b384`.
