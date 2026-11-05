# ES replay on the VM, 2026-11-04

The drift, error, and coverage of each window of the drift job ([#228](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/228), [#229](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/229)) after the replay on the production VM, model `lgbm-basic` `afc3335eefa80dafe3d9ae6ab85c1ae3`.
The replay ran from 22:26 to 23:03 UTC on 2026-11-04 and sent 5,000 control listings, then all 5,779 listings of the `ES` holdout, each with its observed price as feedback ([Replay](../../docs/docs/monitoring.md#replay)).
The figures were read on 2026-11-05 at about 09:45 UTC with `report --all` and `report --all --json` in the monitor container ([Run it by hand](../../docs/docs/monitoring.md#run-it-by-hand)); the JSON is [vm-es-replay-2026-11-04.json](vm-es-replay-2026-11-04.json), verbatim.
The report's figure of the windows is drawn from it by [`reports/analysis/es_replay_figure.py`](../analysis/es_replay_figure.py).

## The replay

```text
phase    sent   answered    feedback
control  5000   200: 5000   201: 5000
es       5779   200: 5779   201: 5779
total    10779  200: 10779  201: 10779
```

## The windows

```text
window  status  seq          es    drift       labelled  mdape   test_mdape  ranges  coverage  nominal  final
1       tested  11940-12939  0%    none        1000      0.0538  0.0673      500     0.9680    0.90     no
2       tested  12940-13939  0%    none        1000      0.0528  0.0673      500     0.9540    0.90     no
3       tested  13940-14939  0%    none        1000      0.0487  0.0673      500     0.9620    0.90     no
4       tested  14940-15939  0%    none        1000      0.0497  0.0673      500     0.9720    0.90     no
5       tested  15940-16939  0%    none        1000      0.0512  0.0673      500     0.9620    0.90     no
6       tested  16940-17939  100%  14 drifted  1000      0.0941  0.0673      500     0.8580    0.90     no
7       tested  17940-18939  100%  14 drifted  1000      0.0928  0.0673      500     0.8540    0.90     no
8       tested  18940-19939  100%  14 drifted  1000      0.0920  0.0673      500     0.8760    0.90     no
9       tested  19940-20939  100%  13 drifted  1000      0.0898  0.0673      500     0.8660    0.90     no
10      tested  20940-21939  100%  15 drifted  1000      0.0930  0.0673      500     0.8660    0.90     no
```

The five control windows are not flagged, and the five `ES` windows are flagged in 13 to 15 of 23 features.
The `ES` windows' MdAPE of 8.98 % to 9.41 % is 1.33 to 1.40 times the test MdAPE of 6.73 % and above `ModelErrorHigh`'s ceiling of 7.48 % at 1,000 labelled answers in every window.
Their coverage of 85.4 % to 87.6 % is below the nominal 90 % and the test coverage of 90.2 %, and below `ModelCoverageLow`'s floor of 86.44 % at 500 labelled ranges in windows 6 and 7 only.
The control listings are training listings, so the control windows' error and coverage are in-sample and no baseline ([The baseline](../../docs/docs/monitoring.md#the-baseline)).
The last 779 `ES` requests are the open window, which has no figures yet.
The figures were not final when read: labels can still arrive for 7 days, but every request of these windows already had its price.

## The findings

```text
# report --window 1
No drift in window 1 (log seq 11940 to 12939, 1000 requests answered 2026-11-04T22:26:32.195+00:00 to 2026-11-04T22:29:51.992+00:00 by lgbm-basic afc3335eefa80dafe3d9ae6ab85c1ae3, 0% ES): none of the 23 features is below p 0.000217 (0.005 / 23, NFR-11). MdAPE 5.38 % over 1000 labelled answers, 0.80 x the test MdAPE of 6.73 %; coverage 96.80 % of 500 labelled ranges at a nominal 90.00 %, against 90.23 % on the test set (labels still arriving).
# report --window 6
Input drift in window 6 (log seq 16940 to 17939, 1000 requests answered 2026-11-04T22:43:12.207+00:00 to 2026-11-04T22:46:31.995+00:00 by lgbm-basic afc3335eefa80dafe3d9ae6ab85c1ae3, 100% ES): 14 of 23 features are below p 0.000217 (0.005 / 23, NFR-11): country_code (p 0), nr_prev_owners__missing (p 0), body_type (p 2.7e-34), nr_seats__missing (p 1.2e-24), gears__missing (p 6.5e-21), age_years (p 3.6e-15), fuel_category (p 7.8e-15), mileage_km_raw (p 6.7e-13), nr_doors (p 1.5e-12), make (p 2.8e-10), transmission (p 1.3e-08), drive_train (p 7.7e-07), cylinders_volume_cc__missing (p 2.9e-06), seller_type (p 4.5e-06). MdAPE 9.41 % over 1000 labelled answers, 1.40 x the test MdAPE of 6.73 %; coverage 85.80 % of 500 labelled ranges at a nominal 90.00 %, against 90.23 % on the test set (labels still arriving).
# report (latest)
Input drift in window 10 (log seq 20940 to 21939, 1000 requests answered 2026-11-04T22:56:32.207+00:00 to 2026-11-04T22:59:51.992+00:00 by lgbm-basic afc3335eefa80dafe3d9ae6ab85c1ae3, 100% ES): 15 of 23 features are below p 0.000217 (0.005 / 23, NFR-11): country_code (p 0), nr_prev_owners__missing (p 0), body_type (p 4.4e-32), nr_seats__missing (p 6.3e-29), age_years (p 5.7e-19), gears__missing (p 1.1e-16), mileage_km_raw (p 9.3e-15), make (p 3.6e-14), fuel_category (p 3e-12), drive_train (p 1.9e-10), transmission (p 4.8e-10), cylinders_volume_cc__missing (p 5.3e-08), nr_doors (p 1.2e-06), cylinders_volume_cc (p 2.8e-06), gears (p 7.1e-05). MdAPE 9.30 % over 1000 labelled answers, 1.38 x the test MdAPE of 6.73 %; coverage 86.60 % of 500 labelled ranges at a nominal 90.00 %, against 90.23 % on the test set (labels still arriving).
```
