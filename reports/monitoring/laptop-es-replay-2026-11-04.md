# ES replay on a laptop stack, 2026-11-04

The error and the coverage per window of the drift job ([#229](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/229)) on a local stack of the branch `feature/error-coverage`, model `lgbm-basic` `afc3335eefa80dafe3d9ae6ab85c1ae3`.
The replay sent 2,000 control listings, then 1,500 of the `ES` holdout, each with its observed price as feedback, and one pass of `run-once` ran after it.
Every field of every window is in [laptop-es-replay-2026-11-04.json](laptop-es-replay-2026-11-04.json), the output of `report --all --json`.

```bash
docker compose -f deploy/compose.yaml --profile replay run --rm replay \
    --control 2000 --then es --then-count 1500 --rate 40
docker compose -f deploy/compose.yaml exec monitor python -m recommenditos_monitoring run-once
docker compose -f deploy/compose.yaml exec monitor python -m recommenditos_monitoring report --all
```

## The replay

```text
phase    sent  answered   feedback
control  2000  200: 2000  201: 2000
es       1500  200: 1500  201: 1500
total    3500  200: 3500  201: 3500
duration 176.9 s, 19.78 listings/s (target 40)
result: ok
```

## The windows

```text
window  status  seq        es    drift       labelled  mdape   test_mdape  ranges  coverage  nominal  final
1       tested  1-1000     0%    none        1000      0.0538  0.0673      500     0.9680    0.90     no
2       tested  1001-2000  0%    none        1000      0.0528  0.0673      500     0.9540    0.90     no
3       tested  2001-3000  100%  14 drifted  1000      0.0941  0.0673      500     0.8580    0.90     no
```

The two control windows are not flagged and their error and coverage are better than the test values (MdAPE 5.38 % and 5.28 % against 6.73 %, coverage 96.8 % and 95.4 % against 90.2 %), because the control listings are training listings.
The export below compares the coverage with 91.0 %, the SC-05 coverage of partial inputs, which the job used when this run was made; it now compares with the full-input test coverage, 90.2 %, since a window holds only requests with every required field ([the baseline](../../docs/docs/monitoring.md#the-baseline)).
The ES window's 85.8 % is 4.4 points below it.
The `ES` window is flagged in 14 of 23 features, its MdAPE is 9.41 %, 1.40 times the test MdAPE, and 500 labelled ranges hold 85.8 % of the prices, 4.2 percentage points below the nominal 90 %.
The last 500 `ES` requests are the open window, which has no figures yet.

## The finding of the ES window

```text
Input drift in window 3 (log seq 2001 to 3000, 1000 requests answered 2026-11-04T22:06:37.551+00:00 to 2026-11-04T22:07:27.416+00:00 by lgbm-basic afc3335eefa80dafe3d9ae6ab85c1ae3, 100% ES): 14 of 23 features are below p 0.000217 (0.005 / 23, NFR-11): country_code (p 0), nr_prev_owners__missing (p 0), body_type (p 2.7e-34), nr_seats__missing (p 1.2e-24), gears__missing (p 6.5e-21), age_years (p 3.6e-15), fuel_category (p 7.8e-15), mileage_km_raw (p 6.7e-13), nr_doors (p 1.5e-12), make (p 2.8e-10), transmission (p 1.3e-08), drive_train (p 7.7e-07), cylinders_volume_cc__missing (p 2.9e-06), seller_type (p 4.5e-06). MdAPE 9.41 % over 1000 labelled answers, 1.40 x the test MdAPE of 6.73 %; coverage 85.80 % of 500 labelled ranges at a nominal 90.00 %, against 91.01 % on the test set (labels still arriving).
```
