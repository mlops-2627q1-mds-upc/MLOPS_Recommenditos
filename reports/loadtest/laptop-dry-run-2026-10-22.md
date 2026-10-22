# Load test: laptop-dry-run-2026-10-22

**Dry run on a shared development laptop, not a measurement of the VM.** The stack ran on the same 4-core laptop as Locust, an IDE and other development sessions, so latency, throughput and energy here say nothing about NFR-02 to NFR-04 on the VM; the run checks that the load test and its measurements work end to end. 60 s instead of NFR-03's 5 minutes, to keep the shared machine usable.

- Started 2026-10-22T13:08:28+00:00 against `http://localhost:8080` from `lukas-NBLB-WAX9N` (8 logical CPUs).
- Target rate 20 requests/s for 60 s, 80 Locust users.
- Load average (1, 5, 15 min) before: [0.59, 0.43, 0.62], after: [0.68, 0.52, 0.63].
- Achieved 19.67 of 20 requests/s (yes, at least 95 % of the target, NFR-03).
- API image `recommenditos-api:local`, 2 worker(s).

## Latency and errors (Locust, client side)

| Endpoint | Requests | Req/s | Errors (< 1 %) | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|
| /v1/price-range | 214 | 3.57 | 0.00 % (yes) | 78 | 190 (yes, target 300) | 250 |
| /v1/predict | 722 | 12.03 | 0.00 % (yes) | 90 | 180 (yes, target 200) | 250 |
| /v1/comparables | 244 | 4.07 | 0.00 % (yes) | 13 | 39 (yes, target 300) | 55 |
| Aggregated | 1180 | 19.67 | 0.00 % (yes) | 71 | 170 | 250 |

## Share of answers within NFR-02's targets (`/metrics`, server side)

| Endpoint | Answers | Within 200 ms | Within 300 ms |
|---|---|---|---|
| /v1/comparables | 244 | 100.00 % | 100.00 % |
| /v1/predict | 722 | 97.65 % | 99.86 % |
| /v1/price-range | 214 | 97.20 % | 99.53 % |

## API container (`docker stats`, NFR-04)

- Memory: max 566 MiB, mean 560 MiB (yes, below 1 GB), 31 samples.
- CPU: mean 93 %, max 147 % (100 % is one core).

## Locust process (psutil)

- CPU: mean 5 %, max 39 % (100 % is one core), on the machine it ran on.

## Energy (CodeCarbon, whole machine, NFR-10)

- Method: cpu_load, Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz, TDP 15.0 W; `machine` tracking mode, power integrated every 15 s.
- Of the run's energy, CPU 96 J and RAM 620 J, the RAM part CodeCarbon's constant power; the figure above the idle baseline leaves that constant out.
- Run: 716 J over 62 s (11.5 W), 0.607 J per request.
- Idle baseline: 11.5 W over 60 s; above it, 0.001 J per request.

## Prediction log (NFR-03)

- 1180 successful valuations; the log gained 1180 records.
- Missing 0, unexpected 0, duplicated 0.
- Exactly one record per successful request: **yes**.
