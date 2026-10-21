# Load test: laptop-dry-run-2026-10-21

**Dry run on a shared development laptop, not a measurement of the VM.** The stack ran on the same 4-core laptop as Locust, an IDE and other development sessions, so latency, throughput and energy here say nothing about NFR-02 to NFR-04 on the VM; the run checks that the load test and its measurements work end to end. 60 s instead of NFR-03's 5 minutes, to keep the shared machine usable.

- Started 2026-10-21T09:36:32+00:00 against `http://localhost:8080` from `lukas-NBLB-WAX9N` (8 logical CPUs).
- Target rate 20 requests/s for 60 s, 20 Locust users.
- Load average (1, 5, 15 min) before: [2.05, 1.66, 1.64], after: [2.07, 1.7, 1.65].
- API image `recommenditos-api:local`, 2 worker(s).

## Latency and errors (Locust, client side)

| Endpoint | Requests | Req/s | Errors (< 1 %) | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|
| /v1/predict | 1184 | 19.72 | 0.00 % (yes) | 710 | 970 (no, target 200) | 1100 |
| Aggregated | 1184 | 19.72 | 0.00 % (yes) | 710 | 970 | 1100 |

## Share of answers within NFR-02's targets (`/metrics`, server side)

| Endpoint | Answers | Within 200 ms | Within 300 ms |
|---|---|---|---|
| /v1/predict | 1200 | 10.42 % | 12.17 % |

## API container (`docker stats`, NFR-04)

- Memory: max 511 MiB, mean 507 MiB (yes, below 1 GB), 31 samples.
- CPU: mean 161 %, max 213 % (100 % is one core).

## Energy (CodeCarbon, whole machine, NFR-10)

- Method: cpu_load, Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz, TDP 15.0 W; `machine` tracking mode.
- Run: 715 J over 61 s (11.7 W), 0.604 J per request.
- Idle baseline: 11.5 W over 30 s; above it, 0.009 J per request.

## Prediction log (NFR-03)

- 1200 successful valuations; the log gained 1200 records.
- Missing 0, unexpected 0, duplicated 0.
- Exactly one record per successful request: **yes**.
