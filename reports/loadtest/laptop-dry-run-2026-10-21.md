# Load test: laptop-dry-run-2026-10-21

**Dry run on a shared development laptop, not a measurement of the VM.** The stack ran on the same 4-core laptop as Locust, an IDE and other development sessions, so latency, throughput and energy here say nothing about NFR-02 to NFR-04 on the VM; the run checks that the load test and its measurements work end to end. 60 s instead of NFR-03's 5 minutes, to keep the shared machine usable.

- Started 2026-10-21T15:44:36+00:00 against `http://localhost:8080` from `lukas-NBLB-WAX9N` (8 logical CPUs).
- Target rate 20 requests/s for 60 s, 80 Locust users.
- Load average (1, 5, 15 min) before: [3.74, 2.82, 2.41], after: [3.96, 3.03, 2.52].
- Achieved 19.55 of 20 requests/s (yes, at least 95 % of the target, NFR-03).
- API image `recommenditos-api:local`, 2 worker(s).

## Latency and errors (Locust, client side)

| Endpoint | Requests | Req/s | Errors (< 1 %) | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|
| /v1/predict | 882 | 14.70 | 0.00 % (yes) | 3100 | 5700 (no, target 200) | 7300 |
| /v1/price-range | 291 | 4.85 | 0.00 % (yes) | 2800 | 5200 (no, target 300) | 6200 |
| Aggregated | 1173 | 19.55 | 0.00 % (yes) | 3100 | 5400 | 7000 |

## Share of answers within NFR-02's targets (`/metrics`, server side)

| Endpoint | Answers | Within 200 ms | Within 300 ms |
|---|---|---|---|
| /v1/predict | 882 | 0.00 % | 0.23 % |
| /v1/price-range | 291 | 0.00 % | 0.34 % |

## API container (`docker stats`, NFR-04)

- Memory: max 597 MiB, mean 574 MiB (yes, below 1 GB), 32 samples.
- CPU: mean 177 %, max 293 % (100 % is one core).

## Locust process (psutil)

- CPU: mean 4 %, max 43 % (100 % is one core), on the machine it ran on.

## Energy (CodeCarbon, whole machine, NFR-10)

- Method: cpu_load, Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz, TDP 15.0 W; `machine` tracking mode, power integrated every 15 s.
- Of the run's energy, CPU 176 J and RAM 640 J, the RAM part CodeCarbon's constant power; the figure above the idle baseline leaves that constant out.
- Run: 816 J over 64 s (12.8 W), 0.696 J per request.
- Idle baseline: 16.6 W over 30 s; above it, not measurable per request.

## Prediction log (NFR-03)

- 1173 successful valuations; the log gained 1173 records.
- Missing 0, unexpected 0, duplicated 0.
- Exactly one record per successful request: **yes**.
