# Load test: vm-2026-11-05-full-stack

Full M6 stack on the VM (1 vCPU, 3.8Gi RAM): api, proxy, prometheus, cadvisor, node-exporter, alertmanager, grafana, monitor. Locust on the laptop (8 logical CPUs) against the public URL http://nattech.fib.upc.edu:40520 over the internet (about 30 ms round trip), so the latency includes the network and nginx, unlike the 2026-10-31 runs with Locust on the VM; one API worker; model lgbm-basic afc3335e; no docker, metrics or prediction-log measurements because the VM cannot be reached from here; X-Client: loadtest on every request.

- Started 2026-11-04T23:08:01+00:00 against `http://nattech.fib.upc.edu:40520` from `lukas-NBLB-WAX9N` (8 logical CPUs).
- Target rate 20 requests/s for 300 s, 80 Locust users.
- Load average (1, 5, 15 min) before: [2.94, 1.45, 1.42], after: [3.15, 2.49, 1.88].
- Achieved 20.04 of 20 requests/s (yes, at least 95 % of the target, NFR-03).

## Latency and errors (Locust, client side)

| Endpoint | Requests | Req/s | Errors (< 1 %) | p50 ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|
| /v1/predict | 3607 | 12.02 | 0.00 % (yes) | 97 | 280 (no, target 250) | 1100 |
| /v1/price-range | 1184 | 3.95 | 0.00 % (yes) | 82 | 270 (yes, target 300) | 1200 |
| /v1/comparables | 1222 | 4.07 | 0.00 % (yes) | 31 | 120 (yes, target 300) | 950 |
| Aggregated | 6013 | 20.04 | 0.00 % (yes) | 85 | 260 | 1100 |

## Locust process (psutil)

- CPU: mean 7 %, max 74 % (100 % is one core), on the machine it ran on.

## Prediction log (NFR-03)

- Not checked: the API has no prediction log.
