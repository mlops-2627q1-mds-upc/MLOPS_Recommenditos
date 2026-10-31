The API
=======

The HTTP API serves the price model: it estimates the asking price of a used car (UC1) and the price range of a car described in part (UC2) with the model bundle the service was started with, and returns real listings comparable to that car (UC2) from the comparables index the service was started with.
This page says how to run it, what it offers and how its errors are shaped, and how to add an endpoint.
The field-by-field reference is the generated OpenAPI document, served by the API itself at `/docs` (Swagger UI), `/redoc` (ReDoc) and `/openapi.json`; this page does not repeat it.
A copy of it is committed as `docs/docs/openapi.json` and rendered on the [API reference](api-reference.md) page, so the contract can be read without running the service.
What each endpoint must do is in the [specification](specification.md) (FR-01 to FR-16, NFR-03, NFR-04, NFR-08, NFR-09).

## Running it

```bash
uv run dvc pull train comparables   # the model bundles and the comparables index, into models/
make api                   # uvicorn on http://127.0.0.1:8000
```

`make api` serves `models/lgbm-basic`.
`RECOMMENDITOS_MODEL_DIR=models/lgbm-extended make api` serves another bundle, and `make api API_PORT=8080` another port.

The service reads its settings from environment variables (`recommenditos/api/settings.py`, with pydantic-settings); locally, the repository's `.env` is loaded into the environment first.

| Variable | Default | What it sets |
|----------|---------|--------------|
| `RECOMMENDITOS_MODEL_DIR` | `models/lgbm-basic` | The bundle to serve |
| `RECOMMENDITOS_COMPARABLES_FILE` | `data/processed/comparables.parquet` | The comparables index `/v1/comparables` searches; the image sets `/app/models/comparables.parquet` |
| `RECOMMENDITOS_API_KEY` | none | The key `/v1/feedback` requires; at least 16 characters. Unset or empty, the endpoint refuses every request |
| `RECOMMENDITOS_PREDICTION_LOG` | `logs/prediction-log.sqlite3` | The SQLite file of the prediction log and the feedback |
| `RECOMMENDITOS_PREDICTION_LOG_MAX_BYTES` | `2000000000` | The size the log and the feedback are pruned to (NFR-04) |
| `RECOMMENDITOS_LOG_FORMAT` | `text` | `json` for one JSON object per log line, as the image sets it ([Logs](#logs)); read by `recommenditos/api/logs.py`, because uvicorn configures the log before the app and its settings exist |

The API key is a secret (NFR-15): `.env.template` lists it without a value, and the settings hold it as a `SecretStr`, which a repr or a log line shows as `**********`.
The bundle and the comparables index, `data/processed/comparables.parquet` or the file `RECOMMENDITOS_COMPARABLES_FILE` names, are loaded once, before the first request, and the API never calls MLflow ([EDN-08](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md), FR-12).
A bundle the API cannot serve stops the service at startup instead of failing on the first request: one that consumes no column for a field FR-01 requires, and one without a SHAP export to explain a valuation with (FR-08), which today means every bundle but a LightGBM one (`b0` and `b1` are refused).

`make api` runs `python -m recommenditos.api`, which passes its options on to uvicorn: `python -m recommenditos.api --workers 2`, or `WEB_CONCURRENCY=2 make api`, starts two worker processes.
With more than one worker, set `PROMETHEUS_MULTIPROC_DIR` too (see [Metrics](#metrics)); the image does.

The deployed service runs the same app in a container behind nginx, on a laptop as on the VM: [Deployment](deployment.md) says how to build, start and check that stack.

A valuation:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/predict -H 'content-type: application/json' -d '{
  "make": "BMW", "model": "320", "registration_date": "2021-03",
  "mileage_km_raw": 60000, "power_kw": 135, "fuel_category": "Diesel",
  "transmission": "Automatic", "country_code": "DE", "body_type": "Sedan",
  "seller_type": "Dealer"}'
```

## Performance

Each worker process scores one request at a time, in the order the requests arrive, on a thread of its own (`ServedModel` hands every valuation and price range to it).
The request threads of uvicorn's pool wait for their answer there without using CPU.
Scored in the request threads themselves, concurrent requests took turns at the GIL every few milliseconds, and every request's latency grew to that of all of them.
In one process with 8 callers in a closed loop on the laptop (issue #179), that cost 86 ms of CPU per valuation against 50 ms with the scoring thread, about 30 % less CPU per valuation, without the multi-second tail a plain lock showed (p99 above 4 s).
A plain lock serialises the requests too, but it is not fair: a thread that releases it can take it straight back, and p99 reached 4.7 s.
The trade-off is that one process uses at most about one core for scoring, so the API scales by workers (`WEB_CONCURRENCY`), each with about 240 MiB of memory, and this pays off when there are about as many workers as vCPUs.
With cores to spare, latency with the scoring thread is similar or slightly worse than without it: in-process percentiles at 10 requests per second varied up to two-fold between runs on the laptop, so only CPU per valuation is quoted.

`python -m recommenditos.api`, and so the image, sets `OMP_NUM_THREADS=1` for its workers unless the environment sets it already.
LightGBM's default is one OpenMP thread per core, and `Booster.predict` ignores the `num_threads` that `booster.txt` records.
In a worker's scoring thread one OpenMP thread costs 47 ms of CPU per valuation against 50 ms with the default, and each worker then no longer holds an OpenMP team for every thread that ever scored (137 to 345 threads per worker under load).
The answers are the same either way; the FR-06 parity test compares the API's prices with the pipeline's, which predicts with the default.
Batch scoring in the pipeline keeps the default, and so does an app started another way, such as the tests' `TestClient`.

A valuation takes about 50 ms: about 18 ms to build the one-row matrix (`build_matrix` and the schema's casts, the code the pipeline uses too), about 20 ms for LightGBM's SHAP export behind the explanation, and the price itself.
The explanation's per-field sums run on the NumPy array, 0.1 ms against 7 ms through pandas.
In the laptop measurement, one worker saturated at about 20 requests per second of the load test's mix; [Load testing](load-testing.md) measures what the VM sustains.

## Endpoints

The client contract is versioned under `/v1`, so an incompatible change can ship as `/v2` beside it.
The operational endpoints are not part of that contract and stay at the root, where a browser, an uptime monitor and Prometheus look for them: `/`, `/health` and `/metrics`.
`/` and `/health` also answer `HEAD`, which uptime monitors often probe with, with the headers of the `GET` answer and no body.

| Endpoint | Purpose | Requirements |
|----------|---------|--------------|
| `POST /v1/predict` | The point estimate of the asking price in EUR, its explanation, the model that made it, a request ID and warnings | FR-01, FR-03 to FR-06, FR-08, FR-13 |
| `POST /v1/price-range` | The typical price and the nominal 90 % price range in EUR for a car described by any fields that include its make, the model, a request ID and warnings | FR-02 to FR-05, FR-07, FR-13 |
| `POST /v1/comparables` | Up to `k` real listings with their price and key features, comparable to a car described as for `/v1/price-range`, the index and the model, a request ID and warnings | FR-02 to FR-04, FR-09, FR-13 |
| `POST /v1/feedback` | Stores the observed price of an answered listing against its request ID; internal, requires the API key | FR-10 |
| `GET /health` | `ok`, the served model (its variant, version and training time), `price_range`, whether the bundle can answer `/v1/price-range`, and `comparables`, the index's version and size | FR-11, FR-12 |
| `GET /` | The service name, the served model and links to `/docs`, `/redoc` and `/openapi.json` | - |
| `GET /metrics` | Request, latency, error and model metrics in Prometheus' text format | FR-11 |

`/v1/predict` values one car per request, because UC1 is one seller asking about one listing and every answer becomes one record of the prediction log; it has no batch form.
`/predict` requires the ten fields of FR-01 and accepts every other input of the basic and the extended feature set as optional.
The served model reads the fields its feature set holds; an omitted one reaches it as absent, an omitted equipment list as an empty one.
A field the request body does not define is refused, so a misspelt optional field does not go unnoticed.

Every string and equipment list has a maximum length, which the OpenAPI document states as `maxLength` and `maxItems`.
The bounds accept every one of the 118,382 raw listings: a string outside the equipment lists gets at least twice its longest value in the data, an equipment list and its items at least 1.2 times theirs.

| Field | Longest in the data | Bound |
|-------|---------------------|-------|
| `make`, `model` and the categorical and free-text fields | 23 characters (`body_type`) | 50 characters |
| `model_version` | 121 characters | 250 characters |
| an equipment item | 37 characters | 45 characters |
| `equipment_comfort`, `equipment_entertainment`, `equipment_extra`, `equipment_safety` | 42, 16, 37 and 36 items | 55, 20, 45 and 45 items |

A body with every field at its bound is at most 9.8 KB (9,804 bytes for `/v1/comparables`), so a longer value is a 422 that names the field rather than a 413 for the whole body.

The values a categorical field accepts come from the loaded bundle, not from the code: the supported makes from its metadata (FR-04), the levels of `fuel_category`, `transmission`, `body_type`, `drive_train` and `seller_type` from its feature space (FR-03).
`/docs` lists them as enums, and a 422 for such a field lists them in `allowed`.
`country_code` accepts the 8 countries of the data.
A country or a `model` the training data does not hold, `ES` today, is answered and reported in `warnings` (FR-05).

### The explanation

`/v1/predict` explains every estimate, so that it can be checked instead of believed (FR-08):

```json
"explanation": {
  "base_price_eur": 38389.95741359577,
  "top_features": [
    {"field": "power_kw", "effect_pct": -35.26084112077258},
    {"field": "model", "effect_pct": -23.360818674067794},
    {"field": "make", "effect_pct": -17.95976142027397},
    {"field": "mileage_km_raw", "effect_pct": 14.580681443498216},
    {"field": "registration_date", "effect_pct": -6.263490267105494}
  ],
  "other_features_effect_pct": -12.776273101340156
}
```

`base_price_eur` is where every estimate starts, the model's average log price over its training listings as a price.
Each effect is a percentage change of the price: `-35.26` for `power_kw` means this car's power lowers the estimate by 35 %.
The effects multiply, because the model predicts the log of the price: the base price times `1 + effect / 100` over the five listed fields and `other_features_effect_pct` gives `price_eur`, 14,639.20 EUR for the example request of `/docs`.

The values are SHAP values from the booster's own export, LightGBM's `predict(pred_contrib=True)`, and not from the `shap` package, which stays out of the API image ([EDN-11](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
A SHAP value φ in log price is reported as `exp(φ) - 1`, in percent.

An effect belongs to a request field, named as the request body names it, because that is what a client can look up and change.
The model reads matrix columns, and a field becomes one or more of them; the field's effect is the sum of its columns' SHAP values, which SHAP's additivity allows:

| Request field | Model columns |
|---------------|---------------|
| `registration_date` | `age_years`, the age at the request date |
| `equipment_comfort`, `equipment_entertainment`, `equipment_extra`, `equipment_safety` | one column per item the training data kept, so the whole list counts as one effect (extended set only) |
| every other field | the column of the same name |

`top_features` lists the five fields with the largest effect, whether it raises or lowers the price, largest first; two equal effects keep the order of the request body.
A field the client left out has an effect too, the effect of the model not knowing it.
`other_features_effect_pct` combines every other field the model reads, and is 0 when the model reads five fields or fewer, in which case `top_features` lists them all.
Nothing in the explanation is rounded, so that the product reproduces the unrounded estimate to float precision, and `price_eur`, which is rounded to the cent, to the cent; a client rounds for display.

When the model's estimate lies outside the price range of its training data, `price_eur` is that range's bound and `warnings` says so with the pointer `#`; the explanation then gives the estimate before the bound.
It is rare: with `lgbm-basic`, 1 of the 103,868 listings of the feature matrices reaches it, a 25-year-old Volvo V70 with 442,000 km in the test split, estimated at 482.02 EUR against the lower bound of 500 EUR.

The explanation is the costliest step of a request, because TreeSHAP visits every leaf of each of the 1,394 trees of `lgbm-basic`, where a prediction follows one path per tree.
Measured in-process with `TestClient` on a shared 4-core development laptop, over 394 requests built from test listings, it took p50 34 ms and p95 49 ms of a request that took p50 63 ms and p95 93 ms; a second run under heavier load from other processes gave 44 and 71 ms of 81 and 140 ms.
Both are inside NFR-02's 250 ms for `/predict`, which the load test on the target VM checked.
`lgbm-extended`, with 162 columns in 33 fields, is close to that limit or over it: under a load average of 2.7 to 5.7 one run gave a whole request of p50 181 ms and p95 195 ms (explanation 78 and 83 ms) and another p50 250 ms and p95 357 ms (explanation 106 and 148 ms), where `lgbm-basic` took p50 65 ms and p95 78 ms in the first run.

### The price range

`/v1/price-range` answers UC2, a buyer asking what a car should cost: only `make` is required (FR-02), and any other field of `/v1/predict`'s request may be added.
A field that is given is checked by the same rules as in `/v1/predict` (FR-03, FR-04), and an unseen `model` or country is answered with a warning (FR-05); a field left out, or sent as `null`, reaches the model as absent.
The request body is `PartialListing` in `schemas/partial_listing.py`, which `/v1/comparables` takes as well.

```json
{"price_eur": 15606.57, "lower_eur": 12790.57, "upper_eur": 27480.29, "confidence_level": 0.9,
 "model": {"variant": "lgbm-basic", "version": "...", "trained_at": "..."},
 "request_id": "3f1bad86-3ff2-489c-8ce8-a764efd6841b", "warnings": []}
```

That is the answer for a 2019 Volkswagen Golf with 45,000 km, scenario P1 of SC-05.
`lower_eur` and `upper_eur` are the conformalized bounds of the three quantile boosters the bundle carries ([EDN-85](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)); `confidence_level` is their nominal coverage.
`price_eur`, the typical price, is the one `/v1/predict` gives when the request carries all ten fields `/v1/predict` requires, and the median of the price range otherwise, because the point model has never seen a required field missing (FR-07).
The bounds are widened where the typical price falls outside them, so `lower_eur <= price_eur <= upper_eur` holds in every answer; on the test split that moves a bound for at most 1 % of the requests.

The range narrows as the request says more.
On the 19,665 listings of the test split, with the fields left out as a client would leave them out (`reports/analysis/price_range_widths.py`), the mean width `(upper_eur - lower_eur) / price_eur` is:

| Request | Typical price | Mean width | Median width | Coverage |
|---------|---------------|------------|--------------|----------|
| make only | median | 2.30 | 2.47 | 90.7 % |
| P1: make, model, registration date, mileage | median | 0.93 | 0.81 | 91.0 % |
| the ten fields `/v1/predict` requires | point model | 0.51 | 0.42 | 90.9 % |
| full | point model | 0.47 | 0.38 | 90.3 % |

Measured in-process with `TestClient` on the shared 4-core development laptop under a load average of 4 to 6, over 900 requests built from test listings, 300 each make-only, P1 and full, a request took p50 38 ms and p95 47 ms, with no shape above p95 49 ms (`reports/analysis/price_range_latency.py`).
That is inside NFR-02's 300 ms, which the load test on the target VM checks.

A bundle without interval models, such as `lgbm-extended`, serves `/v1/predict` and answers `/v1/price-range` with a 503 problem; `/health` then reports `"price_range": false`.

### Comparables

`POST /v1/comparables` takes the body of `/v1/price-range` plus `k`, how many listings to return: 5 when left out, at most 20, and a `k` outside 1 to 20 is a 422 like any other invalid field.
It answers with listings of the training and validation data, never of `ES` (EDN-03) or the test and calibration splits, each with its price in EUR and its key features, and none with a PII column (NFR-08).
A listing is comparable when it has the request's `make` and `model`, its age differs by at most 2 years from the car's and its mileage by at most 25 % of the car's (FR-09).
The listing's age is its age on the snapshot date, the age its price was asked at; the car's is its age today, as the model computes it.
The snapshot is from 2025-11-08, so the window today lies about a year earlier in registration date than the car's own: a car registered in 2020 matches listings registered from about 2017 to 2021.
The closest mileage comes first, then the closest age, then the index's own order, so the same request gets the same listings on the same day; the car's age, and with it the window, moves with the date.
A field the request leaves out sets no filter and adds no distance: a request with only `make` gets listings of every model of the make, one without a mileage is ordered by age alone.
No filter is relaxed, so a car outside every window gets an empty list; a `model` of which the index holds no listing for the make gets one too, with a warning that says so.
Fields other than these four are checked like everywhere else and do not change which listings come back.

The example of the OpenAPI document, a BMW 320 registered 2019-06 with 60,000 km and `k` 3, gets three listings of 2018 to 2020 with 59,000 to 60,000 km.
The Golf of P1 above gets an empty list: of the 70,172 listings, 54 are Volkswagen Golfs, and none of them lies within the car's age and mileage window.

The answer names the index it searched, `index.version` and `index.listings`, as `/health` does: the version is the md5 `dvc.lock` records for `data/processed/comparables.parquet`, which the `comparables` stage writes ([pipeline](pipeline.md#stages)).
A service whose `RECOMMENDITOS_COMPARABLES_FILE` names no file values listings, answers `/v1/comparables` with a 503 problem and reports `"comparables": null` on `/health`.

The real index holds 70,172 listings and takes 1.87 MiB on disk, which is what it adds to the image.
A process that has built the app around `lgbm-basic` holds 338.7 MiB resident with the index and 272.9 MiB without it, so the index costs every uvicorn worker about 65.9 MiB of NFR-04's 1 GB container cap; it is read on one thread, because Arrow's thread pool added about 40 MiB more.
Measured in-process with `TestClient` on the shared development laptop under a load average of about 3.5, over 1,200 requests built from test listings, 300 each make only, make and model, P1, and P1 with `k` 20, a request took p50 5.0 ms and p95 5.9 ms, with no shape above p95 6.0 ms; 2.7 % of the P1 requests got an empty list (`reports/analysis/comparables_footprint.py`).
That is inside NFR-02's 300 ms, which the load test on the target VM checks.

### Model version and request ID

The model version is the hash DVC records for the bundle directory in `dvc.lock`, so `GET /health` and `dvc.lock` can be compared directly.
Every response carries an `X-Request-ID` header; `/predict`, `/price-range` and `/comparables` repeat it as `request_id`, and a problem names it in `instance`.
The service generates it for every request and never takes it from the client; every line of the [service log](#logs) written while the request is answered carries it.

## Metrics

`GET /metrics` serves the metrics below in Prometheus' text exposition format.
In the deployment only Prometheus reads it, inside the Compose network: nginx refuses the path from outside.

| Metric | Type | Labels | What it counts |
|--------|------|--------|----------------|
| `http_requests_total` | counter | `method`, `path`, `status` | Answered requests |
| `http_request_duration_seconds` | histogram | `method`, `path`, `status` | Time from receiving a request to the last byte of its answer |
| `http_request_errors_total` | counter | `method`, `path`, `status` | Answers with a status of 400 or above |
| `recommenditos_predicted_price_eur` | histogram | - | The prices `/v1/predict` answered, for drift monitoring (M6) |
| `recommenditos_prediction_warnings_total` | counter | `field` | `/v1/predict` warnings, by the field they point at (FR-05) |
| `recommenditos_price_range_price_eur` | histogram | - | The typical prices `/v1/price-range` answered |
| `recommenditos_price_range_relative_width` | histogram | - | The width of each answered range over its typical price, which falls the more fields a client sends |
| `recommenditos_price_range_warnings_total` | counter | `field` | `/v1/price-range` warnings, by the field they point at (FR-05) |
| `recommenditos_comparables_listings` | histogram | - | How many listings each `/v1/comparables` answer held, 0 when none lay in every window |
| `recommenditos_comparables_warnings_total` | counter | `field` | `/v1/comparables` warnings, by the field they point at |
| `recommenditos_model_info` | gauge | `variant`, `version` | 1 for the served model; the version is the one `/health` reports |

`path` is the route template, such as `/v1/predict`, never the raw path, so a client cannot grow the label set; a request that matches no route is counted under `<unmatched>`.
A method other than the standard ones is counted as `OTHER`.
The latency histogram has bucket bounds at 0.25 s and 0.3 s, NFR-02's targets, so the share of answers within each target is read from one bucket.
Every request is counted, a body refused before it was read and a last-resort 500 included, because the middleware that counts sits outside all others.
The process's own CPU and memory are not here: cAdvisor or node-exporter report the container's (M6).

The figures are correct for any number of uvicorn workers when `PROMETHEUS_MULTIPROC_DIR` names a writable directory.
Each worker then writes its values to its own files there, and a scrape, answered by whichever worker, adds them up (prometheus_client's multiprocess mode).
`python -m recommenditos.api` deletes the metric files in that directory once, before the workers start, so a restarted service starts from zero; a worker never does, so a worker that uvicorn restarts keeps the others' counts.
Without the variable, as in the tests and a plain `make api`, each process reports its own figures, which is right only for one worker.
Multiprocess mode supports neither custom collectors nor Info metrics, so the model is a gauge labelled with its version rather than an Info metric.

## The prediction log and the feedback

Every answer of a valuation endpoint is one record of the prediction log (FR-13), and `POST /v1/feedback` stores an observed price against the answer's request ID (FR-10).
M6's drift job joins the two by request ID to report the real error and the interval coverage (FR-14).
Both are tables of one SQLite file in WAL mode, on a volume the drift job's container mounts as well ([EDN-86](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
The API reaches them through the interface `PredictionStore` in `recommenditos/api/prediction_log.py`, so a PostgreSQL store would be a second implementation of it and a change in `create_app`.

### What a record holds

| Column | Content |
|--------|---------|
| `request_id` | The ID the answer carries in `request_id` and `X-Request-ID`; unique |
| `endpoint` | The route, such as `/v1/predict` |
| `answered_at` | When the answer was recorded, ISO 8601 in UTC |
| `latency_ms` | From the arrival of the request to the record: validation and the model, without the serialisation of the response |
| `model_variant`, `model_version` | The model that answered, as in the answer's `model` |
| `inputs` | The validated request body as JSON, omitted optional fields left out |
| `outputs` | The answer as JSON, without `model`, `request_id` and `warnings`, which have their own columns |
| `warnings` | The answer's `warnings` as JSON |

The request bodies have no PII field, and no column holds the caller's address (NFR-08).
A refused request (422, 413) is not an answer and has no record; it shows in the access log and the metrics.
`feedback` holds `request_id`, `observed_price_eur` and `received_at`, and the view `labelled_predictions` is the join of the two tables.
A reader opens the file with SQLite in the same way the API does; it needs write access to the directory for WAL's shared-memory file, even when it only reads.

### Exactly once

A route records its answer right before it returns it, with one call:

```python
def predict(
    listing: ValuationRequest, served: Served, request_id: RequestId, log: Log
) -> Valuation:
    ...
    log.record(listing, valuation)
    return valuation
```

The client gets no answer the log does not hold, and the log holds no answer twice (NFR-03).
`request_id` is unique in the table, and `log.record` fails when a route calls it twice.
Several worker processes write the same file: SQLite serialises the writers with a file lock, and a writer that finds it held waits up to 5 s.
When the record cannot be written, the client gets a 503 instead of the answer, and the cause goes to the service log.
An answer the log does not hold would break exactly-once silently, and the load test counts a 503 as an error, so a failing log is visible at once.

Writing a record takes 0.15 ms at the median and 0.23 ms at the 95th percentile, measured over 2,000 answers of `lgbm-basic` on a development laptop, against the 250 ms NFR-02 gives `/predict`.
A record takes about 940 bytes, most of it the explanation in `outputs`, so the 2 GB of NFR-04 hold about 2.1 million answers, fewer when requests fill more optional fields.

### Feedback

`POST /v1/feedback` takes `{"request_id": ..., "observed_price_eur": ...}` with the key in `X-API-Key`.
It is internal: the reverse proxy refuses it from outside, and the key is a second layer ([EDN-13](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
The key is compared with `secrets.compare_digest`, so the comparison takes the same time however much of it matches.

| Case | Answer |
|------|--------|
| A new price for an answer in the log | 201 and the stored price |
| The same price again | 200 and the stored price; nothing changes, so the replay job can retry |
| Another price for the same answer | 409; the stored price is kept |
| An answer that gave no price, such as one of `/v1/comparables` | 409; nothing is stored, because FR-14 has no estimate to compare the price with |
| A request ID the log does not hold, or no longer holds after rotation | 404; nothing is stored, because a price nothing joins is no label |
| A missing or wrong key | 401, checked before the body is validated; malformed JSON is a 422 and a body over 10 KB a 413 either way |

### Rotation

NFR-04 caps the prediction log and the feedback together at 2 GB, oldest first.
Every 1,000 records, and at startup, a background thread of each worker process compares the pages the data occupies with `RECOMMENDITOS_PREDICTION_LOG_MAX_BYTES`.
Over the cap, it deletes the oldest answers, with their feedback, in batches of 1,000 until the data fits 90 % of the cap.
It takes the lock per batch and pauses 50 ms between batches, so a request waits for one batch at most, and no request runs the pruning itself.
A batch of 1,000 held the lock for 2 ms at the median and 26 ms at most, measured on a 47 MB log of 150,000 records on a development laptop.
SQLite reuses the freed pages for new records instead of growing the file, so the file stays at about the cap without a `VACUUM`, and the WAL file is truncated to 64 MB after each checkpoint.
Setting the variable and the volume on the VM is part of the deployment (NFR-04).

## Logs

The service writes its log to stderr, through loguru.
`RECOMMENDITOS_LOG_FORMAT` picks the format: `json` writes one JSON object per line, and `text`, the default, writes loguru's human-readable line.
The image sets `json`, so `docker compose logs api` can be filtered with `jq`; `make api` keeps the text format for reading in a terminal.

Every line has `time` (UTC), `level`, `logger` (the module that wrote it) and `message`, plus `exception` with the traceback when there is one.
A line written while a request is answered also has its `request_id`, the value of the answer's `X-Request-ID`, whichever module wrote it.
The last line of a request says how it was answered:

```json
{"time": "2026-10-21T08:37:38.570+00:00", "level": "INFO", "logger": "recommenditos.api.middleware", "message": "POST /v1/predict 200 54.7 ms", "request_id": "0b9f3c1e-6c0a-4b5e-9d1f-2a3b4c5d6e7f", "method": "POST", "path": "/v1/predict", "status": 200, "duration_ms": 54.7}
```

That line replaces uvicorn's access log, which names the client's address and port.
No line of the service log holds the client's address, a header, the request body or the path the client sent: `path` is the route template, as in the [metrics](#metrics), and `<unmatched>` for a path that matches no route.
uvicorn's own lines, such as its startup and shutdown, and warnings from the libraries go to the same log in the same format, in the supervising process as in every worker, because `python -m recommenditos.api` starts uvicorn with `recommenditos/api/log_config.json`.
uvicorn's `--log-level` sets the level of the whole log: `--log-level warning` leaves out the lines of answered requests.

### Following one request

nginx's access log line ends with `request_id=`, the `X-Request-ID` of the API's answer, so one ID finds the request in the proxy's log, in the API's log and in the prediction log:

```bash
docker compose logs proxy | grep 'request_id=<id>'      # the request as nginx saw it, with the client
docker compose logs api | grep '"request_id": "<id>"'   # every line the API wrote for it
docker compose exec api python -c 'import os, sqlite3, sys; print(sqlite3.connect(os.environ["RECOMMENDITOS_PREDICTION_LOG"]).execute("SELECT * FROM predictions WHERE request_id = ?", sys.argv[1:]).fetchone())' <id>   # its record
```

The client's address is only in nginx's log, which stays on the VM (NFR-08).
A 403, 413 or 503 that nginx answers itself never reached the API: its `X-Request-ID` is nginx's own, 32 hex digits, and only nginx's log names it.
The last command prints the [prediction-log record](#what-a-record-holds) of a valuation, or `None` for a request that is not one; a problem names the ID in `instance`, so a client that reports an error quotes the ID that finds it.

## Errors

Every error is an [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457) Problem Details object with the media type `application/problem+json` ([EDN-84](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
That holds for every error path: a request that fails validation, an unsupported make, a body over the size limit, an unknown route, a wrong method, a price range from a bundle without interval models (503), and an unexpected failure.

```json
{
  "type": "https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/docs/docs/api.md#validation-error",
  "title": "Invalid request",
  "status": 422,
  "detail": "The request has 2 invalid fields.",
  "instance": "urn:uuid:248b5a24-c067-4dd2-8f2b-69d1f88c18ae",
  "errors": [
    {"pointer": "#/make", "detail": "'Trabant' is not a make the served model supports.",
     "allowed": ["Alfa Romeo", "Aston Martin", "Audi", "BMW", "..."]},
    {"pointer": "#/mileage_km_raw", "detail": "Input should be greater than or equal to 0"}
  ]
}
```

`type` identifies the kind of problem, and a client branches on it.
The kinds the API defines are below; any other error has the type `about:blank`, which RFC 9457 reserves for a problem that means no more than its HTTP status, such as 404 or 405.
That includes `/v1/feedback`'s 401, 404 and 409 and the 503 of a prediction log that cannot be written, described [above](#the-prediction-log-and-the-feedback).

### validation-error

Status 422.
The request body is not valid JSON, misses a required field, has a value of the wrong type, outside its range or longer than its bound (FR-03), or names a value the served model does not accept, such as an unsupported make (FR-04).
`errors` has one entry per invalid field, every invalid field of the request in one response.
`pointer` is a JSON Pointer ([RFC 6901](https://www.rfc-editor.org/rfc/rfc6901)) into the request body, written as a URI fragment; `#` alone means the body as a whole.
No endpoint takes query or path parameters yet; the first one that does decides how its errors name them, and documents it here.
`allowed` lists the accepted values where the field takes one of a set.

### payload-too-large

Status 413.
The request body is larger than 10 KB (NFR-09).
A declared `Content-Length` over the limit is refused before the body is read; a chunked body without one is counted as it arrives and refused as soon as it passes the limit.
The largest of the 118,382 raw listings, sent as a valuation request with every field it has, is 3.5 KB.
A body within the [length bounds](#endpoints) of every field fits in the limit, so a 413 means padding, escaped characters or a body that is not a listing.

A 500 says nothing about its cause; the cause is in the [service log](#logs), as the `exception` of the line with the request ID that `instance` names.

## Adding an endpoint

The code is in `recommenditos/api/`:

| Module | What it holds |
|--------|---------------|
| `app.py` | `create_app`: loads the bundle, wires middlewares and error handlers, mounts the routers, finishes the OpenAPI document; `openapi_contract`: the document without a bundle |
| `routes/` | One module per route group, each with a `router` |
| `schemas/common.py` | What every body shares: the field types, `ListingFields`, the model description, warnings and the error shape |
| `schemas/<group>.py` | The request and response bodies of one route group, named after its module in `routes/` |
| `errors.py` | The problem shape, the handlers, and `problem_responses` for a route's documentation |
| `serving.py` | `ServedModel`: the loaded bundle, what it accepts, and the valuation |
| `settings.py` | `Settings`: the environment variables the service reads |
| `prediction_log.py` | `PredictionStore` and its SQLite implementation, and `PredictionLog`, which records an answer |
| `dependencies.py` | `Served`, `Metrics`, `RequestId`, `Log` and `Store`, which a route takes as parameters, and `RequireApiKey` |
| `middleware.py` | The request ID, its arrival time and the line that logs each request, the last-resort 500, and the body-size limit |
| `logs.py`, `log_config.json` | The service log: JSON or text, the request ID on every line, uvicorn's lines in the same log |
| `metrics.py` | The Prometheus metrics, the middleware that counts and times every request, and the multiprocess directory |
| `__main__.py` | `python -m recommenditos.api`: prepares the multiprocess directory, then starts uvicorn with the logging configuration |

1. Add `routes/<group>.py` with `router = APIRouter(tags=[...])` and its operations.
   Take `served: Served` for the model and `request_id: RequestId` for the request ID.
   An endpoint that answers a valuation takes `log: Log` and calls `log.record(body, answer)` right before it returns, and lists `HTTPStatus.SERVICE_UNAVAILABLE` among its errors; an internal endpoint adds `dependencies=[RequireApiKey]`.
2. In `routes/__init__.py`, add the module to the import and its router to `V1_ROUTERS` if it is part of the client contract, or to `OPERATIONAL_ROUTERS` if it is for operating the service.
   Both are one entry per line with a trailing comma, in alphabetical order, so that endpoints added in parallel touch different lines.
3. Put its bodies in `schemas/<group>.py`, built from the field types in `schemas/common.py`.
   A body that describes a listing derives from `ListingFields`, which makes the error handler run the served model's checks on it, so its 422 lists every invalid field like `/predict`'s.
   Give every request and response body an `examples` entry in its `model_config`.
4. Raise `ProblemError`, or `validation_problem([...])` for a 422, for an error; list `problem_responses(...)` with the statuses it can return in the decorator.
5. Add `tests/api/test_<group>.py`.
   The fixtures in `tests/api/conftest.py` give an app on a synthetic bundle (`client`, `served`) and a valid listing (`listing`), and `tests/api/helpers.py` checks the problem shape.
   `tests/api/test_openapi.py` walks every operation, so it fails until the new one has a request and a response example and documents its errors as problems (FR-16).
   `tests/api/test_prediction_log.py` posts a listing to every operation tagged `valuation` and fails until it records its answer once.
   `tests/api/test_contract.py` covers the new operation without a change: Schemathesis generates valid and invalid requests from the OpenAPI document and fails on a 5xx, an undocumented status code, a response that does not match its schema, a valid request refused or an invalid one accepted ([Contract tests](#contract-tests)).
6. Add `tests/integration/test_deployed_<group>.py` with one valid request through the proxy and the documented shape of its answer; `test_deployed_predict.py` is the example ([Deployment](deployment.md#integration-tests)).
7. Run `make openapi` and commit the regenerated `docs/docs/openapi.json`.
   `tests/api/test_openapi.py` fails while the committed document differs from the one the app serves, so every change to the contract shows in its pull request.
   The file leaves out the values a checked listing field accepts, which come from the served bundle, so a retrain does not change it.

A module the API comes to import has to stay within the serving runtime; [Contributing](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/CONTRIBUTING.md#dependencies) explains the check.

## Contract tests

`tests/api/test_contract.py` runs with every `make test` and in CI, against the app in process, with no server and no network.
Schemathesis reads `/openapi.json` and sends each operation the boundary and invalid values its coverage phase derives from the schema (about 370 requests for `/v1/predict`), up to 50 random valid and invalid requests, and the documented example.
Generation is deterministic, so every run sends the same requests, and a failure prints a `curl` line that reproduces it.
For the six operations so far that is about 970 requests in about 18 seconds on a laptop CPU, most of them for `/v1/predict` and `/v1/price-range`.
An operation behind an API key, such as `/v1/feedback`, gets the test app's key, and Schemathesis also checks that it refuses a request without it.

The OpenAPI document is what a client codes against, so the API keeps to it as written: a request body is strict, and `0` is not a boolean nor `"45000"` a number, while a whole float such as `4.0` is an integer, as JSON Schema defines one.
A constraint the schema cannot state needs a hook in the test module.
The 10 KB limit of NFR-09 needs none: the length bounds keep every generated valid body under it, so a 413 for one fails the test.
The one so far is that `registration_date` must not lie after the current month (FR-03): the pattern states the lower bound, 1900-01, and the upper bound moves with the calendar.
The response schemas allow additional properties, so a field an answer gains without being documented is not caught; a documented field that changes its type or goes missing is.
A generated feedback names a request ID the prediction log does not hold, so the stored, repeated and conflicting prices of `/v1/feedback` are left to `tests/api/test_feedback.py`.
