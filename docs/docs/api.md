The API
=======

The HTTP API serves the price model: it estimates the asking price of a used car (UC1) with the model bundle the service was started with.
This page says how to run it, what it offers and how its errors are shaped, and how to add an endpoint.
The field-by-field reference is the generated OpenAPI document, served by the API itself at `/docs` (Swagger UI), `/redoc` (ReDoc) and `/openapi.json`; this page does not repeat it.
What each endpoint must do is in the [specification](specification.md) (FR-01 to FR-16, NFR-09).

## Running it

```bash
uv run dvc pull train      # the model bundles, into models/
make api                   # uvicorn on http://127.0.0.1:8000
```

`make api` serves `models/lgbm-basic`.
`RECOMMENDITOS_MODEL_DIR=models/lgbm-extended make api` serves another bundle, and `make api API_PORT=8080` another port.
The bundle is loaded once, before the first request, and the API never calls MLflow ([EDN-08](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md), FR-12).
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

## Endpoints

The client contract is versioned under `/v1`, so an incompatible change can ship as `/v2` beside it.
The operational endpoints are not part of that contract and stay at the root, where a browser, an uptime monitor and Prometheus look for them: `/`, `/health` and `/metrics`.

| Endpoint | Purpose | Requirements |
|----------|---------|--------------|
| `POST /v1/predict` | The point estimate of the asking price in EUR, its explanation, the model that made it, a request ID and warnings | FR-01, FR-03 to FR-06, FR-08 |
| `GET /health` | `ok` and the served model: its variant, version and training time | FR-11, FR-12 |
| `GET /` | The service name, the served model and links to `/docs`, `/redoc` and `/openapi.json` | - |
| `GET /metrics` | Request, latency, error and model metrics in Prometheus' text format | FR-11 |

`/v1/predict` values one car per request, because UC1 is one seller asking about one listing and every answer becomes one record of the prediction log; it has no batch form.
`/predict` requires the ten fields of FR-01 and accepts every other input of the basic and the extended feature set as optional.
The served model reads the fields its feature set holds; an omitted one reaches it as absent, an omitted equipment list as an empty one.
A field the request body does not define is refused, so a misspelt optional field does not go unnoticed.

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
No listing of the data reaches the bound with `lgbm-basic`.

The explanation is the costliest step of a request, because TreeSHAP walks each of the 1,394 trees of `lgbm-basic` once per feature path.
Measured in-process with `TestClient` on a shared 4-core development laptop, over 394 requests built from test listings, it took p50 34 ms and p95 49 ms of a request that took p50 63 ms and p95 93 ms; a second run under heavier load from other processes gave 44 and 71 ms of 81 and 140 ms.
Both are inside NFR-02's 200 ms for `/predict`, which the load test on the target VM checks.

### Model version and request ID

The model version is the hash DVC records for the bundle directory in `dvc.lock`, so `GET /health` and `dvc.lock` can be compared directly.
Every response carries an `X-Request-ID` header; `/predict` repeats it as `request_id`, and a problem names it in `instance`.

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
| `recommenditos_model_info` | gauge | `variant`, `version` | 1 for the served model; the version is the one `/health` reports |

`path` is the route template, such as `/v1/predict`, never the raw path, so a client cannot grow the label set; a request that matches no route is counted under `<unmatched>`.
A method other than the standard ones is counted as `OTHER`.
The latency histogram has bucket bounds at 0.2 s and 0.3 s, NFR-02's targets, so the share of answers within each target is read from one bucket.
Every request is counted, a body refused before it was read and a last-resort 500 included, because the middleware that counts sits outside all others.
The process's own CPU and memory are not here: cAdvisor or node-exporter report the container's (M6).

The figures are correct for any number of uvicorn workers when `PROMETHEUS_MULTIPROC_DIR` names a writable directory.
Each worker then writes its values to its own files there, and a scrape, answered by whichever worker, adds them up (prometheus_client's multiprocess mode).
`python -m recommenditos.api` deletes the metric files in that directory once, before the workers start, so a restarted service starts from zero; a worker never does, so a worker that uvicorn restarts keeps the others' counts.
Without the variable, as in the tests and a plain `make api`, each process reports its own figures, which is right only for one worker.
Multiprocess mode supports neither custom collectors nor Info metrics, so the model is a gauge labelled with its version rather than an Info metric.

## Errors

Every error is an [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457) Problem Details object with the media type `application/problem+json` ([EDN-84](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
That holds for every error path: a request that fails validation, an unsupported make, a body over the size limit, an unknown route, a wrong method, and an unexpected failure.

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

### validation-error

Status 422.
The request body is not valid JSON, misses a required field, has a value of the wrong type or outside its range (FR-03), or names a value the served model does not accept, such as an unsupported make (FR-04).
`errors` has one entry per invalid field, every invalid field of the request in one response.
`pointer` is a JSON Pointer ([RFC 6901](https://www.rfc-editor.org/rfc/rfc6901)) into the request body, written as a URI fragment; `#` alone means the body as a whole.
No endpoint takes query or path parameters yet; the first one that does decides how its errors name them, and documents it here.
`allowed` lists the accepted values where the field takes one of a set.

### payload-too-large

Status 413.
The request body is larger than 10 KB (NFR-09).
A declared `Content-Length` over the limit is refused before the body is read; a chunked body without one is counted as it arrives and refused as soon as it passes the limit.
A valuation request with every optional field filled is about 2 KB.

A 500 says nothing about its cause; the cause is in the service log under the request ID that `instance` names.

## Adding an endpoint

The code is in `recommenditos/api/`:

| Module | What it holds |
|--------|---------------|
| `app.py` | `create_app`: loads the bundle, wires middlewares and error handlers, mounts the routers, finishes the OpenAPI document |
| `routes/` | One module per route group, each with a `router` |
| `schemas/common.py` | What every body shares: the field types, `ListingFields`, the model description, warnings and the error shape |
| `schemas/<group>.py` | The request and response bodies of one route group, named after its module in `routes/` |
| `errors.py` | The problem shape, the handlers, and `problem_responses` for a route's documentation |
| `serving.py` | `ServedModel`: the loaded bundle, what it accepts, and the valuation |
| `dependencies.py` | `Served`, `Metrics` and `RequestId`, which a route takes as parameters |
| `middleware.py` | The request ID, the last-resort 500, and the body-size limit |
| `metrics.py` | The Prometheus metrics, the middleware that counts and times every request, and the multiprocess directory |
| `__main__.py` | `python -m recommenditos.api`: prepares the multiprocess directory, then starts uvicorn |

1. Add `routes/<group>.py` with `router = APIRouter(tags=[...])` and its operations.
   Take `served: Served` for the model and `request_id: RequestId` for the request ID.
2. In `routes/__init__.py`, add the module to the import and its router to `V1_ROUTERS` if it is part of the client contract, or to `OPERATIONAL_ROUTERS` if it is for operating the service.
   Both are one entry per line with a trailing comma, in alphabetical order, so that endpoints added in parallel touch different lines.
3. Put its bodies in `schemas/<group>.py`, built from the field types in `schemas/common.py`.
   A body that describes a listing derives from `ListingFields`, which makes the error handler run the served model's checks on it, so its 422 lists every invalid field like `/predict`'s.
   Give every request and response body an `examples` entry in its `model_config`.
4. Raise `ProblemError`, or `validation_problem([...])` for a 422, for an error; list `problem_responses(...)` with the statuses it can return in the decorator.
5. Add `tests/api/test_<group>.py`.
   The fixtures in `tests/api/conftest.py` give an app on a synthetic bundle (`client`, `served`) and a valid listing (`listing`), and `tests/api/helpers.py` checks the problem shape.
   `tests/api/test_openapi.py` walks every operation, so it fails until the new one has a request and a response example and documents its errors as problems (FR-16).

A module the API comes to import has to stay within the serving runtime; [Contributing](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/CONTRIBUTING.md#dependencies) explains the check.
