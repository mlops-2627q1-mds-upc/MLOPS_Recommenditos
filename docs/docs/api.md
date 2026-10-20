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
A bundle the API cannot serve, for example one that consumes no column for a field FR-01 requires, stops the service at startup instead of failing on the first request.

A valuation:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/predict -H 'content-type: application/json' -d '{
  "make": "BMW", "model": "320", "registration_date": "2021-03",
  "mileage_km_raw": 60000, "power_kw": 135, "fuel_category": "Diesel",
  "transmission": "Automatic", "country_code": "DE", "body_type": "Sedan",
  "seller_type": "Dealer"}'
```

## Endpoints

Every endpoint is versioned under `/v1`, so an incompatible change can ship as `/v2` beside it.

| Endpoint | Purpose | Requirements |
|----------|---------|--------------|
| `POST /v1/predict` | The point estimate of the asking price in EUR, the model that made it, a request ID and warnings | FR-01, FR-03 to FR-06 |
| `GET /v1/health` | `ok` and the served model: its variant, version and training time | FR-11, FR-12 |

`/predict` requires the ten fields of FR-01 and accepts every other input of the basic and the extended feature set as optional.
The served model reads the fields its feature set holds; an omitted one reaches it as absent, an omitted equipment list as an empty one.
A field the request body does not define is refused, so a misspelt optional field does not go unnoticed.

The values a categorical field accepts come from the loaded bundle, not from the code: the supported makes from its metadata (FR-04), the levels of `fuel_category`, `transmission`, `body_type`, `drive_train` and `seller_type` from its feature space (FR-03).
`/docs` lists them as enums, and a 422 for such a field lists them in `allowed`.
`country_code` accepts the 8 countries of the data.
A country or a `model` the training data does not hold, `ES` today, is answered and reported in `warnings` (FR-05).

The model version is the hash DVC records for the bundle directory in `dvc.lock`, so `GET /v1/health` and `dvc.lock` can be compared directly.
Every response carries an `X-Request-ID` header; `/predict` repeats it as `request_id`, and a problem names it in `instance`.

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
| `schemas.py` | The request and response bodies, and the field types they share |
| `errors.py` | The problem shape, the handlers, and `problem_responses` for a route's documentation |
| `serving.py` | `ServedModel`: the loaded bundle, what it accepts, and the valuation |
| `dependencies.py` | `Served` and `RequestId`, which a route takes as parameters |
| `middleware.py` | The request ID, the last-resort 500, and the body-size limit |

1. Add `routes/<group>.py` with `router = APIRouter(tags=[...])` and its operations.
   Take `served: Served` for the model and `request_id: RequestId` for the request ID.
2. Add its router to `V1_ROUTERS` in `routes/__init__.py`, in alphabetical order, one per line.
3. Build the request body from the field types in `schemas.py`.
   A body that describes a listing derives from `ListingFields`, which makes the error handler run the served model's checks on it, so its 422 lists every invalid field like `/predict`'s.
   Give every request and response body an `examples` entry in its `model_config`.
4. Raise `ProblemError`, or `validation_problem([...])` for a 422, for an error; list `problem_responses(...)` with the statuses it can return in the decorator.
5. Add `tests/api/test_<group>.py`.
   The fixtures in `tests/api/conftest.py` give an app on a synthetic bundle (`client`, `served`) and a valid listing (`listing`), and `tests/api/helpers.py` checks the problem shape.
   `tests/api/test_openapi.py` walks every operation, so it fails until the new one has a request and a response example and documents its errors as problems (FR-16).

A module the API comes to import has to stay within the serving runtime; [Contributing](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/CONTRIBUTING.md#dependencies) explains the check.
