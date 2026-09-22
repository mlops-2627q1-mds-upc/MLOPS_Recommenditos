Requirements
============

What the used-car price component does and under which constraints.
The requirements are derived from the use cases UC1 (car valuation) and UC2 (purchase guidance) in the [project brief](project-brief.md).
What the model learns, its features and its quality targets live in the [problem specification](problem-spec.md); this page links to it instead of repeating it.

Every requirement has an ID (`FR-xx` for functional, `NFR-xx` for non-functional).
Tests reference the IDs they verify in their name, e.g. `test_fr05_rejects_unsupported_make`.

Status markers as in the project brief:

- **[decided]** agreed by the team.
- **[proposed]** proposed and pending team confirmation (see [Decisions pending confirmation](#decisions-pending-confirmation)).
- **[open]** depends on a decision that is not made yet.

## 1. Functional requirements

### Inputs

Input fields are the features of the [problem specification](problem-spec.md#4-features) and use its names.
The API schema only adds the validation rules below; it does not define features of its own.

| ID | Requirement | Verified by |
|----|-------------|-------------|
| FR-01 | **UC1 required fields [proposed]:** `/predict` requires `make`, `model`, `registration_date` (year and month), `mileage_km_raw`, `power_kw`, `fuel_category`, `transmission` and `country_code`. All other features of the basic and extended set are optional; a missing optional field is passed to the model as missing. | API test per required field, e.g. `test_fr01_rejects_missing_power_kw` |
| FR-02 | **UC2 required fields:** `/price-range` requires only `make`; any subset of the other features is accepted. | API test with `make` only and with the partial-input scenario P1 of SC-05 |
| FR-03 | **Validation:** fields are type-checked; `registration_date` lies between 1900-01 and the request month; `mileage_km_raw` is between 0 and 1,000,000; `power_kw` is between 1 and 1,200; low-cardinality categoricals (`fuel_category`, `transmission`, `body_type`, `drive_train`, `seller_type`) must be one of the values in the training data; `country_code` is an ISO 3166-1 alpha-2 code. An invalid request gets HTTP 422 with one error per invalid field. | Parametrised API tests |
| FR-04 | **Scope check:** a `make` outside the supported makes (computed by the pipeline, [EDN-05](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)) is rejected with HTTP 422, and the response lists the supported makes. | API test |
| FR-05 | **Unseen values [proposed]:** a `country_code` or `model` that is valid but missing from the training data (e.g. `ES`, see project brief section 3.2) is accepted, passed to the model as unknown, and reported in a `warnings` field of the response. | API test for `ES` and for an unseen model |

### Outputs and endpoints

| ID | Requirement | Verified by |
|----|-------------|-------------|
| FR-06 | **Valuation (UC1) [proposed]:** `POST /predict` returns the point estimate of the asking price in EUR, the model version and a request ID. | API test |
| FR-07 | **Price range (UC2) [proposed]:** `POST /price-range` returns the point estimate and the lower and upper bound of the nominal 90 % interval in EUR, the model version and a request ID. | API test (lower ≤ estimate ≤ upper) |
| FR-08 | **Explanation:** `/predict` returns the five features with the largest SHAP contribution, each as its effect on the price in percent. | API test: all contributions plus the base value reproduce the prediction |
| FR-09 | **Comparables:** `POST /comparables` takes the same input as `/price-range` and returns the k most similar processed listings (default 5, at most 20) with their price and key features, without PII. | API test, PII test (NFR-08) |
| FR-10 | **Feedback [open]:** `POST /feedback` takes a request ID and an observed price and stores it as a delayed label. It is fed with simulated prices from the `ES` holdout. Whether the endpoint exists depends on the feedback-loop decision in project brief section 5. | API test |
| FR-11 | **Operations:** `GET /health` returns the status and the loaded model version; `GET /metrics` exposes request count, latency and error metrics in Prometheus format. | API test |
| FR-12 | **Model loading:** at startup the API loads the current production model from the MLflow model registry. A new model version needs no change to the API schema. | Integration test with two registered versions |
| FR-13 | **Prediction log:** every request is logged with its validated inputs, outputs, warnings, model version, timestamp and latency, without PII. | Integration test |
| FR-14 | **Drift monitoring:** a separate job compares the logged inputs of each time window with the training reference and reports input drift; when feedback exists, it also reports error and interval coverage. | Test with the `ES` replay (NFR-11) |

### Out of scope

- Batch prediction: the `ES` replay sends single requests, like real clients.
- Free-text input or parsing via an LLM (problem specification section 2).
- User accounts and authentication: the API is public and read-only apart from `/feedback`.

## 2. Non-functional requirements

Targets for latency, throughput and resources (NFR-02 to NFR-05) are estimates for the planned 4 GB, CPU-only VM **[proposed]**.
They are checked with the first load test in M4 and adjusted there if needed.

| ID | Quality | Requirement and target | Verified by |
|----|---------|------------------------|-------------|
| NFR-01 | Model quality | A model is only deployed if it meets all success criteria SC-01 to SC-05 of the [problem specification](problem-spec.md#8-success-criteria). This requirement sets no thresholds of its own. | Model tests (M3) and a gate before a model version is promoted in the registry |
| NFR-02 | Latency | On the target VM, p95 latency is at most 200 ms for `/predict` (including SHAP) and at most 300 ms for `/price-range` and `/comparables`. | Load test and the Prometheus latency histogram |
| NFR-03 | Throughput and scalability | The API sustains 20 requests/s for 5 minutes with less than 1 % errors. The API keeps no state in memory between requests, so it scales by adding replicas. | Load test |
| NFR-04 | Resources | The full stack runs in 4 GB RAM; the API container stays below 1 GB RSS under the NFR-03 load; the API image is at most 1 GB and contains no GPU or deep-learning libraries; the drift job runs in its own container (project brief section 6). | `docker stats` during the load test; image size check in CI |
| NFR-05 | Availability | All services restart automatically; the API is ready at most 30 s after start; uptime is at least 99 % in the weeks before each presentation. | Prometheus `up` metric |
| NFR-06 | Reproducibility | `dvc repro` on a clean clone produces the same splits and metrics within ±0.1 percentage points. Every MLflow run records the git commit, the DVC data version and all parameters. | Re-run before each delivery; MLflow run check |
| NFR-07 | Maintainability | ruff (including the Pylint rules) reports no findings; test coverage of `src/` is at least 80 %; Pynblint reports no issues on the notebooks; CI passes before every merge. | CI |
| NFR-08 | Privacy | No PII column of the problem specification (section 4, excluded columns) appears in the processed data, the prediction log, the comparables or the model artefacts. The raw data is never re-hosted in a public remote. | Great Expectations suite and tests on the response and log schemas |
| NFR-09 | Security | Request bodies are limited to 10 KB; no secrets are committed; containers run as a non-root user; all dependencies are locked in `uv.lock`. | API test, secret scan in CI, Dockerfile review |
| NFR-10 | Energy efficiency | CodeCarbon measures the emissions of every training run and logs them to MLflow; a full training run takes at most 15 minutes on a laptop CPU. | MLflow |
| NFR-11 | Observability | The drift job flags the `ES` replay as drift within its first 1,000 requests. | M6 replay |
| NFR-12 | Portability | `docker compose up` starts the whole stack on a fresh VM; no LLM or paid external service is needed at runtime. | Deployment smoke test |

## Decisions pending confirmation

The items marked **[proposed]** are proposals that could reasonably go differently.
Once the team confirms them, they become **[decided]** and are recorded in the [EDN](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md).

1. **UC1 required fields (FR-01).**
   - **A (proposed):** a small required core that every owner knows; everything else is optional.
     Keeps UC1 clearly different from UC2 without asking for fields users often do not know.
   - **B:** only `make` required on both endpoints.
     Simplest, but `/predict` then returns point estimates for barely described cars and the two use cases blur.
   - **C:** all basic features required.
     Best accuracy, but hostile to users: `nr_prev_owners` is filled in only 55 % of the listings.
2. **Unseen values (FR-05).**
   - **A (proposed):** accept with a warning.
     Needed for the `ES` drift scenario and supported natively by LightGBM.
   - **B:** reject with HTTP 422.
     Stricter, but breaks the new-market scenario.
3. **Endpoint split (FR-06, FR-07).**
   - **A (proposed):** separate `/predict` and `/price-range`, as in the project brief.
     The two use cases have different validation rules, so each endpoint has one clear contract.
   - **B:** one `/valuation` endpoint that returns the estimate and the interval.
     Fewer endpoints, but the required fields would depend on the use case inside one schema.
4. **Performance and resource targets (NFR-02 to NFR-05).**
   - **A (proposed):** commit to the estimated targets now and revisit them after the first M4 load test.
     Gives the load tests and the report a target from the start.
   - **B:** leave the targets open until they are measured.
     Avoids guessing, but leaves M4 without acceptance criteria.
