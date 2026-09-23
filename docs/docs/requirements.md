Requirements
============

What the used-car price component does and under which constraints.
The requirements are derived from the use cases UC1 (car valuation) and UC2 (purchase guidance) in the [project brief](project-brief.md).
What the model learns, its features and its quality targets live in the [problem specification](problem-spec.md); this page links to it instead of repeating it.

Every requirement has an ID (`FR-xx` for functional, `NFR-xx` for non-functional).
Tests mark the IDs they verify with `@pytest.mark.req("FR-04")` (supports multiple IDs per test, e.g. a test that covers both FR-15 and NFR-01).
CI turns those markers into a requirement-to-test matrix (NFR-07), which is what the report cites as the traceability evidence.

The "Verified by" column says how each requirement is checked:

- **[automated]** at least one Pytest test carries the requirement's `req` marker, so the matrix picks it up on its own.
- **[manual]** the requirement cannot be checked from Pytest (a load test, a chaos test, a deployment run, a `dvc repro` on a clean clone, a human drill).
  The cell then names the evidence, and the matrix shows the requirement as manually verified rather than as missing.

Status markers as in the project brief:

- **[decided]** agreed by the team.
- **[proposed]** proposed and pending team confirmation (see [Decisions pending confirmation](#decisions-pending-confirmation)).
- **[open]** depends on a decision that is not made yet.

## 1. Functional requirements

### Inputs

Input fields are the features of the [problem specification](problem-spec.md#4-features) and use its names.
The API schema only adds the validation rules below; it does not define features of its own.
The allowed categorical values and the supported makes are not fixed in the schema: they are read from the metadata of the loaded model version and checked at runtime, so a new model version needs no schema change (FR-12).

| ID | Requirement | Verified by |
|----|-------------|-------------|
| FR-01 | **UC1 required fields [proposed]:** `/predict` requires `make`, `model`, `registration_date` (year and month), `mileage_km_raw`, `power_kw`, `fuel_category`, `transmission` and `country_code`. All other features of the basic and extended set are optional; a missing optional field is passed to the model as missing. **[open]** This assumes the point model is trained to handle each optional field's absence the way it is actually used at serving time; see the modelling plan (project brief section 4) for whether that needs the same random masking already planned for the UC2 interval models. | **[automated]** API test per required field, e.g. `test_fr01_rejects_missing_power_kw`; once trained, a per-field masked-input accuracy check |
| FR-02 | **UC2 required fields:** `/price-range` requires only `make`; any subset of the other features is accepted. | **[automated]** API test with `make` only and with the partial-input scenario P1 of SC-05 |
| FR-03 | **Validation:** fields are type-checked; `registration_date` lies between 1900-01 and the request month; `mileage_km_raw` is between 0 and 1,000,000; `power_kw` is between 1 and 1,200; low-cardinality categoricals (`fuel_category`, `transmission`, `body_type`, `drive_train`, `seller_type`) must be one of the values in the training data of the loaded model; `country_code` is one of the 8 countries of the data (problem specification section 2), so any other country is rejected as out of scope. An invalid request gets HTTP 422 with one error per invalid field. | **[automated]** Parametrised API tests, including a country outside the 8 (e.g. `US`) |
| FR-04 | **Scope check:** a `make` outside the supported makes (computed by the pipeline, [EDN-05](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)) is rejected with HTTP 422, and the response lists the supported makes. | **[automated]** API test |
| FR-05 | **Unseen values [proposed]:** a `country_code` among the 8 countries or a `model` that is missing from the training data (currently the country `ES`, see project brief section 3.2) is accepted, passed to the model as unknown, and reported in a `warnings` field of the response. | **[automated]** API test for `ES` and for an unseen model |

### Outputs and endpoints

All endpoint paths are served under `/v1` (omitted from the table below for brevity).

| ID | Requirement | Verified by |
|----|-------------|-------------|
| FR-06 | **Valuation (UC1) [proposed]:** `POST /predict` returns the point estimate of the asking price in EUR, the model version and a request ID. | **[automated]** API test |
| FR-07 | **Price range (UC2) [proposed]:** `POST /price-range` returns the point estimate and the lower and upper bound of the nominal 90 % interval in EUR, the model version and a request ID. The point estimate and the bounds come from different models, so the bounds are widened where needed to contain the estimate; widening only raises coverage, so the conformal guarantee still holds. | **[automated]** API test (lower ≤ estimate ≤ upper) |
| FR-08 | **Explanation [decided, EDN-11]:** `/predict` returns the base price and the five features with the largest absolute SHAP contribution, plus the combined contribution of all other features, computed from the trained booster's own SHAP export (LightGBM `predict(pred_contrib=True)` or CatBoost `get_feature_importance(type="ShapValues")`), not the `shap` package, which stays a training/notebook-only dependency for offline global analysis. The model predicts `log(price)`, so a contribution φ is reported as its multiplicative effect on the price, `exp(φ) - 1` in percent, and the effects multiply rather than add. | **[automated]** API test: the base price times the product of `1 + effect` over all six reported effects reproduces the estimate; a training-time test asserts the booster's export matches `shap.TreeExplainer` |
| FR-09 | **Comparables:** `POST /comparables` takes the same input as `/price-range` and returns up to k processed listings (default 5, at most 20) with their price and key features, without PII. A listing is comparable when it matches the request in `make` and `model` and its age differs by at most 2 years and its mileage by at most 25 %; the results are sorted by mileage distance, then age distance. Fields the request omits are ignored rather than folded into a distance, and no filter is relaxed: when nothing matches, the list is empty. The index holds the training and validation listings only, never `ES` (EDN-03) or test rows. | **[automated]** API test: every returned listing satisfies the filter, a request outside any window returns an empty list, the order is as specified; PII test (NFR-08) |
| FR-10 | **Feedback [decided, EDN-13]:** `POST /feedback` takes a request ID and an observed price and stores it as a delayed label, which FR-14 joins to the prediction log to report the real error and interval coverage. It is fed with simulated prices from the `ES` holdout; there are no real users. The endpoint is not publicly reachable: the reverse proxy in front of the API rejects `/feedback` from outside, so only the replay job inside the Compose network reaches it. It additionally requires an API key in the `X-API-Key` header as a second layer, in case that proxy rule is ever wrong; a missing or wrong key gets HTTP 401. The key therefore never travels over the public internet (NFR-09). | **[automated]** API test, including `test_fr10_rejects_missing_api_key`; deployment smoke test asserting `/feedback` is refused from outside the VM |
| FR-11 | **Operations:** `GET /health` returns the status and the loaded model version; `GET /metrics` exposes request count, latency and error metrics in Prometheus format. | **[automated]** API test |
| FR-12 | **Model loading [decided, EDN-08]:** the model and FR-09's comparables index are baked into the API image at CI build time via `dvc pull`, pinned to whatever version `dvc.lock` on `main` points to; promoting a model means merging that pointer to `main`. The API never calls the MLflow registry at build or run time; MLflow stays the experiment-tracking and audit record of which run was chosen. A new model version needs no change to the API schema. | **[automated]** CI check on the built image: the model version `GET /health` reports equals the version `dvc.lock` points to; an API test that the categorical values and supported makes come from the loaded model's metadata, not from the schema |
| FR-13 | **Prediction log:** every request is logged with its validated inputs, outputs, warnings, model version, timestamp and latency, without PII. | **[automated]** Integration test |
| FR-14 | **Drift monitoring:** a separate job compares the logged inputs of each time window with the training reference and reports input drift. Joining FR-10's delayed labels to the prediction log by request ID, it also reports the error (MdAPE) and the empirical interval coverage per window, so a degradation is visible as a number, not only as a changed input distribution. | **[automated]** Test with the `ES` replay (NFR-11), asserting all three (drift, error, coverage) are reported |
| FR-15 | **Retraining and promotion [decided, EDN-12]:** when FR-14's drift job flags input drift on the `ES` replay (per NFR-11), a team member retrains the pipeline with `ES` included (`dvc repro`) and opens a PR updating the model pointer. The candidate must pass NFR-01's gate before that PR is merged; merging is the promotion (EDN-08). Retraining and promotion are human-triggered, not automated: no PR is opened or merged without a person reviewing the gate result. If a promoted model regresses after deployment, rollback is redeploying the previous image tag (EDN-08). | **[manual]** `ES` replay integration test: drift flagged → retrain → gate → promotion PR opened; a rollback drill redeploying a previous image tag |
| FR-16 | **API documentation and errors:** the OpenAPI schema (`/docs`, `/redoc`) includes a request and response example for every endpoint. All error responses share one envelope, including validation errors (FR-03), the scope check (FR-04), a missing or invalid API key (FR-10), and the body-size limit (NFR-09); none differ in shape. | **[automated]** OpenAPI schema test: every endpoint has a request and response example; API tests confirm every error path returns the same envelope shape |

### Out of scope

- Batch prediction: the `ES` replay sends single requests, like real clients.
- Free-text input or parsing via an LLM (problem specification section 2).
- User accounts and per-user authentication: the prediction endpoints are public and read-only; `/feedback`, the only endpoint that writes data, is not exposed publicly at all and is protected by a single shared API key behind that (FR-10).

## 2. Non-functional requirements

Targets for latency, throughput and resources (NFR-02 to NFR-04) are estimates for the planned 4 GB, CPU-only VM **[proposed]**.
They are checked with the first load test in M4 and adjusted there if needed.

> **Unverified: what the VM's network allows.**
> Nobody on the team has access to the FIB Virtech VM yet, so anything below that depends on how the VM can be reached from the internet is an assumption, not a decision.
> The assumption on record is that the VM sits behind NAT, reachable through a forwarded port on a shared host name, which is how FIB Virtech is usually described, and nobody has checked it.
>
> Affected are NFR-05 (Better Uptime has to ping the VM from outside), NFR-09 (no TLS, because a Let's Encrypt HTTP-01 challenge needs port 80 on a host name we control) and NFR-13 (SSH from a GitHub-hosted runner, with a self-hosted runner on the VM as the fallback).
> Each of them names its fallback, so none of them blocks work before M4.
>
> Once we have access, one person checks three things, and we settle all three requirements together (decision 5 in [Decisions pending confirmation](#decisions-pending-confirmation)): whether the VM is reachable from the public internet at all, on which ports, and whether there is a host name we control enough to get a certificate for.
> Until then, do not build anything that silently assumes one of the answers, and do not write these assumptions into the report as facts.
> Nothing else on this page changes with the answer: FR-10's `/feedback` stays internal either way, and no target moves.

| ID | Quality | Requirement and target | Verified by |
|----|---------|------------------------|-------------|
| NFR-01 | Model quality | A model is only deployed if it meets all success criteria SC-01 to SC-05 of the [problem specification](problem-spec.md#8-success-criteria). This requirement sets no thresholds of its own. | **[automated]** Model tests (M3) and the gate before a model version is promoted (FR-15) |
| NFR-02 | Latency | On the target VM, p95 latency is at most 200 ms for `/predict` (including SHAP) and at most 300 ms for `/price-range` and `/comparables`. | **[manual]** Load test and the Prometheus latency histogram |
| NFR-03 | Throughput and scalability | The API sustains 20 requests/s for 5 minutes with less than 1 % errors, while still meeting NFR-02's latency targets. Correctness does not depend on how many worker processes or replicas run: the API keeps no per-request or per-session state, and the model and the comparables index are read-only, so they can be shared. Whatever the worker count, every request appears exactly once in FR-13's prediction log, and every FR-10 feedback record is visible to the drift job (FR-14). How the prediction log and the feedback store achieve this is a design decision, still open (project brief section 5). | **[manual]** Load test (throughput, error rate, NFR-02's latency targets); the same load test with 2 worker processes, asserting the prediction log holds exactly one entry per request |
| NFR-04 | Resources | **RAM:** the full stack runs in the VM's 4 GB; the API container stays below 1 GB RSS under the NFR-03 load. **Image:** the API image is at most 1 GB and contains no GPU or deep-learning libraries. **Drift job:** it runs in its own container (project brief section 6) and only while it works, started on a schedule instead of kept running, so its numpy<2 / pandas<3 stack with transformers, opencv and scikit-image never competes with the API for memory. **Disk:** the stack fits the VM's 20 GB with at least 4 GB free at all times: container images together at most 6 GB, Prometheus capped at 2 GB by a retention size, and the prediction log and feedback (FR-13, FR-10) capped at 2 GB by rotation, oldest first. The VM needs no DVC cache, because the model ships inside the image (FR-12). | **[manual]** `docker stats` during the load test; image size check in CI; the disk metric in Grafana with an alert on the 4 GB floor; a deployment check that the retention and rotation caps are configured, not just intended |
| NFR-05 | Availability | **[decided, EDN-10]** All services use `restart: unless-stopped` and the API reaches `/health` = 200 within 30 s of a crash or VM reboot. The API has zero unplanned downtime during the M4-M6 presentation (Dec. 9) and its warm-up, checked immediately beforehand; the API does not exist yet for the M1-M3 presentation (Oct. 14). Outside the presentation window the stack runs unattended on a single VM with no redundancy or SLA, so uptime is monitored and reported, not contractually targeted. | **[manual]** Chaos test (kill the API container, measure recovery); manual health check before the December presentation; Better Uptime external ping (**[proposed]**, needs the VM to be reachable from outside, see the warning at the top of this section) and Grafana dashboard for the monitored period |
| NFR-06 | Reproducibility | `dvc repro` on a clean clone produces the same splits and metrics within ±0.1 percentage points. Every MLflow run records the git commit, the DVC data version and all parameters. | **[manual]** Re-run before each delivery; MLflow run check |
| NFR-07 | Maintainability | ruff (including the Pylint rules) reports no findings; test coverage of the application code in `recommenditos/` (notebooks and one-off scripts excluded) is at least 80 %; Pynblint reports no findings on the notebooks beyond the ones we accept, each of which is documented with its reason; CI passes before every merge. From M3 onward, CI also builds the requirement-to-test matrix from the `req` markers and the tables on this page, publishes it as a docs page and in the PR summary, and fails when a marker names an ID that does not exist here. It does not enforce completeness: that is what the **[automated]** and **[manual]** tags are for, and the matrix is read by a person before each delivery. | **[manual]** CI run on the delivery commit; the published matrix reviewed before each delivery |
| NFR-08 | Privacy | No PII column of the problem specification (section 4, excluded columns) appears in the processed data, the prediction log, the comparables or the model artefacts. This also covers the caller's own IP address: neither FR-13's prediction log nor uvicorn's own access log records it, since uvicorn's default access-log format includes the client address unless reconfigured. **[decided]** The raw data is never re-hosted in our own remote; it is imported from its Zenodo DOI ([EDN-07](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)). | **[automated]** Great Expectations suite and tests on the response and log schemas; CI check that the raw file is absent from the DagsHub remote; log format/config check confirming no client-address field is emitted |
| NFR-09 | Security | Request bodies are limited to 10 KB, enforced by middleware that checks `Content-Length` before the body is read, not left to the reverse proxy alone. There is no per-client rate limit: the load test (NFR-03) and the `ES` replay (NFR-11) are our own traffic from a single address, and behind the VM's NAT external clients can share one address too, so a per-IP limit would throttle our own traffic before it stopped anyone else. **[decided, EDN-13]** No TLS, and no requirement on the VM having a certificate: every publicly reachable endpoint is read-only and carries no secret, because `/feedback`, the only endpoint with a key, is reachable only from inside the Compose network (FR-10). The public endpoints are therefore served over plain HTTP, and the FIB Virtech VM needs neither a domain nor a certificate. No secrets are committed; the API key is passed as an environment variable; containers run as a non-root user. Dependencies are locked in `uv.lock` and watched by Dependabot alerts; the built image is scanned with `trivy` before deploy, failing the build on HIGH or CRITICAL findings that have a fix available, so an unfixable base-image CVE does not block every deploy. | **[automated]** API test (body limit); secret scan and `trivy` in CI; Dockerfile review; deployment smoke test asserting `/feedback` is refused from outside the VM (FR-10) |
| NFR-10 | Energy efficiency | CodeCarbon measures the emissions of every training run and logs them to MLflow; training the final chosen configuration (no hyperparameter search) takes at most 15 minutes on a laptop CPU. CodeCarbon also measures the API's energy during the NFR-03 load test, reported as average energy per request, not tracked per individual request: CodeCarbon's measurement granularity does not match single-digit-millisecond events, and a per-request tracker would risk NFR-02's latency budget. | **[manual]** MLflow; energy-per-request figure from the NFR-03 load test |
| NFR-11 | Observability | The drift job flags the `ES` replay as drift within its first 1,000 requests, including at least one feature other than `country_code`. Replaying an equal-sized sample of the held-out test set through the same harness does not flag drift. | **[manual]** M6 replay (`ES`); M6 replay (test-set control, expecting no alarm) |
| NFR-12 | Portability | `docker compose up` starts the whole stack on a fresh VM; no LLM or paid external service is needed at runtime. | **[manual]** Deployment smoke test |
| NFR-13 | CI/CD | Every merge to `main` deploys, unless it touches nothing but `docs/`, `reports/` and the Markdown files at the top level. The trigger is stated as that exclusion, not as a list of paths that deploy: a forgotten path then costs one needless build instead of silently leaving the VM on an older version. This is what connects a model promotion to the running system, since promoting a model is a merge that changes `dvc.lock` and nothing else (FR-12, EDN-08). CD builds the API image (baking in the `dvc pull`-pinned model and comparables index), tags it with the commit SHA and `latest`, pushes both to GHCR, and deploys the SHA tag to the FIB Virtech VM (`docker compose pull && up -d`, with the tag read from a variable in the VM's environment file). A deployment counts as successful only once NFR-12's smoke test passes against it; if it fails, the previous SHA tag is put back, which is the rollback EDN-08 refers to. **[proposed]** The runner reaches the VM over SSH; if the VM turns out not to be reachable from GitHub-hosted runners, the fallback is a self-hosted runner on the VM itself, which needs no inbound connection. Because the repository is public, such a runner may only ever run on `push` to `main`, never on `pull_request`. | **[manual]** CD pipeline run on a merge to `main`; smoke test result; a rollback drill putting the previous SHA tag back (FR-15) |

### Quality model

The `Quality` column maps to [ISO/IEC 25010](https://www.iso.org/standard/78176.html) (software product quality) and [ISO/IEC 25059](https://www.iso.org/standard/80655.html) (its extension for AI systems), except where noted.
Three rows have no equivalent in either standard; they are here because the course grades them as their own practice, not because a quality model names them.

| Quality (this page) | ISO/IEC 25010 / 25059 characteristic |
|----------------------|----------------------------------------|
| Model quality (NFR-01) | AI-specific functional correctness (25059); 25010 covers software only, not model behaviour |
| Latency, Throughput and scalability, Resources (NFR-02 to NFR-04) | Performance efficiency (Time behaviour, Capacity, Resource utilization) |
| Availability (NFR-05) | Reliability (Availability) |
| Reproducibility (NFR-06) | Reliability (Recoverability), extended: not itself a named 25010 sub-characteristic, but standard in ML-specific quality work |
| Maintainability (NFR-07) | Maintainability |
| Privacy (NFR-08) | Security (Confidentiality) |
| Security (NFR-09) | Security |
| Energy efficiency (NFR-10) | Not in 25010/25059; graded as its own M3 practice |
| Observability (NFR-11) | Maintainability (Analysability), extended for MLOps monitoring |
| Portability (NFR-12) | Portability |
| CI/CD (NFR-13) | Not in 25010/25059; graded as its own M5 practice |

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
4. **Performance and resource targets (NFR-02 to NFR-04).**
   - **A (proposed):** commit to the estimated targets now and revisit them after the first M4 load test.
     Gives the load tests and the report a target from the start.
   - **B:** leave the targets open until they are measured.
     Avoids guessing, but leaves M4 without acceptance criteria.
5. **What the VM's network allows (NFR-05, NFR-09, NFR-13).**
   Deliberately left open: nobody has access to the VM yet, so this is decided once someone has tried it, and not before (see the warning at the top of section 2).
   Three questions, to be answered in one go by whoever gets access first:
   - **Reachable from outside at all, on which ports?**
     If yes, Better Uptime can ping `/health` as NFR-05 assumes.
     If no, NFR-05's external check has to come from inside the UPC network or be dropped, and the monitored-uptime part of EDN-10 shrinks to what Grafana sees from inside.
   - **A host name we control enough for a certificate?**
     If yes, TLS becomes cheap and we can add it to NFR-09 even though nothing depends on it any more.
     If no, NFR-09 stays as it is; this is why FR-10 keeps the key inside the Compose network (EDN-13) instead of relying on TLS.
   - **How does CD reach the VM (NFR-13)?**
     **A (proposed):** SSH from the GitHub-hosted runner, which keeps everything in GitHub Actions and needs nothing running on the VM, but only works if the SSH port is reachable from outside the UPC network.
     **B:** a self-hosted runner on the VM that polls GitHub and needs no inbound connection, which works behind NAT and still leaves the deployment record in Actions, but costs RAM on the 4 GB VM and, since the repository is public, may only run on `push` to `main`, never on `pull_request`.

**Decided:** the feedback endpoint and its exposure (FR-10, NFR-09), recorded as [EDN-13](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md).
`/feedback` stays, because the delayed labels are what let FR-14 report a real error and coverage drop instead of input drift alone, and because the `ES` holdout was chosen for exactly that (EDN-03).
It is not exposed publicly, which removes the need for TLS on a VM that probably cannot get a certificate.

**Decided:** raw data hosting (NFR-08), recorded as [EDN-07](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md).
The raw file contains PII (street, zip, coordinates, seller company name for private sellers) and comes from an immutable, versioned Zenodo DOI, so it is pulled with `dvc import-url` and never pushed to our own DagsHub remote, instead of being tracked like a normal pipeline input (see [Data versioning](data-versioning.md)).
