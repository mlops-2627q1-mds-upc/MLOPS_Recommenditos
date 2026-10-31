Architecture
============

This page shows how the price component is built, in five views:

- the use cases: who uses the system for what;
- the logical view: which components are part of the ML lifecycle, which containers there are and how they talk ([C4](https://c4model.com/) container level);
- the class view: the main classes of the serving code;
- the dynamic view: how a request travels through the stack;
- the physical view: where the containers run.

It then names the architectural patterns, what the design leaves out, and the decisions behind both, with their entries in the [Engineering Decision Notebook][edn] (EDN).
What each part must do is in the [specification](specification.md); how to run the API and the stack is in [The API](api.md) and [Deployment](deployment.md).

Every element is in one of two states, and the diagrams draw them differently:

| State | Drawn as | Meaning |
|-------|----------|---------|
| Built | Solid border | On `main` |
| Planned | Grey, dashed border | Milestone 5 or 6, named in the label |

A dashed line is a connection that does not exist yet.

## Use cases

Mermaid has no use-case diagram, so this one is a flowchart drawn as one: the actors stand outside the system boundary, the use cases are the rounded shapes inside it, and a plain line joins an actor to a use case it takes part in.
It shows the three kinds of user the rest of the page serves: the end users of the public API, the team that changes the model and the code, and the monitoring that watches the running service.

```mermaid
flowchart LR
    subgraph u_left[" "]
        direction TB
        u_seller["<b>Seller</b><br/>«actor», end user"]
        u_buyer["<b>Buyer</b><br/>«actor», end user"]
        u_team["<b>Team member</b><br/>«actor»"]
        u_cd["<b>CD</b><br/>«actor», GitHub Actions"]
        u_replay["<b>Replay job</b><br/>«actor», M6"]
        u_monitor["<b>Monitoring</b><br/>«actor», Better Uptime,<br/>Prometheus, drift job, M6"]
    end

    subgraph u_system["Price component"]
        u_uc1(["Value an owned car<br/>UC1, /v1/predict"])
        u_uc2(["Get a price range<br/>UC2<br/>/v1/price-range"])
        u_comp(["Find comparable listings<br/>UC2<br/>/v1/comparables"])
        u_retrain(["Retrain and promote a model<br/>FR-15"])
        u_gate(["Pass the promotion gate<br/>NFR-01"])
        u_deploy(["Deploy or roll back an image<br/>NFR-13"])
        u_feedback(["Report an observed price<br/>/v1/feedback"])
        u_watch(["Check health, scrape metrics<br/>/health, /metrics"])
        u_drift(["Report drift, error, coverage<br/>FR-14, M6"])
    end

    u_seller --- u_uc1
    u_buyer --- u_uc2
    u_buyer --- u_comp
    u_team --- u_retrain
    u_team --- u_deploy
    u_cd --- u_deploy
    u_retrain -.->|"«include»"| u_gate
    u_replay --- u_feedback
    u_monitor --- u_watch
    u_monitor --- u_drift

    classDef later fill:#f2f2f2,stroke:#888,color:#555,stroke-dasharray:6 4
    classDef bare fill:none,stroke:none
    class u_replay,u_monitor,u_drift later
    class u_left bare
```

- **End users.** A seller values a car they own (UC1); a buyer asks what a car with the fields they care about costs and which listings are like it (UC2); both reach the API only through nginx.
- **The team.** Retraining and promotion are started by a person and pass the promotion gate before the merge (FR-15, [EDN-12][edn]); every merge to `main` is deployed by CD, which puts the previous tag back when the new one fails its smoke check (NFR-13).
- **Monitoring and the replay job (M6).** There are no real users who report the price a car sold at, so the replay job sends the `ES` holdout's prices as feedback (NFR-11); monitoring checks `/health` and `/metrics`, and the drift job reports drift, error and interval coverage per window (FR-14).
  The `/v1/feedback` endpoint is built; its only caller today is the integration suite, from inside the Compose network.

## Logical view

### ML and non-ML components

An ML component trains, evaluates, holds, serves or monitors the model, or holds the data the API answers from, and is drawn with a blue fill; the others would look the same in any web service with a database.
The two boxes are the two halves of the ML lifecycle loop: data to the training pipeline, the pipeline to a model bundle and a comparables index, the verdict through the promotion gate and the merge into serving, every answer into the prediction log, the observed prices back as feedback, and drift back to a person who retrains.

```mermaid
flowchart TB
    subgraph l_serve["Serving and monitoring"]
        direction LR
        l_client["<b>HTTP client</b>"]
        l_replay["<b>Replay job</b><br/>M6"]
        l_api["<b>Price API</b><br/>[FastAPI] contract,<br/>validation, errors"]
        l_served["<b>Served model</b><br/>[ServedModel] price,<br/>explanation, range"]
        l_log[("<b>Prediction log, feedback</b><br/>[SQLite]")]
        l_drift["<b>Drift job</b><br/>[Alibi Detect]<br/>M6"]
        l_client -->|"HTTP, via nginx"| l_api
        l_replay -.->|"feedback"| l_api
        l_api -->|"scores with"| l_served
        l_api -->|"one row per answer"| l_log
        l_log -.->|"read, joined<br/>by request ID"| l_drift
    end

    subgraph l_train["Training and promotion"]
        direction TB
        l_pipe["<b>Training pipeline</b><br/>[dvc repro] on the<br/>DVC-tracked listings"]
        l_mlflow[("<b>Experiment tracking</b><br/>[MLflow on DagsHub]")]
        l_bundle[("<b>Model bundle</b><br/>models/lgbm-basic,<br/>pinned by dvc.lock")]
        l_index[("<b>Comparables index</b><br/>[ComparablesIndex] listings,<br/>pinned by dvc.lock")]
        l_gate{"<b>Promotion gate</b><br/>required CI check"}
        l_pipe -->|"params, metrics"| l_mlflow
        l_pipe -->|"writes"| l_bundle
        l_pipe -->|"builds"| l_index
        l_pipe -->|"evaluate: metrics.json, dvc.lock"| l_gate
    end

    l_train -->|"merge to main, CD bakes the bundle and index into the image"| l_serve
    l_serve -.->|"drift flagged: a person starts the Retrain workflow"| l_train

    classDef ml fill:#e3eefa,stroke:#2a6db0,color:#0b2e4f
    classDef mlLater fill:#eef3f9,stroke:#6f93c0,color:#555,stroke-dasharray:6 4
    classDef later fill:#f2f2f2,stroke:#888,color:#555,stroke-dasharray:6 4
    class l_pipe,l_mlflow,l_bundle,l_index,l_gate,l_served ml
    class l_drift mlLater
    class l_replay later
```

- **Model registry.** There is none at run time ([EDN-08][edn]): `dvc.lock` on `main` pins the bundle the image is built with, so merging a pull request that changes it promotes a model, and `/health` reports the bundle's `dvc.lock` hash (FR-12).
- **Comparables index.** The pipeline builds it (stage `comparables`) and pins it in `dvc.lock` beside the bundle, so it counts as an ML component.
  `/v1/comparables` returns its listings of the same make and model within an age and mileage window, closest first (FR-09).
- **Lineage.** `dvc.lock` ties the bundle to the data and code that produced it, and each model variant has one MLflow run across the pipeline's stages ([EDN-55][edn]).
- **Retraining trigger.** The drift job (M6) flags drift; a person decides to retrain and starts the Retrain workflow (`.github/workflows/retrain.yml`) by hand, which runs `dvc repro` on a GitHub-hosted runner, logs to MLflow and pushes `dvc.lock` to a branch; the person opens the pull request from its link, and the gate decides whether it may merge ([EDN-12][edn], [EDN-92][edn]).
- **Inside the API.** Only `ServedModel`, the `LightGBMModel` it holds and the `ComparablesIndex` are ML; the routes, the validation, the error handling, the metrics and the prediction log are ordinary service code, which is why a model can be replaced without changing them.

### Runtime containers

The system has two halves that meet only in the API image: the containers that answer requests, and the pipeline and checks that build and deliver that image.

```mermaid
flowchart TB
    client["<b>HTTP client</b><br/>front end, script, browser"]
    uptime["<b>Better Uptime</b><br/>external check<br/>M6"]

    subgraph stack["Docker Compose stack"]
        proxy["<b>Gateway</b> [nginx]<br/>only published port<br/>403 for /feedback, /metrics"]
        replay["<b>Replay job</b><br/>ES holdout<br/>M6"]
        prom["<b>Prometheus, Grafana</b><br/>M6"]
        api["<b>Price API</b><br/>[FastAPI, uvicorn workers]"]
        drift["<b>Drift job</b><br/>[Alibi Detect, scheduled]<br/>M6"]
        bundle[("<b>Model bundle</b><br/>in the image")]
        comparables[("<b>Comparables index</b><br/>in the image")]
        log[("<b>Prediction log, feedback</b><br/>SQLite, WAL, volume")]
    end

    client -->|"HTTP"| proxy
    uptime -.->|"/health"| proxy
    proxy -->|"other paths"| api
    replay -.->|"/v1/predict, /v1/feedback"| api
    prom -.->|"/metrics"| api
    api -->|"loads at start"| bundle
    api -->|"looks up"| comparables
    api -->|"writes"| log
    drift -.->|"reads, joins"| log

    classDef later fill:#f2f2f2,stroke:#888,color:#555,stroke-dasharray:6 4
    class uptime,replay,prom,drift later
```

### Build and delivery

```mermaid
flowchart TB
    b_dagshub[("<b>DagsHub</b><br/>DVC remote, MLflow")]
    b_dvc["<b>DVC pipeline</b><br/>[dvc repro]"]
    b_ci["<b>CI</b> [GitHub Actions]<br/>tests, Serving runtime,<br/>Promotion gate, Integration"]
    b_laptop["<b>Team laptop</b><br/>make image<br/>fallback"]
    b_cd["<b>CD</b> [GitHub Actions]<br/>build, push, deploy"]
    b_image[("<b>API image</b><br/>runtime, model bundle,<br/>comparables index")]

    b_dvc <-->|"data, models, runs"| b_dagshub
    b_dvc -->|"dvc.lock, metrics.json<br/>in a pull request"| b_ci
    b_ci -->|"merge to main"| b_cd
    b_dvc -->|"models/lgbm-basic"| b_laptop
    b_laptop -->|"docker build"| b_image
    b_cd -->|"dvc pull, build, push to GHCR"| b_image
```

| Container | Responsibility | State, code |
|-----------|----------------|-------------|
| **Gateway** (nginx 1.30, unprivileged) | Publishes the stack's only port; answers `/feedback` and `/metrics` with 403; sets security headers; its own errors are RFC 9457 problems | Built, `deploy/nginx.conf` |
| **Price API** (FastAPI, uvicorn; `WEB_CONCURRENCY` workers) | Validates and prices a request and explains the price (FR-08); `/v1/predict`, `/v1/price-range`, `/v1/comparables`, `/v1/feedback`, `/`, `/health`, `/metrics`, `/docs` | Built, `recommenditos/api/` |
| **Model bundle** (directory in the image) | UC1 and UC2 boosters, interval offsets, feature pipeline; `/health` reports its `dvc.lock` hash (FR-12) | Built, `Dockerfile`, `modeling/model.py` |
| **Comparables index** (`data/processed/comparables.parquet`, copied into the image) | The training and validation listings `/v1/comparables` searches (FR-09) | Built, `recommenditos/data/build_comparables.py` (stage `comparables`), `recommenditos/api/comparables.py` |
| **Prediction log and feedback** (SQLite in WAL mode, named volume) | One row per answer (FR-13) and per observed price (FR-10), joined by request ID (FR-14) | Built, `api/prediction_log.py` |
| **Replay job** | Sends the `ES` holdout as requests and its prices as feedback (NFR-11) | M6 |
| **Drift job** (Alibi Detect, scheduled container) | Input drift, error and interval coverage per window of logged requests (FR-14, EDN-14) | M6 |
| **Prometheus and Grafana** | Scrape `/metrics` inside the network; show latency, errors and resources | M6 |
| **Better Uptime** | Checks `/health` from outside during the presentation windows (NFR-05, proposed) | M6 |
| **DVC pipeline** | Turns the raw listings into the model bundle, its metrics and the gate's verdict | Built, `dvc.yaml`, [pipeline](pipeline.md) |
| **DagsHub** | DVC remote for data and models; hosted MLflow for every run | Built, [data versioning](data-versioning.md) |
| **CI** (GitHub Actions) | On every pull request: lint, tests on Linux and Windows, Serving runtime, Promotion gate, integration tests through nginx | Built, `.github/workflows/ci.yml` |
| **CD** (GitHub Actions, GHCR) | On a merge to `main`, `cd.yml` builds and pushes the image; `deploy.yml` then deploys it to the VM, runs the smoke check and puts the previous tag back if it fails (NFR-13) | Built, `.github/workflows/cd.yml` and `deploy.yml`; deploying to the VM has been on since 2026-10-30 ([Deployment](deployment.md#turn-continuous-deployment-on)) |

The API calls no other service at request time: no DagsHub, no MLflow.
The state it keeps outside the image is the prediction log on its volume and the workers' metric files in a tmpfs, so a worker or the whole container can be restarted or replaced without losing a request's context.

## Class view: the serving code

The main classes of `recommenditos/api/` and the model they load from `recommenditos/modeling/`, with the members the routes use; the ML classes have the blue fill of the logical view.
`FeatureSpace` is defined in `recommenditos/data/build_features.py`, the one class here outside `api/` and `modeling/`.
The routes are functions, not classes, so they are drawn as one box: FastAPI hands each of them the objects the app holds through `Depends` (`api/dependencies.py`).

```mermaid
classDiagram
    direction TB
    class Routes["v1 routes"] {
        <<api/routes>>
        predict(listing) Valuation
        price_range(car) PriceRange
        comparables(car) Comparables
        feedback(body) FeedbackReceipt
    }
    class ServedModel {
        model: LightGBMModel
        info: ModelInfo
        load(directory)$
        faults(body)
        value(request) Estimate
        price_range(request)
    }
    class ModelInfo {
        variant
        version
        trained_at
    }
    class ComparablesIndex {
        listings: DataFrame
        load(path)$
        search(query, k)
    }
    class ApiMetrics {
        observe_request(...)
        observe_valuation(...)
        exposition()
    }
    class PredictionLog {
        record(request, answer)
    }
    class PredictionStore {
        <<interface>>
        record_answer(AnswerRecord)
        record_feedback(Feedback)
        close()
    }
    class SqliteStore {
        path
        max_bytes
        prune() int
    }
    class Model {
        metadata
        predict_eur(frame) Series
        predict_interval_eur(frame)
    }
    class LightGBMModel {
        booster
        intervals
        log_price_contributions(frame)
    }
    class FeatureSpace {
        schema
        load(directory, name)$
    }

    Routes ..> ServedModel : Depends
    Routes ..> ComparablesIndex : Depends
    Routes ..> ApiMetrics : Depends
    Routes ..> PredictionLog : one per request
    Routes ..> PredictionStore : feedback
    ServedModel *-- LightGBMModel
    ServedModel *-- ModelInfo
    LightGBMModel --|> Model
    Model o-- FeatureSpace : space
    PredictionLog --> PredictionStore
    PredictionStore <|.. SqliteStore

    style ServedModel fill:#e3eefa,stroke:#2a6db0,color:#0b2e4f
    style LightGBMModel fill:#e3eefa,stroke:#2a6db0,color:#0b2e4f
    style Model fill:#e3eefa,stroke:#2a6db0,color:#0b2e4f
    style FeatureSpace fill:#e3eefa,stroke:#2a6db0,color:#0b2e4f
    style Routes fill:#ffffff,stroke:#333333,color:#000000
    style ComparablesIndex fill:#e3eefa,stroke:#2a6db0,color:#0b2e4f
    style ApiMetrics fill:#ffffff,stroke:#333333,color:#000000
    style PredictionLog fill:#ffffff,stroke:#333333,color:#000000
    style PredictionStore fill:#ffffff,stroke:#333333,color:#000000
    style SqliteStore fill:#ffffff,stroke:#333333,color:#000000
    style ModelInfo fill:#ffffff,stroke:#333333,color:#000000
```

- **`create_app`** (`api/app.py`) builds the app once per uvicorn worker: it loads the bundle with `ServedModel.load` and the comparables index, keeps them and the `ApiMetrics` in `app.state`, and its lifespan opens one `SqliteStore` per run of the app and closes it at shutdown.
  A bundle that cannot be served, or a log that cannot be opened, fails the startup instead of the first request.
- **`ServedModel`** (`api/serving.py`) is the boundary between the service and the model.
  It calls `load_model` (`modeling/model.py`), which reads `model.json`, loads the `FeatureSpace` the model was trained on, and returns the `Model` subclass the bundle names; `ServedModel` accepts only a `LightGBMModel`, because only its boosters give the SHAP values of FR-08.
  It scores every request of a worker on one thread, in arrival order (issue #179).
- **`PredictionStore`** is the store interface: routes write through `PredictionLog`, which records one answer per request, and `/v1/feedback` calls `record_feedback`; `SqliteStore` is the only implementation ([EDN-86][edn]).
- **`ApiMetrics`** (`api/metrics.py`) holds the Prometheus metrics that `/metrics` exposes (FR-11); `MetricsMiddleware` times every request with it.
- **Left out:** `Settings` (`api/settings.py`, read once by `create_app`), the Pydantic schemas under `api/schemas/`, the four middleware classes (in the dynamic view below), and the `MedianBaselineModel` and `RidgeModel` subclasses of `Model`, which the pipeline trains as baselines and the API refuses.

## Dynamic view: a request through the stack

### Pricing a listing

`POST /v1/predict` is the main path: it shows which component refuses a bad request, where the model runs, and why a client never gets an answer the prediction log does not hold (NFR-03).
`/v1/price-range` and `/v1/comparables` follow the same path with their own route and `ServedModel.price_range` or `ComparablesIndex.search`.

```mermaid
sequenceDiagram
    autonumber
    actor client as HTTP client
    box Compose network on the VM
        participant proxy as nginx
        participant api as API<br/>routes/predict.py
        participant served as ServedModel
        participant model as LightGBMModel
        participant log as PredictionLog,<br/>SqliteStore
    end

    client->>proxy: POST /v1/predict
    proxy->>api: to api:8000
    Note over api: request ID, 10 KB limit (413),<br/>ValuationRequest (422)
    api->>served: faults(listing)
    served-->>api: faults (may be empty)
    Note over api: any fault: 422
    api->>served: value(listing)
    served->>model: predict_eur,<br/>log_price_contributions
    model-->>served: price, SHAP values
    served-->>api: Estimate
    api->>log: record(listing, valuation)
    alt the log cannot write
        log-->>api: StoreError
        api-->>proxy: 503 problem
    else the row is written
        log-->>api: done
        api-->>proxy: 200 Valuation
    end
    proxy-->>client: the API's answer
```

Every error on this path is an RFC 9457 problem that names the request ID ([EDN-84][edn]), and nginx answers 503 itself while the API is down.

### Storing an observed price

`POST /v1/feedback` stores the price a listing was observed at against the answer's request ID, the delayed label FR-14 needs to compute the real error.
nginx refuses it from outside, so it is called only from inside the Compose network and with the API key as a second layer ([EDN-13][edn]).

```mermaid
sequenceDiagram
    autonumber
    actor outsider as Client outside the VM
    box Compose network on the VM
        participant proxy as nginx
        participant replay as Replay job, M6
        participant api as API<br/>routes/feedback.py
        participant store as SqliteStore
    end

    outsider->>proxy: POST /v1/feedback
    proxy-->>outsider: 403 problem
    replay->>api: POST /v1/feedback to api:8000,<br/>X-API-Key, request ID, price
    Note over api: no or a wrong key: 401
    api->>store: record_feedback(Feedback)
    store-->>api: FeedbackResult
    api-->>replay: 201 stored, 200 same price again,<br/>404 unknown request, 409 conflict,<br/>503 log unavailable
    Note over store: the drift job (M6) reads<br/>the view labelled_predictions
```

A 409 means that another price is stored for the answer, or that the answer gave no price, as one of `/v1/comparables` does.
Until the replay job exists, `tests/integration/test_deployed_feedback.py` takes its place: it calls the endpoint from a sibling container with the key and reads the joined row from the log.

## Physical view: deployment

```mermaid
flowchart LR
    d_user["<b>Client</b>"]
    d_uptime["<b>Better Uptime</b><br/>M6"]
    d_laptop["<b>Team laptop</b><br/>fallback"]
    d_ghcr[("<b>GHCR</b>")]

    subgraph d_vm["FIB Virtech VM"]
        direction LR
        d_proxyc["<b>proxy</b><br/>nginx-unprivileged:1.30<br/>listens on 8080"]
        d_apic["<b>api</b><br/>API_IMAGE:sha<br/>user 10001, port 8000"]
        d_m6["<b>prometheus, grafana,<br/>drift job</b><br/>M6"]
        d_tmpfs[("<b>tmpfs</b><br/>/tmp, /tmp/prometheus")]
        d_vol[("<b>volume</b> prediction-log<br/>/var/lib/recommenditos")]
    end

    d_user -->|"HTTP, host port 8080"| d_proxyc
    d_uptime -.->|"/health"| d_proxyc
    d_laptop -->|"docker save, ssh, docker load"| d_apic
    d_ghcr -.->|"deploy.sh pulls, on a merge to main"| d_apic
    d_proxyc -->|"api:8000"| d_apic
    d_m6 -.->|"/metrics"| d_apic
    d_apic --- d_tmpfs
    d_apic --- d_vol
    d_m6 -.->|"reads"| d_vol

    classDef later fill:#f2f2f2,stroke:#888,color:#555,stroke-dasharray:6 4
    class d_uptime,d_m6 later
```

- **The VM.** One FIB Virtech VM for the semester ([EDN-17][edn]); its size, operating system and reachable ports were verified on 2026-10-30 ([The VM](deployment.md#the-vm)).
  The 4 GB of RAM and 20 GB of disk are what the course announced and what NFR-04 budgets for.
- **The stack.** Docker Compose runs the project `recommenditos` from `/opt/recommenditos` with two containers today: `proxy`, the only one with a published port, and `api`, reachable only inside the Compose network ([EDN-87][edn]).
  The host port is `HTTP_PORT` in the stack's `.env`: 8080 on the VM and on a laptop, where `make stack` runs the same stack.
  Both containers restart after a crash and after a reboot, and `proxy` starts once the API's health check passes (NFR-05).
- **Storage.** The workers' metric files live in a tmpfs, the temporary files of SQLite and Python in another one on `/tmp`, and the prediction log on the named volume `prediction-log`, which survives a new image and which the drift job will mount; all are owned by the service user.
  Everything else in the API container is the image, mounted read-only ([Deployment](deployment.md#the-stack)).
- **How the image gets there.** CD builds the image in GitHub Actions and pushes it to GHCR, and `deploy/deploy.sh` on the VM pulls the commit's tag and starts it, so the VM needs neither the repository, DVC nor a DagsHub token ([Deployment](deployment.md#how-the-image-gets-there)).
  The Deploy workflow's jobs on the VM run on a self-hosted runner there, which may run nothing but `deploy.sh` and hands it the job's token for the private package; the smoke check runs from outside ([EDN-94][edn]).
  Rolling back is starting the previous tag, which `deploy.sh` does on its own when a new tag does not come up healthy; a laptop can still stream an image with `docker save | ssh docker load`.

## Patterns

| Pattern | Where | Why | Decision |
|---------|-------|-----|----------|
| Model as a service behind a stateless API | Price API | Clients depend on the HTTP contract only, so a model can be swapped by a merge without changing them, and any worker can answer any request | FR-12, [EDN-08][edn] |
| Model baked into the image at build time | `Dockerfile` copies `models/lgbm-basic` | The image that was tested is the one that runs, startup needs no network, and a rollback is the previous tag | [EDN-08][edn] |
| Reverse proxy as gateway | nginx in front of the API | One published port; nginx refuses `/feedback` and `/metrics` at the edge, so only services inside the Compose network reach them | FR-10, [EDN-13][edn], [EDN-87][edn] |
| Prediction log with a delayed-label feedback loop | API and store built; replay and drift jobs M6 | Prices are observed after the answer, so the real error and the interval coverage can only be computed by joining the observed price to the logged answer by request ID | FR-13, FR-10, FR-14 |
| Repository (store interface) | `PredictionStore` protocol (`record_answer`, `record_feedback`, `close`), implemented by `SqliteStore` | The API writes through these three methods only, so moving to PostgreSQL is a new class, not a change in the routes; the drift job (M6) reads the tables with SQL | [EDN-86][edn] |
| One endpoint per use case | `/v1/predict` (UC1), `/v1/price-range` (UC2) | The two use cases require different fields and return different answers, so each gets its own schema, validation and documentation | [EDN-15][edn] |
| Versioned client contract | `/v1` routers; `/`, `/health`, `/metrics` at the root | An incompatible change can ship as `/v2` beside `/v1`; monitors and Prometheus find the operational endpoints where they look for them | - |
| Training-only libraries kept out of serving | `[project] dependencies` against dependency groups | MLflow, `shap` and MAPIE stay out of the image, which keeps it under NFR-04's 1 GB (702 MB measured for EDN-87); `make test-serving` fails on a serving import from a group | [EDN-47][edn], [EDN-11][edn], [EDN-85][edn] |
| Promotion as a gated merge | Required CI check `Promotion gate` | A pull request that changes `dvc.lock` fails unless its committed verdict passes NFR-01's gate, so promoting a model is a reviewed merge; the SC-04 waiver for `lgbm-basic` is versioned in `params.yaml` | [EDN-12][edn], [EDN-62][edn], [EDN-06][edn] |
| Health-checked containers with a restart policy | Image `HEALTHCHECK`, `restart: unless-stopped` | A crashed or hung API is back within NFR-05's 30 s, and nginx answers 503 meanwhile; for a hang, the healthcheck kills the server behind Docker's init (`init: true`) | [EDN-10][edn], [EDN-93][edn] |

## What the design leaves out

Each of these would add a moving part that no requirement needs yet:

- **A container orchestrator.** The system runs on one VM ([EDN-17][edn]), where Docker Compose restarts and connects the containers ([EDN-87][edn]).
- **A model registry at run time.** The API never asks MLflow for a model; startup would then depend on DagsHub being reachable ([EDN-08][edn]).
- **A database server.** One SQLite file serves the workers and the drift job on one host; PostgreSQL would cost a container, a secret and memory on the 4 GB VM ([EDN-86][edn]).
- **TLS and an automatic-TLS proxy.** Every publicly reachable endpoint is read-only and carries no secret, so NFR-09 needs no TLS, and Caddy's or Traefik's automatic certificates would bring nothing nginx lacks ([EDN-13][edn], [EDN-87][edn]).

## Decisions

The entries in the [EDN][edn] that shape this architecture:

| EDN | Decision |
|-----|----------|
| EDN-06 | A model is deployable only when it meets the success criteria |
| EDN-08 | The model is baked into the API image; promoting a model is a merge to `main` |
| EDN-10 | The availability target is recovery within 30 seconds |
| EDN-11 | Serving explains predictions without the `shap` package |
| EDN-12 | Retraining and promotion are started by a person, and a failing gate blocks the merge |
| EDN-13 | The feedback endpoint is reachable only inside the Compose network |
| EDN-14 | Drift is tested on windows of 1,000 requests at level 0.005 |
| EDN-15 | `/predict` requires ten fields; UC2 has its own endpoint (absorbs EDN-16) |
| EDN-17 | The component is deployed on the FIB Virtech VM |
| EDN-28 | The pipeline code is one module per DVC stage |
| EDN-47 | The serving runtime is separated by dependency groups |
| EDN-55 | Each model variant has one MLflow run across stages |
| EDN-62 | `lgbm-basic` is the candidate model, with the gate's SC-04 waiver |
| EDN-84 | Every API error is an RFC 9457 Problem Details object |
| EDN-85 | Serving adds MAPIE's offsets itself |
| EDN-86 | The prediction log and the feedback are one SQLite file in WAL mode |
| EDN-87 | The API runs on the VM as a Compose stack behind nginx from Milestone 4 |

[edn]: https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md
