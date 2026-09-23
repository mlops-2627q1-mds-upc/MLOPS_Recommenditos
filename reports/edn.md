# Engineering Decision Notebook (working copy)

This is where we record our EDN entries as we make decisions.
Before each delivery, the entries are transferred to the LaTeX EDN in [latex/edn/](latex/edn/), which becomes `MLOps_Recommenditos_EDN.pdf`.

What belongs here and what each field means: [references/Instruction_EDN_MLOps_v2026.md](../references/Instruction_EDN_MLOps_v2026.md).
Record a decision when it meaningfully affects the ML system and could reasonably have gone differently, not every decision or every use of AI.

How to add an entry:

- Copy the template at the end of this file and append the new entry above the template, oldest entry first.
- Give it the next free ID (`EDN-01`, `EDN-02`, ...).
  IDs never change once assigned, because the report refers to them.
- Fill in every field; write "-" for optional fields that do not apply.
- Record the alternatives with their pros and cons even when the choice looked obvious.
  They are the evidence that the decision was actually weighed.
- Review AI conversation excerpts before quoting or linking them; never include personal or sensitive data.
- Set **In LaTeX** to "yes" once the entry has been transferred to `latex/edn/`.

## Entries

### EDN-01: Main dataset choice: AutoScout24 Car Listings

- **Date:** 2026-09-22
- **Milestone:** M1: Project Inception
- **Activity / Topic:** Dataset Selection
- **Participants:** All team members
- **Decision:** Use the AutoScout24 Car Listings Dataset (2025 snapshot, Zenodo DOI `10.5281/zenodo.17643343`) as the main training dataset for the used-car price component.
- **Alternatives considered:**
  - **Option A (chosen): AutoScout24 Car Listings Dataset.** 118,382 listings, 75 columns, 8 countries, MIT license.
    Pros: real scraped listings, large enough for gradient boosting, rich feature set (equipment, history flags, free-text description), verifiable provenance and license.
    Cons: skewed toward premium brands (BMW, Porsche, Mercedes-Benz, Audi are 83 % of rows), mass-market brands nearly absent, single scrape with no listing date so no temporal split.
  - **Option B: smaller generic Kaggle used-car datasets.** Various single-market, pre-cleaned listing datasets evaluated per issue #3's scope.
    Pros: often smaller and pre-cleaned, lower profiling effort.
    Cons: typically single-country, far fewer rows and features than AutoScout24, weaker or undocumented provenance/license.
- **Rationale:** AutoScout24 is the only candidate large and feature-rich enough to support the planned gradient-boosting model with equipment/history features and SHAP explainability, and it has a clear, citable license. The brand skew is accepted as a stated scope limit (see project brief §2) rather than a reason to switch datasets.
- **AI involvement:** Information seeking, Alternative assessment
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI profiled the raw dataset (row/column counts, country and brand distribution, missingness, known data issues) and summarized the trade-off against smaller generic alternatives; the team reviewed those facts and confirmed the choice in today's sprint planning.
- **AI interaction evidence:** Dataset profiling and trade-off summary in `docs/docs/project-brief.md` §3.1 and §2 (2026-09-22), confirmed in today's sprint planning session.
- **Other evidence:** [PR #13](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/13), [issue #3](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/3).
- **In LaTeX:** no

### EDN-02: Model family: gradient boosting (LightGBM as main model)

- **Date:** 2026-09-22
- **Milestone:** M1: Project Inception
- **Activity / Topic:** Modelling Approach
- **Participants:** All team members
- **Decision:** Use gradient boosting as the model family, with LightGBM as the main model (basic features, then all features) and CatBoost as a challenger for high-cardinality categoricals.
- **Alternatives considered:**
  - **Option A (chosen): Gradient boosting (LightGBM main, CatBoost challenger).**
    Pros: best-in-class on medium tabular data, native categoricals and missing values, trains in minutes on CPU, small artefacts, exact SHAP explanations.
    Cons: more tuning surface than a linear baseline; still needs a simpler baseline (median, Ridge) to prove its value.
  - **Option B: Linear / Ridge regression only.**
    Pros: simple, fast, highly interpretable depreciation baseline.
    Cons: cannot capture non-linear interactions (brand x age x mileage) well, lower expected accuracy.
  - **Option C: kNN, random forest, tabular NNs, hierarchical Bayes, LLM zero-shot.**
    Pros: each offers some property (kNN gives comparables for free, NNs could use text embeddings, LLM needs no training).
    Cons: kNN and random forest scale poorly or lag gradient boosting on this data size; tabular NNs need more data/tuning and lose native categorical handling; hierarchical Bayes is slow to fit at this scale; LLM zero-shot has no calibrated uncertainty and conflicts with the CPU-only, small-image constraint.
- **Rationale:** The course grades MLOps practice over raw accuracy, so the team wants a model that trains fast on CPU, handles categoricals/missing values natively, keeps artefacts small, and supports exact SHAP explanations for UC1/UC2 - gradient boosting (LightGBM) is the best fit, with a median and Ridge baseline kept in the experiment ladder to prove it earns the added complexity.
- **AI involvement:** Information seeking, Alternative assessment
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI laid out the alternatives and the CPU/size/explainability constraints behind the "why gradient boosting" rationale in the project brief; the team reviewed and confirmed the choice, including keeping CatBoost as a challenger rather than the main model, in today's sprint planning.
- **AI interaction evidence:** Modelling plan and "Why gradient boosting" write-up in `docs/docs/project-brief.md` §4 (2026-09-22), confirmed in today's sprint planning session.
- **Other evidence:** [PR #13](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/13).
- **In LaTeX:** no

### EDN-03: New-market drift scenario: hold out AutoScout24 Spain instead of using DataMarket

- **Date:** 2026-09-22
- **Milestone:** M1: Project Inception (data plan; shapes M6: Monitoring)
- **Activity / Topic:** Dataset Selection, Monitoring
- **Participants:** lukas2510, after supervisor feedback (S. del Rey)
- **Decision:** Hold out all AutoScout24 `ES` listings (8,015, 6.8 %) from training, validation and calibration, and use them as the "new market" drift scenario for M6. The DataMarket Spanish second-hand car sample is dropped from the pipeline.
- **Alternatives considered:**
  - **Option A (chosen): Hold out one AutoScout24 country (`ES`).**
    Pros: same schema, scrape date and portal as the training data, so detected drift comes from the market alone; every held-out listing has a price, so we can show real MAE and interval-coverage degradation, not only input drift; the same prices serve as delayed labels for the feedback loop; no mapping stage and no license issue; `ES` costs only 6.8 % of training rows (holding out `IT` would cost 20 %).
    Cons: less training data; the served model does not cover Spain until retrained; `country` becomes a category unseen in training, which the API must handle.
  - **Option B: Keep DataMarket as an optional stress test, restricted to makes both datasets share.**
    Pros: real external data, from a different portal; answers the supervisor's question with an actual experiment.
    Cons: market, time (2021 vs 2025), portal and brand mix still shift together, so a drift alarm cannot be attributed; needs a schema-mapping stage for ~10 overlapping fields with Spanish labels and free-text colour; no license file and a commercial sample, so it cannot live in a public DVC remote.
  - **Option C: Drop DataMarket without a replacement.**
    Pros: simplest.
    Cons: leaves M6 without a realistic drift scenario.
- **Rationale:** The supervisor asked us to check whether DataMarket is feasible as a new-market set given the feature mismatch. Our checks found that Spain is already in AutoScout24 (8,015 `ES` listings), 36.5 % of DataMarket rows are makes the model never sees and the shared mass-market makes have under 400 training rows each, and several factors shift at once (e.g. BMW median price 16,000 EUR in DataMarket vs 24,819 EUR in AutoScout24 `ES`). A drift alarm on DataMarket would therefore not be explainable. The M6 rubric asks for "timely, actionable signals about data changes, model performance degradation", and holding out `ES` gives a drift with a single known cause plus ground-truth prices to show the performance drop, at the cost of one filter in the split stage.
- **AI involvement:** Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** The three options came from our own feasibility check. AI (Claude Code) explained the trade-offs against the M6 rubric and recommended Option A with `ES` as the held-out country (smaller training loss than `IT`), keeping `country` as a feature with unseen countries treated as unknown, and an optional synthetic drift scenario with a controlled cause. We accepted it because it answers the supervisor's concern with evidence, keeps the pipeline to one data source, and lets M6 show both input drift and performance degradation.
- **AI interaction evidence:** Claude Code session, 2026-09-22: prompt "explain this to me again and give me a recommendation [...] what is feasible for our project and still in the scope of the course" on project brief §3.2; the response compared options A-C against the M6 rubric and recommended A with `ES`.
- **Other evidence:** [PR #13](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/13), project brief §3.2 (`docs/docs/project-brief.md`).
### EDN-04: Model scope: used passenger cars only

- **Date:** 2026-09-22
- **Milestone:** M1: Project Inception
- **Activity / Topic:** Problem Specification
- **Participants:** @lukas2510
- **Decision:** The model only covers used passenger cars: `offer_type = U`, not pre-registered, `vehicle_type = Car`. New cars, pre-registered cars and transporters are filtered out in the pipeline and rejected by the API.
- **Alternatives considered:**
  - **Option A (chosen): used cars only.** Drops about 5 % of the cleaned listings (4,252 new, 3,702 pre-registered and 456 transporter listings in the raw data).
    Pros: matches the product ("used-car price"); new cars follow list prices, not a second-hand market, so one model does not have to learn two price mechanisms; a clear, testable scope statement.
    Cons: slightly less data; the API has to reject new cars.
  - **Option B: keep everything, with the flags as features.**
    Pros: more data, broader API.
    Cons: blurs the product and mixes two price mechanisms in one model.
- **Rationale:** An exploratory LightGBM run showed almost no accuracy difference between the two options (MdAPE 6.8 % vs 6.9 %), so the decision rests on product clarity: a used-car price component should be scoped to used cars.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI profiled the data (counts of new, pre-registered and transporter listings), ran the comparison on both scopes and recommended option A. Lukas checked that the numbers came from the real dataset and accepted the recommendation because the accuracy argument was neutral and the product argument decisive.
- **AI interaction evidence:** Claude Code session on 2026-09-22 while working on issue #2: prompt "Which listings are in scope for the model (car type)?" with options A and B and the exploratory results; Lukas chose option A.
- **Other evidence:** [Problem specification](../docs/docs/problem-spec.md), section 2; [issue #2](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/2), [PR #17](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/17).
- **In LaTeX:** no

### EDN-05: Supported makes: minimum listing support per make

- **Date:** 2026-09-22
- **Milestone:** M1: Project Inception
- **Activity / Topic:** Problem Specification
- **Participants:** @lukas2510
- **Decision:** Only makes with at least 300 listings in the cleaned used-car data (counted after the `ES` holdout, before the split) are supported; the pipeline computes the list and the API rejects other makes. With the current data this gives 11 makes covering 98.6 % of the used listings after the `ES` holdout (EDN-03).
- **Alternatives considered:**
  - **Option A (chosen): minimum support threshold, reject the rest.**
    Pros: honest and testable scope; no silent low-quality predictions; easy to state in the model card.
    Cons: smaller product (e.g. Ford, Renault, Opel, Kia are not supported).
  - **Option B: accept all 25 makes and flag rare ones as low confidence.**
    Pros: broader API.
    Cons: weak predictions for makes with a few dozen listings; the warning logic needs its own tests and thresholds anyway.
  - **Option C: accept all makes without any rule.**
    Pros: simplest.
    Cons: no clear scope statement; hidden weaknesses for rare makes.
- **Rationale:** The dataset is skewed toward premium brands (EDN-01 accepts this as a scope limit). A threshold turns that limit into an explicit, testable rule. 300 keeps 11 makes and 98.5 % of the data (98.6 % after the `ES` holdout); 200 would add three makes (0.7 % of the data), 500 would drop Aston Martin and Volkswagen.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI counted the listings per make after scoping and deduplication, compared thresholds of 200, 300 and 500, and recommended option A. Lukas accepted it because it states the dataset's brand skew openly instead of hiding it behind a warning.
- **AI interaction evidence:** Claude Code session on 2026-09-22 while working on issue #2: prompt "How do we handle the 25 makes, many of which have very few listings?" with options A to C; Lukas chose option A.
- **Other evidence:** [Problem specification](../docs/docs/problem-spec.md), section 2; [issue #2](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/2), [PR #17](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/17).
- **In LaTeX:** no

### EDN-06: Success criteria for the price model

- **Date:** 2026-09-22
- **Milestone:** M1: Project Inception
- **Activity / Topic:** Performance Criteria
- **Participants:** @lukas2510
- **Decision:** A model is deployable when it meets SC-01 to SC-05 on the grouped test set: MdAPE ≤ 9 %; at least 85 % of predictions within ±20 %; MdAPE at least 30 % lower than the median baseline B0; MdAPE ≤ 15 % in every input-based segment with at least 500 test rows; empirical coverage of nominal 90 % intervals between 88 % and 92 % for full inputs and for a partial-input scenario.
- **Alternatives considered:**
  - **Option A (chosen): absolute targets plus a baseline comparison, per segment.**
    Pros: says whether the product is useful to users and whether the model earns its complexity; segment criterion catches models that are good on average but bad for a group; grounded in an exploratory run.
    Cons: targets depend on this dataset and must be revisited if the scope changes.
  - **Option B: baseline comparison only.**
    Pros: independent of the data.
    Cons: says nothing about usefulness; a weak baseline makes it easy to pass.
  - **Option C: MAPE-based targets, as in the original plan.**
    Pros: common and familiar.
    Cons: dominated by cheap cars and outliers (the same model has 9.7 % MAPE but 6.8 % MdAPE).
- **Rationale:** An exploratory run on the final scope (102,695 listings, 80/20 split grouped by seller) gave B0 11.6 % MdAPE / 71.4 % within ±20 % and an untuned LightGBM 6.8 % / 90.7 %. The targets keep a margin to the LightGBM values so they hold across splits, while ruling out a model only marginally better than B0. Price buckets are excluded from the segment criterion because they condition on the target. The run already shows one gap: cars older than 20 years reach 16.7 % MdAPE, which is recorded as a known risk rather than hidden by loosening SC-04. Re-checked after the `ES` holdout (EDN-03) on 96,831 listings: B0 11.9 % / 70.9 %, LightGBM 6.7 % / 90.6 %, cars older than 20 years 15.3 %; the criteria stay unchanged.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI ran the exploratory baseline and LightGBM models, proposed the criteria with concrete thresholds and recommended option A. Lukas accepted it because the thresholds are tied to measured values. After the decision, AI's per-segment check found that the proposed SC-04 is not yet met for cars older than 20 years; the criterion was kept as agreed and the gap documented.
- **AI interaction evidence:** Claude Code session on 2026-09-22 while working on issue #2: prompt "Which kind of success criteria?" with options A to C and the exploratory results; Lukas chose option A.
- **Other evidence:** [Problem specification](../docs/docs/problem-spec.md), sections 6 to 8; [issue #2](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/2), [PR #17](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/17).
- **In LaTeX:** no

### EDN-07: Raw data hosting: import from Zenodo, never push to our own remote

- **Date:** 2026-09-22
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Data Versioning, Privacy
- **Participants:** @lukas2510
- **Decision:** The raw AutoScout24 CSV is pulled with `dvc import-url` from its Zenodo DOI, pinned to version 1.0.0, and is never `dvc push`ed to our DagsHub remote. Only the PII-stripped `data/interim` output onward is tracked and pushed like a normal pipeline artefact. Enforced by a CI check that the raw file's hash is absent from the DagsHub remote.
- **Alternatives considered:**
  - **Option A (chosen): `dvc import-url` from Zenodo, raw file never pushed.**
    Pros: the raw file carries PII (`street`, `zip`, exact coordinates, `seller_company_name` for private sellers); the Zenodo DOI is already immutable and versioned, so re-fetching it on `dvc repro` gives the same reproducibility as a cached copy without a second, PII-bearing copy on our own remote (likely public); matches the data-minimization principle for sensitive raw data.
    Cons: `dvc repro` needs live access to Zenodo; reproducibility depends on Zenodo keeping this exact file available; `dvc import-url` still caches the file locally, so the "never pushed" half needs an explicit CI check, not just the command choice, to actually hold.
  - **Option B: `dvc add` the raw file like any other pipeline input and push it to DagsHub.**
    Pros: consistent with how every other artefact in the project is tracked; does not depend on Zenodo staying reachable.
    Cons: re-hosts a PII-bearing file on our own remote, which is likely public; the second copy adds no reproducibility over the immutable Zenodo DOI, only exposure.
- **Rationale:** Option A was already the direction noted (but left `[open]`) in the project brief's data-issues section. It was surfaced as an undecided requirement-vs-brief contradiction during the review of PR #19 ("requirements.md" stated the raw data is never re-hosted as a settled `NFR-08`, while the brief still called the same question open, and the existing `data-versioning.md` example showed the opposite pattern, `dvc add data/raw/cars.csv`). Data minimization for a file containing indirect identifiers of private sellers outweighs the convenience of a uniform tracking pattern; the Zenodo DOI's own immutability and versioning already give the reproducibility that pushing a copy would otherwise provide.
- **AI involvement:** Information seeking, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** While reviewing PR #19 for SOTA fit and course requirements, AI noticed that `NFR-08`'s "never re-hosted" wording silently settled the brief's open `dvc import-url` question without going through the EDN process, and that the project's own `data-versioning.md` example contradicted it. Asked for a recommendation, AI laid out options A and B with pros and cons, recommended A on privacy and data-minimization grounds, and flagged that `dvc import-url` alone does not prevent a re-push and needs a CI check to be enforceable. Lukas reviewed the reasoning and confirmed option A.
- **AI interaction evidence:** Claude Code session on 2026-09-22 while reviewing PR #19: AI flagged the NFR-08/brief contradiction, explained it in detail on request, then gave a recommendation and cited data-minimization and Zenodo DOI immutability as the SOTA rationale for option A; Lukas replied "yes please do so".
- **Other evidence:** [Requirements](../docs/docs/requirements.md) NFR-08 and "Decisions pending confirmation"; [Data versioning](../docs/docs/data-versioning.md); [project brief](../docs/docs/project-brief.md) section 3.3; [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** no

### EDN-08: Model loading: bake into the API image via `dvc pull` at CI build time, not the MLflow registry at runtime

- **Date:** 2026-09-22
- **Milestone:** M4: Model Deployment
- **Activity / Topic:** ML System Design
- **Participants:** @lukas2510
- **Decision:** The API image is built by CI with the current production model already baked in, fetched with `dvc pull` pinned to whatever `dvc.lock` on `main` points to. Promoting a model is a normal merge to `main` that updates that pointer. The API never calls the MLflow registry at build or run time; MLflow remains the experiment-tracking and audit record of which run was chosen as production.
- **Alternatives considered:**
  - **Option A: pull from the MLflow model registry at API startup.** As originally stated in requirements.md FR-12.
    Pros: matches the target architecture's "MLflow tracking + model registry" component as originally drafted; a model can be promoted without a redeploy.
    Cons: startup now depends on DagsHub being reachable, which works against NFR-05 (ready within 30 s) and NFR-12 (a self-contained stack); a DagsHub outage right before a presentation could break the whole demo.
  - **Option B: CD resolves the MLflow registry alias and bakes the model into the image on promotion.**
    Pros: avoids the runtime dependency of option A while still using the registry as the deployment-time source of truth.
    Cons: needs a new trigger connecting an MLflow promotion event to a GitHub Actions build (webhook or poller), which is infrastructure the course does not demo yet and is disproportionate for a 5-person, one-semester, single-VM project; nothing watches it if it silently stops firing.
  - **Option C (chosen): bake into the image via `dvc pull` at CI build time; promotion is a merge to `main`.**
    Pros: matches the course demo's own deployment recipe (`dvc pull` the model before running `uvicorn`), just containerized; reuses GitHub Flow, already the project's branching model, as the promotion mechanism with no new trigger to build; leans on DVC, the single heaviest-weighted practice in the course rubric (15/100), for the thing that most needs to be reproducible; avoids the runtime dependency of option A.
    Cons: MLflow's model registry is no longer part of the deployment path, only tracking/audit, which is a smaller role than the target architecture originally sketched.
- **Rationale:** Given the course's own demo pattern, the rubric's weight on DVC over MLflow, and the team's size and one-semester timeline, option C gives the same startup-independence as option B without inventing new promotion infrastructure. MLflow keeps its full credit for the experiment-tracking practice; it is just not on the deployment-time critical path.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** While reviewing PR #19, AI first flagged that FR-12's runtime registry pull (option A) worked against NFR-05 and NFR-12 and recommended option B. Asked to weigh that against the project's actual scope and tools, AI checked `references/course-demos.md`, found the demo's own recipe pulls the model via DVC rather than MLflow, cross-referenced the rubric's point weights, and revised its recommendation to option C, naming the trade-off (MLflow registry no longer in the deployment path) explicitly. Lukas reviewed the reasoning and confirmed option C.
- **AI interaction evidence:** Claude Code session on 2026-09-22 while reviewing PR #19: AI recommended option B, Lukas asked "considering our project scope and the tools we will use is this the best option for us?", AI re-derived the recommendation from `references/course-demos.md` and the rubric weights and proposed option C instead; Lukas replied "yes please do that".
- **Other evidence:** [Requirements](../docs/docs/requirements.md) FR-12; [project brief](../docs/docs/project-brief.md) section 5 (target architecture); [references/course-demos.md](../references/course-demos.md) "API and deployment (M4)"; [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** no

### EDN-09: Bump to Python 3.12, to use real SHAP instead of a workaround

- **Date:** 2026-09-22
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Tooling, Explainability
- **Participants:** @lukas2510
- **Decision:** `requires-python` is `~=3.12.0` (was `~=3.11.0`). `pyproject.toml`, `uv.lock`, and the Python version in `.github/workflows/ci.yml` are updated. FR-08's explanation endpoint uses `shap.TreeExplainer` directly.
- **Alternatives considered:**
  - **Option A: use LightGBM's native `predict(pred_contrib=True)` instead of the `shap` package.** Gives the same exact TreeSHAP values without touching `shap` or its Python 3.12 requirement, and avoids `numba`/`llvmlite` in the API image.
    Pros: no project-wide version bump needed this early in the semester; smaller API image.
    Cons: reimplements what `shap` already provides; still needs the real `shap` package for any offline/notebook analysis (global importance, summary plots), so the Python constraint would resurface there anyway; more code to maintain for a workaround to a constraint that turned out to be removable.
  - **Option B (chosen): bump to Python 3.12 project-wide.** Verified empirically: `uv lock --python 3.12` and `uv sync --python 3.12` resolved and installed cleanly with the full planned M2-M5 dependency set added (`shap`, `mlflow>=2.22,<3`, `lightgbm`, `catboost`, `mapie`, `fastapi`, `uvicorn`, `pydantic`, `great-expectations`, `dvc`, `pytest-cov`, `codecarbon`, `httpx`), and `shap.TreeExplainer` on a trained LightGBM model produced correct SHAP values under 3.12 in a functional test.
    Pros: uses `shap` directly with no workaround; every other planned dependency (checked against PyPI's `requires_python` metadata for the exact pinned versions, not just "latest") only has a lower bound compatible with 3.12, so nothing else in the stack is put at risk; no Dockerfile exists yet to update, only three lines in `ci.yml`.
    Cons: a project-wide version bump this early, touching `pyproject.toml`, `uv.lock` and CI, for the sake of one requirement.
- **Rationale:** The constraint was checked empirically rather than assumed from documentation: a scratch resolve of the full planned dependency set under Python 3.12 succeeded with no conflicts, and `shap.TreeExplainer` functionally worked. With no blocker found, option B removes the constraint at its source instead of working around it, and the change is small (no Dockerfile yet, three `ci.yml` lines, `pyproject.toml`, `uv.lock`).
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** While reviewing PR #19, AI proposed option A as a way around the Python 3.11/SHAP 0.52 conflict documented in the brief. Asked to check whether the version could be bumped instead, AI verified this empirically (a scratch `uv lock`/`uv sync` under 3.12 with the full planned dependency set, plus a functional `TreeExplainer` test) rather than relying on PyPI metadata alone, found no blocker, and presented both options with the empirical evidence. Lukas reviewed it and chose to bump.
- **AI interaction evidence:** Claude Code session on 2026-09-22 while reviewing PR #19: AI first recommended `pred_contrib` as a workaround, Lukas asked "please check if we can bump our python version," AI ran an empirical `uv lock`/`uv sync`/import/functional check under Python 3.12 with the full planned stack, found no blocker, and reported the evidence; Lukas replied "ok please then bump the version in the PR an adjust the requriemnt."
- **Other evidence:** [Requirements](../docs/docs/requirements.md) FR-08; [project brief](../docs/docs/project-brief.md) section 6 (tooling constraints); `pyproject.toml`, `.github/workflows/ci.yml`; [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** no

### EDN-10: Availability target replaced by recovery time plus a presentation-window commitment

- **Date:** 2026-09-22
- **Milestone:** M6: Monitoring
- **Activity / Topic:** Resource Monitoring, ML System Design
- **Participants:** @lukas2510
- **Decision:** NFR-05 no longer commits to "99 % uptime in the weeks before each presentation." It now commits to: (1) automatic recovery, the API reaches a healthy `/health` within 30 s of a crash or VM reboot, verified by a chaos test; (2) zero unplanned downtime during each live presentation and its warm-up, checked manually right before presenting; (3) outside those windows, uptime on the single, unattended, no-SLA VM is monitored and reported via Better Uptime and Grafana, not contractually targeted. Better Uptime is added to the architecture, the M6 tool list and the reference links, none of which named it before.
- **Alternatives considered:**
  - **Option A: keep 99 % uptime over "the weeks before each presentation."**
    Pros: a single, simple number; easy to write into the report.
    Cons: nothing backs it: one VM, no redundancy, no SLA from FIB Virtech, no on-call team; the only listed check (Prometheus `up`) can't see the host itself going down; the measurement window is unbounded and unverifiable after the fact, so the number could not actually be defended if a teacher asked how it was checked.
  - **Option B: lower the aggregate target (e.g. 95 %) instead of dropping it.**
    Pros: keeps a single reportable number; slightly more honest than 99 %.
    Cons: still as unfalsifiable as 99 % without redundancy or an on-call team; still hides that the requirement was actually about two specific presentation days, not a general SLA the team can neither guarantee nor is expected to for a course project.
  - **Option C (chosen): split into recovery time, a narrow presentation-window commitment, and a monitored-not-targeted general period.**
    Pros: the recovery-time claim is testable today with a chaos test, not only after weeks of passive observation; the presentation-window claim is the one that actually matters for grading and is narrow enough for the team to actively watch; the monitored-general-period framing is honest about the single-VM, no-redundancy, no-SLA deployment while still giving the M6 monitoring practice something to show (Better Uptime + Grafana), which is what the rubric actually grades, not a specific SLA number.
    Cons: three clauses instead of one number, slightly longer to state.
- **Rationale:** M6's Resource Monitoring criterion asks whether the monitoring anticipates real-world challenges and gives actionable insight, not whether a specific SLA percentage was hit. A number the team has no redundancy or on-call capacity to guarantee is a liability if questioned in the presentation, not evidence of the practice. Splitting the requirement matches what is actually controllable (recovery time, presentation-day behaviour) from what can only be observed and reported (general uptime on a single, unattended, no-SLA VM), and it adds Better Uptime, a tool the course's own M6 schedule already names but the docs previously omitted.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** During the PR #19 review, AI had already flagged that Prometheus's `up` metric cannot observe the host going down and that Better Uptime was missing from the docs despite being a named M6 course tool. Asked directly what availability target would actually be feasible given the single-VM, no-redundancy deployment, AI proposed the three-part split (recovery time, presentation window, monitored general period) with rationale grounded in the M6 rubric wording and the deployment's real constraints, and named option B as a weaker alternative. Lukas reviewed it and confirmed the split.
- **AI interaction evidence:** Claude Code session on 2026-09-22 while reviewing PR #19: Lukas said "i dont like that [99% uptime]. what NFR that goes in the uptime direction is feasible with our system and constraints?"; AI proposed the three-part split and the addition of Better Uptime; Lukas replied "ok do that."
- **Other evidence:** [Requirements](../docs/docs/requirements.md) NFR-05; [project brief](../docs/docs/project-brief.md) section 5 (target architecture) and section 7 (milestones); [MLOps-lab.md](../references/MLOps-lab.md) session 11; [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** no

### EDN-11: Keep `shap` out of the API image; serving uses the booster's native SHAP export

- **Date:** 2026-09-22
- **Milestone:** M4: Model Deployment
- **Activity / Topic:** ML System Design, Explainability
- **Participants:** @lukas2510
- **Decision:** Amends EDN-09's serving-path clause. FR-08's per-request explanation is computed from the trained booster's own SHAP export (LightGBM `predict(pred_contrib=True)`, or CatBoost `get_feature_importance(type="ShapValues")` if CatBoost is ever the deployed model), not the `shap` package. `shap` stays a training/notebook-only dependency, used for offline global analysis (summary plots, dataset-level importance), and is never installed in the API image. The Python 3.12 bump from EDN-09 is unaffected and still stands, since it is what lets `shap` run at all for that offline use.
- **Alternatives considered:**
  - **Option A: keep `shap.TreeExplainer` in the serving path, as EDN-09 originally set FR-08 to.**
    Pros: one implementation for both the offline/notebook analysis and the serving path, nothing to keep in sync.
    Cons: `shap` unconditionally imports `numba` and `llvmlite` at module load, even to use only `TreeExplainer`, confirmed by deleting them and watching a bare `import shap` fail; a real Docker build of the serving stack (`shap`, `lightgbm`, `mapie`, `fastapi`, `uvicorn`, `pydantic`, `numpy`, `pandas`) measured 702 MB, leaving only about 30 % of NFR-04's 1 GB budget for the baked-in model artefact, app code and everything else still to be added.
  - **Option B (chosen): compute serving-time explanations from the booster's native SHAP export; keep `shap` for offline analysis only.**
    Pros: the same Docker build without `shap` measured 486 MB, about 51 % headroom instead of 30 %; the native export is verified bit-identical to `shap.TreeExplainer` (max absolute difference 0.0 across all features and the base value in a direct comparison); the pattern generalises to CatBoost, the brief's stated challenger model (EDN-02), via its own native SHAP support, so it does not need revisiting if the challenger wins; `shap` is still fully available where it is actually needed, offline global analysis, unconstrained by image size.
    Cons: two call sites compute the same values (the booster's native export in the API, `shap.TreeExplainer` in training/notebook code) instead of one; a training-time test is needed to keep them provably in sync (added to FR-08's "Verified by").
- **Rationale:** The image-size argument is independent of the Python-version problem EDN-09 solved, and EDN-09 folded both into one decision without weighing the size evidence, which was not yet measured at the time. Measured directly (Docker builds, a deletion test proving `numba`/`llvmlite` are unconditional runtime imports, and a numeric comparison proving the native export and `shap.TreeExplainer` produce identical values), option B costs nothing in correctness and gives back real budget headroom under NFR-04, while also removing a dependency choice tied to a single model family.
- **AI involvement:** Information seeking, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI had proposed the native-export approach earlier in the PR #19 review, before EDN-09 was decided. Asked whether it was still relevant after the Python 3.12 bump, AI separated the two motivations (version conflict, now resolved; image size, still open), then built real Docker images for both options and a deletion test to confirm `numba`/`llvmlite` are unconditional, rather than assuming it from documentation. Asked whether this was the best option or if there were better ones, AI additionally ruled out a leaner Docker build (does not work, proven by the deletion test) and dropping per-instance SHAP entirely (would weaken FR-08 and lose the course's stated M3 credit for using SHAP), and pointed out CatBoost's equivalent native support. Lukas reviewed the evidence and confirmed option B.
- **AI interaction evidence:** Claude Code session on 2026-09-22 while reviewing PR #19: AI's original `pred_contrib` suggestion predated the Python 3.12 bump; Lukas asked "is this still relevant?", AI re-derived it with concrete measurements; Lukas asked "is this what we would want? or are there better options?", AI built real Docker images (702 MB vs. 486 MB) and ruled out the alternatives; Lukas replied "ok apply it."
- **Other evidence:** [Requirements](../docs/docs/requirements.md) FR-08; [project brief](../docs/docs/project-brief.md) section 6 (tooling constraints); [EDN-09](#edn-09-bump-to-python-312-to-use-real-shap-instead-of-a-workaround); [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** no

### EDN-12: Retraining and promotion are human-triggered, not automated

- **Date:** 2026-09-22
- **Milestone:** M6: Monitoring
- **Activity / Topic:** ML System Design, Model Performance
- **Participants:** @lukas2510
- **Decision:** FR-15 formalises the retrain-after-drift loop that EDN-03 was designed to demonstrate: when FR-14's drift job flags input drift on the `ES` replay (per NFR-11), a team member retrains with `ES` included and opens the promotion PR by hand. No automation watches the drift job and triggers this on its own. The candidate still must pass NFR-01's gate before the PR is merged; merging is the promotion (EDN-08), and rollback is redeploying the previous image tag (EDN-08).
- **Alternatives considered:**
  - **Option A: fully automated retraining and promotion.** A watcher on the drift job's output triggers `dvc repro` and opens the promotion PR automatically when drift is confirmed.
    Pros: reads as more "extended/innovative" for the M6 System Design criterion; no manual step to forget.
    Cons: needs new infrastructure (something connecting Alibi Detect's output to a CI trigger) built and debugged in the last two weeks before the Dec 9 presentation, exactly when there is the least slack to recover if it breaks; higher risk for a 5-person team with no prior automation of this kind in the project.
  - **Option B (chosen): human-triggered retraining and promotion.** The drift job surfaces the signal (Grafana); a team member runs the retrain and opens the PR once NFR-01's gate passes.
    Pros: still fully closes the loop EDN-03 was built to demonstrate, and still satisfies the M6 rubric's "enabling confident investigation and response", a human response is a legitimate response, not a lesser one; far less new infrastructure, and safer to demo live; matches the team's size and the amount of time left before M6.
    Cons: less visually impressive than full automation; a person has to actually notice the signal and act on it.
- **Rationale:** The `ES` holdout exists specifically to demonstrate this loop closing (EDN-03), so leaving it as an unformalised sentence in the brief risked it quietly not getting built once M6 crunch hits. Given the timeline, automating the trigger is a real risk for a low grading upside over the human-triggered version, which already gets full credit for "closing the loop" and "responding to a signal" without the extra failure surface this late in the semester.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** While reviewing PR #19, AI noted that the retrain-after-drift plan in the brief had never become a testable requirement, unlike everything else on the requirements page, and that EDN-08 had already defined the promotion and rollback mechanism without it being referenced by any FR. Asked whether this and a related NFR-11 tightening were still relevant and worth doing, AI confirmed both were unaddressed, proposed the NFR-11 fix directly (no real alternative to weigh), and for retraining, named the automation-level choice as a genuine decision, laid out both options with the project's timeline as the deciding factor, and recommended the human-triggered option. Lukas confirmed it.
- **AI interaction evidence:** Claude Code session on 2026-09-22 while reviewing PR #19: AI's original review had flagged both the trivially-passable NFR-11 and the missing retraining requirement; Lukas asked whether they were still relevant and whether they were overkill, AI confirmed relevance for both and recommended the human-triggered option for retraining with reasoning tied to the M6 timeline; Lukas replied "yes."
- **Other evidence:** [Requirements](../docs/docs/requirements.md) FR-14, FR-15, NFR-01, NFR-11; [project brief](../docs/docs/project-brief.md) section 3.2 (`ES` new-market drift scenario) and section 5 (target architecture); [EDN-03](#edn-03-new-market-drift-scenario-hold-out-autoscout24-spain-instead-of-using-datamarket), [EDN-08](#edn-08-model-loading-bake-into-the-api-image-via-dvc-pull-at-ci-build-time-not-the-mlflow-registry-at-runtime); [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** no

### EDN-13: Keep the `/feedback` endpoint, but reachable only from inside the Compose network

- **Date:** 2026-09-23
- **Milestone:** M6: Monitoring
- **Activity / Topic:** API Design, ML System Design, Monitoring
- **Participants:** @lukas2510
- **Decision:** `POST /feedback` (FR-10) stays in the component and is no longer `[open]`. It takes a request ID and an observed price, stored as a delayed label, and FR-14 joins those labels to the prediction log to report the real error (MdAPE) and the empirical interval coverage per window, not only input drift. The endpoint is not publicly reachable: a reverse proxy in front of the API refuses `/feedback` from outside, so only the replay job inside the Compose network reaches it; the `X-API-Key` check stays behind that as a second layer in case the proxy rule is ever wrong. Because the key therefore never crosses the public internet, NFR-09 no longer requires TLS, and the previously `[proposed]` TLS item is dropped.
- **Alternatives considered:**
  - **Option A: keep `/feedback` as a publicly reachable endpoint, protected by the API key alone.** The variant the requirements page described before this decision.
    Pros: one fewer component in the deployment; the endpoint behaves like a real public feedback channel.
    Cons: the key would travel in cleartext unless the VM gets TLS, and the FIB Virtech VM sits behind NAT with a forwarded port, so an HTTP-01 Let's Encrypt challenge on port 80 of a domain we control is most likely not possible; that left the whole point of adding the key resting on an unverified assumption about the VM.
  - **Option B: drop `/feedback`; the drift job reads the `ES` prices directly from the versioned artefact and joins them to the prediction log by row ID.**
    Pros: no write path, no key, no TLS question, no shared store needed for feedback; least code.
    Cons: the labels never enter the system through an interface, so the delayed-label mechanism exists only inside one batch job; the architecture no longer shows how ground truth would arrive in operation, which is the part M6 actually grades ("model performance degradation, and user impact"); EDN-03 chose the `ES` holdout precisely because every held-out listing has a price, and that argument only pays off if those prices flow back in somewhere.
  - **Option C (chosen): keep the endpoint, but make it unreachable from outside (reverse-proxy rule), with the API key as a second layer.**
    Pros: keeps the feedback loop as a real interface, so FR-14 can report error and coverage and FR-15's retraining is triggered by a measured degradation rather than by an input-distribution change alone; removes the TLS dependency entirely instead of leaving it pending on a VM capability nobody has confirmed; the attack surface of the only writing endpoint drops to zero from the internet; the reverse proxy is a component the course demo already suggests (nginx in front of `uvicorn`).
    Cons: one more component in `docker compose` and one proxy rule that has to be right, which is why the key is kept behind it and the deployment smoke test asserts from outside the VM that `/feedback` is refused.
- **Rationale:** The endpoint's existence could not stay `[open]`, because FR-14, FR-15, NFR-03 and NFR-09 all depended on it. On whether to keep it, the decisive argument is what M6 is graded on: without labels the monitoring can only show that the inputs changed, while with them it can show that the error rises and the nominal 90 % intervals lose their coverage, which is also the stronger trigger for FR-15's retraining. The marginal cost is small, since the storage that dominates the work is needed for FR-13's prediction log anyway. On exposure, option C removes an unverified assumption (TLS on the Virtech VM) rather than carrying it as a pending decision, and it keeps the honest framing that traffic and labels are replayed from the `ES` holdout, not produced by real users.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** During the review of PR #19, AI flagged that FR-10 carried `[open]` and `[decided]` markers in the same row and that four other requirements silently depended on it, then laid out options A to C with the M6 rubric wording and the likely NAT setup of the Virtech VM as the deciding factors, and recommended keeping the endpoint while making it internal. Asked to explain the recommendation in more detail, AI also named the counter-argument itself (the labels are simulated, so a critic could call the endpoint a facade) and the mitigation (state plainly in the report that traffic and labels are replayed). Lukas reviewed the reasoning and accepted the recommendation.
- **AI interaction evidence:** Claude Code session on 2026-09-23 while reviewing PR #19: AI recommended keeping `/feedback` and combining it with internal-only exposure; Lukas asked "erkläre mir FR-10 genauer und warum du das eine vorschlägst", AI explained the delayed-label mechanism, the dependencies and the three options; Lukas replied "ok ich mag deine empfehlung ändere es so".
- **Other evidence:** [Requirements](../docs/docs/requirements.md) FR-10, FR-14, NFR-09; [project brief](../docs/docs/project-brief.md) sections 3.2 and 5; [EDN-03](#edn-03-new-market-drift-scenario-hold-out-autoscout24-spain-instead-of-using-datamarket), [EDN-12](#edn-12-retraining-and-promotion-are-human-triggered-not-automated); [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** no

## Template

```markdown
### EDN-NN: Short name of the decision

- **Date:** YYYY-MM-DD
- **Milestone:** M1: Project Inception | M2: Reproducibility | M3: Quality Assurance | M4: Model Deployment | M5: Model Packaging | M6: Monitoring | Other
- **Activity / Topic:** e.g. Testing Strategy, API Design, Containerization, CI/CD, Monitoring
- **Participants:** All team members | names
- **Decision:** What we decided, in one or two sentences.
- **Alternatives considered:**
  - **Option A (chosen):** short description.
    Pros: ... Cons: ...
  - **Option B:** short description.
    Pros: ... Cons: ...
- **Rationale:** Why the chosen option won: constraints, trade-offs, evidence.
- **AI involvement:** Select all that apply: No AI involvement | Information seeking | Alternative generation | Alternative assessment | Recommendation | Solution generation | Other (describe)
- **Response to AI:** If AI contributed, select one: Accepted | Accepted with modifications | Rejected | Used as input for further analysis | Other (describe)
- **Assessment of the AI contribution:** If AI contributed, why we accepted, modified, rejected or otherwise used it.
- **AI interaction evidence:** Link to the conversation (and which part), or the relevant prompt and response.
- **Other evidence:** Issues, PRs, commits, experiments, test results, diagrams.
- **In LaTeX:** no
```
