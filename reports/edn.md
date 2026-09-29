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

> **Amended by EDN-25** (2026-09-26, PR #27): the raw file is tracked with `dvc add` and pushed to our DagsHub remote instead.
> This entry is kept as the record of what was decided on 2026-09-22 and of the reasoning EDN-25 revised.

- **Date:** 2026-09-22
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Data Versioning, Privacy
- **Participants:** @lukas2510
- **Decision:** The raw AutoScout24 CSV is pulled with `dvc import-url` from its Zenodo DOI, pinned to version 1.0.0, and is never `dvc push`ed to our DagsHub remote. Only the PII-stripped `data/interim` output onward is tracked and pushed like a normal pipeline artefact. Enforced by `push: false` on the file's output in `data/raw/cars.csv.dvc`, which makes `dvc push` skip it, and a test that the field stays there.
- **Alternatives considered:**
  - **Option A (chosen): `dvc import-url` from Zenodo, raw file never pushed.**
    Pros: the raw file carries PII (`street`, `zip`, exact coordinates, `seller_company_name` for private sellers); the Zenodo DOI is already immutable and versioned, so re-fetching it with `dvc update` gives the same reproducibility as a cached copy without a second, PII-bearing copy on our own remote (likely public); matches the data-minimization principle for sensitive raw data.
    Cons: fetching the raw data needs live access to Zenodo; reproducibility depends on Zenodo keeping this exact file available; `dvc push` uploads imported files like any other output, so the "never pushed" half needs `push: false` on the output, not just the command choice, to actually hold; because the file is never on the remote, a fresh clone gets it with `dvc update` instead of `dvc pull`, and `dvc repro` fails until it has.
  - **Option B: `dvc add` the raw file like any other pipeline input and push it to DagsHub.**
    Pros: consistent with how every other artefact in the project is tracked; does not depend on Zenodo staying reachable.
    Cons: re-hosts a PII-bearing file on our own remote, which is likely public; the second copy adds no reproducibility over the immutable Zenodo DOI, only exposure.
- **Rationale:** Option A was already the direction noted (but left `[open]`) in the project brief's data-issues section. It was surfaced as an undecided requirement-vs-brief contradiction during the review of PR #19 ("requirements.md" stated the raw data is never re-hosted as a settled `NFR-08`, while the brief still called the same question open, and the existing `data-versioning.md` example showed the opposite pattern, `dvc add data/raw/cars.csv`). Data minimization for a file containing indirect identifiers of private sellers outweighs the convenience of a uniform tracking pattern; the Zenodo DOI's own immutability and versioning already give the reproducibility that pushing a copy would otherwise provide.
- **AI involvement:** Information seeking, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** While reviewing PR #19 for SOTA fit and course requirements, AI noticed that `NFR-08`'s "never re-hosted" wording silently settled the brief's open `dvc import-url` question without going through the EDN process, and that the project's own `data-versioning.md` example contradicted it. Asked for a recommendation, AI laid out options A and B with pros and cons, recommended A on privacy and data-minimization grounds, and flagged that `dvc import-url` alone does not prevent a re-push and needs a CI check to be enforceable. Lukas reviewed the reasoning and confirmed option A. On 2026-09-23, asked whether NFR-08 was overkill, AI proposed `push: false` on the output instead of that CI check against the DagsHub remote, and verified it in a scratch repository with DVC 3.67.1: without the field `dvc push` uploads the imported file, with it the remote stays empty, also after a `dvc update` from a changed source. The same test showed that in a fresh clone `dvc repro` does not re-fetch an imported file and a plain `dvc pull` exits with an error, while `dvc update` does fetch it, which corrected the instructions in `data-versioning.md`. Lukas accepted the change of mechanism; the decision itself stayed the same.
- **AI interaction evidence:** Claude Code session on 2026-09-22 while reviewing PR #19: AI flagged the NFR-08/brief contradiction, explained it in detail on request, then gave a recommendation and cited data-minimization and Zenodo DOI immutability as the SOTA rationale for option A; Lukas replied "yes please do so". Claude Code session on 2026-09-23: Lukas asked (translated from German) "NFR-08 privacy, does that still need adjusting? how would you adjust it so it is no longer excessive", AI proposed `push: false` with the scratch-repository evidence; Lukas replied "yes please do that".
- **Other evidence:** [requirements](../docs/docs/requirements.md) NFR-08; [specification](../docs/docs/specification.md) NFR-08 and "Decisions pending confirmation"; [Data versioning](../docs/docs/data-versioning.md); [project brief](../docs/docs/project-brief.md) section 3.3; [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
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
- **Other evidence:** [Specification](../docs/docs/specification.md) FR-12; [project brief](../docs/docs/project-brief.md) section 5 (target architecture); [references/course-demos.md](../references/course-demos.md) "API and deployment (M4)"; [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** no

### EDN-09: Bump to Python 3.12, to use real SHAP instead of a workaround

- **Date:** 2026-09-22
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Tooling, Explainability
- **Participants:** @lukas2510
- **Decision:** `requires-python` is `~=3.12.0` (was `~=3.11.0`). `pyproject.toml`, `uv.lock`, and the Python version in `.github/workflows/ci.yml` are updated, and `shap` becomes a usable dependency for offline analysis. Which SHAP implementation FR-08's explanation endpoint uses is decided separately in EDN-11, which amends this entry: the API computes contributions from the booster's own export, and `shap` stays out of the API image.
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
- **Other evidence:** [Specification](../docs/docs/specification.md) FR-08; [project brief](../docs/docs/project-brief.md) section 6 (tooling constraints); `pyproject.toml`, `.github/workflows/ci.yml`; [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
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
- **Other evidence:** [requirements](../docs/docs/requirements.md) NFR-05; [specification](../docs/docs/specification.md) NFR-05; [project brief](../docs/docs/project-brief.md) section 5 (target architecture) and section 7 (milestones); [MLOps-lab.md](../references/MLOps-lab.md) session 11; [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
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
- **Other evidence:** [Specification](../docs/docs/specification.md) FR-08; [project brief](../docs/docs/project-brief.md) section 6 (tooling constraints); [EDN-09](#edn-09-bump-to-python-312-to-use-real-shap-instead-of-a-workaround); [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
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
- **Other evidence:** [requirements](../docs/docs/requirements.md) FR-14, FR-15, NFR-01, NFR-11; [specification](../docs/docs/specification.md) FR-14 and FR-15; [project brief](../docs/docs/project-brief.md) section 3.2 (`ES` new-market drift scenario) and section 5 (target architecture); [EDN-03](#edn-03-new-market-drift-scenario-hold-out-autoscout24-spain-instead-of-using-datamarket), [EDN-08](#edn-08-model-loading-bake-into-the-api-image-via-dvc-pull-at-ci-build-time-not-the-mlflow-registry-at-runtime); [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
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
- **Other evidence:** [Specification](../docs/docs/specification.md) FR-10, FR-14, NFR-09; [project brief](../docs/docs/project-brief.md) sections 3.2 and 5; [EDN-03](#edn-03-new-market-drift-scenario-hold-out-autoscout24-spain-instead-of-using-datamarket), [EDN-12](#edn-12-retraining-and-promotion-are-human-triggered-not-automated); [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** no

### EDN-14: NFR-11's drift control is an i.i.d. sample, not a seller-grouped one

- **Date:** 2026-09-23
- **Milestone:** M6: Monitoring
- **Activity / Topic:** Monitoring, Performance Criteria
- **Participants:** @lukas2510
- **Decision:** NFR-11 is rewritten after measuring it on the real dataset. The drift job compares each window against a reference of 10,000 listings sampled from the training data. The `ES` replay must be flagged within its first 100 requests (was 1,000) and must name at least three drifted features other than `country_code`. The control is an i.i.d. sample of held-out listings, of which at most 1 in 20 windows may be flagged; a seller-grouped sample is explicitly not a valid control.
- **Alternatives considered:**
  - **Option A: keep the seller-grouped test split as the control, as NFR-11 originally said.**
    Pros: the strongest possible claim, namely that the detector stays quiet even on sellers it has never seen; reuses the split the evaluation protocol already produces.
    Cons: measured impossible, not merely hard. 200 of 200 seller-grouped control windows of 1,000 rows were flagged, at every reference size from 2,000 to 76,142 rows. Held-out dealers are genuinely a different distribution: their mean effect size is larger than Spain's on `make` (0.117 vs 0.070), `gears` (0.130 vs 0.034), `mileage_km_raw` (0.108 vs 0.091) and `age_years` (0.113 vs 0.099), and equal on `model` (0.245 vs 0.244). Keeping this control would make NFR-11 permanently red for a reason that has nothing to do with monitoring quality.
  - **Option B: keep the seller-grouped control but add an effect-size floor on top of the p-value.**
    Pros: would have kept the stronger claim and is a standard remedy for over-powered statistical tests.
    Cons: measured not to separate the two cases. At a floor of 0.10, `ES` and the control both have 5 features above it; at 0.15 it is 2 against 1. Rejected on evidence, not on principle.
  - **Option C: drop the control clause and only require that `ES` is flagged.**
    Pros: simplest; the `ES` half passes comfortably.
    Cons: a detector that flags everything would pass. The control is what makes the requirement say anything about the monitoring at all.
  - **Option D (chosen): keep the control, but define it as an i.i.d. sample of held-out listings, and cap the reference at 10,000 rows.**
    Pros: measured to work. The same detector gives a 1 to 10 % false-alarm rate on i.i.d. windows (2 % at a 10,000-row reference, against a nominal 5 %) while still flagging `ES` in 100 % of trials, so it separates the two cases the requirement is about. The reference cap keeps the false-alarm rate from growing with the training set, since power against irrelevant differences rises with reference size (1 % at 2,000 rows, 10 % at 78,330).
    Cons: a weaker claim than option A. It states that the detector is quiet on in-distribution traffic, not that it is quiet on unseen dealers, and the requirement now has to say which sample counts as the control, which is one more thing to get right in the M6 harness.
- **Rationale:** NFR-11 asserted a property of the data ("at least one feature other than `country_code` drifts", "the test-set control does not flag") that nobody had checked. Measured on the scoped, deduplicated dataset (105,431 listings; 5,981 `ES` rows that the API would accept under FR-04; reference 76,142), the first half holds with a large margin: 200 of 200 `ES` windows are flagged at every window size from 100 to 1,000, and still 200 of 200 when `country_code` is excluded. Spain is separable for substantive reasons, above all listing completeness: `nr_prev_owners` is missing in 95.5 % of `ES` rows against 40.0 % in the reference, `nr_seats` in 10.9 % against 3.0 %, and the mix differs in `body_type` (station wagon 6.6 % against 14.8 %), `transmission` (semi-automatic 0.1 % against 4.0 %) and `fuel_category` (diesel 42.1 % against 31.9 %). The second half was simply false, and the i.i.d. cross-check shows the detector is not at fault: the same code gives a near-nominal false-alarm rate as soon as the control window is drawn i.i.d. The requirement is therefore aligned with what can be demonstrated, and the finding behind it, that dealer-level shift is as large as country-level shift on this data, is recorded in the project brief because it also justifies the seller-grouped split and is a risk for SC-04.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** While reviewing PR #19, AI flagged NFR-11 as the one requirement that makes an unverified claim about the data rather than about the system. Asked for an in-depth feasibility check, AI downloaded the raw dataset from the Zenodo DOI, reproduced the scope, deduplication and split protocol, emulated alibi-detect's `TabularDrift` with scipy after confirming its tests, defaults and Bonferroni rule against the v0.13 source (including that it does no NaN handling, so missing indicators had to be added), and ran 200 trials per window size. It reported that the first half of the requirement passes with margin and the second half fails, ran two controlled cross-checks to separate a detector fault from a data property (i.i.d. split, reference size), and tested and then rejected its own effect-size remedy on the measurements. Lukas reviewed the evidence and chose option D.
- **AI interaction evidence:** Claude Code session on 2026-09-23 while reviewing PR #19: Lukas asked "NFR-11 please make an in depth analysis if this is feasible [...] we need to be really sure about this"; AI ran the measurement described above, reported the pass, the failure and the limits of the study, proposed options A to D and recommended D; Lukas replied "yes please".
- **Other evidence:** [requirements](../docs/docs/requirements.md) NFR-11; [specification](../docs/docs/specification.md) NFR-11; [reports/analysis/](analysis/) (`nfr11_check.py`, `nfr11_diag.py`, `nfr11_results.json`, run of 2026-09-23); [project brief](../docs/docs/project-brief.md) section 3.2; [EDN-03](#edn-03-new-market-drift-scenario-hold-out-autoscout24-spain-instead-of-using-datamarket); [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** no

### EDN-15: UC1 required fields after measuring the fill rates, plus SC-06 for absent optional fields

> **Evidence corrected on 2026-09-29.** The first run of `fillrates.py` printed fill rates with one decimal, so `seller_type`'s 99.986 % appeared as 100.0.
> The script now prints three decimals and `fillrates_results.txt` was regenerated. The decision is unchanged: 14 missing rows in 97,913 leave it effectively never absent.

- **Date:** 2026-09-29
- **Milestone:** M1: Project Inception (requirements and success criteria; shapes M3: Quality Assurance)
- **Activity / Topic:** Requirements Engineering, API Design, Performance Criteria
- **Participants:** @lukas2510
- **Decision:** `/predict` requires `body_type` and `seller_type` in addition to the core FR-01 asked for before (`make`, `model`, `registration_date`, `mileage_km_raw`, `power_kw`, `fuel_category`, `transmission`, `country_code`). The remaining basic-set fields stay optional. FR-01 no longer prescribes how the model handles an absent optional field; it states the observable outcome instead, and the new SC-06 of the problem specification bounds it: with each optional field masked on its own, and with all of them masked at once, the point model's MdAPE stays at or below 1.5 times its full-input MdAPE. This amends EDN-06, which set SC-01 to SC-05; because NFR-01 gates deployment on all of them, the gate now also covers partial UC1 inputs. Whether the rarely-missing optional fields need the random masking planned for the UC2 interval models is left to M2/M3, when the pipeline exists and the effect can be measured.
- **Alternatives considered:**
  - **Option A: keep all eight basic-set fields optional and rely on LightGBM's native missing-value handling.**
    Pros: the smallest required set, so UC1 stays easy to call and clearly different from UC2; no training-time work at all.
    Cons: measured to be unsupported for two fields. `body_type` is filled in 100.000 % of the training listings and `seller_type` in 99.986 % (14 of 97,913 rows), so the model effectively never sees them absent and a request that omits them is sent to the default side of every split on that feature with no training example backing that direction. The contract would promise behaviour the training data cannot show.
  - **Option B: keep all eight optional and train the point model with the same random masking as the UC2 interval models.**
    Pros: friendliest contract; one masking scheme for both model families, which also keeps the point estimate and the interval bounds from drifting apart on partial inputs (FR-07 widens the bounds when they do).
    Cons: makes an API requirement depend on a pipeline that does not exist yet (no `dvc.yaml`, issue #10 still open), so PR #19 would stay blocked on a modelling decision that cannot be measured before the first report. Masking also costs some accuracy on full inputs, which are the common case for UC1.
  - **Option C: require all basic features.**
    Pros: best accuracy, no missing-value path at serving time at all.
    Cons: hostile to users. `nr_prev_owners` is filled in 61.0 % of the listings, `gears` in 62.5 % and `drive_train` in 76.4 %, so even the sellers behind the listings often do not state them.
  - **Option D (chosen): require the two always-filled fields, keep the rest optional, and bound the outcome with SC-06.**
    Pros: closes the measured gap at zero cost to the user, because whoever owns the car knows its body type and, for UC1, is the seller. Separates the API contract (decidable now) from the training mechanism (decidable in M2/M3), so the requirement can be merged without guessing. SC-06 turns FR-01's previously unquantified "per-field masked-input accuracy check" into a threshold and puts it in the deployment gate.
    Cons: the required set grows from eight fields to ten. SC-06 amends EDN-06, which was already agreed, and it is relative rather than absolute, so it cannot rule out a model that is uniformly mediocre; SC-01 to SC-04 do that.
  - **Option E: adopt D, but report the masked-input error only, without gating on it.**
    Pros: no amendment to EDN-06; the number is still visible in MLflow and in the report.
    Cons: leaves NFR-01's gate blind to exactly the case FR-01 allows the client to trigger, which is how the gap arose in the first place.
- **Rationale:** FR-01 carried an `[open]` marker because it mixed two questions: which fields the client must send, which is an API contract, and how the model is trained to honour the optional ones, which is a modelling mechanism. The fill rates that the second question depended on had never been measured. Measured on the scoped, deduplicated training data (97,913 listings, `ES` held out, 11 supported makes), the risk turns out to sit in exactly two fields: `body_type` and `seller_type` at 100 %, followed by `nr_doors` (98.8 %), `nr_seats` (97.0 %) and `cylinders_volume_cc` (91.1 %), while `nr_prev_owners` (61.0 %), `gears` (62.5 %) and `drive_train` (76.4 %) have plenty of natural missingness. Only 30.7 % of training rows have every one of the six remaining optional fields present and 469 rows (0.5 %) have none of them, so the model sees partial rows including the extreme case, just never a row without `body_type` or `seller_type`. Requiring the two is therefore the cheapest correct fix, and it costs the user nothing. The rest is left to the modelling plan on purpose: the mechanism cannot be chosen on evidence before the pipeline runs, and blocking the requirements page on it would have held up the M1 deliverable for a decision that belongs to M2/M3. SC-06 is what makes that safe, and it is relative to the model's own full-input MdAPE, like SC-03 is relative to the baseline, because no reference value exists yet and an absolute threshold would have been invented rather than derived.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** Asked which decisions had to be settled before PR #19 could be merged, AI named FR-01's open item first and called it a merge blocker. Asked to re-examine that, it downloaded the raw dataset from the Zenodo DOI, reproduced the scope, deduplication and make filter (which matched the brief's 11 makes as a cross-check), measured the per-field fill rates, and then corrected two of its own claims: the item does not block code that exists, because there is no pipeline yet, and FR-01 already promised a masked-input check, so the real gap was that the check had no threshold and no link to NFR-01's gate. It also reframed the requirement as mixing contract and mechanism, which is what made a clean split possible. Lukas accepted the recommendation and the field list; the modification is that the mechanism question stays open on purpose instead of being decided together with the contract.
- **AI interaction evidence:** Claude Code session on 2026-09-29: Lukas asked (translated from German) "which things urgently still need deciding in requirements.md before we can merge, and why?", then "analyse that again, then explain it to me and give me a reasoned recommendation on what the best way is"; AI measured the fill rates, revised its own analysis, proposed options A to E and recommended D with SC-06 in the gate; Lukas replied "I like your recommendation, please change it that way in the PR".
- **Other evidence:** [Specification](../docs/docs/specification.md) FR-01; [problem specification](../docs/docs/problem-spec.md) SC-06; [reports/analysis/](analysis/) (`fillrates.py`, run of 2026-09-29); [project brief](../docs/docs/project-brief.md) section 4; [EDN-06](#edn-06-success-criteria-for-the-price-model); [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** no

### EDN-16: Separate `/predict` and `/price-range` instead of one valuation endpoint

- **Date:** 2026-09-29
- **Milestone:** M1: Project Inception (API contract; implemented in M4: Model Deployment)
- **Activity / Topic:** API Design
- **Participants:** @lukas2510
- **Decision:** UC1 and UC2 keep their own endpoints, `POST /predict` and `POST /price-range` (FR-06, FR-07). They are no longer marked `[proposed]`.
- **Alternatives considered:**
  - **Option A (chosen): one endpoint per use case.**
    Pros: each endpoint has one input contract, which matters because `/predict` requires ten fields (FR-01, EDN-15) and `/price-range` requires only `make` (FR-02); FR-16's per-endpoint OpenAPI examples and FR-01's per-field 422 stay straightforward; the prediction log (FR-13) keeps the two traffic shapes apart, so FR-14's drift job does not read a change in the UC1/UC2 mix as input drift on exactly the completeness features the `ES` scenario relies on (EDN-14); consistent with `/comparables` (FR-09), which is already its own endpoint.
    Cons: two request schemas to keep in step, and a client that wants the estimate and the interval calls twice.
  - **Option B: one `/valuation` endpoint returning the estimate and the interval.**
    Pros: one path, one call for a client that wants both.
    Cons: the required fields would depend on the use case inside one schema, so the two contracts would be rebuilt as a conditional rule instead of disappearing; mixes both traffic shapes in one log; leaving `/comparables` separate would be inconsistent, and merging it too would put SHAP, intervals and the comparables lookup in every request against NFR-02's latency budget.
- **Rationale:** The two endpoints differ in their input contract rather than in their output, and EDN-15 widened that gap from eight required fields to ten against one. A single endpoint would therefore not remove the two contracts, only hide them in a conditional schema. The monitoring side settles it: FR-14 and NFR-11 compare the logged input distribution of a window against the training reference, and UC2 allows every optional field to be absent, so a shift in the client mix would look like drift on the same completeness features that make the `ES` replay detectable. The decision is also the reversible one: a combined endpoint can be added later as a convenience, while splitting one apart under `/v1` would be a breaking change.
- **AI involvement:** Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** Asked to re-examine the split before merging PR #19, AI first called it a merge blocker and then withdrew that: no test or report cites FR-06 or FR-07 yet, so the practical cost of deciding later starts with the M4 API tests, not with the merge. It found that the two endpoints do not differ in their output, since FR-07 already returns the point estimate, and contributed the monitoring argument above, which had not been written down anywhere. Lukas kept the scope deliberately small and dropped AI's further suggestions (a shared base schema stated in the requirement, a combined convenience endpoint) as implementation detail or as unnecessary for a course project.
- **AI interaction evidence:** Claude Code session on 2026-09-29: Lukas asked (translated from German) "analyse that again, then explain it to me and give me a reasoned recommendation on what the best way is" about the endpoint split, and after the recommendation replied "I do not want overkill, it should stay simple, it is only a university course".
- **Other evidence:** [Specification](../docs/docs/specification.md) FR-06, FR-07, FR-09; [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** no

### EDN-17: Deployment target is the FIB Virtech VM

- **Date:** 2026-09-29
- **Milestone:** M4: Model Deployment
- **Activity / Topic:** ML System Design, Deployment
- **Participants:** @lukas2510
- **Decision:** The component is deployed on the FIB Virtech VM the course provides (4 GB RAM, 20 GB disk, CPU-only, one semester). The project brief no longer lists the target as an open point, and the requirements say so once instead of assuming it in every target.
- **Alternatives considered:**
  - **Option A (chosen): the FIB Virtech VM.**
    Pros: free and provided for the course, so no payment details and no personal cloud account for a semester project; 4 GB is enough with the budgets NFR-04 already sets (drift job scheduled in its own container, Prometheus capped by retention, logs capped by rotation, 4 GB disk kept free); every target in the requirements is written for this machine.
    Cons: tight on RAM, a single unattended host with no redundancy, and nobody on the team has access yet.
  - **Option B: another cloud (AWS, DigitalOcean, Oracle Always Free or similar).**
    Pros: more headroom, and a free ARM tier exists at one provider.
    Cons: either paid or tied to a private account with payment details; the grading rewards nothing for the hosting choice; NFR-02 to NFR-04 would have to be re-derived for different hardware, and an ARM tier would also mean multi-architecture images.
- **Rationale:** The requirements already named this VM in eight places (NFR-02, NFR-04, NFR-05, NFR-09, NFR-12, NFR-13 and the network warning), so the decision had been made by writing rather than by deciding, while the brief still listed the target as open. The reason the brief gave for keeping it open, that the stack is tight on 4 GB, has since been planned for in NFR-04, so the open point was stale. What stays genuinely open is not the target but what the VM's network allows, which the requirements already track with a fallback per requirement. Access is a task, not a decision, and it has a date: if nobody has access at the M4a lab on 2026-10-21, we raise it with the teachers.
- **AI involvement:** Information seeking, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** While reviewing PR #19, AI reported the brief and the requirements as contradicting each other on the deployment target. Asked to look again, it corrected its own framing: the `[proposed]` marker in the requirements refers to the performance numbers, not to the machine, and the requirements treat the VM as settled throughout, so the real issue was an implicit decision rather than a disagreement between two pages. It also pointed out that the brief's stated reason for leaving the point open is already covered by NFR-04. Lukas kept the change small and deliberately left the performance targets `[proposed]` until the first M4 load test.
- **AI interaction evidence:** Claude Code session on 2026-09-29: after Lukas asked (translated from German) "analyse that again, then explain it to me and give me a reasoned recommendation on what the best way is" about the deployment target, AI recommended fixing the VM as the target with a dated trigger for the missing access, and Lukas replied "yes do that".
- **Other evidence:** [Specification](../docs/docs/specification.md) section 2 and NFR-04; [project brief](../docs/docs/project-brief.md) section 5; [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** no

### EDN-18: Unseen countries and models are accepted with a warning, not rejected

- **Date:** 2026-09-29
- **Milestone:** M4: Model Deployment (serving contract; enables M6: Monitoring)
- **Activity / Topic:** API Design
- **Participants:** @lukas2510
- **Decision:** FR-05 loses its `[proposed]` marker. A `country_code` among the 8 countries of the data or a `model` that is missing from the training data is accepted, passed to the model as unknown, and reported in the response's `warnings` field.
- **Alternatives considered:**
  - **Option A (chosen): accept and warn.**
    Pros: the `ES` replay reaches the model at all, which is what EDN-03's new-market drift scenario and all of M6 depend on; LightGBM handles an unseen category natively, so nothing has to be built; the caller still learns that the answer is extrapolated, and the warning is in the prediction log (FR-13).
    Cons: the API answers for inputs the model has never seen, and the estimate can be worse than the metrics suggest without the caller noticing the warning.
  - **Option B: reject with HTTP 422.**
    Pros: stricter, and never answers outside what was trained.
    Cons: would reject every `ES` listing, so the drift scenario could not be replayed and the retraining loop (FR-15) could not be demonstrated. The scope check we do want is about the make (FR-04), which stays a 422.
- **Rationale:** The decision was effectively made by EDN-03, which chose Spain as the held-out new market precisely so that unseen traffic can be replayed against the running API in M6. Rejecting it would have removed the scenario the monitoring milestone is built on. Out-of-scope makes are a different case and keep their 422 (FR-04), so "accept" is not a blanket rule.
- **AI involvement:** Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI pointed out that this item was already prejudiced by EDN-03 and cost nothing to confirm, which is why it stayed unmarked longer than necessary. No further analysis was needed.
- **AI interaction evidence:** Claude Code session on 2026-09-29: AI noted that FR-05 was (translated from German) "in fact already prejudged by EDN-03" and recommended lifting it to `[decided]`; Lukas replied "do that".
- **Other evidence:** [requirements](../docs/docs/requirements.md) FR-05 and FR-04; [specification](../docs/docs/specification.md) FR-05; [EDN-03](#edn-03-new-market-drift-scenario-hold-out-autoscout24-spain-instead-of-using-datamarket); [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** no

### EDN-19: Separate the requirements from their specification, keeping one set of FR/NFR IDs

- **Date:** 2026-09-29
- **Milestone:** M1: Project Inception
- **Activity / Topic:** Requirements Engineering
- **Participants:** @lukas2510
- **Decision:** `docs/docs/requirements.md` keeps the `FR-xx` and `NFR-xx` vocabulary but is rewritten at a design-free level: what must be true, for whom, with a priority and an acceptance criterion, and without endpoints, formats, status codes, libraries or deployment mechanics. Everything design-level moves to a new `docs/docs/specification.md`, which carries the same IDs, so `FR-04` there specifies `FR-04` here. Tests, the requirement-to-test matrix and all earlier EDN entries keep pointing at one unchanged ID space.
- **Alternatives considered:**
  - **Option A (chosen): two levels, one ID space.**
    Pros: the requirements answer what the component must do and can be read by someone who does not know the architecture, which is what the feedback asked for; every design decision that was already analysed and settled (EDN-08, EDN-11 to EDN-18) survives in full, at the level where it belongs; a later design change, for example a different endpoint split or a different deployment path, no longer looks like a requirements change; the `FR-xx`/`NFR-xx` IDs that tests and twelve earlier EDN entries refer to keep working, and the requirement-to-test traceability is unaffected.
    Cons: two pages to keep consistent, and a reader has to know which of the two answers their question; every cross-reference in the brief, the problem specification, `data-versioning.md`, `CONTRIBUTING.md` and the EDN evidence links had to be repointed to the level it actually meant.
  - **Option B: rewrite the requirements as user stories (`US-xx`) with a specification behind them.**
    Pros: the most explicit way to show the user's intent, and the standard form in agile requirements engineering, which fits the Scrum setup of the course.
    Cons: introduces a second ID space on top of `FR-xx`/`NFR-xx`, so every test and every earlier EDN entry would reach its requirement through one more hop; the course material and the report template speak of functional and non-functional requirements, so the team would be renaming the artefact the report has to deliver. The team tried this form first and rejected it for these reasons; the abstraction it was meant to bring is achieved by the rewrite in option A without the renaming.
  - **Option C: keep one page and only soften its wording.**
    Pros: no new file, no link changes.
    Cons: the design decisions would still sit inside the requirements, only less visibly, so the criticism would be answered in form and not in substance; and there would be no place left for the endpoint-level detail that M4 needs.
- **Rationale:** The lecturer's feedback on the M1 requirements was that they read as a specification: too technical, and fixing design where they should fix intent. The diagnosis holds. The page stated endpoints, HTTP status codes, header names, container mechanics and image tags, which meant that changing any of those looked like changing a requirement, and that a reader could not tell what the component owes its users from what we happened to choose. The detail itself is not the problem, since it is what the tests and the M4 deployment work are built on, so it was moved rather than dropped. Keeping a single ID space across both levels was the deciding factor against option B: the traceability the report has to show runs from a requirement ID to a test, and adding a second vocabulary on top of that would have cost a hop and gained nothing the rewrite does not already give.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** Lukas relayed the feedback and asked for a more abstract version, with the explicit constraint that the existing analysis must not be lost. AI confirmed the criticism against the actual page and proposed the split into two levels, which the team accepted. Its first execution went further than asked and replaced the requirements with user stories; Lukas corrected this, because the course material, the report template and the existing IDs all speak of functional and non-functional requirements, and the stories had been meant as orientation only. AI then reworked the page to the agreed form. The abstraction of each requirement and the acceptance criteria are its proposal and were reviewed by the team.
- **AI interaction evidence:** Claude Code session on 2026-09-29 (prompts were in German, translated here): Lukas relayed the feedback as "we got feedback on the requirements. The lecturer said they are too technical, they are more of a specification. He would rather have them like user stories, more abstract. Can you do that, but please also write a spec so the work was not for nothing", and after AI delivered user stories corrected it with "user stories were only meant as orientation for you. We still want to call them functional and non-functional requirements. But as said they should not be so technical, rather more abstract, and should not fix architecture decisions yet".
- **Other evidence:** [Requirements](../docs/docs/requirements.md); [Specification](../docs/docs/specification.md); [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
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
