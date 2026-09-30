# Engineering Decision Notebook (working copy)

This is where we record our EDN entries as we make decisions.
Before each delivery, the entries are transferred to the LaTeX EDN in [latex/edn/](latex/edn/), which becomes `MLOps_Recommenditos_EDN.pdf`.

What belongs here and what each field means: [references/Instruction_EDN_MLOps_v2026.md](../references/Instruction_EDN_MLOps_v2026.md).
Record a decision when it meaningfully affects the ML system and could reasonably have gone differently, not every decision or every use of AI.

How to add an entry:

- Copy the template at the end of this file and append the new entry above the template, oldest entry first.
- Give it the next free ID (`EDN-01`, `EDN-02`, ...).
  IDs never change once assigned, because the report refers to them.
  Reserve the ID by adding its heading to this file on `main` before opening the branch that needs it.
  Sprint 1 handed out EDN-19 on two branches at once, and the entry that lost the race had to be renumbered to EDN-27 while merging.
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
- **In LaTeX:** no

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

> **Amended by EDN-26** (2026-09-29): the window is 1,000 requests, not 100, and `model` is excluded from the comparison.
> The entry below stated a window and a control clause that were measured in two different runs and never together.
> **Evidence regenerated on 2026-09-29.** The numbers below are the 2026-09-23 run and are left as they were measured.
> `nfr11_results.json` was re-run after EDN-22 removed the listings registered after the reference date, which moves the
> scoped, deduplicated count from 105,431 to 105,405 and the `ES` rows the API accepts from 5,981 to 5,979.

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
> The script now prints three decimals and `fillrates_results.txt` was regenerated. The decision is unchanged: 14 missing rows in 97,889 leave it effectively never absent.
> The training-scope count moved from 97,913 to 97,889 when the script was brought in line with EDN-22, which drops the listings registered after the reference date.

- **Date:** 2026-09-29
- **Milestone:** M1: Project Inception (requirements and success criteria; shapes M3: Quality Assurance)
- **Activity / Topic:** Requirements Engineering, API Design, Performance Criteria
- **Participants:** @lukas2510
- **Decision:** `/predict` requires `body_type` and `seller_type` in addition to the core FR-01 asked for before (`make`, `model`, `registration_date`, `mileage_km_raw`, `power_kw`, `fuel_category`, `transmission`, `country_code`). The remaining basic-set fields stay optional. FR-01 no longer prescribes how the model handles an absent optional field; it states the observable outcome instead, and the new SC-06 of the problem specification bounds it: with each optional field masked on its own, and with all of them masked at once, the point model's MdAPE stays at or below 1.5 times its full-input MdAPE. This amends EDN-06, which set SC-01 to SC-05; because NFR-01 gates deployment on all of them, the gate now also covers partial UC1 inputs. Whether the rarely-missing optional fields need the random masking planned for the UC2 interval models is left to M2/M3, when the pipeline exists and the effect can be measured.
- **Alternatives considered:**
  - **Option A: keep all eight basic-set fields optional and rely on LightGBM's native missing-value handling.**
    Pros: the smallest required set, so UC1 stays easy to call and clearly different from UC2; no training-time work at all.
    Cons: measured to be unsupported for two fields. `body_type` is filled in 100.000 % of the training listings and `seller_type` in 99.986 % (14 of 97,889 rows), so the model effectively never sees them absent and a request that omits them is sent to the default side of every split on that feature with no training example backing that direction. The contract would promise behaviour the training data cannot show.
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
- **Rationale:** FR-01 carried an `[open]` marker because it mixed two questions: which fields the client must send, which is an API contract, and how the model is trained to honour the optional ones, which is a modelling mechanism. The fill rates that the second question depended on had never been measured. Measured on the scoped, deduplicated training data (97,889 listings, `ES` held out, 11 supported makes), the risk turns out to sit in exactly two fields: `body_type` and `seller_type` at 100 %, followed by `nr_doors` (98.8 %), `nr_seats` (97.0 %) and `cylinders_volume_cc` (91.1 %), while `nr_prev_owners` (61.0 %), `gears` (62.5 %) and `drive_train` (76.4 %) have plenty of natural missingness. Only 30.7 % of training rows have every one of the six remaining optional fields present and 469 rows (0.5 %) have none of them, so the model sees partial rows including the extreme case, just never a row without `body_type` or `seller_type`. Requiring the two is therefore the cheapest correct fix, and it costs the user nothing. The rest is left to the modelling plan on purpose: the mechanism cannot be chosen on evidence before the pipeline runs, and blocking the requirements page on it would have held up the M1 deliverable for a decision that belongs to M2/M3. SC-06 is what makes that safe, and it is relative to the model's own full-input MdAPE, like SC-03 is relative to the baseline, because no reference value exists yet and an absolute threshold would have been invented rather than derived.
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

### EDN-19: Which DagsHub repository the team uses as DVC remote and MLflow server

- **Date:** 2026-09-29
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Data Versioning, Experiment Tracking
- **Participants:** @lukas2510 (Scrum Master), on behalf of the team. Pending confirmation by the full team at the next sprint review.
- **Decision:** A DagsHub **organisation** owned by the team hosts the connected repository, and every team member joins it with push rights (option A).
  The repository is <https://dagshub.com/recommenditos/MLOPS_Recommenditos>, owned by the `recommenditos` DagsHub organisation and connected through DagsHub's GitHub integration.
  A first decision for `pauadal03/MLOPS_Recommenditos` was taken and withdrawn on the same day, once it turned out that pauadal03 is not a member of this team: they hold read-only access to our GitHub repository, have never contributed to it, and are not on the roster.
  Their DagsHub repository is a mirror of our public repository made from outside the team, so it is not a candidate.
  Later the same day `mark.welf.atzberger/MLOPS_Recommenditos` was deleted, so that candidate no longer exists either and the raw object it held is gone with it.
  The practical choice is therefore between a DagsHub organisation owned by the team and a fresh repository under one team member's account.
  Whether the repository stays public was decided separately, see EDN-20.
- **Alternatives considered:**
  - **Option A (chosen): a DagsHub organisation owns the connected repository, every team member is a member with push rights.**
    Pros: nothing the project depends on hangs off one person's private account; push rights follow membership instead of manual collaborator entries, and the team is five people, so inviting everyone is trivial; visibility is set in one place; data and experiment history survive if a member leaves or cleans up their account.
    Cons: one-time setup (create the organisation, connect the GitHub repository, invite four people); under a `dvc add` mechanism the 548.6 MB raw object has to be pushed once more, though under `import-url` there is nothing to move.
  - **Option B: use `mark.welf.atzberger/MLOPS_Recommenditos`, reconnected properly.** No longer available: the repository was deleted on 2026-09-29.
    Pros (while it existed): it belonged to a teammate and already held the raw object; no organisation to create.
    Cons: its mirror was demonstrably not in sync (see the evidence below), and the likely fix was to delete and re-import it through the GitHub integration, which would have lost the pushed object anyway; the project would still have hung off one teammate's private account, with a manual collaborator entry needed per member.
  - **Option C: a fresh repository under one team member's account, connected through the GitHub integration.**
    Pros: one form less than an organisation; whoever creates it is admin and can add the other four without waiting on anyone.
    Cons: identical to option B's structural drawbacks - the project hangs off one private account and rights are granted per person by hand. The saving over option A is a single setup form.
  - **Option D: use `pauadal03/MLOPS_Recommenditos`.** Chosen on 2026-09-29 and withdrawn the same day.
    Pros: it is the only mirror verifiably connected through the GitHub integration, and adopting it would have cost nothing.
    Cons: ruled out on ownership, not on measurements - pauadal03 is not on the team, has `read` permission on our GitHub repository and has never contributed to it, so the project's data and experiment history would sit on the account of someone who cannot even push to the repository they mirrored.
    For the record, no project data was ever on that mirror: the raw object returns 404 there, so it holds our public git history and nothing else.
  - **Option E: keep several mirrors and let everyone push to their own.**
    Pros: no coordination needed.
    Cons: not viable - the remote URL is committed in `.dvc/config`, so the team would overwrite each other's setting, and data and experiment history would be split across accounts.
- **Rationale:** By the time the decision was taken the alternatives had collapsed: option D was disqualified on ownership and option B ceased to exist when that repository was deleted, so every remaining candidate required a fresh connect through the GitHub integration anyway.
  That removed the only real argument against option A, which had been that a correctly connected repository already existed elsewhere at zero cost.
  What was left is a one-form difference - an organisation versus a repository under one member's account, for the same connect and the same invitations - against the fact that fifteen cohort members outside the team hold read access to the GitHub repository and can mirror it at any time, as one of them already had.
  The team's standing rule was to take the better-engineered option unless it is overkill for a one-semester course project; with four teammates to invite rather than the whole twenty-one-person collaborator list an earlier miscount had assumed, option A is not overkill.
  Evidence gathered on 2026-09-29 against the DagsHub and GitHub APIs.
  Ownership: write access to the GitHub repository is held by @lukas2510, @kadameit, @ulasawczuk, @W11W11W11 and @michudud04, plus @martinezmatias and @santidrj, the two lecturers who own the course organisation; @pauadal03 has `read`, zero pull requests and zero commits, and does not appear on the roster in `docs/docs/scrum/index.md`.
  Sync: `pauadal03/MLOPS_Recommenditos` is at `2e902e2143`, identical to GitHub `main`, carries all 7 branches and mirrors all 7 open issues, while `mark.welf.atzberger/MLOPS_Recommenditos` is at `8b1d64fc`, is missing the `docs/data-facts-corrections` and `model-card` branches, still carries `feature/ruff-pl-and-coverage` which GitHub deleted after the merge, and mirrors 0 issues.
  DagsHub only mirrors issues and pull requests for repositories connected through the GitHub integration, so the issue count is the clearest signal that the two were connected in different ways and that mark's is a plain git mirror.
  Visibility: both report `private: false`; EDN-07 as first written on the issue #3 branch assumed the remote was private, a premise found false and recorded in EDN-20.
  The reason that original check misled us: anonymous access is not evidence either way, because DagsHub refuses every anonymous request, including for a known-public control repository (`DAGsHub-Official/dagshub-docs`, also `private: false`) and for a repository that does not exist.
  Access: `lukas2510` was granted `push` on mark's repository on 2026-09-29, shortly before it was deleted, and has `pull` only on pauadal03's; the rest of the team had push on neither.
  State at the end of 2026-09-29: `mark.welf.atzberger/MLOPS_Recommenditos` returns 404 and so does the raw object it held; `pauadal03/MLOPS_Recommenditos` still exists and still holds no data; a repository search returns that mirror as the only one left.
  Structural point behind all of this: fifteen cohort members outside the team hold `read` on our GitHub repository, so anyone of them can create a mirror at any time without doing anything wrong. The rule the team needs is therefore not "check who owns a mirror" but "the project's infrastructure lives somewhere the team owns", which is an argument for option A independent of the measurements.
  Follow-up: [PR #26](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/26) now commits the organisation's URL in `.dvc/config`, and the owner still has to grant push rights to everyone.
  Nothing has to be migrated: the 548.6 MB raw object that had been pushed to mark's repository is gone with that repository, so under the `dvc add` mechanism settled in EDN-25 ([PR #27](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/27)) it is re-downloaded from Zenodo and pushed once to the organisation's remote.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI found the two mirrors, measured them against GitHub and recommended `pauadal03`'s on those measurements. The measurements were correct and the recommendation was wrong, because AI never checked whether that repository belonged to anyone on the team - it ranked the candidates it had found without asking where they came from. Lukas caught it by asking who pauadal03 is, which invalidated the recommendation and the decision taken on it. Recorded here rather than silently corrected, because the failure is the instructive part: a candidate that scores best on every technical measurement can still be disqualified by a question nobody asked. AI's useful contributions were the measurements, the finding that mark's mirror is not properly connected, and the control-repository method behind EDN-20; its first recommendation was not one of them. Its second recommendation, option A, was the one the team accepted, and only after Lukas had rejected the first and asked who the account behind it belonged to.
  This is the second decision in the same week reached on a premise nobody had checked, after "the DagsHub repository is private" in the first version of EDN-07 on the issue #3 branch, and in both cases the check that settled it took one API call and was available before the decision rather than after it. The pattern, not either individual fix, is the thing worth carrying into the working agreements: state the premise a decision rests on, and verify it, before recording the decision.
- **AI interaction evidence:** Claude Code session on 2026-09-29, prompts translated from German: the comparison and the recommendation for pauadal03 followed "isn't this our DagsHub repo, the one that matches our GitHub repo?" and "why did that repo score worst?"; the retraction followed "I think pauadal03 is one of our lecturers - is it bad that everything now has to go through them?".
- **Other evidence:** [PR #26](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/26), [issue #10](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/10), EDN-20, EDN-25, roster in `docs/docs/scrum/index.md`.
- **In LaTeX:** no

### EDN-20: The DagsHub remote stays public

- **Date:** 2026-09-29
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Data Versioning, Privacy
- **Participants:** @lukas2510 (Scrum Master). Pending confirmation by the full team at the next sprint review.
- **Decision:** The DagsHub repository that serves as our DVC remote and MLflow tracking server stays public.
  We accept that the raw file's PII columns are readable by anyone with a DagsHub account, which follows from EDN-25 tracking the raw file with `dvc add` and pushing it to that remote.
  The additional exposure this creates is small, because the identical file is already publicly downloadable from the pinned Zenodo DOI under the MIT license.
  It is not nothing: hosting our own copy makes the team a publisher of that personal data in its own right and under its own name, which the earlier publication mitigates but does not undo.
- **Alternatives considered:**
  - **Option A (chosen): keep the repository public and say so.**
    Pros: graders and supervisors can inspect code, data and experiments without being added as collaborators, which is the reason the course puts the project on DagsHub in the first place; it matches how the source dataset is already published.
    Cons: the raw file, which still carries `vin`, `street`, `seller_company_name` and coordinates, can be pulled by every logged-in DagsHub user; re-hosting it is a publishing act of our own, so "the author already published it" reduces the marginal risk but does not transfer the responsibility; the decision has to be revisited the moment we host data that is not already public elsewhere.
  - **Option B: make the repository private.**
    Pros: the rationale first given for pushing the raw file would hold as written; the PII sits behind an access wall.
    Cons: every grader, supervisor and teammate then needs a manual collaborator entry; it buys little real protection, because the identical file stays publicly downloadable from Zenodo either way.
- **Rationale:** The question only came up because the premise of the raw-data decision turned out to be false.
  The first version of EDN-07, on the issue #3 branch, justified pushing the raw file with "the DagsHub repo is private (anonymous access is redirected to the login page and an anonymous data download returns 401)".
  Checked on 2026-09-29: DagsHub refuses anonymous access to everything, public repositories included.
  A known-public control repository (`DAGsHub-Official/dagshub-docs`, `private: false`) answers anonymous requests exactly like ours, and so does a repository that does not exist, so the observation has no discriminating power.
  The authenticated API reports `private: false` for both mirrors that existed at the time.
  Given a real choice between hiding a file that is already public and saying openly that it is public, Lukas chose the second: the protection gained would be nominal, while the access cost for graders and supervisors would be real.
  Under EDN-19 the organisation is the team's own, so the visibility is ours to change and this decision can be revisited at any time without asking anyone outside the team.
  Recorded deliberately as an accepted risk rather than a solved problem, because the two are not the same thing: the marginal exposure is small, but we are still the ones publishing personal data, and an entry that claimed the concern was spent would not survive a reviewer who cares about privacy.
  State after the decision: `mark.welf.atzberger/MLOPS_Recommenditos`, the repository that held the raw copy, was deleted on 2026-09-29, and the object returns 404. The team therefore publishes no personal data on any DagsHub remote at present, and this decision governs whatever remote EDN-19 settles on rather than an exposure that exists today.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment
- **Response to AI:** Used as input for further analysis
- **Assessment of the AI contribution:** AI found that the privacy premise of the raw-data decision was false, using a control repository to show that anonymous refusal does not distinguish public from private, and put both ways out to Lukas without recommending either, since the trade-off is about how open the team wants to be rather than a technical question. Lukas decided to keep the repository public. The finding is what changed the outcome here; the decision itself was not AI's to make.
- **AI interaction evidence:** Claude Code session on 2026-09-29, prompts translated from German: after the finding was presented with the two options ("either mark switches it to private ... or you keep it public and write that honestly into the raw-data entry"), Lukas answered "we keep it public".
- **Other evidence:** EDN-19, EDN-25, [PR #26](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/26), [PR #27](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/27).
- **In LaTeX:** no

### EDN-21: Report the upstream licence contradiction instead of resolving it

- **Date:** 2026-09-29
- **Milestone:** M1: Project Inception
- **Activity / Topic:** Data Licensing and Provenance
- **Participants:** @lukas2510
- **Decision:** The dataset card reports both licence statements that the Zenodo record makes about the AutoScout24 dataset, and names the contradiction, rather than picking one or asking the author to resolve it. The YAML frontmatter carries `license: mit` because that is the record's structured field, and the body explains the discrepancy.
- **Alternatives considered:**
  - **Option A (chosen): report both statements and flag the contradiction.**
    Pros: it is the only accurate description of what the source actually says; it costs nothing, because both readings permit non-commercial academic use; it warns anyone who later reuses our work for a commercial purpose, which is exactly where the difference would start to matter; it demonstrates the provenance care the course rewards.
    Cons: a reader has to absorb an ambiguity rather than a single answer; the frontmatter still has to commit to one identifier, so the card is slightly inconsistent with itself unless the body is read.
  - **Option B: report MIT only.**
    Pros: simplest; matches the machine-readable metadata field that tooling and aggregators would read; MIT is the more permissive of the two, so it cannot under-claim our own rights.
    Cons: silently discards a restriction the author wrote on the same record; would mislead a future reader who wanted to use the data commercially; asserts a certainty we do not have.
  - **Option C: report the restrictive prose only.**
    Pros: the conservative reading, so it cannot over-claim rights.
    Cons: contradicts the structured licence field; "research, educational or analytical" is not a recognised licence, so it is not expressible in the card's frontmatter; over-restricts what may genuinely be MIT.
  - **Option D: ask the author to resolve it before finishing the card.**
    Pros: would produce a definitive answer and a citable clarification.
    Cons: blocks a sprint deliverable on a stranger's response time with no deadline; disproportionate for a university project whose use is permitted under either reading.
- **Rationale:** The contradiction is in the source, not in our reading of it: the Zenodo record's structured licence field says MIT while the author's description on the same record says "You are welcome to use this dataset for research, educational, or analytical purposes". Since our use is non-commercial academic work, both readings permit it, so resolving the ambiguity would change nothing about what we are allowed to do. Documenting it honestly is therefore strictly better than choosing a side, and materially better than blocking the deliverable on an email. Option D was explicitly declined by the team as disproportionate for the size of the project.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI found the contradiction while re-validating PR #22, and in doing so corrected its own earlier review, which had asserted that the card's restrictive wording was simply wrong and contradicted the MIT licence recorded in the project brief. Fetching the Zenodo API record showed that both statements come from the same source, so the card's author had not made a mistake and the project brief's flat "License: MIT" was itself incomplete. AI laid out the four options above and recommended option A. The team accepted the recommendation but rejected the accompanying suggestion to contact the author, on the grounds that it is disproportionate for a university project.
- **AI interaction evidence:** Claude Code session on 2026-09-29 while reviewing PR #22. AI fetched `https://zenodo.org/api/records/17643343` and reported that `metadata.license.id` is `mit-license` while the description text welcomes use for research, education and analysis, which reads as narrower than MIT; it then flagged the choice as EDN-worthy rather than deciding it. Lukas replied, in German, that the team should not be pedantic and should simply take the recommendation, since this is a small university-course project, which settled option A and ruled out option D.
- **Other evidence:** [PR #22](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/22), the Licensing section of [the dataset card](../docs/docs/dataset-card.md), Zenodo record [10.5281/zenodo.17643343](https://doi.org/10.5281/zenodo.17643343).
- **In LaTeX:** no

### EDN-22: Drop listings registered after the age reference date

> **Evidence corrected on 2026-09-29.** The decision said preprocessing drops "the 164 listings". 164 is the raw-file count;
> preprocessing runs after scoping, so it removes 27 of them. The alternatives below already said 26, measured after deduplication.
> The decision itself is unchanged.

- **Date:** 2026-09-29
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Data Validation and Preprocessing
- **Participants:** @lukas2510
- **Decision:** Preprocessing drops every listing whose `registration_date` is after the age reference date (the 2025-11-08 snapshot date in training, the request date in serving). There are 164 such listings in the raw file, of which **27** survive the EDN-04 scope and the training price range and so are the ones preprocessing actually removes; 26 remain if deduplication runs first, the difference being one duplicate row. The Great Expectations suite asserts `registration_date <= reference date` on the processed data as a hard expectation, and the same rule on the raw data with `mostly=0.99`, so a future scrape that suddenly carries many such rows is flagged instead of silently cleaned away.
- **Alternatives considered:**
  - **Option A (chosen): drop the rows.**
    Pros: 164 of 118,382 raw rows is 0.14 %, and only 26 of them survive the EDN-04 scope, so nothing measurable is lost; makes `age >= 0` a genuine invariant that the pipeline, the tests and the API contract can all rely on; no special case anywhere in the feature code.
    Cons: throws away the other, possibly correct attributes of those listings; the rule has to live in preprocessing, so the raw expectation has to tolerate what preprocessing removes.
  - **Option B: clamp the age at zero.**
    Pros: keeps the rows and their remaining attributes; a single `max(0, ...)` in the feature code.
    Cons: invents a registration date that the source does not support, and does so for rows whose date is demonstrably wrong (the latest is 2026-11-01, almost a year after the snapshot); a clamped zero is indistinguishable from a genuinely new car, which is exactly the group EDN-04 scopes out.
  - **Option C: keep the rows with a negative age.**
    Pros: no data loss, no cleaning rule; gradient boosting tolerates the values.
    Cons: the API would have to accept a negative age too, or training and serving would disagree; removes the cheapest available sanity check on the most important feature.
  - **Option D: repair the date from `production_year`.**
    Pros: would keep the rows with a defensible value.
    Cons: `production_year` is only 19 % filled, so it repairs almost none of them; a repair rule needs its own validation for a group this small.
- **Rationale:** For the rows this decision actually governs, a registration date after the snapshot is a data-entry error rather than a rare but real phenomenon: 137 of the 164 sit in a single month, January 2026, which looks like a year typo. The reading is weaker for the 114 new cars among them, where a dealer entering a planned first-registration date is plausible, but those fall outside the used-car scope before preprocessing runs; only 26 of the 164 reach it. At that share the cost of dropping them is nil, while every alternative either fabricates a value or gives up the `age >= 0` invariant. Bounding the raw data with `mostly=0.99` rather than a hard rule keeps the raw suite honest about what the source actually contains while still catching a systematic worsening, which matters because the `ES` holdout and the M6 drift scenario re-run these checks on data we have not profiled.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI profiled the 164 rows, laid out the four options above and recommended dropping them, together with the split between a hard expectation on the processed data and a `mostly` bound on the raw data. Lukas accepted the recommendation because the dropped share is negligible and the invariant is worth more than the rows.
- **AI interaction evidence:** Claude Code session on 2026-09-29 while analysing [issue #25](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/25): AI presented the options as a decision question, Lukas chose "drop them".
- **Other evidence:** [Issue #25](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/25), [PR #24](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/24), [problem specification](../docs/docs/problem-spec.md) section 2.
- **In LaTeX:** no

### EDN-23: Read the condition flags as one-sided assertions instead of encoding them three-valued

- **Date:** 2026-09-29
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Feature Encoding and Data Validation
- **Participants:** @lukas2510
- **Decision:** `has_full_service_history`, `non_smoking` and `is_rental` stay plain booleans and are read as "the seller asserted this" versus "the seller did not assert this", never as "yes" versus "no". No three-valued (true / false / unknown) encoding is built and no `False` is mapped to missing. `is_used` is excluded entirely because `offer_type` already carries the same information reliably. The Great Expectations suite only checks that the columns are boolean and non-null.
- **Alternatives considered:**
  - **Option A (chosen): keep them binary, fix the interpretation.**
    Pros: no pipeline, API or serialisation complexity; loses nothing, because the ambiguity is in the source and no encoding can resolve it; the risk it addresses (somebody reading a `False` as a denial) is a documentation and naming problem, so it is solved where it actually lives, in the problem spec, the dataset card and the model card.
    Cons: the column name still reads like a two-sided statement, so the documentation has to carry the caveat; a reader who skips it can still misread the feature.
  - **Option B: encode three-valued, or map `False` to missing.**
    Pros: makes the ambiguity visible in the data itself rather than only in prose; the shape a careful reviewer expects.
    Cons: it is not actually three-valued, because we cannot tell a real `False` from an unknown, so the third state would be empty and the encoding would misrepresent the problem as solved; for a gradient-boosted tree a column holding only `{True, NaN}` splits exactly like `{True, False}`, so the model gains no information at all; adds an extra state to the API schema, the training pipeline and the expectations for that zero gain.
  - **Option C: drop all affected flags.**
    Pros: the easiest position to defend; no ambiguous column ever reaches the model.
    Cons: discards the half of the signal that is reliable, since a `True` is a real assertion; full service history and non-smoking are exactly the kind of attributes that move a used-car price.
  - **Option D: recover the true value from `description` or the equipment lists.**
    Pros: would turn the flags into genuine two-sided features.
    Cons: needs multilingual free-text parsing, which the problem spec puts out of scope; produces labels with no ground truth to validate against.
- **Rationale:** The flags are one-sided by construction: a listing form offers a checkbox, so a tick is evidence and an empty box is the absence of evidence. 14,744 rows flagged as neither used, new nor pre-registered, and 18,108 scoped rows with `is_used = False` while `offer_type = U`, are the direct measurement of that. Since no information exists to separate "no" from "not stated", every encoding choice carries identical information, and the one that costs nothing wins. This deliberately overrides the original proposal in issue #25 to encode the flags three-valued.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** The three-valued encoding was AI's own earlier proposal, written into issue #25. On re-reading the issue, AI argued against it: it showed that `{True, NaN}` and `{True, False}` are informationally identical for the tree models we use, so the encoding would add complexity without any effect on the model, and recommended fixing the interpretation instead. Lukas accepted the reversal because the argument was concrete rather than stylistic.
- **AI interaction evidence:** Claude Code session on 2026-09-29 analysing [issue #25](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/25): AI presented the options as a decision question, Lukas chose "keep them binary".
- **Other evidence:** [Issue #25](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/25), [PR #24](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/24), [problem specification](../docs/docs/problem-spec.md) section 4.
- **In LaTeX:** no

### EDN-24: Keep the pre-registered exclusion although the flag behind it is unreliable

- **Date:** 2026-09-29
- **Milestone:** M1: Project Inception
- **Activity / Topic:** Problem Specification (model scope)
- **Participants:** @lukas2510
- **Decision:** The scope of EDN-04 stands: the pipeline keeps excluding listings with `is_preregistered = True` and the API keeps rejecting pre-registered cars. Because a `False` in that flag can also mean "not stated" (EDN-23), an unknown residue of unmarked pre-registered listings stays in the training data; this is recorded as a known limitation in the problem spec instead of being worked around, and is to be carried into the dataset card and the model card. `is_preregistered` is therefore not a feature: a column that defines the scope filter cannot also be a model input.
- **Alternatives considered:**
  - **Option A (chosen): keep the exclusion, document the residue.**
    Pros: keeps the product scope of EDN-04 ("used-car price") intact and the API rule unchanged; removes the 3,695 listings that are demonstrably pre-registered; states the limitation honestly instead of implying a clean filter.
    Cons: the residue cannot be quantified, only bounded by the 14,744 unflagged rows; a small amount of out-of-scope data stays in training.
  - **Option B: revise EDN-04 and keep pre-registered cars in scope, with the flag as a feature.**
    Pros: more data; a pre-registered car really is priced differently, so the flag carries genuine signal; no contradiction between filter and feature.
    Cons: reopens a settled scope decision whose deciding argument was product clarity, not accuracy (EDN-04 measured 6.8 % vs 6.9 % MdAPE, so almost nothing is at stake); the API would have to accept and explain a vehicle condition the component is not meant to price; the feature would be exactly as unreliable as the filter, because the same `False` ambiguity applies.
  - **Option C: detect unmarked pre-registered listings with a heuristic and drop them too.**
    Pros: would shrink the residue.
    Cons: a heuristic on recent registration and low mileage has no ground truth to validate against, so it would trade a known, documented residue for an unknown number of wrongly dropped ordinary used cars.
- **Rationale:** An earlier profiling run suggested promoting `is_preregistered` into the extended feature set, because it is `True` in 3,695 of the 113,708 rows that survive a scoping by `offer_type` and `vehicle_type`. That run simply did not apply the pre-registered filter, so the finding contradicted EDN-04 rather than extending it, and the suggestion was withdrawn. Of the remaining options, reopening the scope buys almost no accuracy (EDN-04's own measurement) while costing the product clarity that decided EDN-04 in the first place, and a heuristic would invent labels. Documenting a bounded, honest limitation is the proportionate answer.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI found the contradiction between its own change in PR #24 and EDN-04 while checking whether issue #25 was still current, and reported that the scope filter itself, not only the feature set, is affected by the unreliable flag. It laid out the three options and recommended keeping EDN-04. Lukas accepted it, because reopening a settled scope decision for an effect EDN-04 had already measured as negligible is not worth it.
- **AI interaction evidence:** Claude Code session on 2026-09-29 analysing [issue #25](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/25): AI reported the contradiction unprompted and presented the options as a decision question; Lukas chose to keep EDN-04.
- **Other evidence:** [Issue #25](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/25), [PR #24](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/24), EDN-04, [problem specification](../docs/docs/problem-spec.md) section 2.
- **In LaTeX:** no

### EDN-25: Raw data acquisition: track with `dvc add` and push to our DagsHub remote

- **Date:** 2026-09-26
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Data Versioning
- **Participants:** @W11W11W11, confirmed by @lukas2510 on 2026-09-29 when the premise below was corrected. Pending confirmation by the full team at the next sprint review.
- **Decision:** Amends EDN-07, which had proposed importing the raw file from Zenodo and never pushing it to our own remote.
  Download the raw file `autoscout24_dataset_20251108.csv` once from Zenodo, track it with `dvc add data/raw/autoscout24_dataset_20251108.csv` and push it to our DagsHub remote.
  Teammates get it with `dvc pull`; the PII columns are removed later in preprocessing.
  The remote is public (EDN-20), so this re-publishes the raw file's PII columns under the team's own name; that is accepted, with the reasoning recorded in EDN-20.
- **Alternatives considered:**
  - **Option A: `dvc import-url` from Zenodo.** The `.dvc` file records the Zenodo URL and the file hash; the raw file is fetched from Zenodo and never stored on our remote.
    Pros: no second copy of the raw PII (`vin`, `street`, `seller_company_name`, coordinates); provenance is recorded in the pointer file itself.
    Cons: every fresh setup downloads 548.6 MB from Zenodo and depends on it being online; Zenodo sends no ETag, so `dvc update` change detection is less reliable; differs from the course demo, which uses `dvc add`.
  - **Option B (chosen): `dvc add` plus `dvc push` to DagsHub.**
    Pros: the workflow of the course demo and of issue #3; one place to pull all data from, independent of Zenodo; fast pulls.
    Cons: we store a copy of the raw PII ourselves, on a remote that is public, so the team becomes a publisher of that personal data in its own right.
  - **Option C: `dvc add` plus push to a remote made private for this purpose.**
    Pros: the raw PII would sit behind an access wall.
    Cons: same access overhead as B, plus a manual collaborator entry for every grader and supervisor; it buys little real protection, because the identical file stays publicly downloadable from Zenodo either way (EDN-20).
- **Rationale:** B follows the DVC workflow the course demo prescribes; `docs/dvc-demo.md` there teaches `dvc add` on the raw file, and its `dvc import`/`dvc get` commands are for data coming from another DVC or Git repository rather than from an arbitrary URL.
  The remote is public, so pushing the raw file re-publishes its PII columns under the team's own name. We accept that because the identical file is already publicly downloadable from the pinned Zenodo DOI under the same licence, and because the marginal exposure is small while the access cost of a private remote for graders and supervisors would be real (EDN-20).
  Option A was also weaker than it looked on the mechanism itself: the Zenodo URL answers a HEAD request with neither an `ETag` nor a `Content-MD5` header, and `dvc import-url` derives its change detection from one of those, so the mechanism was never verified against this source.
  The MD5 of our copy matches the checksum Zenodo publishes (`b23a122cc51baf7de39f449193ff0d28`), so provenance stays verifiable either way.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Rejected
- **Assessment of the AI contribution:** AI (Claude Code) checked the Zenodo endpoint (size, MD5, no ETag), laid out options A to C and recommended A to keep the raw PII off our infrastructure. Mark chose B because it follows the course demo.
  After the choice, AI checked the DagsHub repository's visibility, reported it as private, and recorded that as the mitigation for the main risk it had raised against B. That check was wrong: DagsHub refuses anonymous access to every repository, public ones included, so the observation had no discriminating power, and the authenticated API reports `private: false`.
  The error is recorded rather than quietly fixed, because the decision stands while the reason given for it did not: B is still the right option, but on the grounds in the rationale above rather than on a privacy mitigation that never existed. Verified on 2026-09-29 against a known-public control repository and a repository that does not exist, both of which answer anonymously exactly like ours.
- **AI interaction evidence:** Claude Code session on 2026-09-26 while working on issue #3: prompt "How should the raw AutoScout24 file get into DVC (PII handling)?" with options A to C and the recommendation for A; Mark chose B, reason "because it follows the demo".
- **Other evidence:** [issue #3](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/3), pointer file `data/raw/autoscout24_dataset_20251108.csv.dvc`, [data versioning conventions](../docs/docs/data-versioning.md), project brief §3.3, EDN-07, EDN-19, EDN-20.
### EDN-26: NFR-11's window is 1,000 requests, and `model` is excluded from the drift comparison

> **Amended by EDN-29** (2026-09-29): the drift job runs at a significance level of 0.005, not 0.05.
> The configuration below was measured on a scope that EDN-22 has since narrowed, and on one split; re-measured, its
> control clause does not hold at 0.05.

- **Date:** 2026-09-29
- **Milestone:** M6: Monitoring
- **Activity / Topic:** Monitoring, Requirements
- **Participants:** @lukas2510
- **Decision:** Amends EDN-14. The drift job compares windows of **1,000** requests, not 100, against the 10,000-listing reference, and leaves **`model`** out of the comparison. The other clauses of NFR-11 are unchanged: the `ES` replay must be flagged, at least three features other than `country_code` must be named, and at most 1 of 20 i.i.d. control windows may be flagged.
- **Alternatives considered:**
  - **Option A (chosen): window 1,000 and `model` excluded.**
    Pros: the only configuration measured to satisfy all three clauses as written. Both changes are independently justified: three drifted features do not reliably co-occur in a 100-row window, and a 349-level chi-square against such a window is under-powered rather than merely noisy.
    Cons: the M6 drift demonstration needs 1,000 replayed requests instead of 100, so the monitoring section of the report shows a slower signal.
  - **Option B: window 1,000, keep `model`, relax the control clause to at most 2 in 20.**
    Pros: no feature is removed, so the monitoring watches everything it sees.
    Cons: weakens a promise instead of fixing its cause, and keeps a test whose cell counts do not support the chi-square.
  - **Option C: keep window 100 and relax both clauses (at least one changed feature, at most 3 false alarms in 20).**
    Pros: keeps the fast 100-request demonstration.
    Cons: two promises weakened at once; 3 false alarms in 20 windows is hard to defend for a monitoring requirement.
  - **Option D: keep NFR-11 as EDN-14 wrote it.**
    Cons: measured to fail. At window 100 with all features the control flags 5.4 of 20 windows, and the three-feature clause holds in only 81.5 % of trials.
- **Rationale:** NFR-11 as written was never measured in one configuration: EDN-14 took its `ES` half from `nfr11_check.py` (window 100, full reference) and its control half from `nfr11_diag.py` (window 1,000, reference capped at 10,000), and the latter's output was never committed. Measuring both halves together (`nfr11_model_excluded.py`, 200 trials per cell, 10,000-listing reference, i.i.d. control) gives:

  | Configuration | `ES` flagged | three features besides `country_code` | control windows flagged of 20 |
  |---|---|---|---|
  | all features, window 100 | 1.000 | 81.5 % of trials, minimum 1 | 5.4 |
  | `model` excluded, window 100 | 1.000 | 79.5 % of trials, minimum 1 | 2.2 |
  | all features, window 1,000 | 1.000 | 100 %, minimum 13 | 1.4 |
  | **`model` excluded, window 1,000** | **1.000** | **100 %, minimum 12** | **0.7** |

  Neither change suffices alone. Excluding `model` roughly halves the false alarms at both window sizes, because at window 100 it alone accounts for 0.190 of the 0.270 flag rate, but it does not reach 1 in 20 and does not help the three-feature clause at all. Enlarging the window fixes the three-feature clause and most of the false alarms, but with `model` kept the control still flags 1.4 of 20. Only the combination satisfies every clause, so NFR-11 now states a configuration that has been demonstrated rather than assembled from two runs.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** A reviewing agent reported that the control clause fails at window 100 and proposed stating window 1,000. A second agent argued that the window was the wrong lever, because the false alarms come from `model`, and recommended excluding it while keeping window 100; Lukas chose that option. The measurement then refuted it: the control improves from 5.4 to 2.2 flagged windows of 20 but stays above the promised 1, and the three-feature clause fails in one run in five at window 100 either way. Two claims in that recommendation were also wrong and are recorded rather than quietly dropped: `model` was described as having roughly 10,000 levels, where it has 349 in the reference (84,171 is `model_version`, a different column), and the reviewing agent's control figure at window 1,000 was 0.6 of 20 against the 1.4 measured here, a difference that decides whether the clause holds with `model` kept. The argument that a 349-level chi-square against a 100-row window is under-powered survived the measurement; its effect was simply smaller than claimed. Lukas then chose the combination, which is the only configuration that was demonstrated to work.
- **AI interaction evidence:** Claude Code session on 2026-09-29: a peer agent's review of PR #19 reported the control failure and proposed window 1,000; this session verified the finding by inspection, argued for excluding `model` instead, and Lukas chose that. After the measurement refuted it, this session reported the four measured configurations and recommended the combination, which Lukas chose.
- **Other evidence:** [requirements](../docs/docs/requirements.md) NFR-11; [specification](../docs/docs/specification.md) NFR-11; [reports/analysis/](analysis/) (`nfr11_model_excluded.py`, `nfr11_model_excluded_results.json`, run of 2026-09-29); [EDN-14](#edn-14-nfr-11s-drift-control-is-an-iid-sample-not-a-seller-grouped-one); [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
### EDN-27: Separate the requirements from their specification, keeping one set of FR/NFR IDs

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

- **In LaTeX:** no

### EDN-28: One module per DVC stage, deviating from the flat Cookiecutter layout

- **Date:** 2026-09-29
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Project Structure
- **Participants:** @lukas2510
- **Decision:** The pipeline code is one module per DVC stage, `recommenditos/data/{download_raw_dataset,preprocess,gx_context_configuration,validate_data,split_data,build_features}.py` and `recommenditos/modeling/{train,evaluate}.py`, instead of the flat Cookiecutter Data Science layout of `dataset.py`, `features.py` and `modeling/train.py` that the template generates. This is the structure the course demo repository uses.
- **Alternatives considered:**
  - **Option A (chosen): one module per stage, as in the teachers' demo.**
    Pros: it is the layout the course demonstrates, so it needs no defending against the graders' own example; each `dvc.yaml` stage declares exactly its own module in `deps`, so a code change reruns that stage and no other; and each ticket owns one file, which is what lets five people build a sequential pipeline in parallel without colliding.
    Cons: a deviation from the prescribed template, which has to be justified in the report.
  - **Option B: keep the flat Cookiecutter layout as generated.**
    Pros: zero deviation from the prescribed template.
    Cons: download, preprocessing and the split would share `dataset.py`, so three people would edit one file in the same week; `deps` would be coarse, so any edit to that file reruns every stage that lists it, which weakens exactly the reproducibility property the milestone is about.
  - **Option C: keep the Cookiecutter file names and split only where a collision forces it.**
    Pros: a smaller deviation.
    Cons: neither the template nor the demo, so the report would have to justify a third structure that follows nothing; and the split boundary would be decided by whoever hits the collision first rather than by the pipeline.
- **Rationale:** Cookiecutter Data Science is prescribed "justified deviations allowed", and the teachers' own demo repository deviates in exactly this way, which makes it the best-supported choice rather than a liberty we take. The technical argument is the DVC dependency graph: a stage's `deps` should name the code that stage actually runs, so that `dvc repro` reruns the minimum. The flat layout cannot express that, because one file backs several stages. The organisational argument is the sprint 2 cut: the pipeline is sequential, so parallel work is only possible if each ticket owns its own files, and one module per stage gives that for free.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** Asked whether the sprint plan matched the course demo repository, AI read the demo's file tree and its `dvc.yaml` rather than relying on the existing notes, and reported that the demo splits one module per stage while our repository still carried the flat template. It connected that to the parallelisation problem the sprint was being cut around, laid out the three options with the trade-offs above and recommended A. Lukas reviewed the comparison and chose A. The contribution was useful mainly because it checked the demo directly instead of arguing from the template's documentation.
- **AI interaction evidence:** Claude Code session on 2026-09-29 during sprint 2 planning: Lukas gave the demo repository's URL and asked whether the planned milestones and tickets fit it; AI fetched its tree and `dvc.yaml`, reported what to adopt and what we deliberately add, and presented the three layout options; Lukas chose the demo layout.
- **Other evidence:** [sprint 2 planning notes](../docs/docs/scrum/sprints.md); [issue #32](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/32); [references/course-demos.md](../references/course-demos.md); [PR #45](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/45).
- **In LaTeX:** no

### EDN-29: The drift job runs at a significance level of 0.005, not 0.05

- **Date:** 2026-09-29
- **Milestone:** M6: Monitoring
- **Activity / Topic:** Monitoring, Requirements
- **Participants:** @lukas2510
- **Decision:** Amends EDN-26. The drift job tests each feature at a Bonferroni-corrected threshold of `0.005 / number of features` instead of the conventional `0.05 / number of features`. Everything else stays: windows of 1,000 requests, a 10,000-listing reference, `model` excluded, an i.i.d. control of which at most 1 of 20 windows may be flagged.
- **Alternatives considered:**
  - **Option A (chosen): lower the significance level to 0.005.**
    Pros: measured to hold with margin. Across three independent i.i.d. splits the control flags between 0 and 0.6 windows of 20 against a promise of at most 1, while the `ES` replay is still flagged in every trial of every split with at least twelve features besides `country_code`. It fixes the cause rather than the symptom: the requirement and the detector's threshold were coupled, and nobody had noticed.
    Cons: a lower threshold means less sensitivity to genuinely subtle drift, which is the monitoring's real job. We have no measurement of that sensitivity, so the cost is real but unquantified.
  - **Option B: significance level 0.01.**
    Pros: the more conventional step down, and it does hold: 0 to 1.0 flagged windows of 20 across the three splits.
    Cons: the maximum sits exactly on the promise, so the requirement would again depend on which split is drawn, which is the failure mode being fixed.
  - **Option C: keep 0.05 and relax the control clause to at most 3 of 20.**
    Pros: no change to the detector; the measured maximum is 2.0, so 3 holds.
    Cons: weakens a monitoring promise to fit a threshold nobody chose deliberately. Three false alarms in twenty windows is hard to defend in the report, and the underlying coupling would stay hidden.
  - **Option D: keep the requirement as EDN-26 wrote it.**
    Cons: measured not to hold. On the scope EDN-22 defines, the control flags 2.3 windows of 20 at window 1,000 with `model` excluded.
- **Rationale:** EDN-26's configuration was measured once, on one i.i.d. split, before EDN-22 narrowed the scope. Re-running it after that change gave 2.3 flagged control windows of 20 instead of 0.7, which 27 removed listings cannot explain: the i.i.d. split is drawn from a permutation whose length depends on the row count, so the second run used a different control sample. The estimate was never stable, and the reason is structural. The drift job flags a window when any feature falls below `P_VAL / n_features`, so by Bonferroni's construction the family-wise false-alarm rate **is** `P_VAL`. At `P_VAL = 0.05` that is 5 %, which is exactly the "at most 1 of 20" the requirement promises, so NFR-11 was asking the test to perform at its own theoretical bound with zero tolerance for estimation error. Lowering the level decouples the two. 0.005 was chosen over 0.01 because at 0.01 the measured maximum is exactly 1.0 of 20, which would leave the requirement depending on the draw again. The `ES` side is unaffected: its p-values are in the order of 1e-300, so no threshold in this range changes whether it is detected.

  Control windows flagged of 20, three splits, window 1,000, `model` excluded:

  | `P_VAL` | min | max | mean | `ES` clause holds in every split |
  |---|---|---|---|---|
  | 0.05 | 0.0 | 2.0 | 1.0 | yes |
  | 0.01 | 0.0 | 1.0 | 0.4 | yes |
  | **0.005** | **0.0** | **0.6** | **0.2** | **yes** |
  | 0.001 | 0.0 | 0.6 | 0.2 | yes |

  Window 100 fails the "at least three changed properties" clause at every level tested, so EDN-26's window decision stands on its own.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI re-ran the committed evidence after EDN-22 landed, found that EDN-26's configuration no longer held, and did not simply restate the number. It identified that the Bonferroni construction makes the family-wise false-alarm rate equal to the significance level, which is what the control clause promises, and wrote a measurement that separates split-to-split variability from the threshold by computing each trial's p-values once and evaluating them at several levels. It recommended 0.005 over 0.01 on the grounds that 0.01's maximum sits exactly on the promise. Lukas delegated the choice and accepted the recommendation. This is the third time a measurement has refuted a written form of NFR-11, which is itself the argument for measuring a requirement before committing to it rather than after.
- **AI interaction evidence:** Claude Code session on 2026-09-29: after the scope correction AI reported that EDN-26's configuration measured 2.3 flagged control windows of 20 instead of 0.7, explained the Bonferroni coupling, ran `nfr11_alpha_sweep.py` over three splits and four significance levels, and recommended 0.005; Lukas replied "ohne team mache das was du recommendest".
- **Other evidence:** [specification](../docs/docs/specification.md) NFR-11; [reports/analysis/](analysis/) (`nfr11_alpha_sweep.py`, `nfr11_alpha_sweep_results.json`, `nfr11_alpha_sweep_results.txt`, run of 2026-09-29); [EDN-26](#edn-26-nfr-11s-window-is-1000-requests-and-model-is-excluded-from-the-drift-comparison); [EDN-22](#edn-22-drop-listings-registered-after-the-age-reference-date); [PR #47](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/47).
- **In LaTeX:** no

### EDN-30: Pipeline configuration lives in `params.yaml`, not in `config.py`

- **Date:** 2026-09-29
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Project Structure, Reproducibility
- **Participants:** @lukas2510
- **Decision:** Everything configurable about the pipeline - the seed, the age reference date, the scope bounds, the split ratios, the feature sets, the model variants and the SC-01 to SC-06 thresholds - lives in `params.yaml` and is declared per stage under `params:` in `dvc.yaml`. Paths stay in `recommenditos/config.py`.
- **Alternatives considered:**
  - **Option A (chosen): a `params.yaml` declared per stage.**
    Pros: `dvc repro` reruns exactly the stages a change affects, because DVC hashes the declared keys; `dvc params diff` shows what changed between two commits, which is evidence the report can cite; `dvc exp` becomes usable without a code change; and a value used by two stages cannot drift, because there is one copy of it.
    Cons: one more file to keep in step with the code, and a parameter typo is caught at run time rather than by the interpreter.
  - **Option B: Python constants in `config.py`, as the course demo does.**
    Pros: it is exactly what the demo shows, so it needs no defending; typos are import errors; no YAML parsing.
    Cons: DVC cannot see a constant, so a changed hyperparameter reruns nothing and `dvc repro` reports the pipeline as up to date while the code says otherwise - which is the opposite of what the milestone is graded on. `dvc exp` cannot sweep it, and the demo's own `config.py` also reads an absolute `ROOT` from `.env` and raises at import without one.
  - **Option C: both, with `config.py` reading `params.yaml`.**
    Pros: one import surface for the stages.
    Cons: the indirection hides which stage reads which key, which is precisely the information `dvc.yaml` needs in order to rerun the minimum.
- **Rationale:** The whole point of M2 is that a change reruns what it should and nothing else. Option B cannot deliver that for anything except code, and parameters are where most changes will happen once the stage tickets land. Paths were deliberately left out of `params.yaml`: they are not experiment variables, and deriving them from `__file__` keeps the project free of machine-specific absolute paths.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI read the demo repository's `dvc.yaml` and `config.py` directly rather than from our notes, reported that the demo declares neither `params` nor `metrics` on any stage, and measured the consequence in a throwaway DVC repository: with a coarse `params:` declaration a single changed hyperparameter retrained every variant, and with a per-item declaration it retrained one. Lukas reviewed the measurement and accepted the recommendation.
- **AI interaction evidence:** Claude Code session on 2026-09-29 while building the pipeline skeleton: AI built a scratch DVC repository, ran `dvc repro` and `dvc status` under both declaration styles, and pasted the outputs showing one stage rerunning instead of three.
- **Other evidence:** [`params.yaml`](../params.yaml); [`dvc.yaml`](../dvc.yaml); [issue #32](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/32).
- **In LaTeX:** no

### EDN-31: The processed-data contract is a hand-written `schema.py`, not Pandera

- **Date:** 2026-09-29
- **Milestone:** M2: Reproducibility (shapes M3: Quality Assurance)
- **Activity / Topic:** Data Validation, Testing Strategy
- **Participants:** @lukas2510
- **Decision:** `recommenditos/schema.py` holds the structural contract - column names, dtypes and nullability for the raw, interim and processed frames - as frozen dataclasses with a `validate` and a `conform` method, written by hand rather than with a validation library. Value rules stay in the Great Expectations suites. Every stage validates after reading and conforms before writing.
- **Alternatives considered:**
  - **Option A (chosen): a hand-written `schema.py`.**
    Pros: no dependency, so the same module can later be imported by the API, which NFR-04 caps at 1 GB and which already has Pydantic for its own request schemas; we control the failure message, and it reports every problem at once with the offending row positions; and `conform` casts to the declared dtypes before each write, which is what stops Parquet's dtype drift between stages - a library that only validates cannot do that. Great Expectations already gives us a declarative vocabulary for value rules, so a second declarative framework would overlap it.
    Cons: about 250 lines we own and test ourselves.
  - **Option B: Pandera.**
    Pros: battle-tested, declarative, less code to own, good lazy error reports as a tidy DataFrame.
    Cons: a third validation system next to Great Expectations and Pydantic; it would travel into the API image; and it does not check the datetime unit at all, so a `datetime64[ns]` declaration would be documentation rather than a guarantee - exactly the failure this contract exists to prevent.
  - **Option C: Great Expectations alone, with no structural contract in code.**
    Pros: one tool, and Data Docs for free.
    Cons: the contract would exist only in a generated `gx/` store whose config carries a fresh UUID per object on every run, so it is awkward to diff and keep honest in git. It also arrives too late: the stage tickets need something to code against in week one, and Great Expectations is a later ticket.
- **Rationale:** Structure and value are different questions. A structural break is a bug in our code and should fail the stage that caused it; a value break is a change in the data and belongs in an expectation suite with a `mostly=` tolerance. Splitting them that way let the contract land before Great Expectations does, which is what unblocks the parallel work. The decisive technical point is `conform`: Parquet preserves whatever dtype it is handed rather than normalising it, so without an explicit cast at each boundary the frames drift apart and the stage that notices is never the stage that caused it. Because `conform` selects only the contract's columns, the PII columns NFR-08 forbids are kept out structurally rather than by every stage remembering to drop them.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI ran the compatibility experiments rather than arguing from documentation: it installed Pandera, Great Expectations and pyarrow against our pandas 3.0.6 and measured what survives a Parquet round trip, finding that an `object` string column silently becomes `str` across the boundary (so a schema declaring `object` passes upstream and fails downstream), that Pandera ignores the datetime unit entirely, and that the Great Expectations store regenerates with different UUIDs on every run. Those three measurements are what decided the option, and none of them was predictable from the documentation. Lukas was given the three options with these findings and chose A.
- **AI interaction evidence:** Claude Code session on 2026-09-29: AI ran the probes in a scratch environment, reported the before/after dtype table for the Parquet round trip and the sixteen declared-versus-actual datetime combinations Pandera accepts, and recommended the hand-written contract; Lukas chose it.
- **Other evidence:** [`recommenditos/schema.py`](../recommenditos/schema.py); [`tests/test_schema.py`](../tests/test_schema.py); [issue #32](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/32).
- **In LaTeX:** no

### EDN-32: Split proportions 60/10/10/20 and the project seed

- **Date:** 2026-09-29
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Evaluation Protocol
- **Participants:** @lukas2510
- **Decision:** The non-`ES` listings are split 60 % train, 10 % validation, 10 % calibration and 20 % test, grouped by `seller_group_id`, with the project seed `20251108` (the snapshot date). Both are pinned in `params.yaml`.
- **Alternatives considered:**
  - **Option A (chosen): 60/10/10/20.**
    Pros: keeps the 20 % test share that the model card's SC-04 analysis and `reports/analysis/make_support_results.txt` already assume, so the published evidence stays true without a re-run. Calibration and validation get about 9,800 rows each, which is ample for conformal calibration and for early stopping.
    Cons: the smallest training set of the three options, about 58,700 rows.
  - **Option B: 70/10/10/10.**
    Pros: the most training data.
    Cons: halves the test set. `make_support_results.txt` gives the test share below which each supported make falls under SC-04's 500-row bar: Volvo at 14.9 % and Suzuki at 13.0 %. At a 10 % test share both drop out, so SC-04 would silently stop checking two of the eleven supported makes and the model card would have to be corrected.
  - **Option C: 64/8/8/20.**
    Pros: the same 20 % test share with slightly more training data.
    Cons: a less round number to explain in the report, for about 3,900 extra training rows.
- **Rationale:** The test share is the binding constraint, not the training share. SC-04 only checks a segment once it holds 500 test rows, so the proportions decide how many makes the quality gate actually covers, and the model card already states which four of the eleven supported makes fall short at 20 %. Choosing anything below about 15 % would quietly enlarge that set, which is the kind of change that is invisible until someone re-reads the analysis. The seed is the snapshot date rather than a random number so that it is recognisable in a log and obviously not tuned.
- **AI involvement:** Information seeking, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI found the coupling the ticket did not mention: it read `make_support_results.txt`, noticed that its sensitivity table was computed at an assumed 20 % test share while the four-way split was still unpinned, and pointed out that pinning a smaller test share would invalidate a number already published in the model card. It laid out the three options against that constraint and recommended A. Lukas accepted.
- **AI interaction evidence:** Claude Code session on 2026-09-29: asked to pin the split, AI reported the per-make sensitivity thresholds from the committed analysis output and framed the three options around the 15 % floor they imply.
- **Other evidence:** [`params.yaml`](../params.yaml); [reports/analysis/make_support_results.txt](analysis/make_support_results.txt); [dataset card](../docs/docs/dataset-card.md); [issue #32](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/32).
- **In LaTeX:** no

### EDN-33: A generated synthetic fixture, and a `download.source` parameter so the skeleton runs without the raw file

- **Date:** 2026-09-29
- **Milestone:** M2: Reproducibility (shapes M3: Quality Assurance)
- **Activity / Topic:** Testing Strategy, Data Acquisition
- **Participants:** @lukas2510
- **Decision:** `recommenditos/data/synthetic.py` generates a schema-valid stand-in for the raw snapshot from public catalogues and a seed, never by sampling the real file. The test suite builds every fixture from it, and the `download` stage produces it while `download.source` in `params.yaml` is `synthetic`, so `dvc repro` is green on a clean clone with no credentials and no 548 MB download.
- **Alternatives considered:**
  - **Option A (chosen): generate, and gate the source on a parameter.**
    Pros: `pytest` and `dvc repro` both run with no data access, which is what lets four people build pipeline stages at the same time; no personal data is committed to Git, so NFR-08 is satisfied by construction rather than by a rule someone has to remember; every edge case the pipeline rules exist for is guaranteed present, so a stage test asserts rather than hopes; and because the source is a parameter it is recorded in `dvc.lock` and shows up in `dvc params diff`, so a pipeline still running on synthetic data cannot pass unnoticed.
    Cons: the generator is code we maintain, and a pipeline on `main` can produce fabricated data until the download ticket lands.
  - **Option B: sample the fixture from the real file and commit it.**
    Pros: real distributions for free, and no generator to write.
    Cons: the raw file carries PII (street, postcode, coordinates, seller company name). EDN-25 accepts that file living behind a DVC pointer on our remote, because the identical file is public on Zenodo under the same licence; a *committed sample* is a different thing, because it puts those rows in the Git history of a repository the whole cohort can read and cannot be removed later. `.gitignore` also excludes everything under `data/`, so it could not be committed without weakening that rule.
  - **Option C: no fixture; `dvc repro` and the tests run on the real file.**
    Pros: nothing is fabricated at any point.
    Cons: every test and every `dvc repro` would need `dvc pull` of 548 MB and DagsHub access, so the working agreements' requirement that a ticket have a local oracle would hold for nobody until the download ticket lands - which is the bottleneck the whole sprint was cut to avoid.
- **Rationale:** The two properties that matter are that nobody is blocked and that no personal data enters the Git history, and only a generated fixture gives both. Making the source a parameter rather than a hard-coded branch is what keeps the option honest: the choice is versioned in `dvc.lock`, it is visible in a params diff, and the stage refuses any other value with a `NotImplementedError` naming the ticket that implements it, so nobody can flip it early and silently get synthetic data. The fixture deliberately keeps the real make skew and gives each seller several listings, because a seller-grouped split on a flat distribution would be indistinguishable from a random one and the split ticket's central invariant would be untestable.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI framed the trade-off as the one between the team being unblocked and fabricated data reaching `main`, and proposed the parameter as the way to keep both, rather than presenting the synthetic stub as free. It also derived the fixture's edge-case list from the individual stage tickets so that each ticket's rule has something to act on, and pointed out the skew requirement that a naive uniform generator would have missed. Lukas reviewed the three options and accepted A.

  The first implementation was wrong in a way worth recording, because it is the failure mode this project's working agreements warn about. Every edge-case row was built as a copy of one body row, which made most of them duplicates of it on the deduplication key and gave all of them that row's country. Preprocessing would therefore have deleted several of the cases the fixture exists to provide, and at the project's own seed the whole block inherited `ES` and would have been diverted into the holdout, so the train, validation, calibration and test frames that four tickets build against would have contained none of them. Nothing in the first round of tests could see it: they asserted that each case was present in the raw frame, which was true. A second agent, asked to attack the change rather than describe it, reproduced the fault, and the fix now carries a test that asserts each case *survives* the preprocessing rules rather than merely existing. The lesson is the one already in the working agreements: an agent's claim decides nothing until it is checked, and the check has to be at the point where the value is consumed.
- **AI interaction evidence:** Claude Code session on 2026-09-29: AI presented the three options with the parallel-work and PII consequences of each and recommended A; Lukas chose it. The generated distribution was then checked against the dataset card's published make and country shares.
- **Other evidence:** [`recommenditos/data/synthetic.py`](../recommenditos/data/synthetic.py); [`tests/test_data.py`](../tests/test_data.py); [EDN-25](#edn-25-raw-data-acquisition-track-with-dvc-add-and-push-to-our-dagshub-remote); [issue #32](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/32).
- **In LaTeX:** no


### EDN-34: The derived raw Parquet keeps its PII columns and is pushed like any other stage output

- **Date:** 2026-09-30
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Data Versioning, Personal Data
- **Participants:** @lukas2510. Open to revisit at the next sprint review, like EDN-25.
- **Decision:** `data/raw/listings.parquet`, the output of the `download` stage, keeps the seven PII columns of the published file and is cached and pushed to the DagsHub remote like any other stage output. No `push: false`, and no PII removal before preprocessing. Our public remote therefore holds the same personal data twice, once as the CSV of EDN-25 and once as this Parquet.
- **Alternatives considered:**
  - **Option A (chosen): push it, which is DVC's default for a stage output.**
    Pros: nothing to build. The marginal exposure is close to nothing, because the identical data already sits on the same public remote as a CSV (EDN-25) and on Zenodo under the same licence, so anyone who wants those columns already has them. `dvc pull` stays sufficient for a complete working state, and the pipeline's first artefact is byte-identical for everyone because it is pulled rather than regenerated.
    Cons: we hold a second artefact containing the same personal data under our own name. A deletion request, a licence question or a scope change then touches two artefacts instead of one. EDN-25's argument was that *the identical file* is already public, and a format we produced ourselves is not literally that file.
  - **Option B: `push: false` on the output.** DVC keeps it in the local cache only and each person regenerates it from the CSV.
    Pros: no second copy leaves the machine.
    Cons: it moves the pipeline's first artefact from pulled to locally produced. Parquet writing is not guaranteed byte-stable across pyarrow versions, so a teammate on a different version would see every downstream stage rerun, which is exactly the reproducibility property NFR-06 asks for and that the skeleton was just measured to have.
  - **Option C: drop the PII in `download` and build `seller_group_id` there.**
    Pros: the Parquet every stage reads would carry no PII at all, so NFR-08 would hold from the first derived artefact rather than from the interim frame, and the file would still be pushed, so reproducibility would be unaffected.
    Cons: the raw layer stops being a faithful copy of the published file, which is what `RAW_SCHEMA` and the raw expectation suite of #25 exist to describe; and the group-key hashing moves out of #34's first step into #33, so a ticket boundary shifts for a gain that does not change what is publicly reachable.
  - **Option D: no Parquet at all, every stage reads the CSV.**
    Cons: each run parses 548.6 MB of CSV and re-infers its dtypes, which is the cost the stage exists to remove.
- **Rationale:** The question is marginal exposure, not storage: measured on the real file the CSV is 548.6 MB, the Parquet of all 75 columns is 215.3 MB, and the seven PII columns are 3.1 MB of it. Against a copy of the same data that is already public on our own remote and on Zenodo, none of the alternatives changes what a reader can reach; they only change what we have to build and what we risk. B risks the reproducibility we have just demonstrated, and C buys the same nothing in exposure terms while making the raw layer no longer raw. So the honest reading is that A costs the least and hides the least.

  What A does cost is recorded rather than argued away: we become the publisher of that personal data in two artefacts instead of one, and EDN-25's "the identical file is already public" does not literally cover a format we produced ourselves. The entry exists so that a later decision to restrict the remote, or a request to remove the data, finds both artefacts named in one place.

  NFR-08 is unaffected either way. It forbids PII in the processed data, the prediction log, the comparables and the model artefacts; the raw layer is none of those, and preprocessing removes the columns structurally, because `Schema.conform` selects only the columns the interim contract names.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI found the question in the first place, while reviewing the pipeline skeleton: the `download` stage's output had inherited EDN-25's acceptance without anyone noticing that EDN-25 had only ever weighed the CSV. Its first write-up estimated the Parquet at "roughly another 550 MB"; asked to explain the trade-off, it measured instead and reported 215.3 MB, of which the PII columns are 3.1 MB, which moved the argument off storage entirely. It also found, while explaining, a reproducibility flaw in its own earlier suggestion of `push: false` and withdrew it. It recommended C as the clean option and A as defensible; Lukas chose A as the cheapest of the two it stood behind. The modification is that the cost of A is written down here rather than treated as settled by EDN-25.
- **AI interaction evidence:** Claude Code session on 2026-09-29 and 2026-09-30: the question was raised in the review of PR #48, the sizes were measured against the Zenodo file, and the four options were laid out with the reproducibility catch in B; Lukas replied "mache die einfachste variante die du recommendest".
- **Other evidence:** [EDN-25](#edn-25-raw-data-acquisition-track-with-dvc-add-and-push-to-our-dagshub-remote); [EDN-20](#edn-20-the-dagshub-remote-stays-public); [issue #33](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/33); [PR #48](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/48).
- **In LaTeX:** no

### EDN-46: DagsHub credentials live in two gitignored stores, and the token is entered twice

- **Date:** 2026-09-30
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Credential handling, developer onboarding
- **Participants:** Lukas
- **Decision:** A contributor's DagsHub token is stored twice, in two gitignored files: in `.env` as `MLFLOW_TRACKING_PASSWORD`, which MLflow reads from the environment, and in `.dvc/config.local` via `dvc remote modify origin --local password`, which is the only place DVC's HTTP remote reads a password from.
  Nothing derives one store from the other, and `.env.template` deliberately does not list `DAGSHUB_USERNAME` or `DAGSHUB_USER_TOKEN`.
  `recommenditos/config.py` loads `<repo>/.env` by name rather than searching upwards, so the credentials of a checkout are its own.
- **Alternatives considered:**
  - **Option A (chosen): two stores, the token pasted into each, and a template that lists only what the project reads.**
    Pros: no machinery to maintain; both stores are already gitignored, so no token can reach a commit; each tool reads its credentials the way it documents; the template cannot mislead, because every variable in it is read and a test asserts that.
    Cons: the token is entered twice, and the reader has to be told why or it looks like an oversight; the `dvc remote modify` form puts the token in shell history and in `ps`.
  - **Option B: one source of truth in `.env`, plus a script that writes `.dvc/config.local` from it.**
    Pros: one paste; one file to rotate.
    Cons: a piece of project machinery whose whole job is to write a secret to disk on its own initiative, which is a worse thing to own than a second paste; another step that can fail or drift; nothing in the course requires it.
  - **Option C: `.env` only, relying on an environment variable for DVC.**
    Pros: would be the simplest of all, if it existed.
    Cons: it does not. Measured rather than assumed: with `MLFLOW_*`, `DAGSHUB_USERNAME` and `DAGSHUB_USER_TOKEN` all exported, `dvc status -c` still fails with `configuration error - HTTP 'basic' authentication require both 'user' and 'password'`, and DVC 3.67.1's HTTP remote schema offers exactly `user`, `password` and `ask_password` and no environment route at all.
  - **Option D: `ask_password true`, prompting instead of storing.**
    Pros: nothing on disk; nothing in shell history.
    Cons: `dvc_http` calls `getpass`, which needs a terminal, so it prompts on every `dvc pull` and breaks any unattended `dvc repro` or CI job. Documented as the better choice on a shared machine, not as our default.
- **Rationale:** Option C would have been the right answer and is not available, which is what makes the double entry a property of DVC rather than a choice.
  Between A and B, the property that matters is that no token can reach a commit, and both stores already have it; B buys one fewer paste at the price of owning a secret-writing script.
  Listing `DAGSHUB_*` in the template was actively harmful: nothing reads those names, so filling them in configures nothing while reading as though it configured DVC, which is the confusion the getting-started page exists to prevent.
  Scoping `.env` to the repository belongs to the same decision: a bare `load_dotenv()` searches upwards, so a clone nested under another checkout inherited that one's token and looked configured when it was not.
- **AI involvement:** Information seeking, Alternative assessment, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI established by execution that DVC has no environment-variable route, which is the fact the whole decision rests on, and it wrote the two-store walkthrough. An adversarial review of that work, also by AI, then found three things the first pass had asserted rather than checked: that the `.env` scoping was not repository-local, so the review's own "fresh clone" verification had in fact been running on the parent checkout's credentials; that `mlflow.db` was neither gitignored nor prevented; and that the documented newcomer command printed a traceback where the page promised a message. All three were reproduced before being fixed. The modification is that the template lost the `DAGSHUB_*` block, which the first pass had defended as documentation, once the review showed nothing reads it.
- **AI interaction evidence:** Claude Code sessions on 2026-09-30: the DVC environment-variable question was settled by running `dvc status -c` with and without the variables exported; the review findings were each reproduced before any fix, including a probe showing `find_dotenv` resolving to a `.env` three directories above a credential-free worktree.
- **Other evidence:** [issue #42](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/42); [PR #50](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/50); [EDN-19](#edn-19-which-dagshub-repository-the-team-uses-as-dvc-remote-and-mlflow-server); NFR-09 in [the specification](../docs/docs/specification.md).
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
