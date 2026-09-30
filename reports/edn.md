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
>
> **Description corrected on 2026-09-30.** The decision described the 27 as what survives "the EDN-04 scope and the training price range".
> The rule order implemented in issue #34 puts this rule above the price range, so the 27 survive the scope alone.
> The number is the same either way, because all 27 are inside the range; only the description of how it was obtained was wrong.
> The decision itself is unchanged.

- **Date:** 2026-09-29
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Data Validation and Preprocessing
- **Participants:** @lukas2510
- **Decision:** Preprocessing drops every listing whose `registration_date` is after the age reference date (the 2025-11-08 snapshot date in training, the request date in serving). There are 164 such listings in the raw file, of which **27** survive the EDN-04 scope, the one rule the implemented order puts above this one, and so are the ones preprocessing actually removes; all 27 are inside the training price range, which runs after this rule, so the count is the same whichever of the two goes first; 26 remain if deduplication runs first, the difference being one duplicate row. The Great Expectations suite asserts `registration_date <= reference date` on the processed data as a hard expectation, and the same rule on the raw data with `mostly=0.99`, so a future scrape that suddenly carries many such rows is flagged instead of silently cleaned away.
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

> **Amended by EDN-36** (2026-09-30, PR #54): the `dvc add` pointer is retired, the `download` stage acquires the file from Zenodo itself, and no copy of the CSV is hosted by us any more.
> This entry is kept as the record of what was decided on 2026-09-26 and of the reasoning EDN-36 revised; the rejection of `dvc import-url` is the part of it that still stands.

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
- **Other evidence:** [issue #3](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/3), pointer file `data/raw/autoscout24_dataset_20251108.csv.dvc`, [data versioning conventions](../docs/docs/data-versioning.md), project brief §3.3, EDN-07, EDN-19, EDN-20, EDN-36.
- **In LaTeX:** no

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

  Corrected on 2026-09-30 while reviewing [PR #54](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/54), and recorded rather than quietly edited: "it is visible in a params diff" above, and the same claim in option A, is wrong. `dvc params diff` compares the params files of two Git revisions, not the lock against the workspace, so it prints nothing about a workspace running on the synthetic source. What actually makes such a run visible is the committed `dvc.lock`, which records the source every artefact was built from, and `dvc status`, which reports a workspace that disagrees with it. The decision is unaffected: making the source a parameter is still what puts the choice in the lock.
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

### EDN-35: The `download` stage owns `data/raw/`, and the published CSV is a local cache outside the DAG

- **Date:** 2026-09-30
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Data Versioning, Pipeline Design
- **Participants:** @lukas2510. Open to revisit at the next sprint review, like EDN-25.
- **Decision:** The `download` stage owns `data/raw/`.
  Under `download.source: zenodo` it fetches the pinned file from the Zenodo record itself and writes `data/raw/listings.parquet` as its only output, so acquisition is inside the pipeline instead of a manual step beside it.
  The published CSV is kept as a local cache under `data/external/` and is neither a stage `out` nor a `dep`; `download.md5` in `params.yaml` is the dependency that pins the bytes, and the stage refuses to read a file that hashes to anything else.
  The standalone pointer `data/raw/autoscout24_dataset_20251108.csv.dvc` is deleted, which EDN-36 records as the amendment to EDN-25.
- **Alternatives considered:**
  - **Option A: the pointer stays and the stage reads it (issue #33's option A).** `download` takes the `dvc add`-tracked CSV as a `dep` and writes only the Parquet.
    Pros: a fresh setup gets the CSV from DagsHub rather than from Zenodo, so it is fast and independent of Zenodo's uptime; one place to pull all data from; the snapshot's hash sits in a file at the tip of `main`, so recovering it never involves the Git history.
    Cons: acquisition stays a manual `dvc add` outside the pipeline, so "reproduce from a clean clone" is not true of the pipeline alone, which is what the data-versioning page and the course demo describe; we keep re-publishing the CSV's PII columns on a public remote; and DVC hashes every `dep` to answer `dvc status`, so 548.6 MB is read before the pipeline can say whether anything changed.
  - **Option B (chosen): the stage fetches the file, and the CSV is a local cache outside the graph.**
    Pros: the pipeline acquires its own input, so a clean clone reproduces it with `dvc repro` and no manual step; nothing about the CSV is re-published by us; and `dvc status` stays instant, because the stage's deps are code and params only. A clean clone does not even pay the download, because `data/raw/listings.parquet` is a pushed stage output and the stage is up to date after `dvc pull`.
    Cons: the file the stage really reads is not in the DAG, so DVC cannot report anything about it and a reader has to know that `download.md5` is standing in for it; and the repository stops holding a tracked copy of the raw snapshot (see the consequence below).
  - **Option C: declare the CSV as a plain `out` of the stage.**
    Pros: the acquired file would be in the graph, cached and recoverable with `dvc checkout`.
    Cons: DVC removes a stage's outputs before running it, so every `dvc repro download` would re-download 548.6 MB; and a cached `out` puts the raw PII on our remote again, which is the copy EDN-36 exists to stop.
  - **Option D: declare the CSV as a `dep`.**
    Cons: the same 548.6 MB hash per `dvc status` as option A, plus a worse failure mode: a missing `dep` is an error DVC raises before the stage runs, so the stage could never fetch the file it depends on, and a clean clone would be stuck.
  - **Option E: declare it as `persist: true, cache: false` (`dvc stage add --outs-persist-no-cache`, verified present in DVC 3.67.1).**
    Pros: this is the option the first write-up of this decision missed, and it removes the objection to option C outright. `persist: true` means DVC does not delete the file before the run, so it is fetched once and reused, and `cache: false` keeps it out of the cache and off the remote, so there is no second copy of the PII. It would also make `dvc status` honest about the CSV: the file the stage reads would be named in the graph, and a damaged or swapped local copy would be reported by DVC and not only by our own check.
    Cons: a non-cached out is still hashed to build the lock and to answer `dvc status`, so it costs exactly the 548.6 MB read per status call that ruled out options A and D, on every run including the many that never touch the raw layer. It is also not recoverable despite being in the graph: `dvc checkout` cannot restore a file DVC never cached, so the lock would carry a hash with nothing behind it. And it would put a second, DVC-owned copy of a fact `download.md5` already states, so re-pinning the file would mean editing the parameter and refreshing the lock instead of only the parameter.
- **Rationale:** The honest reason the CSV stays out of the graph is not that DVC cannot express "a file the stage fetches and keeps".
  It can, with `persist: true` and `cache: false` (option E), and the first version of this entry claimed a limitation that does not exist.
  The reason is what that expression costs against what it buys.

  What it buys is a `dvc status` that mentions the CSV and reports a local copy that has changed.
  The stage already has that check and gives it more context: it re-hashes the file on every run, not only after a download, and it fails with a message naming the two things that can have happened, a damaged local copy or a genuinely re-pinned upstream file, because those need different answers.
  And because `download.md5` is a parameter, re-pinning is a change DVC reruns on, which is the dependency edge a graph entry would have drawn.

  What it costs is a 548.6 MB read on every `dvc status`, which is the cost that ruled out the `dep` shape in the first place, plus a lock entry with no recoverable object behind it.
  So the property is already covered by the stage while the cost would be paid by everyone on every status call, and the file stays a cache.

  It lives under `data/external/` because that is the third-party slot of the project layout: a copy of a published artefact, reproducible from the DOI and the pinned MD5, with nothing for DVC to version. Keeping it out of `data/raw/` is what lets that directory belong to the pipeline alone.

  **Consequence, recorded rather than argued away.** After this change no `.dvc` file at the tip of the branch references the raw blob.
  The blob is still on DagsHub and in local caches, and the pointer's hash is still in the Git history of the deleted file, so an earlier commit can still `dvc pull` and the snapshot is recoverable.
  But recovering it without Zenodo now means somebody reading a hash out of the Git history, and a workspace-scoped `dvc gc` would delete the blob without asking.
  That is why the rule never to run `dvc gc` without `--all-commits` is written down in the data-versioning page as part of this change.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI implemented issue #33's option B and proposed the refinement that the CSV is a local cache rather than a stage output, with the measurements behind it: the CSV is 548.6 MB, the Parquet 215.3 MB, the stage 88 s cold and 66 s warm.
  Its option table was wrong in one way that mattered: it argued the CSV could only be an `out` that is re-downloaded every run or a `dep` that is hashed on every `dvc status`, and presented the cache as the only way out, which made a DVC limitation out of a design trade-off.
  An adversarial review of PR #54, also run with AI, found `--outs-persist-no-cache` in DVC 3.67.1 and established that a `persist: true, cache: false` out is neither deleted before the run nor pushed, and the same review found the recoverability consequence above.
  Lukas kept the mechanism and required the rationale to be rewritten so that it rests on the cost of the 548.6 MB hash and on the check the stage already performs, rather than on a limitation that does not exist, and required the consequence to be recorded.
- **AI interaction evidence:** Claude Code session on 2026-09-30 implementing [issue #33](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/33), which produced the first option table in the body of [PR #54](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/54); a second Claude Code session the same day reviewing that PR adversarially, which produced the `persist` finding and the recoverability consequence; Lukas directed that the entry be corrected rather than the mechanism changed.
- **Other evidence:** [Issue #33](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/33); [PR #54](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/54); the `download` stage in `dvc.yaml` and the `download` block of `params.yaml`; [Data versioning](../docs/docs/data-versioning.md), "Raw data"; `tests/test_download.py`; EDN-25, EDN-34, EDN-36.
- **In LaTeX:** no

### EDN-36: Retire EDN-25's `dvc add` pointer, so we no longer host a copy of the raw CSV

- **Date:** 2026-09-30
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Data Versioning, Personal Data
- **Participants:** @lukas2510. Open to revisit at the next sprint review, like EDN-25 itself.
- **Decision:** Amends EDN-25, which had tracked the raw CSV with `dvc add` and pushed it to our DagsHub remote.
  The pointer `data/raw/autoscout24_dataset_20251108.csv.dvc` is deleted, the CSV is neither tracked nor pushed by us any more, and the `download` stage acquires it instead (EDN-35).
  The blob the pointer referenced is left on the remote untouched, so checking out an earlier commit and running `dvc pull` still works.
  What EDN-25 decided about `dvc import-url` is unchanged and was independently confirmed while implementing the stage.
- **Alternatives considered:**
  - **Option A (chosen): delete the pointer with `dvc remove`, leave the blob on the remote.**
    Pros: only one mechanism owns `data/raw/` at a time, which is what the tracking-granularity rule requires and what DVC enforces anyway, since it refuses an output that overlaps a tracked path; every earlier commit stays reproducible, because the blob stays; and we stop re-publishing the CSV's PII columns going forward.
    Cons: nothing at the tip references the blob, so recovering that snapshot without Zenodo means reading the hash out of the Git history of the deleted pointer, and a workspace-scoped `dvc gc` would delete it silently.
  - **Option B: keep the pointer beside the stage as a record of the hash.**
    Pros: the snapshot's hash stays visible at the tip, and `dvc pull` keeps fetching the CSV for anyone who wants it.
    Cons: two owners of one path, which is what the granularity rule forbids and what DVC rejects outright once a stage declares anything under `data/raw/`; and we would go on re-publishing the PII columns for no gain over the pinned Zenodo DOI.
  - **Option C: delete the pointer and the blob (`dvc gc --cloud`), so no copy of the CSV is left anywhere of ours.**
    Pros: the second copy of the personal data would actually be gone, which is what EDN-07 originally wanted.
    Cons: every earlier commit becomes unreproducible without Zenodo, and it removes an exposure that the pinned DOI provides publicly anyway. It trades real recoverability for no real privacy.
- **Rationale:** EDN-25 chose `dvc add` because the course demo teaches it for a raw input that somebody acquires by hand.
  Once the pipeline acquires the file, that premise is gone: the demo's own pattern is that a stage's outputs are tracked by the pipeline rather than by `dvc add`, and the data-versioning page had already said the manual pointer would be replaced by the `download` stage once the pipeline existed.
  Keeping the pointer was rejected because two owners of one path is exactly what the granularity rule exists to prevent, and deleting the blob was rejected because recoverability is worth more than removing a copy of data that is public at its source.

  What EDN-25 got right about the mechanism survives untouched. The Zenodo URL answers a HEAD request with neither an `ETag` nor a `Content-MD5` header, and `dvc import-url` derives its change detection from one of those, so that mechanism was never verifiable against this source. Implementing the stage measured a second instance of the same gap: Zenodo serves this file chunked and sends no `Content-Length` either.

  The personal-data consequence changes shape rather than going away. We stop hosting the CSV, and we host the Parquet derived from it with the same columns, which is what EDN-34 weighed; that takes effect when `download.source` flips to `zenodo` and the lock is refreshed (issue #57).
- **AI involvement:** Information seeking, Alternative assessment, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI retired the pointer as the implementation of issue #33's option B and, rather than assuming the consequence, checked the remote: the blob is still there (`files/md5/b2/3a12...`, HTTP 200, `content-length: 548610318`), so an earlier commit can still `dvc pull`, and nothing was deleted from the remote.
  An adversarial review of the same PR found what that leaves behind: no pointer at the tip references the blob, `dvc gc` and `dvc gc --cloud` both default to workspace scope and would delete it, and `dvc gc` appeared nowhere in the repository, so nobody was warned.
  Lukas accepted the mechanism and required the rule about `dvc gc --all-commits` to be documented as part of the same change.
- **AI interaction evidence:** Claude Code session on 2026-09-30 implementing [issue #33](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/33) and verifying the remote by authenticated request; a second Claude Code session the same day reviewing [PR #54](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/54), which reported the `dvc gc` exposure.
- **Other evidence:** [Issue #33](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/33); [PR #54](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/54); the deleted pointer `data/raw/autoscout24_dataset_20251108.csv.dvc`; [Data versioning](../docs/docs/data-versioning.md), "Raw data" and "Never run `dvc gc` without `--all-commits`"; EDN-07, EDN-20, EDN-25, EDN-34, EDN-35.
- **In LaTeX:** no

### EDN-37: The seller group key stays an unsalted hash, and the residual risk is disclosed

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Data Validation and Preprocessing, Privacy
- **Participants:** @lukas2510, deciding while reviewing PR #56.
- **Decision:** `seller_group_id` stays what it is: an unsalted SHA-256 of the seller's company name, or of `country_code|zip|city` for a private seller, truncated to 16 hex characters.
  No pepper and no surrogate id.
  What changes is what we claim about it: it is the split's grouping key, it keeps the seller's name, street, zip and city out of every column of every artefact, and it is **not** an anonymisation measure.
  The code, the interim contract and the dataset card say so, and no document describes the hash as one-way any more.
- **Alternatives considered:**
  - **Option A (chosen): keep the hash unsalted and disclose what it does not protect.**
    Pros: the split stays reproducible from a clean clone with no secret, which is what `pytest` and `dvc repro` depend on today; nothing to build; the claim we make becomes one that can be verified, namely that no PII column reaches an artefact.
    Cons: an attacker who holds the published file can map every id back to the name or the location it came from, so for a private seller `zip` and `city` are recoverable from an artefact that is supposed to be free of them; we carry that statement in the dataset card rather than being able to say the ids are opaque.
  - **Option B: key the hash with a secret pepper from the gitignored `.env`.**
    Pros: the dictionary attack stops working; the ids become opaque to anyone without the secret, which is what the old wording already claimed.
    Cons: the split's group assignment then depends on a secret that is not in the repository, so `dvc repro` on a clean clone produces a different split, `dvc.lock` stops matching for anyone who lacks the pepper, and every test that asserts a grouped split has to be given one. It also protects only against an attacker who does not already have the source file, which is a public download.
  - **Option C: drop `seller_group_id` from the published artefacts and keep the grouping internal to `split`.**
    Pros: the id would not leave the pipeline, so nothing published could be inverted at all.
    Cons: `split` needs the key from `preprocess` because only `preprocess` sees `seller_company_name`, so the column has to cross an artefact boundary; and the key is the evidence for the property the split exists to have, so a test, a review or a drift analysis could no longer check that no seller appears in two sets.
- **Rationale:** The measurement decided it.
  A dictionary built from the file's own `seller_company_name`, `country_code`, `zip` and `city` columns inverts 17,141 of 17,141 dealer ids and 13,576 of 13,576 private-seller ids in about 50 ms, so the truncated digest offers no protection worth the name: a dictionary attack does not care how short the output is, and the private-seller key space is small enough to enumerate even without the file.
  That is what makes option B's cost the deciding factor rather than its benefit.
  A pepper would buy protection only against someone who does not hold a file that anyone can download from the pinned Zenodo DOI, and it would pay for it with the credential-free reproducibility the fixture, the test suite and `dvc repro` on a clean clone are built on (NFR-06).
  The dealer behind a listing is in any case recoverable from the public file by joining on `make`, `model`, `price`, `mileage_km_raw` and `registration_date`, all of which our processed data publishes, so the group key is not the weakest link.

  The `zip` and `city` recovery is accepted on the same ground, and named rather than argued away.
  They are two of the seven columns `params.preprocess.pii_columns` removes, and for a private seller the group key is exactly their hash, so an artefact that has no location column still carries a value that a public file turns back into one.
  We accept it because the same public file already holds those columns next to the listing itself, and because our own remote holds the raw file too (EDN-20, EDN-25), so the group key adds no exposure that is not already there.
  It is written down so that the moment any of those premises changes - a non-public source, a private remote, a dataset we scrape ourselves - this entry is the place that says the hash was never the control.

  NFR-08 keeps its teeth either way, because it governs columns: no PII column appears in the processed data, the prediction log, the comparables or the model artefacts, and `Schema.conform` enforces that structurally. What this entry corrects is the stronger claim the code and the contract had grown around it.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** The claim had been escalating rather than weakening: `preprocess.py` said "the hash is one-way", `schema.py` said "never reversible to the name", and the description of PR #56 called the rule airtight, none of which anybody had tested. An adversarial review of that PR built the dictionary from the published file and reported the two counts above, together with the timing and the observation that the truncation is irrelevant to a dictionary attack. It laid out the three options with the reproducibility cost of a pepper and the artefact-boundary cost of dropping the column, and recommended keeping the hash and correcting the claims. Lukas accepted that, and the reasoning that settled it is his: the split must stay reproducible without secrets, and the dealer identity is already recoverable by joining on the columns we publish, so the hash is a grouping key and nothing more.
- **AI interaction evidence:** Claude Code session on 2026-09-30, reviewing PR #56 against the real snapshot: AI was asked to attack the stage's own claims, measured the inversion of all 30,717 group ids, reported that the one-wayness claim was false, and presented keeping the hash, peppering it and dropping the column as the three options; Lukas decided to keep the unsalted hash and to state the residual risk instead.
- **Other evidence:** [`recommenditos/data/preprocess.py`](../recommenditos/data/preprocess.py) (`hash_seller_group`); [`recommenditos/schema.py`](../recommenditos/schema.py) (`seller_group_id`); [dataset card](../docs/docs/dataset-card.md); [EDN-20](#edn-20-the-dagshub-remote-stays-public); [EDN-25](#edn-25-raw-data-acquisition-track-with-dvc-add-and-push-to-our-dagshub-remote); [issue #34](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/34); [PR #56](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/56).
### EDN-39: The split's size gate is four rules with an unconditional floor, not one bound derived from the realised data

- **Date:** 2026-09-30
- **Milestone:** M2: Reproducibility (shapes M3: Quality Assurance)
- **Activity / Topic:** Evaluation Protocol, Testing Strategy
- **Participants:** @lukas2510
- **Decision:** The `split` stage judges its realised sizes against four named rules: every set non-empty; every set at least 75 % of its configured share; no seller group above 5 % of the pool; and the share's scatter inside a bound derived from the group sizes at 5.0 sigma, capped at 15 points. The failure names every rule that fired. `DISPERSION_SIGMAS` is chosen from a stated false-alarm budget - at most 1 seed in 1,000 may turn the stage red on a pool that satisfies the concentration limit - measured over 2,000 seeds on three pools in [`reports/analysis/split_gate.py`](analysis/split_gate.py).
- **Alternatives considered:**
  - **Option A: one bound derived from the realised group sizes, at four sigma.** What the first implementation of the stage did.
    Pros: the bound is a property of the data rather than a number someone liked, it tightens as the pool gains sellers, and the derivation is algebraically correct for independent group assignment.
    Cons: measured unable to detect the failure it exists for. The bound takes `sum(n_i^2)` from the *realised* sizes, so it widens exactly when one dealer dominates and the split is least trustworthy. One group holding 12 % of a 10,000-row pool, dumped into validation, realises 20.8 % against a 10 % ratio and passes, because the bound has become 14.4 points. A call with three empty sets passes. A `group_key` naming `make` instead of the seller realises train at 38.7 % and passes against a 89.9-point bound, and `country_code` at 70.8 % against 101 points. Four sigma was also wrong in the other direction: the comment claimed 200 seeds reach at worst 3.5 sigma, but seeds 1 to 400 reach 3.92 and 2,000 seeds reach 4.25, so a routine `dvc exp` seed sweep would have turned the stage red on a sound split.
  - **Option B: a fixed percentage bound.** Rejected in the first implementation and still rejected.
    Pros: nothing about it can be widened by the data.
    Cons: to hold on the 1,865-row synthetic fixture pool it would have to allow about 6.8 points, and that width accepts a calibration set at a third of its intended size on the real snapshot. One number cannot serve a 99,326-row pool with 28,435 groups and a 1,865-row one with 614.
  - **Option C (chosen): keep the derived bound as a scatter check, cap it, and add an unconditional floor plus a declared concentration limit.**
    Pros: each rule catches what it is for, and the rule that catches an undersized set cannot be widened by any property of the realised data, because it is a fraction of the *configured* ratio. Concentration now fires the gate instead of widening it, which is what makes a mis-set `group_key` detectable at all. The cap stops the derived quantity from becoming vacuous. The failure message names the rule, so the reader is sent to the right cause. All three exploits above now fail, and the real snapshot's own split still passes with the sizes the dataset card states.
    Cons: four numbers to justify instead of one, and the floor is a statement about a pool of the real snapshot's granularity - on the much smaller fixture pool the shares scatter far more, so the seed-sweep test there sweeps a looser floor and says why. Three of the four rules are module constants rather than `params.yaml` entries, so they are not swept by `dvc exp`.
  - **Option D: derive the bound from a declared maximum group share instead of the realised sizes.**
    Pros: would answer the objection directly, with one rule instead of three.
    Cons: measured not to work. The available closed form, `sum(n_i^2) <= max(n_i) * N`, assumes every group is the maximum size and gives a 26.8-point bound at a 5 % declared share, which is looser than what it replaces. Modelling the tail as equal-sized groups instead understates the real dispersion (0.0346 against a realised 0.0379) and would fail sound splits at seed 206. The declared expectation is therefore used as its own rule, where it is exact, rather than as an input to a bound it cannot model.
- **Rationale:** The defect was not the formula, which is correct for what it describes, but what it was allowed to do: a single bound that both licensed and was derived from the concentration it should have flagged. Separating the two is what makes each rule checkable. The floor answers "is this set the size we asked for", the concentration limit answers "is the assumption behind the scatter bound still true", and the scatter bound answers "is this more movement than independent assignment explains". Only the third is a statement about noise, and only the third should be derived from the data.

  The constants come from measurement rather than from taste, because the first version showed what taste produces: 4.0 was justified by a sweep that stopped at 200 seeds and was wrong by 2,000. The budget is stated so the number can be re-checked: at most 1 in 1,000 seeds may false-alarm. Over 6,000 seed-pool trials the worst case is 4.25 sigma, so 5.0 gives a measured false-alarm rate of 0 with 0.75 sigma of margin. The floor at 0.75 sits below the smallest share any of 2,000 seeds produced on the real snapshot, 0.820, and rejects the calibration set at 0.58 of its intended size that option A accepted. The concentration limit at 5 % sits above the real snapshot's largest dealer, 3.41 %, and far below the 12 %, 34.6 % and 100 % of the three exploits.

  One thing is deliberately written into the comment rather than left implied: "five sigma" is not a normal-theory quantile here and must not be read as one. The largest dealer alone accounts for 80.7 % of the variance of any set's share, so the distribution is dominated by a single Bernoulli and a Gaussian tail probability would be meaningless. The multiple is an empirical quantile of a measured sweep, and the comment says so.
- **AI involvement:** Alternative generation, Alternative assessment, Solution generation, Recommendation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** An agent asked to attack the stage rather than describe it found that the gate the pull request presented as its main safety mechanism could not detect an undersized set, and confirmed each case by execution rather than by argument: the dominant-group dump, the three empty sets, and both wrong `group_key` values, plus a 13-mutation scorecard showing that deleting the `check_ratios` call in `main` survived the entire test suite. It also found the constant's justification wrong in both directions, which is the part that mattered most, because the stage would have failed a sound split at the next seed sweep.

  The modification is option D. The review asked for the bound to be derived from a declared expectation about concentration; that was tried and measured, and it is either looser than what it replaces or tight enough to fail sound splits, so the declared expectation became its own rule instead. Recording that is the point: the reviewer's prescription was not simply adopted, it was tested and the part that did not survive measurement was replaced by something that did.
- **AI interaction evidence:** Claude Code adversarial review of PR #53 on 2026-09-30, with a 13-mutation scorecard and a reproduction of each exploit; the fixing session re-ran every exploit against the new gate and measured the 2,000-seed sweep that chose the constants.
- **Other evidence:** [`reports/analysis/split_gate.py`](analysis/split_gate.py) and its committed output; [`recommenditos/data/split_data.py`](../recommenditos/data/split_data.py); [`tests/test_split.py`](../tests/test_split.py); [EDN-32](#edn-32-split-proportions-60101020-and-the-project-seed); [issue #35](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/35); [PR #53](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/53).
- **In LaTeX:** no


### EDN-40: The `ES` holdout stays row-selected, so a cross-border dealer is reported rather than moved

- **Date:** 2026-09-30
- **Milestone:** M2: Reproducibility (shapes M6: Monitoring)
- **Activity / Topic:** Evaluation Protocol, Monitoring
- **Participants:** @lukas2510
- **Decision:** The `split` stage selects the holdout per listing, not per seller, so "every `ES` listing is held out" (EDN-03) takes precedence over "no seller appears in two sets" (EDN-14) for a dealer that lists in `ES` and elsewhere. The stage counts any such seller and reports it as a warning with row counts, the disjointness tests include the holdout instead of iterating the split names only, and the count is pinned by a test.
- **Alternatives considered:**
  - **Option A: hold out whole sellers that touch `ES`.** Both invariants would then hold without exception.
    Pros: the holdout would be seller-disjoint from the training sets, so an `ES` replay could not contain a dealer the model trained on, which is the property EDN-14's finding says matters.
    Cons: it changes what the holdout *is*. EDN-03 defines it as the `ES` listings and prices its cost as exactly those rows, so the artefact would gain non-`ES` rows that M6 replays as Spanish traffic, and its size would no longer be the 6,079 rows EDN-03 and the dataset card state. On the real snapshot it would also buy nothing: 0 sellers list both inside and outside `ES`, because `hash_seller_group` keys a dealer by its company name and no name occurs on both sides.
  - **Option B (chosen): keep the row selection, detect the conflict and report it with counts.**
    Pros: EDN-03's definition of the holdout is untouched, and on the real snapshot the resolution costs nothing measurable. The violation becomes visible where it does occur rather than being silently excluded from the tests: the fixtures the tests run on have 91 such sellers at 2,000 raw rows and 898 at 20,000, and no test would have noticed, because both disjointness tests iterated the split names and skipped the holdout.
    Cons: a weaker invariant than option A. The stage does not guarantee seller-disjointness across all five artefacts, only across the four split sets, and a future snapshot with cross-border dealer names would need this decision revisited rather than being handled automatically.
  - **Option C: leave it as it was, selected per row and unmentioned.**
    Cons: this is what the pull request did, and it is the reason the conflict went unnoticed. Two documented invariants cannot both hold, and nothing said which one wins or what it costs.
- **Rationale:** The two rules genuinely conflict and one of them has to lose, so the only bad answer is not saying which. EDN-03 is the authority on what the holdout contains, and its whole purpose is that the replayed rows are a market: rows from other countries in it would change what the drift measurement measures, and the 6,079-row figure is already cited in the dataset card and in EDN-14's count of the `ES` rows the API would accept. Against that, option A's benefit on the real snapshot is exactly zero, because no dealer name crosses the border there.

  What makes option B acceptable rather than a shrug is that the cost is now measured and visible. EDN-14's finding is that dealer-level shift is as large as country-level shift, so a replay containing dealers the model trained on understates drift and confounds both NFR-11's flagging and FR-15's retrain comparison. That is a real limitation of the synthetic fixture as a drift rehearsal, and it is now a reported number with a test pinning it, so if a future snapshot does grow cross-border dealer names the stage says so on every run instead of quietly violating the invariant.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** The adversarial review of PR #53 found the conflict, which the pull request never mentioned, and measured that it occurs 91 and 898 times in the two fixtures the tests run on while occurring 0 times on the real snapshot - the combination that explains why it was invisible. It also identified the mechanism, that both disjointness tests iterate the split names and therefore exclude the holdout by construction. It offered both resolutions and accepted either, provided the invariant became visible. Lukas chose the row selection, because EDN-03 owns the definition of the holdout and option A would have changed a figure already published for no gain on the real data.
- **AI interaction evidence:** Claude Code adversarial review of PR #53 on 2026-09-30, which measured the cross-border counts in both fixtures and on the real snapshot and named the conflict between EDN-03 and EDN-14.
- **Other evidence:** [`recommenditos/data/split_data.py`](../recommenditos/data/split_data.py) (`cross_holdout_sellers`); [`tests/test_split.py`](../tests/test_split.py); [`reports/analysis/split_gate.py`](analysis/split_gate.py); [EDN-03](#edn-03-new-market-drift-scenario-hold-out-autoscout24-spain-instead-of-using-datamarket); [EDN-14](#edn-14-nfr-11s-drift-control-is-an-iid-sample-not-a-seller-grouped-one); [issue #35](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/35); [PR #53](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/53).
- **In LaTeX:** no

### EDN-41: `model_version` becomes the leading token of the normalised trim plus a frequency floor

- **Date:** 2026-09-30
- **Milestone:** M2: Reproducibility (feature engineering; feeds M3: Quality Assurance)
- **Activity / Topic:** Feature Encoding
- **Participants:** @lukas2510
- **Decision:** `model_version` enters the extended feature set as the first `features.model_version_tokens` tokens of its normalised form - case and accents folded, separators collapsed - keeping only the levels above `features.model_version_min_frequency` of the training rows and treating everything else as missing. Both numbers stay parameters so the experiment ladder can sweep them. The two finer variants measured below are rejected for now, and the justification in the code says what the token actually is rather than what it was assumed to be.
- **Alternatives considered:**
  - **Option A (chosen): normalise, keep the leading token, then a frequency floor.**
    Pros: 279 levels covering 85.9 % of the training rows, from a column with 80,332 distinct raw values, at the cost of one regular expression. Both knobs are parameters, so the ladder measures whether the coarseness costs accuracy rather than anyone arguing about it. The level set is an artefact the API loads, so a request is normalised exactly as a training row was.
    Cons: coarse, and unevenly so. The leading token names the body style for Audi and Porsche (`avant`, `coupe`) and the engine letter for the German premium models that are most of the data: `d`, `911` and `e` are the three most frequent levels and 12.4 % of the training rows lead with an engine letter, which largely repeats `fuel_category` while discarding the trim a buyer cares about (`M Sport`, `Touring`). Inside one make and model the merging is heavy - Porsche 992 level `911` covers 743 distinct raw trims, Audi A6 `avant` 473, BMW 320 `d` 331 - and 329 pairs of levels survive where one is a prefix of the other, so `('20d', '320d')` are separate levels for the same car. 14.1 % of the rows have no level at all.
  - **Option B: prefix the token with make and model, so a level is `BMW|320|d`.**
    Pros: reads as the obvious fix for the prefix-pair problem, and it does reduce those from 329 pairs to 90.
    Cons: measured, it does not fix the merging it was meant to fix and it costs a lot. The worst merge inside one make and model stays at exactly 743, because Porsche 992 level `911` is already inside one make and model, and the number of groups holding over 50 raw trims is unchanged at 203. Meanwhile each qualified level is rarer, so the same frequency floor drops far more of them: coverage falls from 85.9 % to 62.8 %, and the median merge rises from 3 to 47.
  - **Option C: keep two tokens when the first is an engine letter, one otherwise.**
    Pros: the only variant that touches the specific weakness. BMW 320's worst merge falls from 331 to 200, BMW 520's from 327 to 130, Mercedes C 220's from 74 to 35, and the groups over 50 raw trims fall from 203 to 179.
    Cons: not clearly better overall. It does nothing for the two largest merges (Porsche 992 `911` at 743, Audi A6 `avant` at 473, neither an engine letter), it costs 3 points of coverage (85.9 % to 82.9 %) and takes the level count from 279 to 318, and 1,797 training rows (2.9 %) lose their level entirely. It also hard-codes a per-make letter list into a rule that is otherwise a swept parameter, so it cannot be turned off from `params.yaml`.
  - **Option D: two tokens throughout, which is option A at `model_version_tokens: 2`.**
    Pros: the best merging of the four - worst merge 358 rather than 743, only 64 groups over 50 trims - and it needs no code at all, because it is a parameter change.
    Cons: coverage collapses to 44.0 %, so more than half the column is missing. Kept as the parameter value the ladder will try, not as the default.
  - **Option E: normalise only, with no token cut.**
    Pros: loses nothing.
    Cons: unusable as a level set. 78,738 distinct values survive normalisation over 105,405 listings, and a frequency floor on the full string keeps 7 levels covering 0.8 % of the rows.
  - **Option F: hash the trim into a fixed number of buckets.**
    Pros: bounded level count with no vocabulary to version.
    Cons: destroys the SHAP explainability FR-08 promises, because a bucket has no meaning to show a user.
- **Rationale:** A is the cheapest thing that produces a usable level set, and the alternatives were measured rather than argued. B and C both looked better on the face of it and neither survived the measurement: B leaves the merging it targets untouched while costing 23 points of coverage, and C buys a real improvement on 12.4 % of the rows by making the other 87.6 % slightly worse and by hard-coding a rule the ladder cannot sweep. The decision that actually matters is that both knobs are parameters, so the question "is the coarse trim good enough" is answered by the experiment ladder with an MdAPE rather than by anyone's judgement here, and D is already the configuration it will compare against. What had to change regardless of the outcome is the justification: the code claimed the leading token "is the one that names the trim", which is true for Audi and Porsche body styles and false for the German premium engine codes that dominate the data, and a comment that claims a property the data does not have is worse than no comment.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** The normalisation and the frequency floor were AI's own proposal in the `features` ticket, and the overstated justification was AI's wording. An adversarial review, also by AI, measured the collisions on the real column and showed the claim was false for most of the data; it proposed B and C as cheap improvements and asked for them to be implemented only if they measured better. They were then measured, and both were rejected on their own numbers - which is the part worth recording, because the suggestion was plausible enough that it would have been accepted without the measurement. Lukas accepted keeping A with the corrected justification.
- **AI interaction evidence:** Claude Code session on 2026-09-30: an adversarial review of [PR #55](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/55) reported that "the docstring's claim that the leading token is the one that names the trim is true for Audi and Porsche body styles and false for the German premium engine codes, which are the bulk of the data", and asked for the two variants to be evaluated and implemented "only if it is clearly better on the real data and you can show the numbers". The numbers are in `reports/analysis/extended_features_results.txt`.
- **Other evidence:** [`reports/analysis/extended_features.py`](analysis/extended_features.py) and its results file (2026-09-30); [model card](../docs/docs/model-card.md) "Feature space"; [problem specification](../docs/docs/problem-spec.md) section 4; [EDN-15](#edn-15-uc1-required-fields-after-measuring-the-fill-rates-plus-sc-06-for-absent-optional-fields); [issue #36](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/36); [PR #55](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/55).
- **In LaTeX:** no

### EDN-42: A categorical value the training rows never saw becomes missing, and the holdout's country is allowed to vanish

- **Date:** 2026-09-30
- **Milestone:** M2: Reproducibility (feature encoding; the holdout feeds M6: Monitoring)
- **Activity / Topic:** Feature Encoding, Monitoring
- **Participants:** @lukas2510
- **Decision:** Every categorical feature leaves the `features` stage as a `category` over the levels the training split holds, for every split and for a request alike. A value outside those levels becomes missing rather than a level of its own. The accepted consequence is that the `ES` holdout's `country_code` is missing in **every row** of its feature matrix, because `ES` is held out by construction and is therefore not a training level. This is documented rather than worked around: the stage warns, naming any column a written split never fills, a test pins the holdout case as intended, and the problem specification states it.
- **Alternatives considered:**
  - **Option A (chosen): training levels only, an unseen value is missing, and the holdout's empty column is documented.**
    Pros: a code means the same car in training, in test, in the holdout and at request time, which is the only way a fitted model can be applied to anything but its training frame. It is exactly what [EDN-18](#edn-18-unseen-countries-and-models-are-accepted-with-a-warning-not-rejected) already decided for the serving path - an unseen country is accepted and passed to the model as unknown - so the pipeline and the API agree without a second rule. The `ES` matrix being country-blind is the new-market scenario stated honestly: the model has never seen that market, which is the whole point of [EDN-03](#edn-03-new-market-drift-scenario-hold-out-autoscout24-spain-instead-of-using-datamarket). Costs nothing to implement.
    Cons: a column that is empty in every row looks exactly like a defect, and the holdout is not replayed until M6, months after anyone remembers why. Mitigated by the warning, the test and the specification sentence rather than by changing the encoding.
  - **Option B: give the holdout country a level of its own.**
    Pros: the column is no longer empty, so nothing looks broken.
    Cons: hands the model a code it was never fitted on, which is worse than missing: a tree splits on it with no training example behind the branch. It also renumbers every other level relative to the fitted model unless the level is appended, and appending a level nothing was fitted on is a contract the artefact cannot honour. And it would make the drift scenario measure something other than a new market.
  - **Option C: drop `country_code` from the feature sets, so the holdout matrix has no such column.**
    Pros: no empty column anywhere.
    Cons: throws away a feature the problem specification names for every other split, where it is filled in 99.98 % of rows, in order to tidy up one holdout frame. Country is a real price driver across seven markets.
  - **Option D: let each frame decide its own levels.**
    Pros: no frame ever has an empty categorical.
    Cons: the failure this whole mechanism exists to prevent. Every split would get different codes, so a model fitted on `train` could not be applied to `test` at all, and a request could not be scored. It is also a leak: validation, test and the holdout would be deciding the feature space.
- **Rationale:** The encoding rule was effectively settled by EDN-02 (native categoricals) and EDN-18 (accept an unseen value, do not reject it); what was open is whether the holdout's fully empty `country_code` is a defect to fix or a consequence to document. It is a consequence, and the two ways of "fixing" it are both worse: option B feeds the model an unfitted code and option C deletes a real feature. The actual defect was that nothing said so - the reasoning existed only in a pull-request description, which does not survive a squash merge - so the fix is the warning, the test and one sentence in the specification. Whether the model should instead be fitted and gated only on the makes and markets the API serves is a separate question that is deliberately not settled here.
- **AI involvement:** Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** An adversarial review found the empty column by running the stage on the real snapshot and checking every column of every split, and established that `country_code` is the only column fully null in the holdout and not also fully null in training. It read the situation as faithful to the drift scenario rather than as a bug, and recommended documenting it in three places rather than changing the encoding, explicitly declining to drop the column or to add a level because those are modelling changes nobody had asked for. That framing is what made this an entry rather than a patch.
- **AI interaction evidence:** Claude Code session on 2026-09-30, adversarial review of [PR #55](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/55): "this is faithful to the drift scenario rather than a defect, because `ES` is held out by construction, so the model genuinely has never seen that market, and EDN-18 already requires an unseen value to be accepted rather than rejected. What is unacceptable is that nothing says so."
- **Other evidence:** [`recommenditos/data/build_features.py`](../recommenditos/data/build_features.py) (`_warn_about_unobserved_columns`); [`tests/test_features.py`](../tests/test_features.py) (`test_the_holdout_country_is_missing_throughout_its_matrix_on_purpose`); [problem specification](../docs/docs/problem-spec.md) section 2; [EDN-03](#edn-03-new-market-drift-scenario-hold-out-autoscout24-spain-instead-of-using-datamarket); [EDN-18](#edn-18-unseen-countries-and-models-are-accepted-with-a-warning-not-rejected); [PR #55](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/55).
- **In LaTeX:** no

### EDN-43: The persisted contract carries a categorical's level list, and refuses to infer one

- **Date:** 2026-09-30
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Data Validation, Feature Encoding
- **Participants:** @lukas2510
- **Decision:** `Column` gains a `levels` field that a `category` cannot be declared without and that no other dtype may carry, so the artefact the `features` stage writes is lossless for the only data-dependent dtype it produces. `Schema.validate` compares the level list, not only `str(dtype)`. `Schema.conform` applies the declared levels instead of inferring them from the frame in front of it, and `build_features.FeatureSpace` is the single public way for a consumer to load the contract and the vocabulary and cast a frame against them.
- **Alternatives considered:**
  - **Option A (chosen): put the levels on `Column`, in the artefact, and make the cast apply them.**
    Pros: the contract becomes self-sufficient - a consumer holding only the loaded `Schema` and a frame has a correct cast, which is what `train` (#37) and `evaluate` (#39) both need. The invariant is structural rather than remembered: `Column` refuses a `category` without levels, so no code path can produce a contract that says `'category'` and leaves the levels to be guessed. `validate` now catches a reordered level set, which changes every code in the column while the dtype string stays the same.
    Cons: the contract file grows by the level lists, including 279 trims, so the extended set's artefact is larger and noisier to read. Nullable columns keep no `levels` key, which keeps the noise to the 15 columns that need it.
  - **Option B: leave the levels in the vocabulary and make the private applier public.**
    Pros: no change to the contract at all, and the vocabulary already holds them.
    Cons: two objects have to be loaded and kept in step for a cast that is conceptually one contract's business, and `validate` still cannot check the levels because the schema still does not know them. It also leaves the trap in place: `conform` would keep silently inferring levels for anyone who used it.
  - **Option C: document that `.cat.codes` is unsafe and leave the mechanism alone.**
    Pros: no code change; the pipeline has no active bug today, because pyarrow preserves the dictionary order through Parquet and LightGBM remaps categories by value.
    Cons: relies on every future consumer reading the warning. #37's `ridge` variant encodes the categoricals itself and #39's masking sweep rebuilds frames per masked field, and `.cat.codes` is the obvious idiom for both, so the trap would be walked into twice within the milestone, each time producing a model scored on a different numbering with no error anywhere.
- **Rationale:** The absence was reproduced three ways before it was fixed: `validate` accepted a frame with reversed levels, `conform` turned a 50-row subset of 25 levels into 5, and a one-row serving frame for `make="BMW"` came back with code 0 where the training code is 3. None of it broke anything today, which is exactly why it was worth fixing now rather than after #37 and #39 had each written their own cast. Option B was the smaller change and was rejected because it leaves `conform` - the function every stage already calls - silently wrong; a contract that cannot check the only data-dependent part of itself is not the boundary this module claims to be.
- **AI involvement:** Information seeking, Alternative assessment, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** The lossy artefact was AI's own design in the `features` ticket, and an adversarial review by AI found it, reproduced all three consequences by execution and named the two tickets that would walk into it. The modification is the shape of the fix: the review suggested either putting the levels in the contract or having the schema take them from the vocabulary, and the first was chosen because it is the only one that also lets `validate` check them. The `Column` invariant and the single `FeatureSpace` entry point were added on top, so that the property holds by construction rather than by review.
- **AI interaction evidence:** Claude Code session on 2026-09-30, adversarial review of [PR #55](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/55): the review reported that "`Schema.conform` INFERS a categorical's levels from the data" and "`Vocabulary` does hold the levels, but the only function that applies them is private, so a consumer holding the loaded `Schema` and a frame has no correct way to cast", with a reproduction of each case.
- **Other evidence:** [`recommenditos/schema.py`](../recommenditos/schema.py); [`tests/test_schema.py`](../tests/test_schema.py); [EDN-02](#edn-02-model-family-gradient-boosting-lightgbm-as-main-model); [EDN-31](#edn-31-the-processed-data-contract-is-a-hand-written-schemapy-not-pandera); [issue #37](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/37); [issue #39](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/39); [PR #55](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/55).
- **In LaTeX:** no

### EDN-44: The traceability gate is milestone-scoped, and its expected-coverage set is a validated YAML file

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Testing Strategy, CI/CD
- **Participants:** Lukas
- **Decision:** NFR-07's requirement-to-test matrix is gated per milestone rather than over the whole table, and which requirement owes evidence when lives in `tools/expected_coverage.yaml`, a schema-validated file the generator checks, rather than in a `Due` column of the specification's tables. Milestone names are validated against M1 to M6 and the enforced set is read off that known order, never off the order the file lists them in.
- **Alternatives considered:**
  - **Option A (chosen): a small validated YAML file, gate enforced per milestone up to `current`.**
    Pros: the due date of a requirement is a parameter of a CI gate, not a statement about the system, so it does not belong in a requirement document; the generator can enforce a strict schema (every documented requirement booked exactly once, no unknown ID, no unknown or out-of-order milestone) where a table cell cannot be checked at all; bumping `current` is a one-line, dated, reviewable diff; the file carries the reasoning per milestone in comments.
    Cons: a second place to look; the gate is only as honest as the bookings, so a requirement can be deferred by moving one line, which is why the M3 membership and the verification route of every already-due requirement are pinned by a test.
  - **Option B: a `Due` column in the specification's tables.**
    Pros: everything in one document, visible next to the requirement it belongs to.
    Cons: mixes a CI parameter into a document that is supposed to describe the system; `current` would still have needed a home in prose; the specification's tables are already the widest thing in the docs; nothing can validate a table cell, so a typo or a missing cell would be invisible.
  - **Option C: no completeness gate at all, matrix published for a human to read.**
    Pros: never a red build nobody can fix; zero maintenance.
    Cons: leaves NFR-07's promise to a review that nothing reminds anyone to do, which is the state that let the matrix go unimplemented until this PR.
  - **Option D: enforce completeness over the whole table from the start.**
    Pros: no bookkeeping, no bookings to argue about.
    Cons: before M4 there is no API, so fourteen of the sixteen functional requirements cannot have a test; the build would be red from the first commit until M6, and a gate that is always red stops being read.
- **Rationale:** A gate is only useful if someone acts on it, which rules out D, and only trustworthy if something enforces it, which rules out C. Between A and B, the deciding argument is that the generator can refuse a malformed expected-coverage set and cannot refuse a malformed table cell: the file makes the gate tighten by itself, because a requirement added to the documents fails the build until someone decides when its evidence is due. The decision rests on two specification amendments, both recorded in this PR: NFR-07 said CI "does not enforce completeness" and now describes this gate, and the "Verified by" legend claimed **[automated]** means a test already carries the marker, which was false for fourteen of sixteen functional entries and is now described as the route to the evidence with the due date pointing at `tools/expected_coverage.yaml`. Neither amendment changes what the system must do; both make the documents describe what CI actually does. An adversarial review of the first implementation found three ways to drop a requirement out of the gate with a one-line edit, one of which - appending an out-of-order milestone block, because the enforced set was positional in the file - was not a reviewable statement at all; validating the names against M1 to M6 and reading the order off that closes it, and the other two are now pinned by a test so the argument for loosening the gate has to be written down.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI laid out A to D with the trade-offs and recommended A, and implemented it. An adversarial AI review of that implementation then found, by execution rather than by reading, that the gate could be loosened three ways without a reviewable statement, and that the enforced set was positional in the YAML. That finding is the modification: the validated milestone order and the tests that pin the gate's parameters were not in the first design and are the reason the decision is defensible as written.
- **AI interaction evidence:** Claude Code sessions on 2026-09-30: the options were laid out and A implemented for issue #40; a second, adversarial review session reproduced each loosening route by running the generator against a modified file and reported the exit codes, and the fixes were then verified the same way.
- **Other evidence:** [issue #40](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/40); [PR #51](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/51); `tools/expected_coverage.yaml`; `DUE_AT_M3` in `tests/test_requirement_matrix.py`.
- **In LaTeX:** no

### EDN-45: A `req` marker is verification only where the specification says a test is the evidence

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Testing Strategy
- **Participants:** Lukas
- **Decision:** The matrix reports four statuses, not two, and the specification's "Verified by" tag decides which one a `req` marker earns: a marker on an **[automated]** entry is *verified by a test*, a marker on a **[manual]** entry is *named by a test* and does not stand in for the drill its cell names. A marker on a **[manual]** entry whose cell names nothing does not satisfy the gate.
- **Alternatives considered:**
  - **Option A (chosen): the specification's tag decides, and "named by a test" is a status of its own.**
    Pros: a fully checkable rule, no judgement in the tool; catches the real failure the review found, where NFR-06's determinism and MLflow provenance were reported as covered by a test whose whole body asserts that two files exist; closes the cheapest way to fake coverage, which was to tag an entry **[manual]**, leave its cell empty and put a marker on any test; makes the matrix state plainly that no status means a person agreed the evidence is enough.
    Cons: one more concept for a reader; a marker on a **[manual]** entry with named evidence still satisfies the gate, so the distinction changes the report and not the build in that case.
  - **Option B: keep one `covered` status, fix only the two wrong markers.**
    Pros: smallest diff; the two specific lies are gone.
    Cons: the mechanism that produced them stays, so the next marker on a **[manual]** entry reports as verified again; and the matrix keeps asserting something a tool cannot know.
  - **Option C: require the specification cell to name the test, and count a marker only when the named test exists.**
    Pros: the strongest claim of the three, and would also catch a marker on the wrong test.
    Cons: most cells name a category and not a test ("API test"), so it would need every cell rewritten before the API exists; "does this cell name a test" is a judgement a parser cannot make without a heuristic, and a heuristic in the middle of the traceability evidence is worse than a coarser rule that is exact.
  - **Option D: drop the automated/manual distinction from the gate and treat any marker as a claim needing human sign-off.**
    Pros: honest about the tool's limits.
    Cons: gives up the one thing the tool can check, and turns the gate back into a reminder.
- **Rationale:** The matrix is what the report cites for NFR-07, so its worst failure mode is reporting a requirement as verified when nothing verified it, and that is exactly what happened: NFR-06 and NFR-01 were both green while their evidence was partly or wholly absent. A tool cannot read a test body and judge whether it verifies a criterion, so the honest move is to let the document that already says which evidence counts decide, and to say out loud in the generated table that a status is about the route being walked and not about a person agreeing it is enough. C would claim more than the documents can back today; B leaves the mechanism in place. The same reasoning is why the human review stays in NFR-07 rather than being replaced by this gate: NFR-01 still reads *verified by a test* while two of its six success criteria have no thresholds, and only a person reading the table can notice that.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** An adversarial AI review found the flaw and proved it by execution, naming the test whose body asserts that two files exist and the covering test whose own comment says four of the six criteria are still null. A second AI session generated A to D, recommended A over C on the grounds that C needs a heuristic, and implemented it. The modification is in the markers rather than the tool: AI's first instinct was to strip NFR-01's markers so the requirement reads as uncovered, which would have thrown away real evidence and turned the M3 gate red with nothing to fix; instead the marker moved onto the test that keeps SC-04 to SC-06 from ever counting as met, and the residual over-claim is stated in the pull request rather than hidden.
- **AI interaction evidence:** Claude Code adversarial review of PR #51 on 2026-09-30, which reproduced the finding by running the generator and reading the covering tests; the follow-up session laid out A to D and verified each fix by re-running the reproduction.
- **Other evidence:** [PR #51](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/51); `Entry.status` and `Entry.has_the_promised_evidence` in `tools/requirement_matrix.py`; the status table in `CONTRIBUTING.md`; [issue #39](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/39) for the SC-04 to SC-06 thresholds.
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


### EDN-48: `split` records the supported-make list and stays a lossless partition; the downstream stages apply it

- **Date:** 2026-09-30
- **Milestone:** M2: Reproducibility (shapes M3: Quality Assurance)
- **Activity / Topic:** Evaluation Protocol, Data Preparation
- **Participants:** @lukas2510, decided by the repository owner
- **Decision:** The `split` stage computes the supported-make list (EDN-05) and writes it as `data/processed/supported_makes.json`, but it does **not** remove the rows of unsupported makes: every interim row lands in exactly one of the five artefacts. The `features`, `train` and `evaluate` stages read that artefact and apply it, so the model is fitted and the SC-01 to SC-06 gate measured only on the makes the API serves. `dvc.yaml` gives `features` the artefact as a dependency, so changing `split.min_listings_per_make` reruns the matrices, the models and the metrics.
- **Alternatives considered:**
  - **Option A: filter in `split`, and write the removed rows to a sixth artefact.**
    Pros: the frames leaving `split` are already in scope, so no downstream stage can forget the filter, which is the failure this decision is most exposed to. The skeleton pointed this way: `schema.py` said "the supported-make filter and the ES holdout belong to `split`", `preprocess.py` said the filter "is deliberately NOT here [...] which happens in `split`", and every "training scope" figure in the docs is post-filter.
    Cons: it makes `split` lossy, so "the split loses no row" stops being a testable invariant and becomes a claim about two artefacts plus a sixth one. EDN-05's threshold could no longer be revisited without re-running the split, which is the irreversible direction: a downstream stage can always apply a list, but it cannot recover rows the split threw away. The sixth artefact also needs its own `dvc.yaml` output and its own contract, which is outside issue #35.
  - **Option B: record the list, and measure the metrics over every make anyway.**
    Pros: nothing to wire; the split stays lossless and no stage has to remember anything.
    Cons: it reports a number the product cannot deliver. The gate would be computed over 1,437 listings whose make the API refuses with a 422 (FR-04), and `evaluate`'s per-make segments would include makes that are out of scope, so SC-04 would check a segment the API never answers.
  - **Option C (chosen): record the list in `split`, and restrict the model input and the metrics downstream.**
    Pros: keeps both properties that matter. `split` stays a lossless partition, so the threshold stays revisitable and no row disappears without an artefact saying where it went; and the reported population is the served population, because the stages that build model input and measure the gate apply the list. It is also the reversible direction, and `dvc.lock` ties the artefact's hash to the stages that consume it.
    Cons: the guarantee now lives in three stages rather than in one, so it has to be written down and tested rather than being true by construction. That is the cost this entry accepts, and `docs/docs/pipeline.md` carries the four obligations so they are not only in a squash-merge commit message.
- **Rationale:** The authority on what the split does is problem-spec section 5, the evaluation protocol, and it removes nothing but `ES`: "The remaining listings are split into train, validation, calibration and test sets grouped by seller". Section 2 scopes the *model*, not the split, and EDN-05 already places the rejection at the API: "the pipeline computes the list and the API rejects other makes". The repository owner read the two documents the same way and confirmed option C.

  The measurements say the choice is cheap in both directions. Applying the list moves every realised share by at most **0.08 pp** (99,326 rows to 97,889), so restricting downstream does not disturb the split proportions the gate of EDN-39 checks. Of the **6,079** holdout rows, **5,979** are a supported make and **100** are not, which matches EDN-14's count of the `ES` rows the API would accept under FR-04, so the holdout needs the same filter and the drift and retrain work is unaffected by where the filter sits.

  One framing from the original write-up is corrected here, because it would have gone into the report wrong. The 1,437 out-of-scope rows would **not** inflate the reported metrics: rare makes are harder to price, so a pooled figure computed over them is if anything pessimistic. The real problem is a different one, and it is about comparability rather than optimism: the reported population would not be the served population, and it would not be comparable to the reference values in problem-spec section 8, which are all post-filter. That is why the metrics are restricted even though leaving them unrestricted would not flatter the model.

  What the chosen option owes, and what `docs/docs/pipeline.md` records under "Known gaps", is four guarantees from #36, #37 and #39, none of them optional: the filter applied to train, validation, calibration, test **and** the `ES` holdout; applied **before** any training-derived vocabulary or statistic is computed, since fitting on rows the API refuses puts unservable makes into the model's own inputs; `evaluate`'s per-make segments restricted to the listed makes; and a test asserting that the filtered frames contain only supported makes, so the guarantee is checked rather than intended.

  Two stale claims were a consequence of this decision, and are now applied: the comments in `recommenditos/schema.py` and `recommenditos/data/preprocess.py` said the supported-make filter belongs to, or happens in, `split`. Both now say that `split` computes the list and the stages building model input apply it.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI found that the skeleton contained evidence for both readings and that the two documents behind them disagree, rather than implementing one and moving on: `schema.py` and `preprocess.py` place the filter in `split`, while problem-spec section 5 and EDN-05 place it at the API. It laid out the options with the irreversibility argument that decided it, and flagged the choice as an EDN candidate before acting on it. It also measured the consequences instead of asserting them, which is what produced the 0.08 pp share shift and the 5,979 of 6,079 holdout figure, and it corrected its own earlier framing that the out-of-scope rows would inflate the metrics once it checked the direction of the effect. Lukas confirmed option C.
- **AI interaction evidence:** Claude Code sessions on 2026-09-30 while implementing and then adversarially reviewing issue #35: the first flagged the two readings and asked for a decision rather than choosing silently; the adversarial review verified the numbers and found that the obligation existed only in the pull request body, with issues #36, #37 and #39 mentioning the make list nowhere at all.
- **Other evidence:** [`recommenditos/data/split_data.py`](../recommenditos/data/split_data.py); the `supported_makes.json` dependency on the `features` stage in [`dvc.yaml`](../dvc.yaml); [pipeline docs](../docs/docs/pipeline.md), "Known gaps", for the four downstream guarantees; [`reports/analysis/split_gate.py`](analysis/split_gate.py) for the 0.08 pp and 5,979 measurements; [EDN-05](#edn-05-supported-makes-minimum-listing-support-per-make); [EDN-14](#edn-14-nfr-11s-drift-control-is-an-iid-sample-not-a-seller-grouped-one); [EDN-39](#edn-39-the-splits-size-gate-is-four-rules-with-an-unconditional-floor-not-one-bound-derived-from-the-realised-data); [issue #35](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/35); [PR #53](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/53).
- **In LaTeX:** no

### EDN-49: No bias correction on the log-to-euro inverse transform

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Modelling Approach, Evaluation Protocol
- **Participants:** @lukas2510
- **Decision:** `predict_eur` is `exp` of the model's log-space prediction, with no Duan smearing factor and no lognormal variance correction. The target functional is the conditional **median** price, and that is what every metric this project reports measures.
- **Alternatives considered:**
  - **Option A (chosen): naive `exp`.**
    Pros: it is the conditional median of price, which is exactly what MdAPE, the +/-20 % share and the conformal intervals are about; it needs no statistic fitted on the training residuals, so nothing extra has to be persisted, versioned or kept in step between `train`, `evaluate` and the API; and it is what the model card already told readers the model does.
    Cons: it is *not* an unbiased estimate of the mean price, so anyone reading the estimate as an expected sale value is reading it wrong. The model card and the response schema say "typical price" for that reason.
  - **Option B: Duan's smearing estimator.** Multiply by the mean of `exp(residual)` over the training rows.
    Pros: the textbook correction for a log-linear model, non-parametric, and it is what a reviewer who knows the retransformation problem will ask about.
    Cons: it targets the conditional mean, so it shifts every prediction upward, away from the typical asking price. Measured on the real snapshot's 19,665 test rows it makes every variant worse: factor 1.0258 and MdAPE 9.52 % to 9.83 % for `b1`; 1.0052 and 6.83 % to 6.86 % for `lgbm-basic`; 1.0028 and 6.32 % to 6.37 % for `lgbm-extended`. It also adds a fitted constant to the artefact that has to travel with the model and be applied identically in three places.
  - **Option C: lognormal variance correction,** multiply by `exp(sigma^2 / 2)`.
    Pros: closed form, no residual pass needed.
    Cons: everything option B has, plus a distributional assumption the residuals do not have to satisfy; it is strictly weaker than smearing, which estimates the same quantity without the assumption.
- **Rationale:** The correction answers a question nobody here asks. Fitting with squared error on `log(price)` estimates `E[log P | x]`, so `exp` of it is the geometric mean, which for the conditional distribution of a price is the median. Every number in problem-spec section 6 and every criterion in section 8 is median- or quantile-flavoured: MdAPE is the primary metric, SC-02 is a share within a band, and SC-05's intervals are conformal and therefore calibrated after the fact on held-out data, so they need no correction either. A mean-targeting correction would make each of those numbers worse, and the measurements say it does, on all three variants that have an inverse transform at all.

  `b0` is the reason the decision is visible in the code rather than implicit. The median baseline is fitted on `price` directly, not on `log_price`, because a median commutes with a monotone transform: `exp(median(log price))` and `median(price)` are the same number, and the readable one is the one worth storing in a lookup table a person can audit. So B0 needs no inverse transform, B1 and the two LightGBM variants share one, and the seam puts it inside the model - `predict_eur` returns euros and `predict_log_price` is defined as its log - so `evaluate` and the API cannot come to disagree about it.

  The honest limit of this entry: the effect is small. The largest degradation measured is 0.32 pp, on the interpretable baseline rather than on the deployed candidate. It is recorded because the retransformation problem is a standard question about a log-target model, the answer has numbers behind it, and the alternative would have quietly cost accuracy.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI raised the retransformation question unprompted, named the two standard corrections and argued from the target functional rather than from convention, which is the argument that decides it. Its first write-up carried the effect sizes from a different run (0.5 to 2.2 pp) as if they were ours; those were re-measured on the real snapshot and are an order of magnitude smaller, so the entry now claims what this project measured and says that the effect is small. The direction of the effect - worse on every variant - reproduced.
- **AI interaction evidence:** Claude Code session on 2026-09-30 implementing issue #37: the smearing factor and the paired MdAPE per variant were computed on the real snapshot in a scratchpad run of the whole chain, with the numbers reproduced in the model card.
- **Other evidence:** [`recommenditos/modeling/model.py`](../recommenditos/modeling/model.py), `Model.predict_eur` and `MedianBaselineModel`; [model card](../docs/docs/model-card.md), Training Procedure, Training; `tests/test_model.py::test_predict_log_price_is_exactly_the_log_of_predict_eur`; [issue #37](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/37).
- **In LaTeX:** no

### EDN-50: One-hot encoding for the Ridge baseline only, against EDN-02's "no one-hot"

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Feature Engineering, Modelling Approach
- **Participants:** @lukas2510
- **Decision:** The `b1` Ridge variant one-hot encodes its categoricals, with `handle_unknown="infrequent_if_exist"` and `min_category_rows: 5` as the threshold below which levels share one column. This is confined to `b1`: the LightGBM variants and the CatBoost challenger take the categoricals natively, as `category` codes over the levels the `features` stage fixed, and no one-hot column exists anywhere near them.
- **Alternatives considered:**
  - **Option A (chosen): one-hot, confined to B1.**
    Pros: it is the standard encoding for a linear model, it needs no target information, and it keeps B1 able to do the job it exists for - telling a Porsche from a Dacia as part of an interpretable depreciation baseline. `handle_unknown="infrequent_if_exist"` is also what makes an unseen level an answer rather than an error, which EDN-18 requires of the serving path.
    Cons: it is the encoding EDN-02 rules out, so a reader who takes that rule as project-wide sees a violation. The column count grows with the level count: 503 columns at real scale before any folding, against 16 input features.
  - **Option B: target encoding with cross-fitting.**
    Pros: one column per categorical however many levels it has, and it usually beats one-hot for a linear model on high-cardinality data.
    Cons: it encodes the target into a feature, so it leaks unless it is cross-fitted, and cross-fitting means a second resampling scheme inside the stage, a second thing to persist and a second thing that can differ between training and serving. For a baseline whose purpose is to be simple and auditable that is the wrong trade, and the leak is the kind of defect this project would only find by not finding it.
  - **Option C: drop the categoricals from B1 and fit on the numerics alone.**
    Pros: no encoding decision at all, and the ladder still has a linear baseline.
    Cons: it destroys the baseline. A depreciation model that cannot see the make is not a weaker version of B1, it is a different and much worse thing, and SC-03's comparison against B0 would be against a baseline nobody would have built.
  - **Option D: read EDN-02 as project-wide and drop B1 from the ladder.**
    Pros: no apparent contradiction to explain.
    Cons: problem-spec section 7 names B1 as a baseline, and the interpretable linear reference is what makes the tree models' gain legible. Removing a baseline to preserve the letter of a rule about a different model family is the wrong direction.
- **Rationale:** EDN-02 is a decision about the *model family*, and "categoricals stay categorical, no one-hot" is stated there as one of the reasons gradient boosting was chosen over the alternatives - LightGBM and CatBoost handle them natively, so the pipeline does not have to blow up the matrix. It is not a constraint on how a linear baseline encodes its inputs, because a linear model has no native handling to use: the choice is one-hot, target encoding or nothing. Recording this keeps the next reader from either "fixing" B1 to match the rule or reading the rule as broken.

  Two details of the encoding are decisions in their own right and are measured rather than assumed.

  Absence is a level. Categoricals are mapped to a literal `"__missing__"` value before encoding, deterministically at fit and at predict time, so an absent value gets a coefficient like any other level instead of failing the fit or being dropped. That is the same treatment EDN-15 asks for, expressed in the only way a linear model can express it.

  `min_category_rows` is an absolute row count rather than a share, so that what the threshold means does not change as the training set grows. Measured on the real snapshot (60,378 training rows, MdAPE on the 19,665 test rows): no folding gives 503 encoded columns at 9.48 %, 5 gives 404 at 9.52 %, 30 gives 294 at 9.71 %, 100 gives 187 at 10.61 % and 1,000 gives 68 at 13.45 %. So folding costs accuracy monotonically, and the reason to fold is not accuracy.

  What the reason is, read off the fitted encoder rather than assumed: at 5, folding touches **2 of the 8 categoricals**. `model` folds 100 of its 438 levels into one infrequent column and `fuel_category` folds 1. The other six have no infrequent group at all, and `make` and `body_type` have no `__missing__` level either, because the real training rows never leave those two empty. An unseen or an absent value in one of those six therefore encodes as all zeros rather than into a group.

  EDN-18 is satisfied either way, and checked rather than assumed: masking each of the eight in turn, and giving each an unseen value in turn, the real `b1` answers with a finite positive price all sixteen times. It is simply satisfied by the all-zero encoding for six of the eight - a coherent encoding of "no level applies" - and not by the infrequent group the threshold produces. The threshold is what stops `handle_unknown="infrequent_if_exist"` from being pointless for `model`, which is the one column where rare levels are the norm, and nothing more.

  So the justification for 5 is what is left: folding is nearly free there, 0.04 pp, and it stops a level seen in a handful of rows from getting a coefficient fitted on those rows. At real scale that is 47 `model` levels seen exactly once, among the 100 the threshold folds. An infrequent group does not need 5 to exist - 2 already gives one, folding 48 levels for 0.02 pp - so 5 is a round number inside a flat region rather than a boundary, and the value is a parameter precisely so the ladder can revisit it.

  One claim that did **not** survive measurement and is recorded so nobody repeats it: the design this work followed justified a much larger threshold with a small-data pathology, a mid-range car with an unseen model and country predicted at 94,323 EUR. That did not reproduce here at either scale. On the fixture the same probe gives 13,993 EUR unfolded and 15,413 EUR at a threshold of 30; at real scale, 32,332 EUR and 33,104 EUR. The threshold is justified by the infrequent group and by the one-row-coefficient argument, not by a pathology this project observed.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI spotted the apparent conflict with EDN-02 before writing the estimator and flagged it as an EDN candidate rather than resolving it silently, and its reading - that EDN-02 is about the tree family - is the one the entry records. It was overruled on the threshold: it recommended 30 on the strength of a measurement from another run and asserted the value was free at real scale. Sweeping it here showed folding is not free and that its pathology did not reproduce, so the value is 5 and the entry says what the evidence actually supports.
- **AI interaction evidence:** Claude Code session on 2026-09-30 implementing issue #37: the threshold sweep was run on the fixture and then on the real snapshot, and the "unseen model and country" probe was re-measured at both scales before the recommended value was changed.
- **Other evidence:** [`recommenditos/modeling/model.py`](../recommenditos/modeling/model.py), `RidgeModel`; the `min_category_rows` comment in [`params.yaml`](../params.yaml); [model card](../docs/docs/model-card.md), Training Procedure, Training; `tests/test_model.py::test_an_unseen_category_is_treated_as_missing_not_an_error`; [EDN-02](#edn-02-model-family-gradient-boosting-lightgbm-as-main-model); [EDN-18](#edn-18-unseen-countries-and-models-are-accepted-with-a-warning-not-rejected); [issue #37](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/37).
- **In LaTeX:** no

### EDN-51: Mean fill plus a per-feature missingness indicator for the Ridge numerics, which is not imputation

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Feature Engineering, Modelling Approach
- **Participants:** @lukas2510
- **Decision:** The `b1` Ridge variant's numeric branch is the union of two paths: a mean-filled, standardised copy of every numeric feature, and `MissingIndicator(features="all")`, which emits one indicator column per feature whether or not that feature was ever missing at fit time. No other estimator fills anything, and no artefact of the pipeline is imputed: the fill exists only inside B1's fitted pipeline.
- **Alternatives considered:**
  - **Option A (chosen): mean fill plus an indicator for every feature.**
    Pros: information-preserving for a linear model. For a row where feature `j` is absent the contribution is `beta_j * mean_j + gamma_j`, and `gamma_j` is free to absorb whatever the absence is worth, so the fill value is a numerically neutral placeholder rather than a guess at the value. Nothing about the missingness is destroyed, which is what EDN-15 is protecting. `features="all"` also makes SC-06 honest: the criterion masks a field and measures what it costs, and it can only do that if masking a field the training rows happened to fill still produces an indicator.
    Cons: a reader who sees `SimpleImputer` in the pipeline will conclude the project imputes, which is exactly the misreading this entry exists to prevent. It also doubles the numeric column count.
  - **Option B: `SimpleImputer` alone, or `SimpleImputer(add_indicator=True)` at its default.**
    Pros: less to explain, fewer columns.
    Cons: this is the option that silently breaks SC-06, and it is the default, which is why it is worth an entry. scikit-learn emits an indicator only for features that were missing **at fit time**, so a column complete in the training split is mean-filled with no indicator the moment the criterion masks it - the model then reports the training mean for that field, the prediction barely moves, and SC-06 measures the model's own imputation rather than the cost of the missing input. Without the indicator at all, an absent value is indistinguishable from an average one, which is imputation in the sense EDN-15 rejects.
  - **Option C: drop the rows or the columns with missing values.**
    Pros: no fill anywhere, so no explaining to do.
    Cons: dropping rows would fit B1 on a different and much smaller population than the other three variants, so the ladder would stop being a comparison. Dropping columns would throw away `nr_prev_owners` and `gears`, which are missing in a third and a quarter of the training rows respectively, and the missingness itself carries signal (EDN-15).
  - **Option D: give up on B1 taking numerics with gaps and use a model that handles them natively.**
    Pros: no encoding at all.
    Cons: that model is LightGBM, which is ladder steps 3 and 4. B1's value is that it is a *linear* reference.
- **Rationale:** EDN-15's rule is that a missing value is never imputed, because the missingness itself carries signal. Ridge cannot consume NaN, so B1 either encodes the gaps or does not exist. The construction above is the encoding under which nothing is lost: the pair `(filled value, indicator)` is a bijection with `(value, present)` for a present value and carries the absence explicitly for an absent one, so the linear model has exactly the information the tree models get from a NaN split. The fill is a placeholder chosen to be numerically harmless after standardisation, not an estimate of the missing value, and the entry exists because those two look identical in the code.

  `features="all"` is the part a test has to hold, because it is invisible from the outside. It is also invisible on the synthetic fixture: every numeric column there has a missing value in the training split, so the default emits the same set of indicators and a test written against the fixture cannot tell the two apart. That was found by breaking the implementation on purpose and watching the test suite pass. The test now fills one column completely before fitting and fails with the default.

  What that test constructs is **not** the situation on the real snapshot, and the entry says so rather than borrowing the stronger claim. Of the 8 numeric features of the `basic` set - the only set `b1` uses - **none** is complete in the real training split: the emptiest is `age_years` with 1 gap in 60,378 rows and the fullest `nr_prev_owners` with 22,697. So scikit-learn's default would emit the same eight indicators here, and `features="all"` changes nothing in the shipped configuration. It is insurance, not a measured fix, and what it insures against is real: 136 of the `extended` set's 147 numeric features are complete at fit time, so a Ridge on `extended` - which the ladder may well want, since B1 is the interpretable reference for the winning feature set - would lose 136 of its 147 indicators to the default, and SC-06 would measure the model's own imputation for every one of them. The guard is kept because it costs nothing and because the configuration it protects is one change away.

  The other half of the same discipline is what the estimator does *not* do. `SimpleImputer(keep_empty_features=True)` keeps an all-missing column in place rather than dropping it, so the matrix shape does not depend on which columns the training rows happened to fill; and nothing outside B1's pipeline fills anything, so no artefact on disk and no other variant carries a filled value.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI identified that the obvious construction would make SC-06 measure its own imputation, which is a subtle failure that would have produced a passing criterion and a wrong conclusion, and it named `features="all"` as the fix and flagged the whole construction as an EDN candidate because of how it reads. It also proposed the test for it. The test it proposed was then shown, by mutating the implementation, to pass either way on this fixture, and it was replaced with one that fills a column first; that correction came from the mutation exercise rather than from the design.
- **AI interaction evidence:** Claude Code session on 2026-09-30 implementing issue #37: the mutation battery run before the pull request recorded `MissingIndicator(features="all")` to `MissingIndicator()` as a surviving mutation, which is what produced the replacement test.
- **Other evidence:** [`recommenditos/modeling/model.py`](../recommenditos/modeling/model.py), `RidgeModel` and `_ridge_pipeline`; `tests/test_model.py::test_the_ridge_emits_a_missing_indicator_for_every_numeric_feature` and `::test_a_numeric_feature_complete_in_training_still_gets_an_indicator`; [model card](../docs/docs/model-card.md), Training Procedure, Training; [EDN-15](#edn-15-uc1-required-fields-after-measuring-the-fill-rates-plus-sc-06-for-absent-optional-fields); [issue #37](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/37).
- **In LaTeX:** no

### EDN-52: `predict_eur` bounds every prediction to the training price range

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Modelling Approach, API Design
- **Participants:** @lukas2510, decided by the repository owner
- **Decision:** `Model.predict_eur` clips the estimator's log-space output to `log` of the observed minimum and maximum `price` of that variant's own training rows, before the exponential, and records both bounds in `models/<variant>/model.json` under `training.price_min_eur` and `training.price_max_eur`. The clip is in the base class and `predict_eur` is final, so no estimator can bypass it, and a bundle whose recorded range cannot bound a price is refused on load.
- **Alternatives considered:**
  - **Option A: return the estimator's output unbounded.**
    Pros: nothing hidden; whatever the model believes is what the caller sees, which is the right default for a component that reports its own errors.
    Cons: the interface cannot then promise what `evaluate` and the API rely on. `exp` overflows to `inf` above about 710 in log space and underflows to exactly `0.0` below -746, and both are reachable from a linear model with no bounded link. An `inf` prediction makes MdAPE `nan`, and `nan <= 0.09` is `False`, so the gate would record a failure with no indication that the number was not a number; a `0.0` prediction breaks the "strictly positive" promise the API's response schema rests on.
  - **Option B (chosen): clip to the training price range, in log space, before the exponential.**
    Pros: makes "finite and strictly positive" a property of the arithmetic rather than something hoped for, because the bounds are finite and positive by construction. Bounding before the exponential rather than after means the overflow is unreachable instead of merely clipped, so no `RuntimeWarning` and no `inf` ever exists. The bound is the range the training data supports, which is the only range about which the model has evidence. And because both edges are recorded in the bundle, the share of predictions sitting exactly on an edge can be reported, so the clip cannot hide a pathology it is covering up.
    Cons: a clipped prediction is wrong in a way that no longer looks wrong. It is a plausible number, so a reader cannot tell it from an estimate unless someone reports the share on the bound - which is why that reporting is part of the decision rather than a nice-to-have.
  - **Option C: clip to a quantile range of the training prices, say the 0.1st to the 99.9th percentile.**
    Pros: a tighter and more defensible band than the extremes, and less sensitive to one collector car in the training rows.
    Cons: it would move real predictions. The bound stops being a guard rail and becomes a silent recalibration of the top and bottom of the distribution, which is a modelling decision dressed up as a safety check, and it would bias the metrics for exactly the segments SC-04 is watching.
  - **Option D: raise on a prediction outside the training range.**
    Pros: the loudest possible signal, and no wrong number is ever served.
    Cons: it turns a model's extrapolation into a 500 from the API for a request that is in scope, which contradicts FR-05's promise that an in-scope car is always answered. It would also make `evaluate` unable to finish a run over the test set, so the one place the behaviour can be measured would be the place that cannot run.
- **Rationale:** The measurement is what settled it, and it also corrected the reason the bound was originally proposed. The design this work followed justified the clip with a Ridge returning **487,417 EUR** for a make-only request. That number would **not** be bounded: on the real snapshot the training range is **500 to 1,814,750 EUR**, so 487,417 is comfortably inside it and the clip would never fire. Repeating the make-only probe here gives 6,529 EUR for B1, nowhere near the bound.

  What the clip does catch, measured over the 19,665 test rows of the real snapshot, is a genuine extrapolation: B1's unbounded prediction leaves the range **once**, at **10,635,538 EUR**. None of the other three variants leaves it at all, with unbounded maxima of 1,549,910 EUR for B0, 1,055,373 for `lgbm-basic` and 731,089 for `lgbm-extended`. So the clip is load-bearing for exactly one variant on roughly one row in twenty thousand - rare, and worth having anyway, because the alternative is one impossible number per twenty thousand answers in a product whose whole claim is a credible price.

  The honest limit is stated in the code and in the model card rather than left for a reader to discover: bounding that 10.6 million turns it into 1,814,750 EUR, which is the price of the most expensive car in the training rows and not a sensible estimate for the car that was asked about. The bound makes the number representable, not right. The reserved `predict_interval_eur` of SC-05 is where a caller will eventually see that such a prediction is not to be trusted.

  One guard came out of writing this down. `predict_eur` promises a strictly positive value and keeps that promise *through* the bound, so a recorded range that cannot bound a price - a lower edge of 0, which would clip to `log(0)` and exponentiate to exactly `0.0` - has to be refused rather than served. `preprocess.price_min_eur: 500` makes it unreachable from the pipeline; the guard exists for a hand-edited bundle, and it is what keeps the promise unconditional instead of conditional on a parameter somewhere else in the system.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI proposed the clip and the float-limit argument for applying it in log space, which is the part that makes the interface's promise structural. Its justifying measurement did not hold up: the 487,417 EUR case it cited is inside the training range and would never be clipped, which was found by measuring the range instead of assuming it. Re-measuring over the whole test split produced the case that does justify the clip, and the entry and the code now claim that one and explicitly disclaim the other. AI also proposed the test; the test was strengthened to replace the estimator with one returning an absurd log price, so that it fails when the clip is removed rather than depending on how far a particular fit happens to extrapolate.
- **AI interaction evidence:** Claude Code session on 2026-09-30 implementing issue #37: the unbounded predictions of all four variants were computed over the real test split, and the make-only probe was re-run at both data scales before the justification in the entry was rewritten.
- **Other evidence:** [`recommenditos/modeling/model.py`](../recommenditos/modeling/model.py), `Model.predict_eur` and the range guard in `Model.__init__`; `tests/test_model.py::test_a_prediction_that_would_overflow_is_bounded_to_the_training_range`, `::test_a_training_price_range_that_cannot_bound_a_price_is_refused` and the range assertions in `::test_predict_eur_contract`; [model card](../docs/docs/model-card.md), Training Procedure, Training; [PR #62](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/62); [issue #37](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/37).
- **In LaTeX:** no

### EDN-53: `train.num_threads` is pinned to 1, trading fit speed for a machine-independent artefact

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Reproducibility, Modelling Approach
- **Participants:** @lukas2510, decided by the repository owner
- **Decision:** `params.yaml` carries `train.num_threads: 1` and `train` passes it to LightGBM as `n_jobs`. It is never left at LightGBM's default of `-1`, which means every core. The reproducibility it buys is a `booster.txt` that is byte-identical on any machine. What it costs is narrower than first recorded: measured against a pin of 2 or 4 threads it costs about 1.5x the fit, and measured against the default it actually *saves* time and, more to the point, removes an enormous variance.
- **Alternatives considered:**
  - **Option A: leave `n_jobs` at its default of `-1`.**
    Pros: nothing to configure. Not, as it turns out, the fastest fit: see the rationale.
    Cons: LightGBM writes the thread count into `booster.txt` as a `[num_threads: N]` line, so the artefact - and therefore its DVC hash and `dvc.lock` - depends on the core count of whoever ran `dvc repro`. Two contributors with different laptops would produce different artefacts from identical data and parameters, and `dvc status` could never say the model was unchanged. On top of that, on an 8-core machine the default is both slower and wildly variable at real scale, and it makes the test suite's cost unpredictable, which matters because the suite fits all four variants on 744 rows.
  - **Option B (chosen): pin to 1.**
    Pros: makes `booster.txt` byte-identical across thread counts, which is a stronger claim than NFR-06's "metrics within 0.1 percentage points" and is the one the reproducibility test asserts. It also removes oversubscription if the four independent `train@*` stages are ever run at once, and it is the value with by far the smallest spread between repeated fits at either data scale - which is what makes a reported fit time worth reporting.
    Cons: it is measurably slower than a **pinned** 2 or 4. On the 60,378-row training split `lgbm-extended` fits in a median 18.3 s at 1 thread against 12.2 s at 4, so option C would buy back about a third of the fit per variant. It is not slower than the default it actually replaces.
  - **Option C: pin to a fixed larger number, 2 or 4.**
    Pros: the same machine independence as option B, because the value is pinned rather than discovered, with most of the speed back. This, and not option A, is the alternative that the pin genuinely costs something against.
    Cons: it is only machine-independent on machines that *have* that many cores. LightGBM silently uses what is available, so a 2-core CI runner asked for 4 would write a different `booster.txt` than an 8-core laptop, and the property would hold right up until the first machine that breaks it - the worst kind of guarantee. 1 is the only value every machine can honour.
- **Rationale:** The measurement that decides the direction is that the **trees are already thread-independent and only the recorded setting is not**. Fitting `lgbm-basic` on the fixture in separate processes at 1, 2, 4 and 8 threads gives the same `best_iteration` (98) and a byte-identical model file once the `[num_threads: N]` line is normalised away, at every thread count; the md5 of the whole file differs at every one of them. So pinning the value does not make the model reproducible - it already is - it makes the *file* reproducible, which is what DVC hashes.

  That also corrects a claim worth not repeating. The design this work followed said `deterministic=True` and `force_row_wise=True` are what keep the trees stable across thread counts. They are not, at this data size: with both flags **off**, the normalised file is still byte-identical across 1, 2, 4 and 8 threads. What the flags do change is *which* trees are built - the normalised md5 differs between the flags on and off - so they are not free of consequence, they are simply not what makes the result thread-independent here. They are kept as insurance documented by LightGBM, and the code and the model card now say that rather than claiming a measurement that does not exist.

  The cost side was measured rather than assumed, and the first version of this entry stated it against a configuration the default never produces. The full real-scale curve, medians of three fits per thread count on the 60,378-row training split, on an 8-core machine with nothing else running:

  | Variant | 1 thread | 2 | 4 | 8 (the default here) |
  |---|---|---|---|---|
  | `lgbm-basic` | 7.1 s (7.0 to 7.2) | 5.6 s (5.3 to 6.8) | 6.8 s (6.4 to 7.4) | 17.7 s (16.5 to 36.5) |
  | `lgbm-extended` | 18.3 s (17.9 to 18.9) | 12.8 s (12.7 to 13.1) | 12.2 s (11.3 to 14.3) | 29.2 s (12.6 to 38.7) |

  Two things follow, and the second is the correction. More threads do help, so the design's "there is no performance reason to use more" is wrong: 2 to 4 threads are consistently the fastest, and the pin costs about 1.5x the fit for `lgbm-extended` and 1.27x for `lgbm-basic` against them. But the alternative this decision actually rejects is not a pin of 2 or 4, it is LightGBM's **default of every core**, and against that the pin is free or a gain: 18.3 s against 29.2 s for `lgbm-extended`, 7.1 s against 17.7 s for `lgbm-basic`. So the trade is with option C, not with option A.

  The spread matters more than the median. At 1 thread the three fits of a variant land within 0.2 s and 1.0 s of each other; at 8 they span 20 s and 26 s, and the fastest `lgbm-extended` fit of the whole sweep (12.6 s) and the slowest (38.7 s) are both at 8 threads. A number that varies threefold between identical runs is not a fit time anyone can report, which is a reproducibility argument for the pin quite apart from the artefact's hash.

  The earlier entry attributed the reversal at 8 threads to the 744-row fixture, where the direction is even sharper: 0.33 s at 1 thread, 0.27 s at 2, 0.48 s at 4 and a median of **28.58 s at 8**, with one fit of 143.71 s, on a machine already at load 13 to 19. That reading was too narrow. The reversal is not about the row count but about the work per thread: it happens at 60,378 rows too, and on `lgbm-basic` - 16 features - more decisively than on `lgbm-extended`'s 162, which is where the extra threads have something to divide. Since the test suite fits every variant on the fixture, on whatever CI runner picks up the job, the pin is what keeps the suite's cost predictable as well.

  Absolute seconds here are machine- and load-dependent and are not comparable with the ladder table's one-off fit times in the model card; the ratios within one sweep are the measurement.
- **AI involvement:** Information seeking, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI identified that LightGBM records the thread count in the artefact, which is the fact the whole decision rests on, and proposed pinning it. Three of its supporting claims were wrong and were corrected by measuring. That `deterministic=True` and `force_row_wise=True` are what make the trees thread-independent: they are not at this size, the trees are identical with the flags off, though the flags do change which trees are built. That pinning to 1 costs nothing at real scale: it costs about 1.5x the fit against a pin of 2 to 4. And then the correction to that correction - the entry's own second version stated the cost against 2 to 4 as if that were what the default does, and put eight threads at real scale under "not measured". Measuring it reversed the comparison that matters: against the default the pin is free or a gain. The entry now names which alternative the cost is against, and the code comments were rewritten to match.
- **AI interaction evidence:** Claude Code session on 2026-09-30 implementing issue #37: `booster.txt` was hashed with the `[num_threads: N]` line normalised away across four thread counts and both settings of the determinism flags. The thread sweep was then redone during the review of pull request #62, as medians of three fits at 1, 2, 4 and 8 threads on both LightGBM variants at real scale on an idle machine, after a first attempt was discarded for having shared the machine with a test run - which is also how the size of the variance at 8 threads came to light.
- **Other evidence:** [`recommenditos/modeling/model.py`](../recommenditos/modeling/model.py), `LightGBMModel.fit`; the `num_threads` comment in [`params.yaml`](../params.yaml); `tests/test_model.py::test_num_threads_is_pinned_from_params`, which trains at a value the project does not use so a hard-coded 1 would fail it, and `::test_the_payload_of_a_refit_is_byte_identical`; [model card](../docs/docs/model-card.md), Training Procedure, Determinism; [requirements](../docs/docs/requirements.md) NFR-06 and NFR-10; [issue #37](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/37).
- **In LaTeX:** no

### EDN-54: Tracking degrades silently when it is not configured, with an environment variable to make it fatal

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** CI/CD, Experiment Tracking
- **Participants:** @lukas2510, decided by the repository owner
- **Decision:** A pipeline stage opens its MLflow run through `optional_run`, which yields a real run handle when the server is configured and reachable and a `DisabledRun` that logs nothing when it is not - whether because nothing is configured, because the credentials are refused or because the server cannot be reached. The stage runs to completion either way and records `mlflow.tracking_mode` in `model.json`. Setting `RECOMMENDITOS_REQUIRE_TRACKING=1` turns any of those failures back into a failed stage. There is never a fallback to a local tracking store.
- **Alternatives considered:**
  - **Option A: fail the stage whenever tracking is not available,** which is what `tracked_run` does and what EDN-46's helper was built for.
    Pros: a run that was not recorded never happened, which is the strongest possible reading of NFR-06; and the failure is the cheapest one, before the fit.
    Cons: CI has no DagsHub credentials and a fresh clone has none either, so `pytest` and `dvc repro` would both fail on the one path every contributor and every pull request takes. The suite's whole premise, stated in `tests/conftest.py`, is that it runs on a clean clone with no credentials and no network; this option contradicts it.
  - **Option B: fall back to a local tracking store when the server is not configured.**
    Pros: the run is recorded somewhere, so nothing is lost and the code path is the same.
    Cons: since MLflow 3.16 an unset tracking URI resolves to `sqlite:///$PWD/mlflow.db`, and for a DVC stage `$PWD` is the repository root. The fallback therefore drops a database into the repository on every run of every contributor, gitignored so nothing would ever point it out. `tests/conftest.py` already guards `mlflow.db` and `mlruns/` as forbidden paths for exactly this reason, and writing the guard for `models/` caught a real instance of it: a local store puts its *artifacts* in `./mlruns` too.
  - **Option C (chosen): degrade silently, with `RECOMMENDITOS_REQUIRE_TRACKING=1` as the escape hatch.**
    Pros: every contributor and CI can train, which is a property of the whole pipeline rather than of this stage. And the runs whose numbers are cited can be made to fail loudly, so "the runs are on the server" is something a person can verify instead of something a reader has to take on trust.
    Cons: two behaviours instead of one, and the quiet one is the default. A contributor who *meant* to record a run and mistyped a variable gets a warning line in a long stage log and a model that looks finished. The mitigation is that `model.json` carries `tracking_mode: "disabled"` and a null `run_id`, so the artefact itself says it was not recorded, and `evaluate` says so again when it finds no run to resume.
  - **Option D: degrade silently with no escape hatch.**
    Pros: one behaviour, nothing to remember.
    Cons: it makes the report's claim unfalsifiable. Nobody could demonstrate that a given set of runs reached the server, because the successful and the skipped path are indistinguishable from the outside.
- **Rationale:** The two halves of the decision answer two different audiences and neither can be dropped.

  Silence is right for the default audience, which is CI and a contributor on a fresh clone. Training with no credentials is not a degraded mode for this project; it is the normal mode for everything except the handful of runs that produce numbers for the report, and a gate that turns red for everyone stops being read.

  The escape hatch is right for the other audience, which is whoever produces those numbers. It was not hypothetical: the live verification of this stage against the team's DagsHub server was run with `RECOMMENDITOS_REQUIRE_TRACKING=1` precisely so that a misconfiguration could not let the check pass while logging nothing. Without it the verification would have proved that the stage finishes, which it does in either case, rather than that the runs landed.

  The rule that there is never a local fallback is what makes "no tracking URI means tracking is off" a statement with content, and it is the same reasoning `recommenditos/config.py` already applies to loading `.env` from a named path rather than searching upwards.

  One consequence was measured rather than assumed, and it is recorded in EDN-57: detecting "the server is not reachable" is not free, because MLflow retries. That is why the retry budget is pinned, and the two decisions have to be read together.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI identified that the existing helper's fail-fast behaviour, which EDN-46 chose deliberately for a credential check, is the wrong behaviour for a pipeline stage, and that the obvious repair - a local fallback - would put a database in the repository root under MLflow 3.16's changed default. It proposed the null-object handle so the branch exists once rather than in front of every log call, and the environment variable so the quiet default stays falsifiable. It also used the variable in its own live verification rather than only testing it, which is what showed the hatch earns its place.
- **AI interaction evidence:** Claude Code session on 2026-09-30 implementing issue #37: the DagsHub verification was run with `RECOMMENDITOS_REQUIRE_TRACKING=1` and the throwaway experiment deleted afterwards; the `mlruns` leak was found by the new `models/` guard in `tests/conftest.py` failing.
- **Other evidence:** [`recommenditos/tracking.py`](../recommenditos/tracking.py), `optional_run`, `Run` and `DisabledRun`; `tests/test_model.py::test_a_tracking_failure_does_not_fail_training`, `::test_require_tracking_makes_a_failure_fatal`, `::test_tracking_is_disabled_without_a_tracking_uri` and `::test_require_tracking_refuses_a_model_with_no_run_to_resume`; the `FORBIDDEN_PATHS` guard in [`tests/conftest.py`](../tests/conftest.py); [EDN-46](#edn-46-dagshub-credentials-live-in-two-gitignored-stores-and-the-token-is-entered-twice); [EDN-57](#edn-57-the-mlflow-retry-budget-is-pinned-to-three-requests); [issue #37](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/37).
- **In LaTeX:** no

### EDN-55: `evaluate` resumes the run `train` created instead of opening its own

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Experiment Tracking, Evaluation Protocol
- **Participants:** @lukas2510, decided by the repository owner
- **Decision:** `train@<variant>` creates the MLflow run and writes its id into `models/<variant>/model.json`. `evaluate` reads that id and reopens the same run with `resume_run`, appending the test metrics and the gate verdict. One experiment holds exactly one run per variant, and no stage opens a run of its own except the stage that first produces one.
- **Alternatives considered:**
  - **Option A: one run per DVC stage, joined by a shared tag.** This is what the course demo does and what its notes warn about.
    Pros: the simplest possible wiring; each stage owns what it logs and nothing is shared; no stage depends on another's identifier.
    Cons: four variants become eight runs, and the thing anyone actually wants - the hyperparameters next to the MdAPE they produced - is in two different rows that a reader has to join by hand. MLflow's compare view compares runs, not pairs of runs, so the one view the rubric asks for ("help you and others compare, understand and extend your results") cannot be built from it. It also doubles again as soon as #38 adds energy figures.
  - **Option B: nested runs, a parent per pipeline state with four children.**
    Pros: MLflow supports it natively and it expresses the ladder's structure.
    Cons: the parent has no natural owner. The four `train@<variant>` stages are independent DVC stages that can run in any order or alone, so whichever one created the parent would be doing it on behalf of the others, and `dvc repro train@lgbm-basic` on its own would either create a parent with one child or reuse a stale one. MLflow's compare view already handles four siblings, so the structure buys nothing it does not cost.
  - **Option C (chosen): `train` creates the run, `evaluate` resumes it.**
    Pros: one run per variant carries its hyperparameters, its train and validation loss, its model artefact, #38's energy figures and #39's test metrics and gate verdict, so the comparable view is the experiment's own table with no joining. It is also the natural place for the artefact, because the run that holds the metrics holds the model they were measured on.
    Cons: `evaluate` now depends on an identifier produced by another stage, so a model trained with tracking off has no run to append to and its metrics are dropped. That is visible rather than silent - `model.json` says `tracking_mode: "disabled"` and `run_id: null`, and `resume_run` logs which variant it could not resume - and `RECOMMENDITOS_REQUIRE_TRACKING=1` makes it fatal for the runs that matter (EDN-54). A resumed run's recorded duration is also the training run's, not the evaluation's, so run duration in the UI means "how long the fit took".
- **Rationale:** The demo's own notes flag one-run-per-stage as a wart, and the rubric asks for results that can be compared and understood. Resumption is the only one of the three options where the comparison needs no post-processing, and the cost it adds - a cross-stage identifier - is small and is carried in an artefact both stages already depend on.

  Two details were raised by the #39 side of the interface and are part of the decision rather than of the implementation. `resume_run` must be a context manager that ends the run it opened, because `evaluate` is one stage covering all four variants in one process and a run left open would collect the next variant's metrics. And a metric a stage could not measure is dropped rather than logged, because SC-04 to SC-06 are `null` until their machinery exists and MLflow has no representation for "not measured" - so the absence of the key is the record, which is consistent with the metrics artefact recording `null` and never a fabricated pass.

  Verified against the team's real DagsHub server rather than only against a local store: four runs for four variants, each with 9 to 12 parameters, the fit metrics, the four tags this stage sets and the three provenance tags the seam sets, and the whole model bundle uploaded under `model/`. Resuming one of them appended the test metrics to it, left its status `FINISHED`, correctly skipped a `null` criterion, and **did not create a fifth run**. Filtering the experiment by `tags.git_commit` returned exactly those four, which is the comparable view, and the filter string is written into the model card so a grader can reproduce it.

  One property of MLflow that the decision does not fix, and that the model card states: there is no upsert by run name, so a second `dvc repro` of one variant creates a second run rather than replacing the first. The earlier run stays as history, which is what tracking is for, and the view is filtered by the commit of the state being looked at.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** The resumption design came out of a negotiation between the agents working on issues #37 and #39 rather than from one of them, which is also where the context-manager requirement came from - the #39 side pointed out that one stage covering four variants would otherwise leak a run across variants. AI then verified the result against the real server instead of asserting it, including the run count before and after resuming, which is the assertion that distinguishes this option from option A.
- **AI interaction evidence:** Claude Code sessions on 2026-09-30 designing and implementing issues #37 and #39: the seam was agreed between the two over three rounds, and the live DagsHub verification of the run count, the parameters, the tags and the uploaded artefacts was run in a throwaway experiment that was deleted afterwards.
- **Other evidence:** [`recommenditos/tracking.py`](../recommenditos/tracking.py), `resume_run`; [`recommenditos/modeling/train.py`](../recommenditos/modeling/train.py) and [`recommenditos/modeling/evaluate.py`](../recommenditos/modeling/evaluate.py); `tests/test_model.py::test_one_variant_is_one_run_that_evaluate_appends_to`; [model card](../docs/docs/model-card.md), Training Procedure, Experiment tracking; [EDN-54](#edn-54-tracking-degrades-silently-when-it-is-not-configured-with-an-environment-variable-to-make-it-fatal); [issue #37](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/37); [issue #39](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/39).
- **In LaTeX:** no

### EDN-56: The model bundle carries its own provenance, so `models/` is deliberately not byte-reproducible

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Reproducibility, Data Versioning
- **Participants:** @lukas2510, decided by the repository owner
- **Decision:** `models/<variant>/model.json` carries `trained_at` and `mlflow.run_id` alongside the hyperparameters, the feature list and the training statistics. Both change between two otherwise identical fits, so the model directory is **not** byte-reproducible by design. The reproducibility claim NFR-06 rests on is made about the predictions and about the estimator's own payload file instead, and both are asserted.
- **Alternatives considered:**
  - **Option A (chosen): provenance inside `model.json`.**
    Pros: the bundle is self-describing wherever it ends up. EDN-08 bakes it into the API image at build time, so at serving time there is no MLflow client, no `dvc.lock` and no repository - and `GET /health` still has to be able to say which model version it serves and where its numbers came from. It is also what lets `evaluate` resume the right run (EDN-55) without a second artefact to keep in step.
    Cons: `dvc repro train@<variant>` always produces a new directory hash even when the model is identical, so `evaluate` always reruns after a retrain, and `dvc status` can never say "the model is unchanged". A reviewer comparing two pipeline states cannot tell a real change from a re-run by the hash alone.
  - **Option B: keep `models/` byte-stable and move the provenance to a second, non-cached output.**
    Pros: the model directory's hash then means exactly "the model changed", which is the property a reviewer and `dvc status` both want, and a re-run that changes nothing reruns nothing downstream.
    Cons: it needs a second `outs` entry per variant, and it puts the run id in a file `evaluate` and the API would each have to find beside the model rather than in it - a cross-stage dependency for #39 and a second file the API image has to be told to copy. It also splits one artefact that is restored together into two that might not be.
  - **Option C: drop the provenance and rely on MLflow for it.**
    Pros: `models/` becomes byte-stable at no structural cost, and MLflow already records the commit, the data version and the parameters of every run.
    Cons: it works only where MLflow is reachable. A bundle obtained with `dvc pull` on a machine with no credentials - which is the normal case for this project, see EDN-54 - could then not say which run produced it, and neither could the running API. It would also break the resumption of EDN-55 entirely, because the id would live only on the server whose runs it is meant to identify.
- **Rationale:** This is a trade against NFR-06 and it is worth being exact about which half of NFR-06 it trades. NFR-06 asks for two things: that the documented steps from a clean clone reproduce the same splits and metrics within 0.1 percentage points, and that **every training run records which code, which data version and which parameters produced it**. The provenance fields serve the second half directly. What they cost is a property NFR-06 never asks for, which is that the output directory hash is stable.

  The reproducibility that is claimed is therefore claimed about the right things and is tested. Two fits of the same variant on the same rows give bit-identical predictions for all four variants, and a byte-identical payload file - `booster.txt`, `pipeline.joblib` or `lookup.parquet`, which is the whole of the fitted model. `model.json` is excluded from that comparison explicitly, in the test and in its docstring, so nobody reads the exclusion as an oversight.

  The cost is real and is accepted rather than argued away: every `dvc repro` of a `train` stage reruns `evaluate`, even when the model is identical. At the measured scale that is about three seconds for the evaluation, against the 43 seconds of the fit that preceded it, so it changes nothing about how the pipeline is used. If the evaluation ever becomes expensive - the SC-06 masking sweep of #39 is the candidate, at roughly 25 predict calls per variant - option B becomes worth revisiting, and this entry is the record of what it would buy.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI raised the trade as an explicit decision rather than letting a timestamp drift into an artefact unremarked, which is the failure mode this repository has already had once with a test run's validation summary. It named the alternative that preserves byte stability and the cost of that alternative, and it separated the reproducibility that is claimed from the reproducibility that is not, then wrote the test so that the exclusion of `model.json` is visible in the assertion instead of being silently absent.
- **AI interaction evidence:** Claude Code session on 2026-09-30 implementing issue #37: two fits of each variant were compared byte for byte in a scratchpad before the test was written, and the payload files were confirmed identical while `model.json` differed.
- **Other evidence:** [`recommenditos/modeling/model.py`](../recommenditos/modeling/model.py), `fit_variant` and `Model.save`; `tests/test_model.py::test_the_payload_of_a_refit_is_byte_identical` and `::test_training_is_reproducible`; [requirements](../docs/docs/requirements.md) NFR-06; [EDN-08](#edn-08-model-loading-bake-into-the-api-image-via-dvc-pull-at-ci-build-time-not-the-mlflow-registry-at-runtime); [EDN-55](#edn-55-evaluate-resumes-the-run-train-created-instead-of-opening-its-own); [issue #37](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/37).
- **In LaTeX:** no

### EDN-57: The MLflow retry budget is pinned to three requests

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Experiment Tracking, CI/CD
- **Participants:** @lukas2510, decided by the repository owner
- **Decision:** `recommenditos/tracking.py` sets `MLFLOW_HTTP_REQUEST_MAX_RETRIES` to `3` with `setdefault`, rather than leaving MLflow's default of 5. A caller who wants MLflow's own budget can still ask for it by setting the variable.
- **Alternatives considered:**
  - **Option A: leave MLflow's default of 5 retries.**
    Pros: no deviation to explain, and the most patience with a flaky connection.
    Cons: measured against a closed port, giving up takes **246 s**. Because tracking degrades rather than failing (EDN-54), that is 246 seconds a stage spends before training normally, once per stage. Over the four-variant ladder it is **16.4 minutes of waiting on a server that is not there**, on top of the 67 seconds the fits actually take.
  - **Option B (chosen): 3 retries.**
    Pros: measured at **13.7 s** to give up, so the four-variant ladder loses under a minute instead of a quarter of an hour, and three attempts still ride out a connection that drops a packet or two rather than being down. `setdefault` keeps the decision overridable in the one case where patience is worth more than latency, which is a large artefact upload over a bad line.
    Cons: a deviation from the library's default, so an upload that would have succeeded on the fourth attempt now fails and that run's artefact is missing while its metrics are present. It is a warning in the log rather than a failure, which is the same quiet path EDN-54 chose and inherits the same mitigation.
  - **Option C: 0 or 1 retries.**
    Pros: essentially free to detect an unreachable server - measured at 0.00 s and 0.12 s - which is what the tests use.
    Cons: it makes a single dropped packet lose a run's tracking. For the runs whose numbers are cited that is the opposite of what is wanted, and those are exactly the runs that upload a 6.8 MB booster.
  - **Option D: probe the server with a short bounded request before configuring MLflow.**
    Pros: the fastest possible detection, with MLflow's own budget left intact for the calls that matter.
    Cons: a hand-rolled health check that duplicates what the client already does, has to know DagsHub's URL shape, and can disagree with the client about whether the server is usable. More code and one more thing that can be wrong.
- **Rationale:** The numbers make the choice, and they are steeply non-linear because MLflow backs off exponentially. Measured against `http://127.0.0.1:1/`, a port that is closed so the connection is refused rather than left hanging: 5 retries take 246.48 s, 3 take 13.68 s, 2 take 4.33 s, 1 takes 0.12 s and 0 takes 0.00 s. Three is the knee of that curve - an order of magnitude cheaper than the default while still retrying.

  One thing this entry deliberately does **not** claim, because it would be an overstatement. NFR-10 budgets 15 minutes for "training the chosen configuration", singular. At the default budget the chosen configuration alone would take about 4.8 minutes - 246 s of waiting plus a 43 s fit - which is inside NFR-10, although 85 % of it would be spent on a server that is not there. It is the **whole ladder** that would exceed the figure, at about 17.5 minutes for four variants, and the ladder is not what the requirement measures. The honest statement is that the default makes `dvc repro` unusable in practice and distorts every wall-clock number anyone reports, not that it breaks NFR-10.

  The variable is set beside the existing `MLFLOW_DISABLE_AGENT_HINT` and for the same reason: both are MLflow environment settings that this project has a view on, and one place that sets them is better than each contributor discovering them. Both use `setdefault`, so neither takes the choice away.
- **AI involvement:** Information seeking, Alternative assessment, Recommendation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** The 246-second figure came out of a measurement AI ran because a test against an unreachable server had not returned, rather than from the design, which had listed "does a bogus URI fail fast?" as an open question. AI swept the budget to find the knee instead of picking a number. Its first write-up claimed the default would eat NFR-10's training budget; checking the requirement's wording showed it budgets one configuration rather than the ladder, and the entry now says what the numbers support and explicitly disclaims the stronger version.
- **AI interaction evidence:** Claude Code session on 2026-09-30 implementing issue #37: the retry sweep against a closed port produced 246.48 s, 13.68 s, 4.33 s, 0.12 s and 0.00 s for 5, 3, 2, 1 and 0 retries, and the same measurement is why the tests pin the budget to 0.
- **Other evidence:** [`recommenditos/tracking.py`](../recommenditos/tracking.py), the `MLFLOW_HTTP_REQUEST_MAX_RETRIES` default; `tests/test_model.py::_point_at_an_unreachable_server`; [requirements](../docs/docs/requirements.md) NFR-10; [EDN-54](#edn-54-tracking-degrades-silently-when-it-is-not-configured-with-an-environment-variable-to-make-it-fatal); [issue #37](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/37).
- **In LaTeX:** no

### EDN-61: An omitted equipment list is absent, not empty, and the seam owns the masking

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Feature Encoding, API Design, Evaluation Protocol
- **Participants:** @lukas2510
- **Decision:** A request that omits one of the four equipment lists reaches the estimators as **missing** across every multi-hot column that list produced, not as an empty list. An explicitly empty list stays what it always was, every item `False`. `Model.mask_absent` in `recommenditos/modeling/model.py` is the single implementation of what an omitted field is, for the API, for SC-06's masking sweep and for the tests, and `Model.columns_of` is the single implementation of which columns a field became.
- **Alternatives considered:**
  - **Option A (chosen): omitted means absent, and one shared helper implements it.**
    Pros: it is what EDN-15 asks for - a missing value is never imputed, because the missingness carries signal - and an omitted list mapped to `False` is an imputation, and a strong one: it asserts that the car has none of roughly 146 features nobody said anything about. It is also already what the pipeline's own encoder does: `build_features.encode_equipment`, the function the API calls with a one-row frame, writes `pd.NA` across a list's columns when the source cell is null, so the serving path needs no special case to agree with the training path. And it is what makes SC-06 measure what it claims: masking a field and reading the cost only means something if the model is told the field is absent.
    Cons: the equipment columns are complete in every training row - the source is a repr string and an empty list is the literal `"[]"`, never null - so an absent equipment column is a value the estimators never saw at fit time. For LightGBM that is the missing branch with a learned default direction; for a Ridge on `extended` it is the case `MissingIndicator(features="all")` exists for (EDN-51). Both handle it, neither was fitted on it.
  - **Option B: omitted means the empty list, every item `False`.**
    Pros: it is the state the training rows actually contain, so no estimator ever sees a value it was not fitted on. It is also the reading a web form suggests: a user who ticks no checkbox has an empty list.
    Cons: it is imputation, and the confident kind. A Porsche whose comfort list was not sent would be scored as a Porsche with no comfort equipment. The form analogy is EDN-23's argument, and EDN-23 is explicitly about the three condition flags, where the source **cannot** distinguish "no" from "not stated"; for equipment the source can and does, so borrowing the argument imports a limitation that is not there. It would also make SC-06 report the cost of a false assertion rather than the cost of a missing input.
  - **Option C: leave it to each caller, as it was.**
    Pros: no new code.
    Cons: this is where the entry comes from. The documentation said both things at once - `_align`'s docstring said an absent equipment list must arrive as NaN rather than `False`, `predict_eur`'s said EDN-23 fixes it as empty - and the test helper implemented the second. Worse, both plausible one-liners a caller reaches for are half wrong and neither is caught: `frame[column] = np.nan` is refused for the three flags, because they are non-nullable, and silently correct for equipment, because a nullable `boolean` column upcasts to float64, survives `conform` and comes back as `pd.NA`; `frame[column] = False` is silently wrong for equipment and, in a numeric column, becomes `0.0` - a mileage of zero kilometres that nothing reports.
- **Rationale:** No measurement decides this one, and that is the reason it needs a decision recorded rather than a number. Measured on the real snapshot's `lgbm-extended`, masking `equipment_comfort` as absent gives 6.8982 % MdAPE and masking it as an empty list 6.8975 %: a 0.0007 pp difference, so no metric this project has can tell the two apart and nothing would ever fail. The choice therefore has to be made on what the two encodings *mean*, and the meanings are not interchangeable - one says "nobody told us", the other says "the car has none of these".

  EDN-15 settles that direction, and the contract already says it in the code: the column description `build_features` writes for every equipment feature is "Nullable because that list is: an empty list is every item False, a null list is null throughout." The two states exist in the domain; the decision is only which one an omitted request field maps to, and mapping it to the asserting one would be the project imputing at exactly the seam where it promised not to.

  The API consequence is worth stating explicitly, because it is the mistake this entry is most likely to have to prevent: a request model that defaults an omitted equipment list to `[]` would silently choose option B. An omitted list has to arrive as `None`.

  One thing this entry does **not** claim: that the seam can refuse the wrong masking. It cannot, and that is why a shared helper is the answer rather than a guard. An equipment column of all `False` is indistinguishable from a genuine empty list, and a numeric column of `0.0` from a genuine zero, so there is nothing for `predict_eur` to detect. What is left is to have one correct implementation, use it everywhere, and make the wrong value impossible to reach by accident - which is what `mask_absent` does and what `tests/test_model.py` now goes through, having previously had a private helper of its own that got the equipment case wrong.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Used as input for further analysis
- **Assessment of the AI contribution:** An adversarial review found the contradiction and the masking trap and recommended option B, reading the model card's sentence as the project's position. Following the evidence instead pointed the other way: EDN-23 does not cover equipment lists, the feature contract distinguishes the two states in so many words, `encode_equipment` already maps a null list to `pd.NA`, and `MissingIndicator(features="all")` exists for exactly the columns option B would never mask. So the recommendation was inverted and the documentation fixed in the other direction, and the shared helper - which was AI's proposal and is the durable part - was kept.
- **AI interaction evidence:** Claude Code session on 2026-09-30 reviewing pull request #62: the review reported the contradiction between `_align`'s and `predict_eur`'s docstrings as "one is wrong" without saying which, and separately measured the 0.09 % metric shift that makes the two indistinguishable; the direction was settled against the review's own recommendation by re-reading EDN-23, `_equipment_columns` and `encode_equipment`.
- **Other evidence:** [`recommenditos/modeling/model.py`](../recommenditos/modeling/model.py), `Model.mask_absent`, `Model.columns_of` and `Model._align`; [`recommenditos/data/build_features.py`](../recommenditos/data/build_features.py), `_equipment_columns` and `encode_equipment`; `tests/test_model.py::test_mask_absent_is_the_one_answer_to_what_an_omitted_field_is`, `::test_align_turns_every_boolean_column_into_a_float` and `::test_predict_eur_does_not_raise_with_every_optional_field_absent`; [EDN-15](#edn-15-uc1-required-fields-after-measuring-the-fill-rates-plus-sc-06-for-absent-optional-fields); [EDN-23](#edn-23-read-the-condition-flags-as-one-sided-assertions-instead-of-encoding-them-three-valued); [EDN-51](#edn-51-mean-fill-plus-a-per-feature-missingness-indicator-for-the-ridge-numerics-which-is-not-imputation); [pull request #62](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/62); [issue #39](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/39).
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
