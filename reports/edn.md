# Engineering Decision Notebook (working copy)

This is where we record our EDN entries as we make decisions.
`make edn` transfers the entries to the LaTeX EDN in [latex/edn/](latex/edn/), which becomes `MLOps_Recommenditos_EDN.pdf`.
This file stays the only place an entry is written or edited; the LaTeX files are generated from it.
Run `make edn` in the same pull request as any change to this file and commit what it writes: the required Tests job fails, and blocks the merge, when the LaTeX is behind.

What belongs here and what each field means: [references/Instruction_EDN_MLOps_v2026.md](../references/Instruction_EDN_MLOps_v2026.md).
The course asks for a selected set of short entries.
Record a decision when it meaningfully affects the design, quality, development, deployment, or operation of the ML system and could reasonably have gone differently.
A bug fix, an implementation detail, the use of a tool the course prescribes, and a choice without a realistic alternative get no entry; they belong in the code, the docs, or the pull request.
An entry for a milestone we have not built yet states the design we have settled on, not something already running; the preamble of the PDF says so once, so an entry does not have to repeat it.

On 2026-10-12 and 2026-10-13 the 79 entries EDN-01 to EDN-81 were trimmed to the 37 that meet this criterion, and every kept entry was rewritten to the shape below.
An entry that amended, reversed, or continued another one was merged into the surviving entry, which states the decision as it stands now.
The originals are kept word for word in [edn-archive.md](edn-archive.md), which the PDF does not include, and [Retired and merged IDs](#retired-and-merged-ids) at the end of this file says what became of each ID.

How to add an entry:

- Copy the template at the end of this file and append the new entry after the last one, oldest entry first.
- Give it the next free ID (`EDN-01`, `EDN-02`, ...).
  IDs never change once assigned and are never reused, because the report, the docs, and the code refer to them.
  Reserve the ID by adding its heading to this file on `main` before opening the branch that needs it.
  EDN-38 and EDN-76 were never assigned: they are skipped numbers, not removed entries.
- Keep to the shape.
  An entry is read in about a minute, so it has about 220 words and never more than 300.
  The 300 are the one hard limit: `python -m tools.edn_latex --check` and the tests fail on an entry above it.
  Counted are the title, the note line, and the fields from **Decision** to **Other evidence**; a link counts by the words of its text, and text in double quotes does not count.
  The figures per field are guide values:
  - title: one clause of about ten words that states the decision;
  - **Decision**: what we do, in about 45 words;
  - **Alternatives considered**: only alternatives we really weighed, one line each with the reason it lost; one is enough, and more than three is rare;
  - **Rationale**: the deciding argument, in about 60 words and with one or two numbers, each checked against its source; link the analysis and do not retell it;
  - **Assessment of the AI contribution**: what AI contributed and what we checked, changed, or rejected, in about 45 words;
  - **AI interaction evidence**: the session's date and topic and the prompt that led to the decision, in double quotes: word for word when we have it, otherwise its core, marked "(paraphrased)"; the professor asked on 2026-10-27 for this much detail, a session and topic alone is not enough, and entries recorded before that date keep their evidence as it is;
  - **Other evidence**: about four links.
- When a decision is revised, edit its entry so that it states the current decision, and say in one note line above the fields what changed and when.
  Do not add a second entry for the same decision.
- Fill in every field; write "-" for an optional field that does not apply.
- Review AI conversation excerpts before quoting or linking them; never include personal or sensitive data.
- Leave **In LaTeX** at "no" and run `make edn`, which sets it to "yes" as it transfers the entry to `latex/edn/`.
  A heading with no fields yet is a reservation and is not transferred.

## Entries

### EDN-01: The main dataset is the AutoScout24 listings snapshot

> **Revised on 2026-10-18**: records who found the dataset; the decision is unchanged.
>
- **Date:** 2026-09-22
- **Milestone:** M1: Project Inception
- **Activity / Topic:** Dataset Selection
- **Participants:** @michudud04 found the dataset and proposed it; all team members confirmed the choice
- **Decision:** Use the AutoScout24 Car Listings Dataset (2025 snapshot, Zenodo DOI `10.5281/zenodo.17643343`) as the training data for the used-car price component.
- **Alternatives considered:**
  - Smaller generic Kaggle used-car datasets: mostly one country, fewer rows and features, weaker or undocumented provenance and licence.
- **Rationale:** With 118,382 listings and 75 columns from 8 countries it is the only candidate large and rich enough for gradient boosting with equipment and history features.
  Its provenance and licence are citable.
  Its skew toward premium brands is accepted as a stated scope limit (EDN-05).
- **AI involvement:** Information seeking, Alternative assessment
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** @michudud04 found the dataset on Zenodo after
  a preliminary inspection of related datasets. It was selected because of its large
  number of rows. AI then profiled the raw file (rows, columns, countries, brand mix,
  missing values) and summarised the trade-off against the smaller datasets.
  The team reviewed the figures and confirmed the choice in sprint planning.
- **AI interaction evidence:** Dataset profile and trade-off in `docs/docs/project-brief.md`, sections 2 and 3.1 (2026-09-22).
- **Other evidence:** [PR #13](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/13), [issue #3](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/3).
- **In LaTeX:** yes

### EDN-02: The model family is gradient boosting with LightGBM

- **Date:** 2026-09-22
- **Milestone:** M1: Project Inception
- **Activity / Topic:** Modelling Approach
- **Participants:** All team members
- **Decision:** Use gradient boosting, with LightGBM as the main model on a basic and an extended feature set and CatBoost as a challenger for high-cardinality categoricals.
  A median and a Ridge baseline stay in the experiment ladder.
- **Alternatives considered:**
  - Linear or Ridge regression only: simple and interpretable, but it misses interactions such as make by age by mileage.
  - kNN, random forest, tabular neural networks, hierarchical Bayes, LLM zero-shot: slower or weaker at this data size, or no fit for a CPU-only deployment with a small image.
- **Rationale:** The course grades MLOps practice over raw accuracy.
  LightGBM trains in minutes on a CPU, takes categoricals and missing values natively, keeps artefacts small, and gives exact SHAP explanations.
  The baselines show whether it earns its added complexity.
- **AI involvement:** Information seeking, Alternative assessment
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI laid out the alternatives and the CPU, size, and explainability constraints.
  The team reviewed them and confirmed the choice in sprint planning, including CatBoost as a challenger only.
- **AI interaction evidence:** "Why gradient boosting" in `docs/docs/project-brief.md`, section 4 (2026-09-22).
- **Other evidence:** [PR #13](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/13).
- **In LaTeX:** yes

### EDN-03: The Spanish listings are held out as the new-market scenario

> **Absorbs EDN-40** (2026-09-30): the holdout is selected per listing.

- **Date:** 2026-09-22
- **Milestone:** M1: Project Inception
- **Activity / Topic:** Dataset Selection, Monitoring
- **Participants:** @lukas2510, after supervisor feedback (S. del Rey)
- **Decision:** All AutoScout24 `ES` listings are held out of training, validation, and calibration and replayed in M6 as a new market.
  The DataMarket sample of Spanish cars is dropped.
  The holdout is selected per listing, and the `split` stage reports any seller that also lists outside `ES`.
- **Alternatives considered:**
  - DataMarket as a stress test on the shared makes: market, year, portal, and brand mix shift together, so an alarm could not be attributed, and it has no licence file.
  - Holding out whole sellers that touch `ES`: puts non-Spanish rows into the holdout and gains nothing, since no seller crosses the border in the real snapshot.
- **Rationale:** The supervisor asked whether DataMarket is feasible as a new market.
  It is not: 36.5 % of its rows are makes the model never sees.
  Spain is already in our data with 8,015 raw listings, same schema and scrape date, and every listing has a price, so M6 can show input drift and the error it causes.
- **AI involvement:** Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** The options came from our own feasibility check.
  AI compared them against the M6 rubric and recommended `ES` over `IT`, which would cost about three times the rows.
  We accepted because it answers the supervisor's concern with evidence and keeps one data source.
- **AI interaction evidence:** Claude Code session of 2026-09-22 on project brief section 3.2.
- **Other evidence:** [PR #13](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/13), [PR #53](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/53), [project brief](../docs/docs/project-brief.md) section 3.2, [`split_gate_results.txt`](analysis/split_gate_results.txt).
- **In LaTeX:** yes

### EDN-04: The model covers used passenger cars only

> **Absorbs EDN-24** (2026-09-29): the pre-registered exclusion stays although its flag is one-sided.

- **Date:** 2026-09-22
- **Milestone:** M1: Project Inception
- **Activity / Topic:** Problem Specification
- **Participants:** @lukas2510
- **Decision:** The model covers used passenger cars: `offer_type = U`, not pre-registered, `vehicle_type = Car`.
  The pipeline filters out new cars, pre-registered cars, and transporters, and the API rejects them.
  `is_preregistered` defines the scope and is therefore not a feature.
- **Alternatives considered:**
  - Keep everything, with the flags as features: more data, but one model would learn two price mechanisms, list prices and the second-hand market.
  - Detect unmarked pre-registered cars with a heuristic: there is no ground truth to validate it, so it would drop ordinary used cars.
- **Rationale:** An exploratory LightGBM run showed almost no accuracy difference between the two scopes, so product clarity decides: a used-car price component prices used cars.
  Known limit: a `False` in `is_preregistered` can mean "not stated" (EDN-23), so some unmarked pre-registered listings remain in the training data.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI profiled the data, ran the comparison on both scopes, and recommended this scope.
  Lukas checked that the numbers came from the real dataset.
  AI later found that its own proposal to use `is_preregistered` as a feature contradicted this entry, and withdrew it.
- **AI interaction evidence:** Claude Code sessions of 2026-09-22 (issue #2) and 2026-09-29 (issue #25).
- **Other evidence:** [Problem specification](../docs/docs/problem-spec.md) section 2, [issue #2](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/2), [PR #17](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/17), [issue #25](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/25).
- **In LaTeX:** yes

### EDN-05: A make is supported from 300 listings

> **Absorbs EDN-48 and EDN-67** (2026-09-30, 2026-10-05): `features` is the one stage that applies the list.

- **Date:** 2026-09-22
- **Milestone:** M1: Project Inception
- **Activity / Topic:** Problem Specification, Data Preparation
- **Participants:** @lukas2510
- **Decision:** Only makes with at least 300 listings in the cleaned used-car data, counted after the `ES` holdout and before the split, are supported.
  The `split` stage computes the list and removes no rows.
  The `features` stage restricts every frame to it, the `ES` holdout included, before it builds any vocabulary.
  The API rejects every other make.
- **Alternatives considered:**
  - Accept all 25 makes and flag rare ones as low confidence: weak predictions for makes with a few dozen listings.
  - Filter in `split`: the split becomes lossy, and the threshold cannot be revisited without a new split.
  - Record the list and measure the metrics over every make: the gate would cover listings the API refuses with a 422.
- **Rationale:** The dataset is skewed toward premium brands (EDN-01).
  A threshold turns that limit into an explicit, testable rule.
  At 300 it keeps 11 of the 25 makes and 98.6 % of the split listings.
  Applying it in `features` makes the reported population the served one, and `train` and `evaluate` fail when that does not hold.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI counted the listings per make after scoping and deduplication, compared thresholds of 200, 300, and 500, and recommended the threshold.
  Lukas accepted because it states the brand skew openly instead of hiding it behind a warning.
  AI later found that the code and two documents disagreed on where the filter belongs, raised that as a decision, and corrected its own claim that out-of-scope rows would inflate the metrics.
- **AI interaction evidence:** Claude Code sessions of 2026-09-22 (issue #2), 2026-09-30 (issue #35), and 2026-10-05 (issue #63); prompt: "How do we handle the 25 makes, many of which have very few listings?".
- **Other evidence:** [Problem specification](../docs/docs/problem-spec.md) section 2, "The supported makes" in [the pipeline docs](../docs/docs/pipeline.md), [`served_vocabulary_results.txt`](analysis/served_vocabulary_results.txt), [issue #2](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/2), [PR #17](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/17), [PR #71](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/71).
- **In LaTeX:** yes

### EDN-06: A model is deployable only when it meets the success criteria

- **Date:** 2026-09-22
- **Milestone:** M1: Project Inception
- **Activity / Topic:** Performance Criteria
- **Participants:** @lukas2510
- **Decision:** A model is deployable when it meets SC-01 to SC-05 on the seller-grouped test set: MdAPE at most 9 %, at least 85 % of predictions within ±20 %, MdAPE at least 30 % below the median baseline, MdAPE at most 15 % in every segment level with 500 or more test rows, and 88 % to 92 % coverage of the nominal 90 % intervals.
  EDN-15 added SC-06, which the gate requires as well.
- **Alternatives considered:**
  - Baseline comparison only: says nothing about usefulness, and a weak baseline makes it easy to pass.
  - MAPE-based targets: dominated by cheap cars and outliers.
- **Rationale:** The thresholds come from an exploratory run on 96,831 listings, where the median baseline reached 11.9 % MdAPE and an untuned LightGBM 6.7 %.
  They leave a margin, so they hold across splits and still rule out a model barely better than the baseline.
  Price buckets are excluded from the segments because they condition on the target.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI ran the exploratory models and proposed the criteria with their thresholds.
  Lukas accepted because the thresholds are tied to measured values.
  AI's segment check then showed SC-04 missed for cars older than 20 years; we kept the criterion and documented the gap (EDN-62).
- **AI interaction evidence:** Claude Code session of 2026-09-22 on issue #2; prompt: "Which kind of success criteria?".
- **Other evidence:** [Problem specification](../docs/docs/problem-spec.md) sections 6 to 8, [issue #2](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/2), [PR #17](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/17).
- **In LaTeX:** yes

### EDN-08: The model is baked into the API image

- **Date:** 2026-09-22
- **Milestone:** M4: Model Deployment
- **Activity / Topic:** ML System Design
- **Participants:** @lukas2510
- **Decision:** CI builds the API image with the production model inside, fetched with `dvc pull` at the state `dvc.lock` on `main` points to.
  Promoting a model is a merge to `main`.
  The API never calls the MLflow registry; MLflow stays the record of the experiments and of the run that was promoted.
- **Alternatives considered:**
  - Pull from the MLflow registry at API startup: startup then depends on DagsHub being reachable, against NFR-05 and NFR-12.
  - CD resolves the registry alias and bakes the model on promotion: needs a trigger from MLflow to GitHub Actions that nobody would watch.
- **Rationale:** The course demo deploys by pulling the model with DVC before it starts the server, and we do the same in a container.
  GitHub Flow, already our branching model, becomes the promotion mechanism, so there is no new infrastructure to build or to fail before a presentation.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI first recommended the registry-alias option.
  Asked whether that fits the project's scope, it read `references/course-demos.md`, found that the demo pulls with DVC, and revised its recommendation, naming the cost: the MLflow registry leaves the deployment path.
  Lukas confirmed the revised option.
- **AI interaction evidence:** Claude Code session of 2026-09-22, review of PR #19; Lukas asked: "considering our project scope and the tools we will use is this the best option for us?".
- **Other evidence:** [Specification](../docs/docs/specification.md) FR-12, [project brief](../docs/docs/project-brief.md) section 5, [course demo notes](../references/course-demos.md), [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** yes

### EDN-10: The availability target is recovery within 30 seconds

- **Date:** 2026-09-22
- **Milestone:** M6: Monitoring
- **Activity / Topic:** Resource Monitoring, Requirements
- **Participants:** @lukas2510
- **Decision:** NFR-05 no longer promises 99 % uptime.
  The API recovers by itself within 30 seconds of a crash or a reboot, which a rehearsed crash verifies.
  It has no unplanned downtime during a graded presentation and its warm-up.
  Outside those windows, uptime is monitored with Better Uptime and Grafana and reported, not promised.
- **Alternatives considered:**
  - Keep 99 % uptime: nothing backs it on one VM without redundancy, SLA, or on-call, and its measurement window was unbounded.
  - Lower the number to 95 %: as unverifiable as 99 %.
- **Rationale:** M6 grades whether monitoring gives actionable insight, not whether a percentage was hit.
  The new requirement separates what we control, recovery and the presentation days, from what we can only observe.
  The recovery clause is testable today.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI had flagged that Prometheus's `up` metric cannot see the host going down and that Better Uptime, a course tool, was missing from the docs.
  Asked which target is feasible, it proposed the three-part requirement.
  Lukas confirmed it.
- **AI interaction evidence:** Claude Code session of 2026-09-22, review of PR #19; Lukas asked: "what NFR that goes in the uptime direction is feasible with our system and constraints?".
- **Other evidence:** [Requirements](../docs/docs/requirements.md) NFR-05, [specification](../docs/docs/specification.md) NFR-05, [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** yes

### EDN-11: Serving explains predictions without the `shap` package

> **Absorbs EDN-09** (2026-09-22): the project moved to Python 3.12 so that `shap` runs for offline analysis.

> **Revised on 2026-10-20**: `shap` became a `pipeline`-group dependency with FR-08's implementation (#136), and the rationale now says why the equality with `shap.TreeExplainer` is no independent check; the decision is unchanged.

- **Date:** 2026-09-22
- **Milestone:** M4: Model Deployment
- **Activity / Topic:** ML System Design, Explainability
- **Participants:** @lukas2510
- **Decision:** The API computes FR-08's per-request explanation from the booster's own SHAP export, LightGBM's `predict(pred_contrib=True)`.
  The `shap` package is a `pipeline`-group dependency for offline analysis and the tests, and is never installed in the API image.
  The project runs on Python 3.12, which `shap` requires.
- **Alternatives considered:**
  - `shap.TreeExplainer` in the serving path: one implementation everywhere, but `shap` imports `numba` and `llvmlite` at load, and both would enter the image.
  - Stay on Python 3.11 without `shap`: the offline analysis, such as summary plots, still needs the package.
- **Rationale:** The native export returns the same TreeSHAP values, so serving loses nothing and the image stays inside the size budget of NFR-04 with more headroom.
  For LightGBM the equality holds by construction, because `shap.TreeExplainer` calls `pred_contrib` itself; the real check is local accuracy, each row's values adding up to the model's log price.
  CatBoost has an equivalent export, so the choice survives a change of model.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI first proposed the native export as a workaround for a conflict under Python 3.11.
  Lukas asked it to check a version bump instead, and a scratch resolve and install under 3.12 with the planned dependencies found no blocker.
  AI then built Docker images with and without `shap` and compared the two explanations.
  Lukas confirmed on that evidence.
- **AI interaction evidence:** Claude Code session of 2026-09-22, review of PR #19; Lukas asked: "is this what we would want? or are there better options?".
- **Other evidence:** [Specification](../docs/docs/specification.md) FR-08, [project brief](../docs/docs/project-brief.md) section 6, [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19), [PR #165](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/165).
- **In LaTeX:** yes

### EDN-12: Retraining and promotion are started by a person, and a failing gate blocks the merge

> **Revised on 2026-10-20**: a required CI check now enforces the verdict, which skips criteria an EDN entry waived; before, only the reviewer read it.

- **Date:** 2026-09-22
- **Milestone:** M6: Monitoring
- **Activity / Topic:** ML System Design, Model Performance
- **Participants:** @lukas2510
- **Decision:** When the drift job flags the `ES` replay, a team member retrains with `ES` included and opens the promotion pull request by hand.
  The merge is the promotion, and redeploying the previous image tag is the rollback (EDN-08).
  A required CI check fails a pull request that changes `dvc.lock` unless the committed `metrics.json` has `gate_passed: true` and names the promoted model as `deployable_variant`.
  A criterion waived in `params.yaml` does not count; EDN-62 records the SC-04 waiver.
  FR-15 states this; the reviewer's approval stays on top.
- **Alternatives considered:**
  - A watcher that runs `dvc repro` and opens the pull request on confirmed drift: new infrastructure to debug just before the final presentation, with little slack if it breaks.
  - The gate as a manual check only: one mistaken merge deploys a rejected model.
  - A check in CD: it fails only after the merge.
- **Rationale:** The `ES` holdout exists to show this loop closing (EDN-03).
  A person reacting to the signal closes it too, and is safer live.
  From M5 every merge to `main` deploys (NFR-13), and only the CI check stops a rejected model before the merge.
  It needs no data and reads `gate_passed`, so a new criterion changes nothing in it.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI noticed that the retrain-after-drift plan had never become a testable requirement and recommended the manual trigger because of the timeline.
  On 2026-10-20 it found that no check read the verdict, laid out the options and recommended the CI check.
  Lukas decided.
- **AI interaction evidence:** Claude Code sessions of 2026-09-22 (review of PR #19) and 2026-10-20 (issue #129).
- **Other evidence:** [Requirements](../docs/docs/requirements.md) FR-15, [specification](../docs/docs/specification.md) FR-14 and FR-15, [issue #129](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/129), [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** yes

### EDN-13: The feedback endpoint is reachable only inside the Compose network

- **Date:** 2026-09-23
- **Milestone:** M6: Monitoring
- **Activity / Topic:** API Design, Monitoring
- **Participants:** @lukas2510
- **Decision:** `POST /feedback` (FR-10) stays.
  It takes a request ID and an observed price, and FR-14 joins these delayed labels to the prediction log to report the real error and the interval coverage.
  A reverse proxy refuses the endpoint from outside, so only the replay job reaches it; the API key stays as a second layer.
  NFR-09 therefore needs no TLS.
- **Alternatives considered:**
  - A public endpoint protected by the API key alone: the key would travel in cleartext unless the VM gets TLS, which its NAT setup probably does not allow.
  - No endpoint, the drift job reads the `ES` prices from the artefact: least code, but ground truth never enters the system through an interface.
- **Rationale:** Without labels the monitoring shows only that the inputs changed.
  With them it shows the error rising and the intervals losing coverage, which is what M6 grades and the stronger trigger for retraining.
  An internal endpoint removes an unverified assumption about TLS on the VM.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI flagged that FR-10 was marked both open and decided, and that four requirements depended on it.
  It laid out the options and named the counter-argument itself: the labels are replayed, so the report must say so.
  Lukas accepted the recommendation.
- **AI interaction evidence:** Claude Code session of 2026-09-23, review of PR #19; Lukas asked (in German) for a closer explanation of FR-10 and of the recommendation.
- **Other evidence:** [Specification](../docs/docs/specification.md) FR-10, FR-14, and NFR-09, [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** yes

### EDN-14: Drift is tested on windows of 1,000 requests at level 0.005

> **Absorbs EDN-26 and EDN-29** (2026-09-29): the window, the exclusion of `model`, and the significance level, each set after a new measurement.

- **Date:** 2026-09-23
- **Milestone:** M6: Monitoring
- **Activity / Topic:** Monitoring, Requirements
- **Participants:** @lukas2510
- **Decision:** The drift job compares windows of 1,000 requests with 10,000 training listings, feature by feature, Bonferroni-corrected from a family-wise level of 0.005 over the 23 tested features, without the `model` feature.
  The `ES` replay must be flagged with at least three drifted features besides `country_code`.
  Of 20 i.i.d. control windows, at most 1 may be flagged.
- **Alternatives considered:**
  - A seller-grouped control, as NFR-11 first said: measured, it is flagged as well, because unseen dealers differ as much as an unseen country.
  - Windows of 100 requests: three drifted features show up in at most 82 % of trials.
  - The usual level of 0.05: its false-alarm rate equals the 1 in 20 the requirement allows, and the control showed 0 to 2 flagged windows of 20, depending on the split.
- **Rationale:** NFR-11 claimed properties of the data that nobody had measured.
  With these settings the `ES` replay is flagged in every trial of three splits, and at most 0.6 of 20 control windows are.
  `model` is left out because its 349 levels are too many for a chi-square test on a window of this size.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI flagged the unverified claim and measured it on the real data.
  Its own later measurements refuted two of its proposals: a window of 100 requests, and excluding `model` at that window.
  A re-run after a scope change then showed that the 0.05 level was unstable.
  We adopted each change only once a committed script reproduced it.
- **AI interaction evidence:** Claude Code sessions of 2026-09-23 and 2026-09-29, review of PR #19.
- **Other evidence:** [Specification](../docs/docs/specification.md) NFR-11, [`nfr11_alpha_sweep_results.txt`](analysis/nfr11_alpha_sweep_results.txt), [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19), [PR #47](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/47).
- **In LaTeX:** yes

### EDN-15: `/predict` requires ten fields, with missing inputs bounded by SC-06

> **Absorbs EDN-16 and EDN-59** (2026-09-29, 2026-09-30): UC2's own endpoint, and how SC-06 is measured.

- **Date:** 2026-09-29
- **Milestone:** M1: Project Inception
- **Activity / Topic:** Requirements Engineering, API Design, Performance Criteria
- **Participants:** @lukas2510
- **Decision:** `/predict` requires `body_type` and `seller_type` on top of the eight core fields; the other basic fields stay optional.
  UC2 keeps its own endpoint, `/price-range`, which requires far fewer fields.
  SC-06 bounds the result: with each optional input field masked on its own, and with all of them masked at once, MdAPE stays at or below 1.5 times the full-input MdAPE.
- **Alternatives considered:**
  - All basic fields optional: the model never sees the two fields absent, so the contract would promise untested behaviour.
  - Require every basic field: `nr_prev_owners` is filled in only 61.0 % of the listings.
  - SC-06 over feature columns: it would mask the car's age and single equipment items, which no request can omit.
  - One `/valuation` endpoint for both use cases: two contracts in one conditional schema, and a changed mix of clients would look like drift (EDN-14).
- **Rationale:** In the 97,889 listings of the training scope, `body_type` is filled in 100.000 % and `seller_type` in 99.986 %.
  The model cannot learn their absence, and whoever owns the car knows both.
  With SC-06 in the NFR-01 gate, the gate also covers the partial requests FR-01 allows.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI measured the fill rates and separated the API contract from the training mechanism.
  It found that a criterion over feature columns would count the car's age as optional.
  Its proposal to mask the condition flags as missing values was rejected, because it contradicts EDN-23 and the contract.
  For the endpoints it contributed the monitoring argument; Lukas dropped its further suggestions, a shared base schema and a combined convenience endpoint.
- **AI interaction evidence:** Claude Code sessions of 2026-09-29 (PR #19) and 2026-09-30 (issues #37 and #39).
- **Other evidence:** [Specification](../docs/docs/specification.md) FR-01, FR-06, and FR-07, [problem specification](../docs/docs/problem-spec.md) SC-06, [`fillrates_results.txt`](analysis/fillrates_results.txt), [`evaluate.py`](../recommenditos/modeling/evaluate.py).
- **In LaTeX:** yes

### EDN-17: The component is deployed on the FIB Virtech VM

> **Revised on 2026-10-12**: the comparison of the free tiers was added from section 2.1.5 of the report; the entry of 2026-09-29 weighed "another cloud" on cost and account grounds only.

- **Date:** 2026-09-29
- **Milestone:** M4: Model Deployment
- **Activity / Topic:** ML System Design, Deployment
- **Participants:** @lukas2510
- **Decision:** The component is deployed on the FIB Virtech VM that the course offers: 4 GB of memory, 20 GB of disk, CPU only, free for one semester.
  Every target in the requirements is written for this machine.
- **Alternatives considered:**
  - The free tiers of AWS and GCP: 1 GB and 0.6 GB of memory do not hold an API container, Prometheus, Grafana, and a scheduled drift job.
  - Okteto or Azure: 3 GB, and 1 to 4 GB depending on the instance, so neither has more memory than the FIB machine.
  - A paid cloud or a free ARM tier: tied to a private account with payment details, and an ARM tier would need multi-architecture images.
- **Rationale:** Memory decided the choice, together with the fact that the course provides the machine.
  4 GB is enough with the budgets of NFR-04: the API container stays below 1 GB, the drift job runs in its own container, and metrics and logs are capped.
  The machine is a single unattended host without redundancy, which EDN-10 accounts for.
- **AI involvement:** Information seeking, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI first reported the brief and the requirements as contradicting each other on the target.
  Asked to look again, it corrected its own framing: the requirements treated the VM as settled, so the issue was a decision nobody had written down.
  Lukas fixed the target and left the performance figures as proposed until the first load test.
- **AI interaction evidence:** Claude Code session of 2026-09-29, review of PR #19; Lukas replied "yes do that".
- **Other evidence:** [Specification](../docs/docs/specification.md) section 2 and NFR-04, [course demo notes](../references/course-demos.md), [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** yes

### EDN-18: Unseen countries and models are accepted with a warning

> **Absorbs EDN-42** (2026-09-30): the pipeline encodes an unseen categorical value as missing.

- **Date:** 2026-09-29
- **Milestone:** M4: Model Deployment
- **Activity / Topic:** API Design, Feature Encoding
- **Participants:** @lukas2510
- **Decision:** A `country_code` or a `model` that the training data does not contain is accepted, passed to the model as unknown, and reported in the response's `warnings` field (FR-05).
  The `features` stage follows the same rule: a categorical has the training levels only, and any other value becomes missing.
  The `ES` holdout's `country_code` is therefore missing in every row, on purpose.
- **Alternatives considered:**
  - Reject with HTTP 422: every `ES` listing would be refused, so the drift scenario of EDN-03 could not be replayed.
  - Give the holdout country a level of its own: the model would receive a code it was never fitted on.
  - Drop `country_code` from the features: discards a real price driver to tidy one holdout frame.
- **Rationale:** EDN-03 holds Spain out so that unseen traffic can be replayed against the running API.
  LightGBM handles a missing category natively, so nothing has to be built.
  The scope check we do want is the make, which keeps its 422 (FR-04).
  The `features` stage warns about a column that a split never fills, and a test pins the holdout case.
- **AI involvement:** Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI pointed out that EDN-03 had already settled the API rule.
  An adversarial AI review later found the empty holdout column on the real snapshot and recommended documenting it instead of changing the encoding.
  We accepted both.
- **AI interaction evidence:** Claude Code sessions of 2026-09-29 (PR #19) and 2026-09-30 (review of PR #55).
- **Other evidence:** [Specification](../docs/docs/specification.md) FR-05, [problem specification](../docs/docs/problem-spec.md) section 2, [`tests/test_features.py`](../tests/test_features.py), [PR #55](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/55).
- **In LaTeX:** yes

### EDN-19: The DagsHub repository belongs to a team organisation

- **Date:** 2026-09-29
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Data Versioning, Experiment Tracking
- **Participants:** @lukas2510, on behalf of the team; confirmed by the full team at the laboratory session of 2026-10-13, where all five members also confirmed that they are members of the organisation and can push
- **Decision:** The DVC remote and MLflow server is `recommenditos/MLOPS_Recommenditos` on DagsHub, owned by a DagsHub organisation of the team and connected through the GitHub integration.
  All five members have write access, granted per repository.
- **Alternatives considered:**
  - An existing mirror under another account: it was the best in sync with GitHub, but its owner is not on the team.
  - A repository under one member's account: the project's data and experiment history would hang off one private account.
  - One mirror per member: the remote URL is committed in `.dvc/config`, so members would overwrite each other's setting.
- **Rationale:** Cohort members outside the team have read access to our GitHub repository and can mirror it at any time.
  The rule is therefore that the project's infrastructure lives where the team owns it.
  An organisation costs one more form than a personal repository.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI found two existing mirrors, measured them against GitHub and recommended the better-synced one.
  The measurements were right and the recommendation was wrong: AI never checked who owned the mirror.
  Lukas asked, the owner turned out to be outside the team, and the decision taken on it was withdrawn the same day.
  We accepted AI's second recommendation.
- **AI interaction evidence:** Claude Code session of 2026-09-29; the retraction followed Lukas's question about who owns the recommended mirror.
- **Other evidence:** [PR #26](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/26), [issue #10](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/10), `.dvc/config`.
- **In LaTeX:** yes

### EDN-20: Personal data stays on a public DagsHub remote

> **Absorbs EDN-34 and EDN-37** (2026-09-30): the raw Parquet keeps its personal-data columns, and the seller key is an unsalted hash.

- **Date:** 2026-09-29
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Data Versioning, Privacy
- **Participants:** @lukas2510; confirmed by the full team at the laboratory session of 2026-10-13
- **Decision:** The DagsHub repository that serves as DVC remote and MLflow server stays public.
  The raw Parquet of the `download` stage keeps the seven personal-data columns of the published file and is pushed like any stage output; preprocessing removes them.
  `seller_group_id` is an unsalted hash that groups the split and anonymises nothing, as the dataset card says.
- **Alternatives considered:**
  - A private repository: every grader and supervisor needs a collaborator entry, and the identical file stays public on Zenodo.
  - `push: false` on the raw Parquet: each machine regenerates the pipeline's first artefact, which puts byte-identical reproduction at risk.
  - A secret pepper in the hash: the split would depend on a secret outside the repository, so a clean clone could not reproduce it.
- **Rationale:** The same data is public under the pinned Zenodo DOI, so the added exposure is small.
  It is not nothing: hosting a copy makes the team a publisher of that personal data in its own right.
  We accept this risk and revisit it once we host data that is not public elsewhere.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Used as input for further analysis
- **Assessment of the AI contribution:** AI showed that its own earlier check, "the repository is private", was wrong, and laid out both options without recommending one; Lukas decided.
  For the Parquet, AI preferred removing the columns in `download`; Lukas took the simpler option.
  An adversarial AI review inverted every seller id with a dictionary, so we corrected the claim instead of adding a pepper.
- **AI interaction evidence:** Claude Code sessions of 2026-09-29 and 2026-09-30 (reviews of PR #48 and PR #56); Lukas answered (translated from German): "we keep it public".
- **Other evidence:** [Dataset card](../docs/docs/dataset-card.md), [PR #26](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/26), [PR #48](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/48), [PR #56](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/56).
- **In LaTeX:** yes

### EDN-23: Condition flags are read as one-sided assertions

- **Date:** 2026-09-29
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Feature Encoding and Data Validation
- **Participants:** @lukas2510
- **Decision:** `has_full_service_history`, `non_smoking`, and `is_rental` stay plain booleans.
  They mean "the seller asserted this" or "the seller did not assert this", never "yes" or "no".
  No three-valued encoding is built.
  `is_used` is not a feature, because `offer_type` carries the same information reliably.
- **Alternatives considered:**
  - A three-valued encoding, or `False` mapped to missing: the third state would be empty, and a tree splits `{True, NaN}` exactly like `{True, False}`.
  - Drop the flags: discards the reliable half, since a `True` is a real assertion.
  - Recover the value from the free text: multilingual text parsing is out of scope and has no ground truth.
- **Rationale:** A listing form offers a checkbox, so a tick is evidence and an empty box is the absence of evidence.
  No encoding can separate "no" from "not stated".
  The risk is a reader who takes `False` as a denial, so the fix belongs in the problem specification, the dataset card, and the model card.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** The three-valued encoding was AI's own earlier proposal in issue #25.
  On re-reading the issue, AI argued against it and showed that it adds a state without adding information for tree models.
  Lukas accepted the reversal because the argument was concrete.
- **AI interaction evidence:** Claude Code session of 2026-09-29 on issue #25; Lukas chose "keep them binary".
- **Other evidence:** [Issue #25](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/25), [problem specification](../docs/docs/problem-spec.md), [dataset card](../docs/docs/dataset-card.md).
- **In LaTeX:** yes

### EDN-25: The `download` stage fetches the raw file from Zenodo

> **Absorbs EDN-07, EDN-35, and EDN-36** (2026-09-22 to 2026-09-30).

- **Date:** 2026-09-26
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Data Versioning, Pipeline Design
- **Participants:** @W11W11W11 decided the `dvc add` form of 2026-09-26, @lukas2510 the current form of 2026-09-30; the full team confirmed the current form at the laboratory session of 2026-10-13
- **Decision:** The `download` stage fetches the pinned AutoScout24 CSV from its Zenodo record and writes `data/raw/listings.parquet`, its only output.
  The CSV stays a local cache under `data/external/`, and `download.md5` in `params.yaml` pins its bytes.
  We neither track nor host the CSV.
  This replaced, on 2026-09-30, the `dvc add` pointer chosen on 2026-09-26.
- **Alternatives considered:**
  - `dvc import-url` with `push: false`, the first decision of 2026-09-22: Zenodo sends neither an `ETag` nor a `Content-MD5` header, which DVC's change detection relies on.
  - `dvc add` and push to DagsHub, chosen on 2026-09-26: the course demo's pattern, but acquisition stays a manual step and we re-host the CSV.
  - The CSV as a dependency or output of the stage: DVC would hash the whole file on every `dvc status`.
- **Rationale:** Once the pipeline exists it should acquire its own input, so a clean clone reproduces with `dvc repro` and no manual step.
  The stage hashes the cached file on every run and refuses any other bytes.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Other: rejected on 2026-09-26, partly adopted on 2026-09-30
- **Assessment of the AI contribution:** AI recommended `dvc import-url`, to keep the raw personal data off our remote.
  @W11W11W11 rejected that for `dvc add`, because the course demo teaches it.
  Four days later @lukas2510 retired the pointer, so that the pipeline owns its input.
  That adopts the recommendation only in part: the file comes from Zenodo again, but `dvc import-url` stays rejected and its aim is not met, because the raw Parquet with the same seven personal-data columns is pushed to our public remote (EDN-20).
  AI's claim that DVC cannot keep such a file in the graph was wrong; an AI review found the option.
- **AI interaction evidence:** Claude Code sessions of 2026-09-26 (issue #3) and 2026-09-30 (issue #33, review of PR #54).
- **Other evidence:** [Data versioning](../docs/docs/data-versioning.md), [PR #27](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/27), [PR #54](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/54).
- **In LaTeX:** yes

### EDN-27: Requirements are separated from their specification

- **Date:** 2026-09-29
- **Milestone:** M1: Project Inception
- **Activity / Topic:** Requirements Engineering
- **Participants:** @lukas2510
- **Decision:** `docs/docs/requirements.md` states what must be true, for whom, with a priority and an acceptance criterion, and names no endpoints, formats, status codes, or deployment mechanics.
  The design detail moves to `docs/docs/specification.md` under the same `FR-xx` and `NFR-xx` IDs.
- **Alternatives considered:**
  - User stories (`US-xx`) with a specification behind them: a second ID space, while the course material and the report template speak of functional and non-functional requirements.
  - One page with softened wording: the design decisions would stay inside the requirements, only less visibly.
- **Rationale:** The lecturer's feedback on M1 was that our requirements read as a specification.
  The page named endpoints, status codes, and container mechanics, so a design change looked like a requirement change.
  One ID space keeps the trace from a requirement to its test intact.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI confirmed the criticism against the page and proposed the two levels, which we accepted.
  Its first execution went further than asked and replaced the requirements with user stories.
  Lukas corrected that, and AI reworked the page to the agreed form.
- **AI interaction evidence:** Claude Code session of 2026-09-29; Lukas's correction (translated from German): "user stories were only meant as orientation for you. We still want to call them functional and non-functional requirements."
- **Other evidence:** [Requirements](../docs/docs/requirements.md), [specification](../docs/docs/specification.md), [PR #19](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/19).
- **In LaTeX:** yes

### EDN-28: The pipeline code is one module per DVC stage

- **Date:** 2026-09-29
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Project Structure
- **Participants:** @lukas2510
- **Decision:** Each DVC stage has its own module under `recommenditos/data/` or `recommenditos/modeling/`, instead of the flat Cookiecutter Data Science layout with `dataset.py`, `features.py`, and `modeling/train.py`.
  The course demo repository is structured the same way.
- **Alternatives considered:**
  - The flat Cookiecutter layout as generated: download, preprocessing, and split would share one file, so an edit to it reruns every stage that depends on it.
- **Rationale:** The course prescribes Cookiecutter Data Science and allows justified deviations.
  A stage's `deps` should name the code that stage runs, so that `dvc repro` reruns the minimum.
  One file per ticket also lets several people build a sequential pipeline in parallel.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** Asked whether the sprint plan matched the course demo, AI read the demo's file tree and `dvc.yaml` instead of our notes and reported the difference.
  It laid out the options and recommended the demo's layout.
  Lukas chose it.
- **AI interaction evidence:** Claude Code session of 2026-09-29, sprint 2 planning.
- **Other evidence:** `dvc.yaml`, [course demo notes](../references/course-demos.md), [issue #32](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/32), [PR #45](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/45).
- **In LaTeX:** yes

### EDN-30: Pipeline configuration lives in `params.yaml`

- **Date:** 2026-09-29
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Project Structure, Reproducibility
- **Participants:** @lukas2510
- **Decision:** Everything configurable about the pipeline lives in `params.yaml` and is declared per stage under `params:` in `dvc.yaml`: the seed, the reference date, the scope bounds, the split ratios, the feature sets, the model variants, and the thresholds of the success criteria.
  Paths stay in `recommenditos/config.py`.
- **Alternatives considered:**
  - Python constants, as the course demo keeps in `train.py`: DVC sees a constant only through the hash of its module, so a changed hyperparameter retrains every variant, and `dvc params diff` and `dvc exp` cannot see it.
  - `config.py` reading `params.yaml`: hides which stage reads which key.
- **Rationale:** M2 is about a change rerunning what it should and nothing else.
  DVC hashes the declared keys, `dvc params diff` shows what changed between two commits, and `dvc exp` can sweep a value without a code change.
  Paths are not experiment variables and are derived from the module's location.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI read the demo's `dvc.yaml` and `config.py` directly and reported that the demo declares no `params` on any stage.
  It measured the consequence in a scratch DVC repository: with a declaration per key, a changed hyperparameter retrained one variant and not all of them.
  Lukas reviewed the measurement and accepted.
- **AI interaction evidence:** Claude Code session of 2026-09-29, building the pipeline skeleton.
- **Other evidence:** [`params.yaml`](../params.yaml), [`dvc.yaml`](../dvc.yaml), [issue #32](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/32).
- **In LaTeX:** yes

### EDN-31: The data contract is a hand-written `schema.py`

- **Date:** 2026-09-29
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Data Validation, Testing Strategy
- **Participants:** @lukas2510
- **Decision:** `recommenditos/schema.py` holds the structural contract of the raw, interim and processed frames: column names, dtypes, and nullability, with a `validate` and a `conform` method.
  Value rules stay in the Great Expectations suites.
  Every stage validates after reading and conforms before writing.
- **Alternatives considered:**
  - Pandera: a third validation system next to Great Expectations and Pydantic, and it does not check the datetime unit.
  - Great Expectations alone: the contract would live only in a generated store, and the stage tickets needed something to code against before that ticket.
- **Rationale:** A structural break is a bug in our code and should fail the stage that caused it.
  A value break is a change in the data and belongs in a suite with a tolerance.
  `conform` casts to the declared dtypes before each write, because Parquet keeps whatever dtype it is handed.
  It also selects only the contract's columns, which keeps the personal-data columns out of later frames.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI installed Pandera, Great Expectations, and pyarrow against our pandas version and measured what survives a Parquet round trip.
  Two findings decided the option: a string dtype that changes across the boundary, and Pandera ignoring the datetime unit.
  Lukas chose from the three options with these findings.
- **AI interaction evidence:** Claude Code session of 2026-09-29, probes in a scratch environment.
- **Other evidence:** [`schema.py`](../recommenditos/schema.py), [`tests/test_schema.py`](../tests/test_schema.py), [issue #32](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/32).
- **In LaTeX:** yes

### EDN-32: The split is 60/10/10/20 by seller, with the snapshot date as seed

> **Absorbs EDN-79** (2026-10-12): the split was reviewed after training and kept; spread re-measured 2026-10-13 (EDN-82).

- **Date:** 2026-09-29
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Evaluation Protocol
- **Participants:** @lukas2510
- **Decision:** The listings outside `ES` are split 60 % train, 10 % validation, 10 % calibration, and 20 % test, with the seed `20251108`, both pinned in `params.yaml`.
  Whole sellers are assigned by a hash of the seed and `seller_group_id`, without stratification.
  The model card states the spread over 20 other splits and, as a limitation, that the snapshot has no listing dates for a temporal split.
- **Alternatives considered:**
  - 70/10/10/10: halves the test set, so Volvo and Suzuki would fall under the 500 test rows SC-04 needs.
  - A stratified-group split on `make`: better make balance, but about half of the existing rows change set when the `ES` listings join, which confounds FR-15's retrain comparison.
- **Rationale:** SC-04 checks a segment only from 500 test rows, so the test share decides how many makes the quality gate covers.
  The seed is the snapshot date, so it is obviously not tuned.
  Over 20 other splits the candidate's test MdAPE is 6.72 % on average, with a standard deviation of 0.26 pp, and SC-01 and SC-02 hold on all 20.
  Cars older than 20 years miss SC-04 on all 20 as well, at 18.2 % on average, so that miss is no artefact of the draw.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI found a coupling the ticket did not mention: the published per-make analysis assumed a 20 % test share while the split was still unpinned.
  It recommended 60/10/10/20, and Lukas accepted.
  For the review of 2026-10-12 it wrote and ran the evidence script, corrected the premise of the issue, which Lukas had written, and recommended keeping the split, which Lukas did.
- **AI interaction evidence:** Claude Code sessions of 2026-09-29 (pinning the split) and 2026-10-12 (issue #89).
- **Other evidence:** [`params.yaml`](../params.yaml), [`make_support_results.txt`](analysis/make_support_results.txt), [`split_strategy_results.txt`](analysis/split_strategy_results.txt), [issue #32](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/32), [issue #89](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/89), [PR #107](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/107).
- **In LaTeX:** yes

### EDN-33: Tests and a clean clone run on a generated fixture

- **Date:** 2026-09-29
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Testing Strategy, Data Acquisition
- **Participants:** @lukas2510
- **Decision:** `recommenditos/data/synthetic.py` generates a schema-valid stand-in for the raw snapshot from public catalogues and a seed, never by sampling the real file.
  The test suite builds its fixtures from it.
  `download.source` in `params.yaml` selects `synthetic` or `zenodo`, so the pipeline also runs without network and credentials.
- **Alternatives considered:**
  - A sample of the real file committed to Git: puts personal data into the history of a repository the whole cohort can read.
  - No fixture: every test and every `dvc repro` would need the real file and DagsHub access.
- **Rationale:** Nobody is blocked, and no personal data enters the Git history.
  Because the source is a parameter, `dvc.lock` records which source every artefact was built from, and `dvc status` reports a workspace that disagrees.
  The fixture keeps the real make skew and several listings per seller, so the seller-grouped split is testable.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI proposed the parameter and derived the fixture's edge cases from the stage tickets.
  Its first implementation was wrong: the edge-case rows were copies of one row, so preprocessing would have removed most of them, and the tests only looked at the raw frame.
  A second agent, asked to attack the change, found it.
  A test now asserts that each case survives preprocessing.
- **AI interaction evidence:** Claude Code session of 2026-09-29; Lukas chose from three options.
- **Other evidence:** [`synthetic.py`](../recommenditos/data/synthetic.py), [`tests/test_data.py`](../tests/test_data.py), [issue #32](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/32).
- **In LaTeX:** yes

### EDN-44: The traceability gate is enforced per milestone

> **Absorbs EDN-45** (2026-09-30): a marker counts as verification only where the specification names a test as the evidence.

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Testing Strategy, CI/CD
- **Participants:** @lukas2510
- **Decision:** The requirement-to-test matrix of NFR-07 is gated per milestone.
  `tools/expected_coverage.yaml`, a schema-validated file, states from which milestone on each requirement owes evidence.
  A `req` marker counts as "verified by a test" only on a requirement the specification tags as automated; on a manual one it is "named by a test".
- **Alternatives considered:**
  - A `Due` column in the specification's tables: nothing can validate a table cell.
  - No completeness gate: leaves NFR-07 to a review nobody is reminded to do.
  - Completeness over the whole table from the start: red until M6, because most functional requirements need the API.
- **Rationale:** A gate is useful only if someone can act on it, and trustworthy only if something enforces it.
  The generator refuses a malformed coverage file, and a requirement added to the documents fails the build until someone books when its evidence is due.
  The two statuses keep the matrix from reporting a requirement as verified by a test that does not verify it.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI laid out the options and implemented the file-based gate.
  An adversarial AI review then showed, by running the generator, three ways to drop a requirement from the gate, and a marker that counted as verification on a test asserting only that two files exist.
  The validated milestone order and the statuses come from that review.
- **AI interaction evidence:** Claude Code sessions of 2026-09-30 on issue #40 and on the review of PR #51.
- **Other evidence:** [Issue #40](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/40), [PR #51](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/51), `tools/expected_coverage.yaml`, `tools/requirement_matrix.py`.
- **In LaTeX:** yes

### EDN-47: The serving runtime is separated by dependency groups

- **Date:** 2026-10-05
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Containerization, Dependency Management
- **Participants:** @lukas2510
- **Decision:** `[project] dependencies` in `pyproject.toml` holds only what serving needs to load a model bundle and price a listing.
  Everything else is in PEP 735 dependency groups (`pipeline`, `notebook`, `docs`, `test`, `dev`), and `default-groups = "all"` keeps a plain `uv sync` unchanged.
  The API image will install the runtime alone.
  `make test-serving`, a required CI check, fails when a serving module imports a package from a group.
- **Alternatives considered:**
  - Optional extras: `uv sync` leaves them out by default, so every documented command would change or quietly install less.
  - One flat list with `mlflow-skinny`: measured, it still pulls in most of what MLflow does.
  - A hand-written requirements file for the image: a second list that nothing ties to `uv.lock`.
- **Rationale:** What matters is which dependencies reach the image at all.
  Built from the lock, the runtime is 421 MB of site-packages against 815 MB for the full environment, and NFR-04 caps the image at 1 GB.
  One lock and one resolution mean the image installs a subset of exactly what CI tested.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** Lukas chose groups over extras before the implementation.
  AI traced the serving imports by running them, built both environments from the lock, and showed that the check catches an injected training import.
  An independent AI review found three ways the check could still be wrong; each is fixed and tested.
- **AI interaction evidence:** Claude Code sessions of 2026-10-05 on issue #61 and on the review of PR #69.
- **Other evidence:** [Issue #61](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/61), [PR #69](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/69), [`runtime_footprint_results.txt`](analysis/runtime_footprint_results.txt), the Dependencies section of [CONTRIBUTING.md](../CONTRIBUTING.md).
- **In LaTeX:** yes

### EDN-55: Each model variant has one MLflow run across stages

> **Absorbs EDN-54** (2026-09-30): a stage runs without MLflow unless tracking is required.

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Experiment Tracking, Evaluation Protocol
- **Participants:** @lukas2510
- **Decision:** `train@<variant>` creates the MLflow run and writes its id into `models/<variant>/model.json`.
  `evaluate` reopens that run and appends the test metrics and the gate verdict.
  Without a reachable server a stage runs untracked and says so in `model.json`; `RECOMMENDITOS_REQUIRE_TRACKING=1` turns a tracking failure into a failed stage.
- **Alternatives considered:**
  - One run per DVC stage joined by a tag, as in the course demo: the hyperparameters and the error they produced land in different rows.
  - Nested runs under a parent: the four `train` stages are independent, so the parent has no natural owner.
  - Fail whenever tracking is unavailable: CI and a fresh clone have no DagsHub credentials, so `pytest` and `dvc repro` would fail for everyone.
- **Rationale:** The rubric asks for results that can be compared and understood.
  With one run per variant, MLflow's own table shows parameters, losses, energy figures, and test metrics side by side, without joining.
  The cost is that `evaluate` depends on an id from `train`: a model trained without tracking has no run to append to.
  The switch serves whoever produces the numbers for the report: with it, a misconfiguration cannot pass while logging nothing.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** The design is AI's: two agents working on issues #37 and #39 agreed it.
  AI verified against the real DagsHub server that resuming creates no additional run.
  It proposed the switch and saw that a fallback to a local store would leave a database in the repository.
  Lukas accepted the design as proposed; no check by a person is recorded.
- **AI interaction evidence:** Claude Code sessions of 2026-09-30 on issues #37 and #39.
- **Other evidence:** `resume_run` and `optional_run` in [`tracking.py`](../recommenditos/tracking.py), [model card](../docs/docs/model-card.md), [issue #37](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/37), [issue #39](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/39).
- **In LaTeX:** yes

### EDN-58: An unmeasured success criterion blocks the release gate

> **Absorbs EDN-60** (2026-09-30): which segments may gate is enforced in code.

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Testing Strategy, Model Evaluation
- **Participants:** @lukas2510
- **Decision:** A criterion the `evaluate` stage could not measure is recorded as `null` and blocks the NFR-01 gate.
  `metrics.json` names the criteria that were not measured and counts the variants that pass everything measurable.
  The rules for which segment levels SC-04 may gate on are checks in the stage: a segment that conditions on the target, or one over an optional input field, raises.
- **Alternatives considered:**
  - Leave an unmeasurable criterion out of the gate: a model would pass NFR-01 without any interval ever being calibrated.
  - Count SC-04 as met when no segment level reaches 500 rows: a check nobody ran would be reported as passed, on every CI run with the small fixture.
  - Document the segment rules and rely on review: one line in `params.yaml` could add the price buckets and change the number the gate blocks on.
- **Rationale:** NFR-01 says "every" criterion.
  SC-05 is not measurable yet, because the UC2 intervals are a later ticket, so the gate is closed and the artefact says why.
  Promotion is a pull-request review (EDN-12) and the reviewer reads the metrics diff, so the artefact has to explain itself.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI laid out the readings and recommended the blocking one.
  It measured that no segment level reaches 500 rows on the test fixture, so the suite evaluates a second time with a lower bar and gets a real verdict.
  We moved the worst segment and the per-field sweep out of `metrics.json` into the per-variant records, and added a check AI had not proposed: a gating segment must be a required input field.
- **AI interaction evidence:** Claude Code session of 2026-09-30 on issue #39.
- **Other evidence:** [`metrics.json`](../metrics.json), [`evaluate.py`](../recommenditos/modeling/evaluate.py), [`tests/test_pipeline.py`](../tests/test_pipeline.py), [issue #39](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/39).
- **In LaTeX:** yes

### EDN-61: An omitted equipment list is read as empty

- **Date:** 2026-10-01
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Feature Encoding, API Design, Evaluation Protocol
- **Participants:** @lukas2510
- **Decision:** A request that omits one of the four equipment lists is read as an empty list, so every column that list produces reaches the model as `False` and not as missing.
  `Model.mask_absent` is the single implementation of what an omitted field is, for the API, for SC-06's masking sweep, and for the tests.
- **Alternatives considered:**
  - Omitted means absent, with missing values across the list's columns: what EDN-15 asks for in general, but no listing has a null list, so the model would score a state it was never fitted on.
  - Leave it to each caller: the docstrings said both things at once, and the obvious one-line masking silently chose the missing-value reading.
- **Rationale:** No measurement decides: on the test set the two readings give identical predictions for the LightGBM variants.
  That is a property of the tree models and would not hold for a model with missingness indicators, so the rule is fixed on meaning.
  An empty list is a state the training data holds; an absent one is not.
  An API model that passes the field on as `None` would choose the other reading, so it defaults to `[]`.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Rejected
- **Assessment of the AI contribution:** A first AI review of PR #62 found the contradiction and recommended the empty list.
  A second AI pass argued from the contract's wording for missing values, recommended that, and implemented it.
  Lukas overruled the second pass on a ground neither had weighed: the training data has no null list.
  A figure the second pass gave for its option came from an error in its own probe.
  The shared helper is AI's and stayed.
- **AI interaction evidence:** Claude Code sessions of 2026-09-30 (review of PR #62) and 2026-10-01.
- **Other evidence:** `Model.mask_absent` in [`model.py`](../recommenditos/modeling/model.py), [`tests/test_model.py`](../tests/test_model.py), [PR #62](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/62).
- **In LaTeX:** yes

### EDN-62: `lgbm-basic` is the candidate model

> **Revised on 2026-10-20**: adds the gate's SC-04 waiver; since EDN-82 `lgbm-basic`'s ratio is 1.02, not 1.21.

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Modelling Approach, Model Evaluation
- **Participants:** @lukas2510
- **Decision:** `lgbm-basic` is the candidate the model card and the report put forward, although `lgbm-extended` has the lower pooled MdAPE.
  `lgbm-extended` fails SC-06.
  SC-04 is missed by every variant on cars over 20 years and is recorded as missed: the threshold stays, the scope gets no age bound, and the segment stays in the breakdown.
  The gate waives it through `params.yaml`, citing this entry; `metrics.json` still reports it.
- **Alternatives considered:**
  - `lgbm-extended` as the candidate, with SC-06 as a known limitation: overrides the criterion the first time it fires.
  - Routing complete and partial requests to different models: two models and a rule; left open for deployment.
  - An upper age bound in the scope, or a higher SC-04 threshold: the criterion would pass by removing the rows it fails on.
  - Keeping the gate closed until a variant meets SC-04: blocks every promotion.
- **Rationale:** With only the fields FR-01 requires, `lgbm-extended` rises to 1.65 times its full-input MdAPE and `lgbm-basic` to 1.02; the bound is 1.5.
  Then the ranking reverses.
  The extended features worsen the over-20 segment and do not fix SC-04.
  The waiver makes NFR-01, this entry and the M4 deployment consistent, and is versioned and reviewed like any change; every other criterion still gates.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** Two AI sessions measured SC-06 independently and agreed; one saw that masking every optional field leaves a required-fields-only request.
  A failure it had predicted did not reproduce, and it reported that.
  AI recommended this option and left the serving and scope questions to Lukas; on 2026-10-20 it laid out the gate options and recommended the waiver, and Lukas decided.
- **AI interaction evidence:** Claude Code sessions on issues #37 and #39 (2026-09-30) and #137 (2026-10-20).
- **Other evidence:** [Model card](../docs/docs/model-card.md), [`masked-inputs.csv`](metrics/masked-inputs.csv), [PR #65](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/65), [issue #137](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/137), [PR #158](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/158).
- **In LaTeX:** yes

### EDN-68: The Great Expectations store is generated and cached

> **Absorbs EDN-77** (2026-10-06): fill rates are asserted by the suites, not by the raw contract.

- **Date:** 2026-10-05
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Data Validation, Testing Strategy
- **Participants:** @lukas2510
- **Decision:** The `configure_gx` stage rebuilds the Great Expectations context under `gx/` from code, `schema.py`, and `params.yaml` on every run.
  `gx/` is a DVC-cached output and is not committed.
  The suites' column, dtype, and non-null checks are generated from `schema.py`.
  Fill rates are tolerant rules in both suites, and the contracts keep only structural nullability.
  Any failed expectation fails `validate-data`.
- **Alternatives considered:**
  - Commit `gx/` as configuration, as the course demo does: Great Expectations writes fresh ids on every save, so two builds of the same code differ and every rerun is a diff.
  - No store, an ephemeral context: no `configure_gx` stage and no inspectable suites.
  - Fill rates as non-null columns of the raw contract: a gap in a scrape would stop `download` like a bug in our code.
- **Rationale:** An artefact that changes on every rebuild without changing its meaning is a build product, and we track build products with DVC.
  Generating the structural checks keeps `schema.py` the one place the contract is written.
  Both suites require a minimum row count, so no suite passes on an empty frame.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** Before building, AI tested whether Great Expectations can bound a text date (it cannot), what the store writes across two builds, and whether each rule can hold at its place in the pipeline.
  Each result changed the design; the second turned "commit `gx/`" into a cached output.
  Lukas confirmed the setup as proposed.
  An independent AI review then found four faults, among them rules that passed on an empty frame; all were fixed before merging.
- **AI interaction evidence:** Claude Code sessions of 2026-10-05 (issue #25, review of PR #76) and 2026-10-06 (issue #78).
- **Other evidence:** [`gx_context_configuration.py`](../recommenditos/data/gx_context_configuration.py), [`gx_probe_results.txt`](analysis/gx_probe_results.txt), [PR #76](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/76), [issue #78](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/78).
- **In LaTeX:** yes

### EDN-71: Secrets are scanned with gitleaks before commit and in CI

- **Date:** 2026-10-05
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** CI/CD, Security
- **Participants:** @lukas2510
- **Decision:** gitleaks scans every staged change in a pre-commit hook and the whole published history in a required CI job.
  `.gitleaks.toml` adds one rule of our own for the DagsHub token.
  A canary fails the CI job when the scan no longer finds planted tokens.
- **Alternatives considered:**
  - detect-secrets: by default it finds the DagsHub token in 11 of 13 places, where gitleaks' defaults find it in 7, but it reads files and not history.
  - GitHub secret scanning: it has no DagsHub pattern and leaves nothing in the repository to review.
  - A Pytest grep of the working tree: sees only the latest commit.
- **Rationale:** The repository is public, so a secret in any pushed commit is published even if the next commit deletes it.
  Only a scan of the history catches a commit that skipped the hook.
  The detection gap of gitleaks closes with one rule in a file we own: with it, every planted token is found in all 13 places.
  The history gap of detect-secrets cannot be closed.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI recommended gitleaks partly because detect-secrets would detect worse.
  Measured, that was false, so the choice rests on the history argument alone.
  An independent AI review then found two blind spots, merge commits and a canary that could not fail, and both were fixed before merging.
- **AI interaction evidence:** Claude Code session of 2026-10-05 on issue #60.
- **Other evidence:** [Issue #60](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/60), [PR #70](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/70), `.gitleaks.toml`, [`secret_scan_dagshub_token_results.txt`](analysis/secret_scan_dagshub_token_results.txt).
- **In LaTeX:** yes

### EDN-73: Hyperparameters change only under a written tuning protocol

> **Absorbs EDN-70 and EDN-78** (2026-10-05, 2026-10-06): the tree ceiling, and the search over both LightGBM variants. **Revised on 2026-10-20**: a correction across sweeps, set before issue #117's search.

- **Date:** 2026-10-05
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Hyperparameter Tuning, Evaluation Protocol
- **Participants:** @lukas2510; @kadameit decided the search over both variants (2026-10-06) and the correction (2026-10-20)
- **Decision:** `learning_rate` 0.05, `num_leaves` 63, `min_child_samples` 20, and `reg_lambda` 0 stay, because no search found a point measurably better on the validation split.
  A value changes only under the pipeline docs' protocol: selection on the validation split by L1, when a point's simultaneous interval lies below zero at level 1 - 0.05/k, for k sweeps on that split.
  `n_estimators` is a ceiling of 5,000 with early stopping after 50 rounds.
  `tune.py` searches outside `dvc repro`, recorded in MLflow only.
  NFR-10's 15 minutes cover the final fit, and a search's energy is reported apart.
- **Alternatives considered:**
  - Adopt the point with the lowest validation L1: its gain lies inside the noise.
  - Optuna or seller-grouped cross-validation: about five times the fits, for a region found flat.
  - No correction across sweeps: k sweeps adopt a lucky point in up to k times 5 % of cases.
- **Rationale:** No sweep's interval lies below zero.
  At 5,000, early stopping ends every fit.
  The split, unchanged since 2026-10-01, held seven sweeps before issue #117, so its two `reg_lambda` sweeps decided at k = 9: best points -0.00001 and +0.00014 in L1, half-widths 0.0009 and 0.0010.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI designed the sweeps and stopped before each decision.
  An independent AI review replaced a per-point interval, unsafe across eleven comparisons, with the simultaneous one.
  AI recommended committing sweep records beside MLflow.
  Kevin had it check the course demo, which keeps them in MLflow only, and chose that.
  AI counted k from MLflow and `dvc.lock`, and Kevin chose Bonferroni over grouped cross-validation.
- **AI interaction evidence:** Claude Code sessions on #64, #88, and #117.
- **Other evidence:** "Tuning a hyperparameter" in [the pipeline docs](../docs/docs/pipeline.md), [`tuning_sweep_results.txt`](analysis/tuning_sweep_results.txt), [`early_stopping_ceiling_results.txt`](analysis/early_stopping_ceiling_results.txt), [PR #77](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/77), [issue #88](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/88), [issue #117](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/117).
- **In LaTeX:** yes

### EDN-75: The test suite also runs on Windows in CI

- **Date:** 2026-10-06
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Testing Strategy, CI/CD
- **Participants:** @lukas2510; @kadameit found the Windows failures
- **Decision:** CI runs the whole test suite on `windows-latest` as well as on Linux, as the required job `Tests (Windows)`.
  Coverage and the test reports still come from the Linux job.
- **Alternatives considered:**
  - The same job, not required: a red check that does not block is one people learn to merge past.
  - Emulate Windows in the Linux suite: covers only the failure modes it was written for.
  - Rely on contributors who develop on Windows: a failure is found after the merge, by one person.
- **Rationale:** One of us develops on Windows, and NFR-06 asks that a clean clone reproduces the pipeline.
  On Windows it did not: JSON artefacts came out with CRLF line endings and a different hash.
  The job's first run also found a failure nobody knew of, paths that span two drives.
  The cost is a longer wait for the last required check of a pull request.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI traced the two known Windows failures to their causes, reproduced one end to end, and recommended the required job, which Lukas chose.
  The fixes, the tests written to fail first, and the analysis of the failures the new job found were AI's.
- **AI interaction evidence:** Claude Code session of 2026-10-06 on issue #84 and on the review of PR #73 and PR #75.
- **Other evidence:** [Issue #84](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/84), [PR #85](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/85), the [failing run](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/actions/runs/37438994409) before the fix, `.github/workflows/ci.yml`.
- **In LaTeX:** yes

### EDN-80: The project is the used-car price component

- **Date:** 2026-09-21 at the latest (the first commit's README names car price estimation); the day itself is not recorded
- **Milestone:** M1: Project Inception
- **Activity / Topic:** Project Selection
- **Participants:** The team, as @lukas2510 recalls it; who took part in the discussion was not recorded
- **Decision:** We build a used-car price component: a model behind an API that estimates the market price of a car and the price range a buyer should expect to pay.
- **Alternatives considered:**
  - A route-safety model, a recommender, and a grocery idea: none was investigated in depth.
    We looked for no dataset and defined no target for them, so their pros and cons were never established.
- **Rationale:** This is @lukas2510's recollection, written down on 2026-10-12, and not a record from the time.
  Four ideas were on the table, and the team did not weigh them at length.
  The used-car price idea sounded the most feasible and interested us, so we checked that it was feasible and started.
  The other three were not rejected for a reason of their own: they were not picked.
  Confirmed afterwards, and no reason at the time: a public dataset with a DOI (EDN-01), a measurable target (EDN-06), and fits of under a minute on a laptop.
- **AI involvement:** Information seeking
- **Response to AI:** Used as input for further analysis
- **Assessment of the AI contribution:** As far as @lukas2510 recalls, AI did not take part in choosing among the four ideas.
  It was used for the feasibility check, and its output was not taken at its word: the plan drafted with an AI assistant rested on assumptions about the data that a profile of the real files contradicted ([PR #13](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/13)).
  The three confirming reasons are AI's, checked against the repository on 2026-10-12.
- **AI interaction evidence:** Claude Code sessions of 2026-09-22 (the drafted plan checked against the data) and 2026-10-12 (@lukas2510's account).
- **Other evidence:** [Issue #95](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/95), [problem specification](../docs/docs/problem-spec.md), [`energy-vs-error.csv`](figures/energy-vs-error.csv) (`fit_seconds`).
- **In LaTeX:** yes

### EDN-81: Coding agents work under a written team rule

- **Date:** 2026-09-29 (rule written down); review rule reworded on 2026-10-12 and confirmed on 2026-10-13
- **Milestone:** Other (team process)
- **Activity / Topic:** Use of AI Coding Agents, Code Review
- **Participants:** @lukas2510 wrote the rule down and proposed the rewording; the full team confirmed it at the laboratory session of 2026-10-13; no earlier discussion of the rule by the other four members is on record
- **Decision:** Agents write and review most changes, people decide and are accountable, and no person signs off each pull request.
  The rule is one of four in the [working agreements](../docs/docs/scrum/working-agreements.md), and its review clause, reworded on 2026-10-12, reads: "Agent-written code goes through the same pull request as anything else. An independent agent with a fresh context reviews every substantive change by reproducing its claims and trying to break it, and the required checks gate the merge. A person decides the questions that belong in the EDN and is accountable for the change."
- **Alternatives considered:**
  - Keep the old wording, "The reviewer is a person": a rule the team does not follow.
  - A second person approves every pull request: GitHub blocks approving one's own, and an agent's runs under its author's account, so most merges would wait.
- **Rationale:** The rule now says what the trace shows: of the 56 pull requests merged by 2026-10-11, 2 carry a review by another person, and branch protection requires 9 checks and 0 approvals.
  People decide: of the 76 original entries EDN-01 to EDN-78, the AI's proposal was accepted in 46, modified in 27, rejected in 2, and used as input in 1; of the 35 kept after the trim, in 23, 9, 1, and 1, with EDN-25 rejected and then partly adopted.
  Comments and pull requests under a member's account are often typed by an agent on that member's instruction, and the member answers for them.
  The weak spot: 48 of the 56 were opened and merged under one account without a second person's review, so in sprint 3 every teammate opens the pull request for their own ticket.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** The rule's text is AI's, adopted by @lukas2510 on 2026-09-29.
  On 2026-10-12 AI measured the numbers above, found that the review rule and the trace disagreed, and recommended the rewording, which @lukas2510 chose.
- **AI interaction evidence:** Claude Code sessions of 2026-09-21, 2026-09-29, and 2026-10-12.
- **Other evidence:** [PR #45](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/45), [issue #95](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/95), [PR #101](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/101#issuecomment-6124299563), `gh pr list`, and the branch protection of `main`, read on 2026-10-12.
- **In LaTeX:** yes

### EDN-82: `lgbm-basic` is trained on blanked copies of its listings

> **Revised on 2026-10-13**: first recorded as a measurement only; the adoption followed the same day.

- **Date:** 2026-10-13
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Modelling Approach, Model Evaluation
- **Participants:** @kadameit proposed the measurement ([issue #116](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/116)); @lukas2510 decided
- **Decision:** `lgbm-basic`, the candidate, is fitted on its training rows plus two copies of each with optional input fields set to what an omitted field is: one with all six omitted and one with a random subset.
  `lgbm-extended` is trained as before and does not become the candidate.
- **Alternatives considered:**
  - `lgbm-extended` with the copies as the candidate: better on most requests, but worse than `lgbm-basic` as then trained on one giving exactly the basic set's six fields (7.12 % against 6.96 % validation MdAPE), which the rule fixed before the measurement excludes.
  - Serving both models, routed by how full a request is: two models, a threshold, and still 8.57 % on a required-only request.
  - No change before the first delivery: the candidate stays worse than necessary on a request FR-01 allows.
- **Rationale:** All six optional fields are empty together in 324 of 60,378 training rows, so a required-only request was almost absent from what the model saw.
  With the copies its validation MdAPE on that request falls from 8.57 % to 7.19 %, and no tested request is measurably worse.
  The test split, read once after the decision, agrees: 8.26 % to 6.86 % there, 6.81 % to 6.73 % on a complete request, and SC-06 at 1.02 times instead of 1.21.
- **AI involvement:** Information seeking, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI corrected the issue's premise (fields are often empty singly, almost never all together), proposed the rule, and measured and recommended this option, which @lukas2510 took.
  Independent agents reviewed the protocol before the fits, adding the condition that excluded `lgbm-extended`, and the code before the pipeline run; no person did.
- **AI interaction evidence:** Claude Code session of 2026-10-13 on issue #116: measurement, three options, recommendation.
- **Other evidence:** [Measurement results](analysis/blanked_fields_results.txt), [`masked-inputs.csv`](metrics/masked-inputs.csv), [PR #118](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/118), [PR #121](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/121), MLflow sweep `116-blanked-fields` in `recommenditos-price-tuning`.
- **In LaTeX:** yes

### EDN-83: Python code is type-checked with mypy in its default mode

- **Date:** 2026-10-20
- **Milestone:** M4: Model Deployment
- **Activity / Topic:** Testing Strategy, CI/CD
- **Participants:** @lukas2510 decided
- **Decision:** mypy, in its default mode plus `check_untyped_defs`, checks `recommenditos/` and `tools/` in a pre-commit hook and in the required `Lint & format` job.
  Untyped third-party packages are ignored module by module, `pandas-stubs` is not installed, and a finding is fixed in the code, not silenced.
- **Alternatives considered:**
  - No type checker: ruff and the tests catch a lot, but annotations drift unnoticed and a forgotten `None` shows only in a test.
  - basedpyright: infers types from pandas' source and sees more there, but reported 193 errors, mostly pandas noise, and agents would scatter `ignore` comments.
  - ty: 19 diagnostics, but at version 0.0.89 its rules change between releases and CI would be unstable.
- **Rationale:** 307 of 350 functions in the package and tools carry a return annotation that nothing checked.
  mypy reports 19 findings in 9 files, none a real bug: each is a value the code does not show cannot be `None`, or a library type.
  A check, an `assert`, a corrected annotation or a small helper fixes each one.
  It has a pydantic plugin for the M4 API.
  It sees every DataFrame as `Any`, so column names and dtypes stay with `schema.py` and the tests.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI measured the error counts of the three checkers, laid out the options and recommended mypy, which @lukas2510 chose.
  It fixed the findings and wired the hook and the CI job; a deliberate type error turned the hook red, and a reviewer agent checked the pull request.
- **AI interaction evidence:** Claude Code session of 2026-10-20 on issue #133: three checkers measured, four options, recommendation.
- **Other evidence:** [Issue #133](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/133), [PR #149](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/149), [`[tool.mypy]`](../pyproject.toml).
- **In LaTeX:** yes

### EDN-84: Every API error is an RFC 9457 Problem Details object

> **Revised on 2026-10-21**: request bodies are validated strictly against the documented schema.

- **Date:** 2026-10-20
- **Milestone:** M4: Model Deployment
- **Activity / Topic:** API Design
- **Participants:** @lukas2510 decided ([issue #135](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/135))
- **Decision:** Every error response of the API is `application/problem+json` as RFC 9457 defines it: `type`, `title`, `status`, `detail`, `instance`, and for a 422 an `errors` list with one entry per invalid field, holding a JSON Pointer, a message and, for a field with a fixed set of values, the accepted ones.
  A request body is validated strictly against its documented schema, because the Schemathesis contract tests found that lax coercion accepted `0` as a boolean and `"20500"` as a price.
- **Alternatives considered:**
  - An error envelope of our own: free to shape, but one more format for every client to learn and for us to document, and FastAPI's 422 needs rewriting for it too.
  - Lax coercion of request values: forgiving to clients, but the server then accepts what the schema refuses, and `0` silently becomes `false`.
- **Rationale:** FR-16 requires one shape for validation errors, the scope check, the API key and the body-size limit, so FastAPI's default 422 had to be rewritten whichever shape we chose.
  Only the standard gives a client something it may already parse, which is what the rubric's "easy to adopt by others" asks for.
  Every error path, from a missing field to an unknown route, the 413 of NFR-09 and an unexpected failure, is tested to return the same shape.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI laid out both options in the issue and recommended the standard, which @lukas2510 chose. AI then built the handlers and the tests that hold every error path to the shape, and on 2026-10-21 laid out strict against lax coercion and recommended strict, which @lukas2510 chose.
- **AI interaction evidence:** Claude Code session of 2026-10-20 planning Milestone 4: the issue's two options and recommendation.
- **Other evidence:** [Issue #135](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/135), [PR #151](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/151), [PR #175](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/175), [API docs](../docs/docs/api.md), [`errors.py`](../recommenditos/api/errors.py), [RFC 9457](https://www.rfc-editor.org/rfc/rfc9457).
- **In LaTeX:** yes

### EDN-85: Serving adds MAPIE's offsets itself, and partial requests take the median's price

- **Date:** 2026-10-20
- **Milestone:** M4: Model Deployment
- **Activity / Topic:** ML System Design, Model Deployment
- **Participants:** @lukas2510 decided ([issue #137](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/137))
- **Decision:** MAPIE's `ConformalizedQuantileRegressor` calibrates UC2's quantile LightGBM models in the `train` stage only.
  The `lgbm-basic` bundle stores the three boosters and the two conformity offsets, and `predict_interval_eur` adds them up with LightGBM alone, so the API image carries no MAPIE.
  `/price-range` takes its typical price from `lgbm-basic` when FR-01's required fields are all present, and otherwise from the 50 % quantile model (FR-07).
- **Alternatives considered:**
  - MAPIE in the serving runtime: the library computes every interval, but it adds a package to the image NFR-04 caps at 1 GB, for two predictions and an addition.
  - Our own conformalized quantile regression without MAPIE: no dependency anywhere, but we would own the correctness of the calibration.
  - Typical price always from `lgbm-basic`, or always from the 50 % model: the first is 98.5 % MdAPE off on a make-only request, the second 0.2 to 0.7 pp worse on complete ones.
- **Rationale:** The pattern EDN-11 set for SHAP: the statistics stay in a maintained library, and serving replays a constant.
  A test shows that the range served from a bundle on disk equals MAPIE's on the same rows to a relative 1e-12.
  On the test split the nominal 90 % range covers 90.2 % of full inputs and 91.0 % of P1 requests, so SC-05 passes.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI laid out the options for both questions, measured the typical-price ones, and recommended these, which @lukas2510 took.
  AI implemented it and checked that MAPIE 1.6 moves each bound by one constant before storing it.
- **AI interaction evidence:** Claude Code session of 2026-10-20 on issue #137: options, measurements, implementation.
- **Other evidence:** [PR #158](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/158), [`lgbm-basic.json`](metrics/lgbm-basic.json), [Typical-price analysis](analysis/price_range_point_estimate_results.txt), MLflow run `train@lgbm-basic` in `recommenditos-price`.
- **In LaTeX:** yes

### EDN-86: The prediction log and the feedback are one SQLite file in WAL mode

- **Date:** 2026-10-20
- **Milestone:** M4: Model Deployment
- **Activity / Topic:** API Design, Monitoring
- **Participants:** @lukas2510 decided ([issue #140](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/140))
- **Decision:** Every answer is one row of `predictions` and every observed price one row of `feedback`, in one SQLite file in WAL mode on a volume the API's workers and the drift job share.
  A small store interface keeps PostgreSQL a local change.
- **Alternatives considered:**
  - PostgreSQL in its own container: works across hosts, but costs a service, a secret and memory on the 4 GB VM for a multi-host setup we do not have.
  - JSON Lines, one file per worker: simplest to write, but exactly-once, the join and the rotation would be ours to build.
- **Rationale:** NFR-03 wants one record per answer whatever the worker count, FR-14 joins log and feedback by request ID, and NFR-04 caps both at 2 GB on one VM.
  SQLite serialises the workers with its file lock, a unique request ID refuses a second record, and the join is plain SQL.
  Two uvicorn workers recorded each of 200 concurrent answers once, and a write takes 0.15 ms at the median of the 200 ms that NFR-02 gives `/predict`.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI laid out the three options in the issue and recommended SQLite, which @lukas2510 chose.
  AI then built the store, the rotation and the tests, one with two worker processes, and measured the cost of a write.
- **AI interaction evidence:** Claude Code session of 2026-10-20 planning Milestone 4: the issue's three options and recommendation.
- **Other evidence:** [Issue #140](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/140), [PR #166](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/166), [API docs](../docs/docs/api.md#the-prediction-log-and-the-feedback), [`prediction_log.py`](../recommenditos/api/prediction_log.py), [project brief](../docs/docs/project-brief.md) section 5.
- **In LaTeX:** yes

### EDN-87: The API runs on the VM as a Compose stack behind nginx from Milestone 4

> **Revised on 2026-10-21**: the image size measured again with the interval boosters of #158; the decision is unchanged.
>
- **Date:** 2026-10-20
- **Milestone:** M4: Model Deployment
- **Activity / Topic:** ML System Design, Containerization
- **Participants:** @lukas2510 decided ([issue #142](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/142))
- **Decision:** From M4 on, the API runs on the VM as a Docker Compose stack: the API image, with the serving runtime and the model bundle, behind nginx, the only published container, which refuses `/feedback` and `/metrics`.
  Until CD exists, the image is built on a laptop and copied with `docker save | ssh docker load`.
- **Alternatives considered:**
  - The course demo's recipe (`uv sync`, `dvc pull`, uvicorn under systemd, nginx): M5 replaces it with containers, so two deployments to build and document.
  - Building the image on the VM: needs the repository, DVC and a DagsHub token on a shared machine.
- **Rationale:** The report describes containers under M5 either way, so the demo's recipe builds something we plan to discard, while M5 adds CI/CD, the image scan and the hardening to this stack.
  Measured locally, the image is 702 MB (NFR-04: 1 GB), and `/health` answers again 3 s after the API process is killed (NFR-05: 30 s).
  nginx stays over Caddy and Traefik, whose main gain, automatic TLS, EDN-13 does not need.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI laid out both options in the issue and recommended Compose, which @lukas2510 chose.
  AI then built and measured the stack and chose to copy the image; nothing about the VM is verified yet.
- **AI interaction evidence:** Claude Code session of 2026-10-20 planning Milestone 4: the issue's two options and recommendation.
- **Other evidence:** [Issue #142](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/142), [PR #162](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/162), [deployment docs](../docs/docs/deployment.md), [`compose.yaml`](../deploy/compose.yaml), [`nginx.conf`](../deploy/nginx.conf).
- **In LaTeX:** yes

### EDN-88: The latency, throughput and resource targets after the load test (#143)

### EDN-89: How the UC2 intervals handle makes whose coverage misses the band (#108)

- **Date:** 2026-10-21
- **Milestone:** M4: Model Deployment
- **Activity / Topic:** Model Evaluation, ML System Design
- **Participants:** @lukas2510 decided ([issue #108](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/108))
- **Decision:** The UC2 intervals keep the single pooled calibration of EDN-85.
  `evaluate` reports their coverage and width per make beside the pooled SC-05 figure, and the model card names the judged makes outside 88 % to 92 %.
  Makes under 500 test rows are reported and not judged.
- **Alternatives considered:**
  - Mondrian calibration, one offset pair per make: it fixes Mercedes-Benz on P1 (93.9 % to 91.7 %) and mostly Suzuki (85.9 % to 90.4 % full), but moves Alfa Romeo and Volvo out of the band and lowers Audi to 88.9 % on full inputs; four makes have too few rows to calibrate.
  - A larger calibration split: reopens the split of EDN-32 and needs a full retrain.
- **Rationale:** The pooled figures pass SC-05 (90.2 % full, 91.0 % P1). Mercedes-Benz is in the band on full inputs (90.8 %) and over-covered on P1 (93.9 %), the safe direction. Suzuki is the real miss (85.9 % and 81.7 %). Calibrating per make on 329 to 3,453 calibration listings each only trades misses between makes.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI built the per-make table, simulated Mondrian offsets offline on the calibration split and scored them on test, and recommended this option.
  @lukas2510 chose it after the numbers were laid out.
- **AI interaction evidence:** Claude Code session of 2026-10-20 on issue #108: measurement, offline Mondrian simulation, recommendation.
- **Other evidence:** [PR #169](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/169), [per-make coverage table](metrics/interval-coverage-by-make.csv), [split analysis](analysis/split_strategy_results.txt), EDN-32 and EDN-85.
- **In LaTeX:** yes

### EDN-90: CI warns about a stale `dvc.lock` instead of blocking the merge

- **Date:** 2026-10-22
- **Milestone:** M4: Model Deployment
- **Activity / Topic:** CI/CD, Reproducibility
- **Participants:** @lukas2510 decided ([issue #193](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/193))
- **Decision:** A non-blocking CI job reports every stage whose git-tracked dependencies, parameters or command differ from what `dvc.lock` records, in its job summary and as one warning on the pull request, on every pull request and every push to `main`.
  It filters `dvc status` and skips the data and models CI does not have.
- **Alternatives considered:**
  - A required check: every edit of stage code, comments included, would need a 15-30 minute tracked re-run with new MLflow runs before the merge.
  - No check and a re-sync before each delivery: a stale lock stays invisible until the delivery, as it did during M4.
- **Rationale:** During M4, five changes to stage code left the lock stale until #183 re-ran the whole pipeline, and no check noticed.
  One of them changed only `recommenditos/config.py`, which every stage lists, so even a comment there makes all 16 stages stale.
  The warning shows this on the pull request that causes it, and the re-run can wait for a convenient moment, at the latest the delivery.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI drafted the three options and recommended the warning in #193, from the stale-lock incident of #183; @lukas2510 accepted it and asked for it to be implemented.
  AI then built the check, and on the real repository a comment-only edit of `config.py` reported all 16 stages while `main` reported none.
- **AI interaction evidence:** Claude Code session orchestrating M4, prompt "please do you recommendations here: #193".
- **Other evidence:** [Issue #193](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/193), [PR #183](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/183), [PR #205](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/205), [`tools/stale_lock.py`](../tools/stale_lock.py).
- **In LaTeX:** yes

### EDN-91: A green CI run on `main` publishes the API image to a public GHCR package

- **Date:** 2026-10-22
- **Milestone:** M5: Model Packaging
- **Activity / Topic:** CI/CD, Containerization, Security
- **Participants:** @lukas2510 decided ([issue #197](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/197))
- **Decision:** A workflow of its own, `cd.yml`, runs after CI passed on a push to `main` that changes more than documentation.
  It pulls the model with a DagsHub token from the GitHub Environment `dagshub`, which only `main` may use, tests the image as a Compose stack, and pushes it to a public GHCR package.
- **Alternatives considered:**
  - A read-only DagsHub account for CI: one more account, against a threat a three-person team with branch protection does not face.
  - A private package with a read token on the VM: a credential on a shared university machine, for an image without secrets or personal data.
  - A job in `ci.yml`: the secret would sit in the workflow pull requests run, and the job would have to wait for every other job.
- **Rationale:** DagsHub tokens have no read-only scope, so the protection is where the token can be used: one step of one job, after a push to `main`, never on a pull request (NFR-15).
  The API key reaches the container at run time, so the public image holds only code and the model the model card describes.
  Built locally with the workflow's steps, the image is 704 MB (NFR-04: 1 GB) and passes the 27 integration tests against `dvc.lock`'s hashes.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI laid out the three questions with their options in the issue and recommended the chosen ones; @lukas2510 decided all three as recommended.
  AI then wrote the workflow and checked the build, the size gate, and the integration suite locally with the real model.
- **AI interaction evidence:** Claude Code session of 2026-10-22 planning Milestone 5: the options and recommendations of issue #197.
- **Other evidence:** [Issue #197](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/197), [PR #207](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/207), [`cd.yml`](../.github/workflows/cd.yml), [deployment docs](../docs/docs/deployment.md#continuous-delivery), EDN-08.
- **In LaTeX:** yes

### EDN-92: A manually started GitHub Actions workflow retrains and prepares the promotion pull request

- **Date:** 2026-10-22
- **Milestone:** M5: Model Packaging
- **Activity / Topic:** CI/CD, Reproducibility
- **Participants:** @lukas2510 decided ([issue #201](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/201))
- **Decision:** A `workflow_dispatch` workflow, refused outside `main`, runs `dvc repro` on a GitHub-hosted runner, logs to the DagsHub MLflow, pushes to the DVC remote and pushes the new `dvc.lock` to a branch.
  Its summary gives the evidence and a link that opens the prefilled promotion pull request; a person opening it is the approval step, and CI runs on it.
  Runner energy figures are kept, labelled as estimates; merging such a promotion replaces the laptop's NFR-10 figures.
- **Alternatives considered:**
  - Only CI's `Pipeline lock (dvc.lock)` check, training on laptops: the pipeline would never run anywhere but a laptop.
  - The workflow opening the pull request: needs the repository-wide setting that lets Actions create and approve pull requests, which the goal does not need.
  - Correcting CodeCarbon's CPU power for the runner: more work, unclear accuracy.
  - CodeCarbon off in CI: no energy figures.
- **Rationale:** A forced run of every stage took 333 s and 388 s, at most 3.3 GB of the runner's 16 GB of RAM and 2.4 GB of disk, and cost no Actions minutes on this public repository. It reproduced every committed metric except per-segment shares of the Ridge baseline, which differ between the two CPUs.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI measured the pipeline on the runner, compared the outputs with the committed ones, checked the billing API when @lukas2510 asked about the Actions limits, laid out the options and recommended this one.
  @lukas2510 chose it, the energy labelling, and to keep the Actions pull request setting off.
- **AI interaction evidence:** Claude Code session of 2026-10-22 on issue #201: runner measurement, billing check, recommendation, workflow.
- **Other evidence:** [measurement on #201](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/201#issuecomment-6347639135), runs [39825393405](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/actions/runs/39825393405) and [39826534385](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/actions/runs/39826534385), EDN-12.
- **In LaTeX:** yes

### EDN-93: A hung API is ended by its healthcheck and restarted by Docker

- **Date:** 2026-10-22
- **Milestone:** M5: Model Packaging
- **Activity / Topic:** Containerization, Reliability
- **Participants:** @lukas2510 decided ([issue #185](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/185))
- **Decision:** Compose runs Docker's init in front of the API (`init: true`).
  Once the API has answered since its start, or after the 60 s start period, a probe that gets no answer from `/health` in two attempts of 6 s kills the init's children; the init exits and `restart: unless-stopped` restarts the container.
  Known limit: with several workers one hung worker goes unnoticed, so the API runs one worker until the VM's load test (#143).
- **Alternatives considered:**
  - An autoheal sidecar (18.4 s): the Docker socket it needs is root on the host, against the hardening of #200.
  - A watchdog on the VM (19.2 s): VM changes that CI cannot test.
  - gunicorn's worker timeout: catches one hung worker among several, but adds two serving dependencies and misses a full threadpool.
  - uvicorn's supervisor: misses a blocked event loop.
- **Rationale:** With the server frozen by SIGSTOP, the old stack was still down after 75 s.
  With the probe, on the stack hardened by #200, `/health` answered again after 19.4 to 20.8 s in four runs, the hang placed right after the API turned healthy.
  The worst case is 5 s interval + 2 × 6 s + 4 s restart = 21 s; under load, a kill needs about 128 requests of 47 ms queued ahead of `/health`, more than the load test's 80 users.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI found the options, measured each on the stack or a small test server, and recommended the probe, which @lukas2510 chose.
  AI then wrote the probe and a hang drill that fails on the old stack and passes on the new one.
- **AI interaction evidence:** Claude Code session of 2026-10-22 on Milestone 5: the measured options for restarting a hung API.
- **Other evidence:** [Issue #185](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/185), [deployment docs](../docs/docs/deployment.md), [`healthcheck.py`](../recommenditos/api/healthcheck.py), [hang drill](../tests/integration/test_deployed_recovery.py).
- **In LaTeX:** yes

## Retired and merged IDs

These IDs had entries until the notebook was trimmed on 2026-10-12 and 2026-10-13.
A merged entry lives on in the entry named in the last column; a retired one did not meet the criterion at the top of this file.
The original text of each is in [edn-archive.md](edn-archive.md), and no ID is reused.

| ID | Former entry | Fate | Now in, or why retired |
|---|---|---|---|
| EDN-07 | Raw data hosting: import from Zenodo, never push | merged | EDN-25 |
| EDN-09 | Bump to Python 3.12 | merged | EDN-11 |
| EDN-16 | UC1 and UC2 have separate endpoints | merged | EDN-15 |
| EDN-21 | Report the upstream licence contradiction | retired | documentation wording with no effect on the system |
| EDN-22 | Listings registered after the reference date are dropped | retired | a cleaning rule for 27 listings; it is in the dataset card |
| EDN-24 | Keep the pre-registered exclusion | merged | EDN-04 |
| EDN-26 | NFR-11 window of 1,000 requests, `model` excluded | merged | EDN-14 |
| EDN-29 | Drift significance level of 0.005 | merged | EDN-14 |
| EDN-34 | Raw Parquet keeps its PII columns and is pushed | merged | EDN-20 |
| EDN-35 | The `download` stage owns `data/raw/` | merged | EDN-25 |
| EDN-36 | Retire the `dvc add` pointer | merged | EDN-25 |
| EDN-37 | Seller group key stays an unsalted hash | merged | EDN-20 |
| EDN-39 | Split size gate with four rules | retired | implementation detail of the `split` stage |
| EDN-40 | The `ES` holdout stays row-selected | merged | EDN-03 |
| EDN-41 | `model_version` token and frequency floor | retired | feature detail of the extended set, which is not the candidate |
| EDN-42 | An unseen categorical value becomes missing | merged | EDN-18 |
| EDN-43 | The contract carries a categorical's levels | retired | fix of a defect in our own design |
| EDN-45 | A `req` marker is verification only for automated evidence | merged | EDN-44 |
| EDN-46 | DagsHub credentials live in two stores | retired | DVC offers no alternative, so nothing was chosen |
| EDN-48 | The supported-make list is applied in the `features` stage | merged | EDN-05 |
| EDN-49 | Predictions are not bias-corrected after the log transform | retired | modelling detail; it is in the model card |
| EDN-50 | One-hot encoding for the Ridge baseline | retired | implementation detail of a baseline |
| EDN-51 | Mean fill and missingness indicators for the Ridge baseline | retired | implementation detail of a baseline |
| EDN-52 | `predict_eur` bounds predictions to the training range | retired | implementation guard |
| EDN-53 | `train.num_threads` is pinned to 1 | retired | a single configuration value |
| EDN-54 | A pipeline stage runs without MLflow unless tracking is required | merged | EDN-55 |
| EDN-56 | The model bundle carries its own provenance | retired | was part of EDN-53, retired with it |
| EDN-57 | MLflow retry budget of three requests | retired | a single configuration value |
| EDN-59 | SC-06 masks API input fields | merged | EDN-15 |
| EDN-60 | Gating segments are enforced in code | merged | EDN-58 |
| EDN-63 | `metrics.json` carries the data source and the estimator | retired | fix after stub metrics, implementation detail |
| EDN-64 | Pynblint runs from a locked environment behind a wrapper | retired | how a prescribed tool is run; it is on the Notebooks page |
| EDN-65 | The Pynblint rule policy | retired | was part of EDN-64, retired with it |
| EDN-66 | NFR-06 and NFR-09 are each split in two | retired | requirement bookkeeping; the result is in the requirements |
| EDN-67 | `features` applies the supported-make list to every frame | merged | EDN-05 |
| EDN-69 | CodeCarbon's figure is reported as an estimate | retired | how a prescribed tool's figure is reported; it is in the model card |
| EDN-70 | `n_estimators` is a ceiling of 5,000 | merged | EDN-73 |
| EDN-72 | The mileage range is a tolerant check | retired | was part of EDN-22, retired with it |
| EDN-74 | Runs are tagged with the hashes of their stage's inputs | retired | fix of a provenance tag, implementation detail |
| EDN-77 | Fill rates are asserted by the Great Expectations suites | merged | EDN-68 |
| EDN-78 | The hyperparameter search covers both LightGBM variants | merged | EDN-73 |
| EDN-79 | The seller-grouped hash split stays | merged | EDN-32 |

## Template

```markdown
### EDN-NN: The decision in one clause of about ten words

- **Date:** YYYY-MM-DD
- **Milestone:** M1: Project Inception | M2: Reproducibility | M3: Quality Assurance | M4: Model Deployment | M5: Model Packaging | M6: Monitoring | Other
- **Activity / Topic:** e.g. Testing Strategy, API Design, Containerization, CI/CD, Monitoring
- **Participants:** All team members | names
- **Decision:** What we do, in about 45 words.
- **Alternatives considered:**
  - An alternative we really weighed: the reason it lost, in one sentence.
- **Rationale:** The deciding argument, in about 60 words and with one or two numbers, each checked against its source.
- **AI involvement:** Select all that apply: No AI involvement | Information seeking | Alternative generation | Alternative assessment | Recommendation | Solution generation | Other (describe)
- **Response to AI:** If AI contributed, select one: Accepted | Accepted with modifications | Rejected | Used as input for further analysis | Other (describe)
- **Assessment of the AI contribution:** If AI contributed: what it contributed and what we checked, changed, or rejected, in about 45 words.
- **AI interaction evidence:** Claude Code session of YYYY-MM-DD on the topic, prompt "the literal prompt, or its core (paraphrased)".
- **Other evidence:** About four links: issue, PR, analysis result, document.
- **In LaTeX:** no
```
