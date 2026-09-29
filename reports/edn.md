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

### EDN-20: Which DagsHub repository the team uses as DVC remote and MLflow server

- **Date:** 2026-09-29
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Data Versioning, Experiment Tracking
- **Participants:** @lukas2510 (Scrum Master), on behalf of the team. Pending confirmation at the next sprint review.
- **Decision:** `pauadal03/MLOPS_Recommenditos` is the single DVC remote and MLflow tracking server for the project (option B).
  `mark.welf.atzberger/MLOPS_Recommenditos` is retired.
  The related question of whether that repository stays public was decided separately, see EDN-22.
- **Alternatives considered:**
  - **Option A: a DagsHub organisation owns the connected repository, every team member is a member with push rights.**
    Pros: mirrors the GitHub organisation, so "our repository" does not depend on one person's private account; push rights follow org membership instead of manual collaborator entries; data and experiment history survive if a member leaves or cleans up their account; visibility is set in one place.
    Cons: one-time setup (create the org, connect the GitHub repository through the GitHub integration, invite everyone, retire the two existing mirrors) and the data has to be pushed again; the remote URL in PR #26 has to change.
  - **Option B (chosen): use `pauadal03/MLOPS_Recommenditos`.**
    Pros: verifiably connected through the GitHub integration, see the evidence below; only the remote URL in PR #26 changes.
    Cons: all data and experiments hang off one member's private account; every member needs a manual collaborator entry with write access; the raw data would have to be pushed again.
  - **Option C: use `mark.welf.atzberger/MLOPS_Recommenditos`, what PR #26 configures and where the raw data already sits.**
    Pros: no change to PR #26; the 548.6 MB raw dataset is already pushed there, so nothing has to be re-uploaded.
    Cons: its mirror is demonstrably not in sync, see the evidence below; same account and access drawbacks as option B.
  - **Option D: keep both mirrors and let everyone push to their own.**
    Pros: no coordination needed.
    Cons: not viable - the remote URL is committed in `.dvc/config`, so the team would overwrite each other's setting, and data and experiment history would be split across accounts.
- **Rationale:** Option B is the only one of the two workable candidates that is verifiably connected the right way, and it costs nothing to adopt.
  Option A is the better long-term structure and the team agrees with that in principle, but for a one-semester course project it buys little: creating an organisation, reconnecting the GitHub repository and re-inviting 21 people is setup whose payoff (surviving account churn, rights by membership rather than by hand) lands after the course has ended.
  The rule the team applied here was to take the better-engineered option unless it is overkill for the scope, and on that test A is overkill while B is not.
  Option C was ruled out on the measurements below, not on preference.
  The residual risk of B is stated rather than hidden: the remote hangs off one member's private account, and every teammate needs a manual collaborator entry with write access.
  Evidence gathered on 2026-09-29 against the DagsHub API.
  Sync: `pauadal03/MLOPS_Recommenditos` is at `2e902e2143`, identical to GitHub `main`, carries all 7 branches and mirrors all 7 open issues.
  `mark.welf.atzberger/MLOPS_Recommenditos` is at `8b1d64fc`, one merge behind, is missing the `docs/data-facts-corrections` and `model-card` branches, still carries `feature/ruff-pl-and-coverage` which GitHub deleted after the merge, and mirrors 0 issues.
  DagsHub only mirrors issues and pull requests for repositories connected through the GitHub integration, so the issue count is the clearest signal that the two repositories were connected in different ways and that mark's is a plain git mirror.
  Visibility: both repositories report `private: false`, so every logged-in DagsHub user can read them and pull the raw data, which still carries `vin`, `street`, `seller_company_name` and coordinates.
  EDN-07 as originally written assumed the remote was private; that premise was found false and is being revised on the issue #3 branch, and the outcome is recorded in EDN-22.
  The reason the original check misled us: anonymous access is not evidence either way, because DagsHub refuses every anonymous request, including for a known-public control repository (`DAGsHub-Official/dagshub-docs`, also `private: false`) and for a repository that does not exist.
  Access: `lukas2510` has `push` on mark's repository since 2026-09-29 and `pull` only on pauadal03's; the rest of the team is most likely still without push rights on either.
  Follow-up the decision requires: the DagsHub remote URL in [PR #26](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/26) has to be changed to `https://dagshub.com/pauadal03/MLOPS_Recommenditos.dvc` before that PR merges, pauadal03 has to add every teammate as a collaborator with write access, and mark's retired mirror has to be dealt with, including the 548.6 MB raw object already pushed to it.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI recommended option A; the team took option B instead, accepting AI's finding that mark's mirror is unreliable but judging the organisation setup disproportionate for a course project.
  The part of the contribution that decided the outcome was the measurement, not the recommendation: without the issue-mirroring comparison the team would have kept option C, which is what the open PR configured.
  AI queried the DagsHub API, found that two mirrors of the GitHub repository exist, compared their commits, branches and mirrored issues against GitHub, checked the team's access rights, verified the repositories' visibility against a public control repository, and laid out the four options above with a recommendation for option A.
- **AI interaction evidence:** Claude Code session on 2026-09-29: prompts "see if you can connect to dagshub i gave you token and usernme", "ist nicht das unser dagshub repo was zu unserem github repo passt?" and "warum hat das repo am schlchtesten abgeschnitten"; the session verified the credentials against the DagsHub API, found the two competing mirrors and produced this comparison.
- **Other evidence:** [PR #26](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/26), [issue #10](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/10), EDN-07.
- **In LaTeX:** no

### EDN-22: The DagsHub remote stays public

- **Date:** 2026-09-29
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Data Versioning, Privacy
- **Participants:** @lukas2510 (Scrum Master). Pending confirmation by the full team at the next sprint review.
- **Decision:** The DagsHub repository that serves as our DVC remote and MLflow tracking server stays public.
  We accept that the raw file's PII columns are readable by anyone with a DagsHub account.
  The additional exposure this creates is small, because the identical file is already publicly downloadable from the pinned Zenodo DOI under the MIT license.
  It is not nothing: hosting our own copy makes the team a publisher of that personal data in its own right and under its own name, which the earlier publication mitigates but does not undo.
- **Alternatives considered:**
  - **Option A (chosen): keep the repository public and say so.**
    Pros: graders and supervisors can inspect code, data and experiments without being added as collaborators, which is the reason the course puts the project on DagsHub in the first place; it matches how the source dataset is already published; no action needed on an account we do not administer.
    Cons: the raw file, which still carries `vin`, `street`, `seller_company_name` and coordinates, can be pulled by every logged-in DagsHub user; re-hosting it is a publishing act of our own, so "the author already published it" reduces the marginal risk but does not transfer the responsibility; the decision has to be revisited the moment we host data that is not already public elsewhere.
  - **Option B: make the repository private.**
    Pros: the original rationale of EDN-07 would hold as written; the PII sits behind an access wall.
    Cons: only the repository owner can flip it, and that is not us; every grader, supervisor and teammate then needs a manual collaborator entry; it buys little real protection, because the identical file stays publicly downloadable from Zenodo either way.
- **Rationale:** The question only came up because the premise of EDN-07 turned out to be false.
  That entry justified pushing the raw file with "the DagsHub repo is private (anonymous access is redirected to the login page and an anonymous data download returns 401)".
  Checked on 2026-09-29: DagsHub refuses anonymous access to everything, public repositories included.
  A known-public control repository (`DAGsHub-Official/dagshub-docs`, `private: false`) answers anonymous requests exactly like ours, and so does a repository that does not exist, so the observation has no discriminating power.
  The authenticated API reports `private: false` for both of our mirrors.
  Given a real choice between hiding a file that is already public and saying openly that it is public, Lukas chose the second: the protection gained would be nominal, while the access cost for graders and supervisors would be real.
  Recorded deliberately as an accepted risk rather than a solved problem, because the two are not the same thing: the marginal exposure is small, but we are still the ones publishing personal data, and an entry that claimed the concern was spent would not survive a reviewer who cares about privacy.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment
- **Response to AI:** Used as input for further analysis
- **Assessment of the AI contribution:** AI found that the privacy premise of EDN-07 was false, using a control repository to show that anonymous refusal does not distinguish public from private, and put both ways out to Lukas without recommending either, since the trade-off is about how open the team wants to be rather than a technical question. Lukas decided to keep the repository public. The finding is what changed the outcome here; the decision itself was not AI's to make.
- **AI interaction evidence:** Claude Code session on 2026-09-29: after the finding was presented with the two options ("entweder mark stellt es auf private ... oder ihr lasst es public und schreibt in EDN-07 ehrlich rein"), Lukas answered "wir lassen es public".
- **Other evidence:** EDN-07 and its revision on the issue #3 branch, EDN-20, [PR #27](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/27), [PR #28](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/28).
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
