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

- **Date:** 2026-09-29
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Data Validation and Preprocessing
- **Participants:** @lukas2510
- **Decision:** Preprocessing drops the 164 listings whose `registration_date` is after the age reference date (the 2025-11-08 snapshot date in training, the request date in serving). The Great Expectations suite asserts `registration_date <= reference date` on the processed data as a hard expectation, and the same rule on the raw data with `mostly=0.99`, so a future scrape that suddenly carries many such rows is flagged instead of silently cleaned away.
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
