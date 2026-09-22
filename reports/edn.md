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
