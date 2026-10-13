# Engineering Decision Notebook - archive of retired and merged entries

This file holds, word for word, the 42 entries that left [edn.md](edn.md) when the notebook was trimmed on 2026-10-12 and 2026-10-13.
It is not part of the graded notebook: `make edn` does not read it and the PDF does not include it.
It exists so that nothing is lost and an ID cited in an older commit, comment or document can still be looked up.
The table "Retired and merged IDs" in [edn.md](edn.md#retired-and-merged-ids) says which entry a merged one lives on in and why a retired one was dropped.
The entries are as they stood on `main` at commit `d8392d6`; links between entries that pointed into edn.md no longer resolve here.
Do not edit them.

## Archived entries

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

### EDN-34: The derived raw Parquet keeps its PII columns and is pushed like any other stage output

- **Date:** 2026-09-30
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Data Versioning, Personal Data
- **Participants:** @lukas2510. Confirmed by the full team at the laboratory session of 2026-10-13, together with EDN-25.
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
- **In LaTeX:** yes

### EDN-35: The `download` stage owns `data/raw/`, and the published CSV is a local cache outside the DAG

- **Date:** 2026-09-30
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Data Versioning, Pipeline Design
- **Participants:** @lukas2510. Confirmed by the full team at the laboratory session of 2026-10-13, together with EDN-25.
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
- **In LaTeX:** yes

### EDN-36: Retire EDN-25's `dvc add` pointer, so we no longer host a copy of the raw CSV

- **Date:** 2026-09-30
- **Milestone:** M2: Reproducibility
- **Activity / Topic:** Data Versioning, Personal Data
- **Participants:** @lukas2510. Confirmed by the full team at the laboratory session of 2026-10-13, together with EDN-25.
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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **Other evidence:** [issue #42](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/42); [PR #50](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/50); [EDN-19](#edn-19-which-dagshub-repository-the-team-uses-as-dvc-remote-and-mlflow-server); NFR-15 in [the specification](../docs/docs/specification.md), which held this as part of NFR-09 when this entry was written ([EDN-66](#edn-66-nfr-06-and-nfr-09-are-each-split-in-two-and-each-id-keeps-the-half-its-citations-mean)).
- **In LaTeX:** yes

### EDN-48: `split` records the supported-make list and stays a lossless partition; the downstream stages apply it

> **Amended by EDN-67** (2026-10-05): `features` is now the one stage that applies the list, to all five frames and before the vocabulary is built, and `train` and `evaluate` check it rather than filtering again.
> All four guarantees listed below are discharged; `docs/docs/pipeline.md` records where under "The supported makes".

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

  What the chosen option owes, and what `docs/docs/pipeline.md` records under "The supported makes", is four guarantees from #36, #37 and #39, none of them optional: the filter applied to train, validation, calibration, test **and** the `ES` holdout; applied **before** any training-derived vocabulary or statistic is computed, since fitting on rows the API refuses puts unservable makes into the model's own inputs; `evaluate`'s per-make segments restricted to the listed makes; and a test asserting that the filtered frames contain only supported makes, so the guarantee is checked rather than intended.
  How each was discharged: the third by `evaluate` in #39 (`check_make_levels_are_served`); the first and second together by `features` in #63 ([EDN-67](#edn-67-features-applies-the-supported-make-list-to-every-frame-the-es-holdout-included-before-the-vocabulary-is-built)), which replaced the train-and-validation filter #37 had put in `train` and the test filter #39 had put in `evaluate` with checks that fail instead of filtering; and the fourth by a test for each of the three, every one of which first asserts that its input holds an unsupported make.

  Two stale claims were a consequence of this decision, and are now applied: the comments in `recommenditos/schema.py` and `recommenditos/data/preprocess.py` said the supported-make filter belongs to, or happens in, `split`. Both now say that `split` computes the list and the stages building model input apply it.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI found that the skeleton contained evidence for both readings and that the two documents behind them disagree, rather than implementing one and moving on: `schema.py` and `preprocess.py` place the filter in `split`, while problem-spec section 5 and EDN-05 place it at the API. It laid out the options with the irreversibility argument that decided it, and flagged the choice as an EDN candidate before acting on it. It also measured the consequences instead of asserting them, which is what produced the 0.08 pp share shift and the 5,979 of 6,079 holdout figure, and it corrected its own earlier framing that the out-of-scope rows would inflate the metrics once it checked the direction of the effect. Lukas confirmed option C.
- **AI interaction evidence:** Claude Code sessions on 2026-09-30 while implementing and then adversarially reviewing issue #35: the first flagged the two readings and asked for a decision rather than choosing silently; the adversarial review verified the numbers and found that the obligation existed only in the pull request body, with issues #36, #37 and #39 mentioning the make list nowhere at all.
- **Other evidence:** [`recommenditos/data/split_data.py`](../recommenditos/data/split_data.py); the `supported_makes.json` dependency on the `features` stage in [`dvc.yaml`](../dvc.yaml); [pipeline docs](../docs/docs/pipeline.md), "The supported makes", for the four downstream guarantees; [`reports/analysis/split_gate.py`](analysis/split_gate.py) for the 0.08 pp and 5,979 measurements; [EDN-05](#edn-05-supported-makes-minimum-listing-support-per-make); [EDN-14](#edn-14-nfr-11s-drift-control-is-an-iid-sample-not-a-seller-grouped-one); [EDN-39](#edn-39-the-splits-size-gate-is-four-rules-with-an-unconditional-floor-not-one-bound-derived-from-the-realised-data); [issue #35](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/35); [PR #53](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/53).
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

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
- **In LaTeX:** yes

### EDN-59: SC-06 masks API input fields, not feature columns, and an absent field's value follows EDN-23

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Model Evaluation (SC-06), Feature Encoding
- **Participants:** @lukas2510
- **Decision:** SC-06's masking sweep is defined over the **API input fields** of the evaluated variant's own feature set, not over the model's feature columns. Every input field is mapped to the column or columns it produced, so `registration_date` covers the derived `age_years` and one equipment field covers every multi-hot column built from it; the optional set is then that mapping minus `evaluate.required_input_fields`, which is 6 fields for the basic set and 23 for the extended one. What "masked" means is decided per field from the contract rather than being one missing value for everything, and it follows EDN-23: an assertion flag becomes `False`, an equipment multi-hot column becomes `False`, a categorical becomes a null code over the declared levels, and a number becomes `NaN`. Any other encoding raises, naming the column and its dtype. The mask is applied to the feature matrix rather than to the raw frame, and the artefact records which fields were masked so the approximation is visible.
- **Scope, after the review of pull request #62:** what an omitted equipment list is was re-opened there and is recorded by [EDN-61](#edn-61-an-omitted-equipment-list-is-the-empty-list-and-the-seam-owns-the-masking), not here. It settled as **the empty list, every derived column `False`**, which is the value this entry was already written against and the one its tests assert, so nothing measured here changed. This entry therefore cites EDN-61 for that value and keeps what is its own: SC-06 defined over input fields rather than feature columns, the mapping from an input field to the columns it produced, the per-dtype rule for the other three kinds of column, and the record below of why nullable-boolean masking was rejected for the three condition flags. `Model.mask_absent` is the single implementation of the rule and the masking sweep calls it.
- **Alternatives considered:**
  - **Option A (chosen): input fields, mapped to their columns, masked in the feature matrix, absence per EDN-23.**
    Pros: it measures the request FR-01 actually permits. A caller cannot leave out one equipment *item*, so masking one multi-hot column measures something the API can never be asked. The mapping also makes the sweep cost stable across the encoding: 33 input fields and 23 optional ones for the extended set whether the equipment columns number 16 (fixture) or 133 (real snapshot), where a column-wise sweep would have grown from 24 predict calls to 162. And it removes the defect the naive reading has: `age_years` is not in `required_input_fields`, so a plain set difference over feature columns makes the single most important feature in the model look optional and SC-06 would have reported the cost of hiding a car's age.
    Cons: it needs one explicit mapping in `params.yaml` (`input_field_to_feature`), which is a line somebody has to maintain, and the mask is applied to the matrix rather than by re-running the feature build, so for a field whose encoding is not one-to-one it is an approximation of the serving path.
  - **Option B: mask feature columns, one at a time.**
    Pros: no mapping, no parameter, and the sweep is derivable from `model.features` alone.
    Cons: measures requests the API cannot make, masks `age_years` as if it were optional, and its cost and its result both change when issue #36's encoding changes, so the criterion would not be comparable across feature-set revisions.
  - **Option C: mask the raw processed frame and re-run the feature build per scenario.**
    Pros: literally the serving path, so exact for every field including the equipment ones.
    Cons: needs a reusable transform the `features` stage does not expose yet, and it is 25 feature builds per variant instead of 25 predict calls. Worth revisiting when the API needs that same seam.
  - **Option D: mask an assertion flag to `pd.NA` on a nullable `boolean` column, so the criterion measures absence rather than a denial.**
    Pros: the shape a careful reviewer expects, and it distinguishes "not asserted" from "asserted false".
    Cons: it is the encoding EDN-23 examined and rejected, for the reason that in this data there is no third state to encode: a listing form offers a checkbox, a tick is evidence and an empty box is the absence of evidence, so `False` **is** absence. It also does not work: the three flags are non-nullable in the persisted contract, so `Schema.conform` refuses a null one and `predict_eur` raises rather than predicting. Reversing EDN-23 for one criterion would have made the pipeline and the API disagree about what an omitted flag is.
- **Rationale:** The field-level definition is the one that matches the requirement: FR-01 is a statement about a request, so a criterion about "a missing optional field" has to be measured one request at a time. The two assertions that make the mapping safe are what justify keeping it in code rather than in a document: every required input field must map to at least one column of the trained model, and no column may be claimed by two fields, so a params edit that breaks either fails the stage instead of quietly changing which fields the criterion covers. Option D was chosen first and then withdrawn on re-reading EDN-23; the deciding facts were that the flags are declared non-nullable, so `pd.NA` raises rather than being measured, and that the regression test for the alternative encoding demonstrates what would have happened instead: assigning `NaN` into a `bool` column on pandas 3 replaces it with a float64 column of `NaN`, on which `.all()` returns `True`, so a consumer reading it as a flag sees "asserted for every row" and the column is destroyed rather than masked. The mask therefore derives its value from the contract's dtype and refuses an encoding it does not recognise, so an encoding choice nobody anticipated fails loudly instead of masking the wrong way. Option A's approximation is recorded rather than hidden: for every field of the basic set the encoding is one-to-one, so A and C are identical there, and the difference is confined to the equipment fields.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI found the `age_years` defect in the written criterion before any code was written, by computing the naive set difference and noticing it yields seven optional fields for the basic set where EDN-15 states six; two independent sessions found it separately, which is the cross-check that it is a defect in the specification rather than a reading of it. AI also proposed option D and wrote it into the design as settled, and the reversal to option A came from checking it against EDN-23 and against the persisted contract, where it fails outright. That is the part of the contribution that was rejected, and it is worth recording as rejected: the argument for `pd.NA` was about what a reviewer would expect to see rather than about this data, and EDN-23 had already answered it with measurements. The per-dtype rule, the two mapping assertions and the "raise on an encoding you do not recognise" rule were accepted as proposed.
- **AI interaction evidence:** Claude Code sessions on 2026-09-30 on issues #37 and #39. The #39 design session reported "the naive set difference yields 7 optional fields for `basic` where EDN-15 says 6, and the spurious one is `age_years`, the single most important feature in the model", and separately measured the mapping against a simulated post-#36 matrix to show the optional-field count is stable at 23. The same session recommended the nullable-boolean masking of option D; the implementation session checked it against EDN-23 and against `Schema._null_problems`, found that it raises `SchemaError`, and implemented `False`.
- **Other evidence:** [`recommenditos/modeling/evaluate.py`](../recommenditos/modeling/evaluate.py), `input_field_columns`, `optional_input_fields`, `sc06_scenarios` and `masking_sweep`; [`recommenditos/modeling/model.py`](../recommenditos/modeling/model.py), `Model.mask_absent` and `Model.columns_of`, which this entry defers to for the value and calls for the expansion; `params.yaml`, `evaluate.input_field_to_feature`; `tests/test_model.py::test_the_optional_fields_of_the_basic_set_are_the_six_of_edn15`, `::test_an_equipment_field_covers_every_multi_hot_column_it_produced`, `::test_every_feature_column_is_claimed_by_exactly_one_input_field` and `::test_mask_absent_is_the_one_answer_to_what_an_omitted_field_is`; [problem specification](../docs/docs/problem-spec.md) section 8 and [specification](../docs/docs/specification.md) FR-01, both reworded by this decision; [EDN-15](#edn-15-uc1-required-fields-after-measuring-the-fill-rates-plus-sc-06-for-absent-optional-fields), [EDN-23](#edn-23-read-the-condition-flags-as-one-sided-assertions-instead-of-encoding-them-three-valued); [issue #39](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/39).
- **In LaTeX:** yes

### EDN-60: Which segments may gate is enforced in code that fails, not by a rule in a document

- **Date:** 2026-09-30
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Model Evaluation (SC-04), Testing Strategy
- **Participants:** @lukas2510
- **Decision:** The three rules that decide which segment levels SC-04 is allowed to gate on are checks that fail the stage rather than conventions a reader has to remember. A module constant names the segments that condition on the target (`price_bucket`), and adding one to `evaluate.segments` raises; the row builder reads that constant itself, so no caller can produce a price-bucket row that counts. A segment over an input field FR-01 does not require also raises, because the rule that excludes a `"(missing)"` level is only correct for a required field. And a per-make level naming a make the model was not fitted on raises, so EDN-48's third obligation is a checked property rather than an intention. Every excluded level stays in the exported table with its metrics, its row count and the name of the rule that excluded it.
- **Alternatives considered:**
  - **Option A (chosen): a constant plus three refusals, and every exclusion recorded in the table.**
    Pros: the rule lives where it is applied, so it cannot be forgotten by someone editing `params.yaml` a month from now, and each refusal names the reason rather than the symptom. Keeping the excluded rows in the table with `excluded_because` means the report can show that `Volkswagen` had 81 test rows and `fuel_category=(missing)` had 5, which is the difference between a breakdown a reader can audit and one that silently omits what did not fit.
    Cons: more code than a comment, and three error paths that only fire on a configuration nobody has written yet.
  - **Option B: document the rules in the problem specification and rely on review.**
    Pros: no code, and the specification is where the criterion is defined anyway.
    Cons: the failure it has to prevent is a one-line `params.yaml` edit that produces a passing gate. SC-04's worst qualifying level for `lgbm-extended` is 17.8 %; its worst price bucket is 22.3 %, so adding `price_bucket` to the segment list changes the number the gate blocks on while looking like a reporting improvement. A rule that is only written down is checked exactly as often as somebody reads it.
  - **Option C: drop the price buckets from the stage entirely.**
    Pros: nothing to exclude, so nothing to enforce.
    Cons: they are the diagnostic that shows *why* they may not gate. Measured on the real snapshot, their profile is U-shaped (22.3 % under 5,000 EUR and 6.3 % over 80,000 EUR against 5.5 % in the 40,000 to 80,000 EUR bucket), which is the regression-to-the-mean artefact of conditioning on the target. Without the table the exclusion is an assertion; with it, it is evidence.
  - **Option D: exclude a `"(missing)"` level for every segment, without checking that the field is required.**
    Pros: one rule, no second check.
    Cons: it would be wrong the first time somebody segments by an optional field. For a required field the absence cannot occur at serving time, because FR-01 answers such a request with a 422; for an optional field it is a case the API answers every day, and excluding it would hide exactly the rows worth looking at.
- **Rationale:** All three rules share one shape: a statement about what the criterion means that would otherwise be maintained by discipline. The M3 rubric rewards exactly this distinction, and the measurement makes it concrete rather than stylistic: the price buckets would change SC-04's verdict if they gated, so the difference between option A and option B is a number in the report. Option D was rejected on the same grounds as the price buckets: it is right today only because all five configured segments happen to be required fields, and an accidental correctness is what the check turns into a guaranteed one.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI proposed the target-conditioned constant and the "absence of a required field" exclusion, and measured the price-bucket profile that shows why the exclusion matters rather than asserting that it does. Two modifications were made to what it proposed. Its construction passed an `eligible=False` flag into the row builder for the price-bucket rows, which leaves the exclusion dependent on the caller passing the flag; the flag was removed and the builder now reads the segment's own name, so the exclusion holds however the rows are built. And it did not propose the third check at all: the requirement that a segment be a required input field came from asking what makes the `"(missing)"` exclusion correct, which turned out to be an assumption nothing enforced.
- **AI interaction evidence:** Claude Code session on 2026-09-30 designing issue #39, which measured the price-bucket breakdown at 22,551 test rows and reported "the U shape is the regression-to-the-mean artefact of conditioning on the target, not a fairness finding, and the 490k MAE in the top bucket would dominate any aggregate"; and the implementation session on the same day, which replaced the `eligible` flag with the name-based rule while reducing the row builder's argument count.
- **Other evidence:** [`recommenditos/modeling/evaluate.py`](../recommenditos/modeling/evaluate.py), `TARGET_CONDITIONED_SEGMENTS`, `criterion_segments`, `_sc04_eligibility` and `check_make_levels_are_served`; `tests/test_model.py::test_a_target_conditioned_segment_cannot_become_a_criterion`, `::test_a_segment_over_an_optional_input_field_is_refused`, `::test_a_price_bucket_row_is_reported_and_cannot_count`, `::test_sc04_refuses_rows_from_a_segment_it_may_not_gate`, `::test_a_per_make_level_the_api_would_reject_is_refused`; `reports/metrics/segments.csv`; [model card](../docs/docs/model-card.md), the per-segment breakdown; [EDN-48](#edn-48-split-records-the-supported-make-list-and-stays-a-lossless-partition-the-downstream-stages-apply-it); [issue #39](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/39).
- **In LaTeX:** yes

### EDN-63: `metrics.json` carries the data source and the fitted estimator, at the cost of five more columns

- **Date:** 2026-10-01
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Experiment Tracking, Release Gate
- **Participants:** @lukas2510
- **Decision:** `metrics.json` gains a `data_source` leaf, the resolved `download.source`, and an `estimator` leaf per variant, the model actually fitted; each `reports/metrics/<variant>.json` carries both as well. That widens the DVC metrics file from 24 leaves to 29, and `evaluate` declares `download.source` under `params:` so the recorded value cannot disagree with the lock.
- **Alternatives considered:**
  - **Option A (chosen): the provenance goes into `metrics.json` itself, five leaves wide.**
    Pros: `dvc metrics show` and `dvc metrics diff` are where a promotion reviewer reads a result (EDN-12), so the one place a number can be quoted from is the one place that now says what produced it. `dvc metrics diff` lists a changed leaf, so a source or estimator that changes between two commits shows up beside the metric it moved.
    Cons: the file was already past the width a terminal table keeps, and this is five more columns of something that is not a measurement.
  - **Option B: the provenance goes only into the per-variant records under `reports/metrics/`.**
    Pros: costs nothing in the metrics table; the per-variant record already held `estimator`.
    Cons: it answers the question in the file nobody reads first. The defect this exists to prevent was a reader quoting `dvc metrics show`, and a second file they have to open is exactly the step that was skipped.
  - **Option C: an MLflow tag on each run and nothing in the artefacts.**
    Pros: the experiment is where the ladder is compared, and a tag is free.
    Cons: tracking is optional by construction - it degrades to disabled with no credentials - so the one artefact that is always written would be the one that says nothing. It also puts the evidence behind a login, while `metrics.json` is in the pull request diff.
  - **Option D: nothing, and rely on `dvc.lock` recording the source.**
    Pros: the lock already records it, so the information exists.
    Cons: it exists in a file whose `download` stage entry nobody reads to interpret an MdAPE, and it is a different file from the one holding the number. That is the state that let a constant-median stub's figure on generated rows sit on `main` as `best_mdape` for a week.
- **Rationale:** The honest version of the trade-off is that the metrics file is already too wide to read in a terminal and this makes it wider, so the question is only whether provenance earns five of those columns. It does, because the failure it prevents happened: `metrics.json` on `main` carried one MdAPE under all four variant names, computed by a constant-median stub on a synthetic fixture, and nothing in the artefact said either thing. The width argument also has a limit that matters here: the file is already read with `dvc metrics show --md`, where five more columns cost nothing a reader notices, while `dvc metrics diff` is long-format and lists only what changed, so the new leaves are invisible until they are the answer.
  Declaring `download.source` under the stage's `params:` is the part that makes it a guarantee rather than a label. Without it the stage would read a key DVC does not watch, and a recorded value could drift from the lock; with it, the convention `dvc.yaml` states for every other stage holds for this one too.
  One consequence was found while implementing it and is worth recording: the end-to-end test fixture ran the stages against the project's own `params.yaml`, so after the flip to `zenodo` the fixture's artefacts would have claimed `data_source: zenodo` over generated rows - the exact mislabel the field exists to prevent. The fixture now runs against a params copy pinned to `synthetic`, and a test asserts the artefacts say so, which is what keeps the new field from being decorative.
- **AI involvement:** Alternative generation, Alternative assessment, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** Issue #57 had already identified the defect and asked for the source and the estimator in `metrics.json`, so the problem was not AI's to find. What AI contributed is the shape and the two things that make it hold: declaring `download.source` under the stage's `params:` rather than reading an undeclared key, and noticing that the test fixture would start mislabelling its own artefacts as soon as the project's source flipped. The second one is the kind of defect this project has repeatedly produced - a claim the code does not back - and it would have shipped as a passing test suite.
- **AI interaction evidence:** Claude Code session on 2026-10-01 implementing issue #57: the four options were laid out against the existing docstring's 24-leaf constraint, and the fixture's mislabelling was found by asking what `params["download"]["source"]` would resolve to inside the test run rather than inside the pipeline.
- **Other evidence:** [`recommenditos/modeling/evaluate.py`](../recommenditos/modeling/evaluate.py), `gate_summary` and `_evaluate_variant`; `tests/test_pipeline.py::test_the_metrics_artefacts_say_which_data_and_which_estimator_produced_them`; `metrics.json`; [EDN-12](#edn-12-retraining-and-promotion-are-human-triggered-not-automated), [EDN-33](#edn-33-a-generated-synthetic-fixture-and-a-downloadsource-parameter-so-the-skeleton-runs-without-the-raw-file); [issue #57](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/57).
- **In LaTeX:** yes

### EDN-64: Pynblint runs from its own locked environment, and a wrapper that reads its JSON report is the gate

- **Date:** 2026-10-01
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Static Analysis, CI/CD
- **Participants:** @lukas2510, decided by the repository owner
- **Decision:** Pynblint 0.1.6 runs from a dedicated tool environment in `tools/pynblint-env/`, with its own `pyproject.toml` (`pynblint==0.1.6`, `click<8.2`, Python 3.12) and its own hash-verified `uv.lock`, invoked as `uv run --project tools/pynblint-env --locked`.
  A stdlib-only wrapper, `tools/notebook_lint.py`, runs inside that environment, lints each notebook, reads Pynblint's JSON report and is what fails the build, because Pynblint itself exits 0 whatever it finds.
  `make notebook-lint` is the one command, and the CI job `Notebook lint (Pynblint)` runs exactly that target.
- **Alternatives considered:**
  - **Option A (chosen): a locked tool environment, plus a wrapper that is the gate.**
    Pros: all 63 packages are pinned and their hashes verified, so the tool cannot change under us; `--locked` fails when the `pyproject.toml` and the lock drift apart; the root `uv.lock` is untouched and the directory is not a workspace member; and the lock is a file Dependabot can read.
    The wrapper runs Pynblint with an allowlisted environment in an empty working directory, so no stray variable or `.pynblint` file can reconfigure it, and it fails loudly when Pynblint crashes instead of reading a crash as a clean run.
    Cons: two more files, one of them a generated 75 KB lock, a wrapper of its own with tests, and a deviation from the issue's literal "via `uvx pynblint`".
  - **Option B: `uvx --python 3.12 --with 'click==8.1.8' pynblint==0.1.6`.**
    Pros: one line, and the closest to what the issue asked for.
    Cons: about 59 transitive dependencies float, and that is exactly how the tool broke: a click release nobody chose crashed it, so CI can turn red on a pull request that changed nothing near it, and a run cannot be reproduced later.
  - **Option C: the same, plus `--exclude-newer <date>`.**
    Pros: the whole resolution is frozen in time, still in one line.
    Cons: no hash verification, a magic date in the Makefile, and no lock file for the dependency graph or Dependabot to see.
  - **Option D: a hashed requirements file passed to `uvx -c`.**
    Pros: looks like a lock without a second project.
    Cons: `uvx` silently ignores the hashes; a deliberately tampered hash still ran with exit 0, while `uv pip install -r` on the same file failed with "Hash mismatch".
  - **Option E: a pre-commit hook with `additional_dependencies`.**
    Pros: feedback at commit time, and CI's existing lint job would run it.
    Cons: the dependencies float as in option B, and the raw hook reported "Passed" on a notebook with findings, so it needs the same wrapper anyway.
  - **Option F: a dependency group in the root project, kept apart with uv's `conflicts`.**
    Pros: one lock file for everything.
    Cons: it ties our lock to an unmaintained tool's `typer<0.13` and `ipython<9`, and every resolution of the project would carry that weight.
  - **For the gate: `jq -e` over the JSON in the Makefile, or the wrapper run in the project environment.**
    Pros: `jq` needs no code; the project environment would let the wrapper use typer and loguru like `tools/requirement_matrix.py`.
    Cons: `jq` is not on every contributor's machine and prints no reasons; the project environment would make a seconds-long CI job install mlflow, lightgbm and DVC just to lint.
- **Rationale:** The decision rests on three measured facts about Pynblint 0.1.6, each reproduced rather than read from its documentation.
  First, it does not run as the issue proposed: it pins `typer<0.13` but not click, and typer 0.12 crashes on click 8.2 and newer (released 2025-05-10) with `TypeError: Secondary flag is not valid for non-boolean flag`, on Python 3.12, 3.13 and 3.14 alike.
  A tool that broke once through a dependency nobody pinned should not be run in a way that lets it break again, which rules out options B and E and makes the lock worth its two files.
  Second, it exits 0 with findings: a notebook with eleven findings still returned 0 in both file and directory mode, so a CI step that only runs it can never fail and something has to read the report.
  Third, its configuration leaks in from outside: its settings are read from environment variables with no prefix, so an ordinary `INCLUDE=/usr/include` crashed a run with a validation error, and a `.pynblint` file in the current directory reconfigures it silently.
  The wrapper is the smallest thing that answers all three: it pins nothing itself, it isolates the process, and its exit code is the finding count.
  It is stdlib only so that it can run in the tool environment, which keeps the CI job free of the project's own dependencies.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI did the experiments the decision rests on: it reproduced the click crash and pinned it to click rather than to the Python version, measured the exit code with findings, found that `uvx -c` ignores hashes by tampering with one, and timed every isolation option cold and warm.
  It also found that the issue's own premise, "run it isolated via `uvx pynblint`", did not work as written.
  AI then laid out options A to F with a recommendation, and the owner chose A.
  AI also wrote the wrapper and its tests, test-first, and two further AI review passes, one on the code and one on the documents, found defects that were fixed before the pull request was opened.
  A mutation run showed that no test noticed when the gate stopped passing the exclusions to Pynblint; an enforced repository rule would have been reported as checked although it never runs on a single notebook; and a report of an unexpected shape exited 1, which reads as findings, instead of 2.
  Each now has a test.
  An independent AI review of the pull request then found two more: the Pynblint answers the tests replay were recorded once and never checked against the real tool, so a release that stopped reporting findings would have passed every test, and a report that could not be written exited 1, which reads as findings.
  A test that runs the real Pynblint from the tool environment and a fix for the exit code followed.
  The owner reviews the result in the pull request.
- **AI interaction evidence:** Claude Code session on 2026-10-01 implementing issue #41: a research run on Pynblint 0.1.6 in a scratch directory (rule catalogue, configuration, exit codes, isolation options with timings), then an architecture pass that turned the measurements into options with pros and cons; the owner was given the isolation and gate options with these measurements and chose the locked environment with the wrapper.
  Claude Code review session on 2026-10-05 of PR #67, which ran the gate on deliberately broken copies of the notebook and checked the tests' recorded answers against the real Pynblint.
- **Other evidence:** [`tools/pynblint-env/pyproject.toml`](../tools/pynblint-env/pyproject.toml), [`tools/notebook_lint.py`](../tools/notebook_lint.py), `tests/test_notebook_lint.py`, the `Notebook lint (Pynblint)` job in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml); [EDN-65](#edn-65-the-pynblint-policy-enforce-what-can-fire-on-a-committed-notebook-and-show-execution-by-running-it); [issue #41](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/41), [PR #67](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/67).
- **In LaTeX:** yes

### EDN-65: The Pynblint policy: enforce what can fire on a committed notebook, and show execution by running it

- **Date:** 2026-10-01
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Static Analysis, Testing Strategy
- **Participants:** @lukas2510, decided by the repository owner
- **Decision:** Of Pynblint 0.1.6's 22 rules, 13 are enforced at their default thresholds and 9 are excluded, each with its reason, in the table in [Notebooks](../docs/docs/notebooks.md#pynblint-rules).
  That table is the configuration: the gate reads it, and refuses to run unless it classifies exactly the rules the installed Pynblint has.
  The three rules that read execution counts are excluded, because nbstripout removes those counts on every commit, and that a notebook runs top to bottom is shown by running it with `make notebook-run` instead.
  The five repository-level rules are not run, each with the reason and with what enforces the same property here.
  Pynblint lints every notebook git would commit, one file at a time, and two rules of our own require every notebook to sit directly in `notebooks/` under the Cookiecutter name `<number>.<version>-<initials>-<description>.ipynb`, and to carry none of the flags that make nbstripout keep its outputs.
  There is no warning tier and no per-notebook waiver: a finding fails the build, and a rule that is wrong for us is excluded in the table, with its reason, in a reviewed pull request.
- **Alternatives considered:**
  - **Option A (chosen): exclude the execution rules, keep nbstripout as it is, and evidence execution by running the notebook.**
    Pros: nbstripout 0.9.1 sets every `execution_count` to null, so on a committed notebook `non-executed-notebook` always fires, `non-linear-execution` can never fire, and `non-executed-cells` fires only on an empty code cell and then flags every other code cell; enforcing them would either keep the build red forever or claim checks that do not happen.
    Stripping also keeps rows of a dataset with personal data out of git, which is more than diff hygiene.
    The profiling notebook's summary cell fails when a dataset card figure does not reproduce, so the drill checks the card as well as the code.
    Cons: execution is not gated in CI; `make notebook-run` is a local drill, because the notebook needs the 548 MB source file.
  - **Option B: `nbstripout --keep-count`, so the three rules have something to read.**
    Pros: three more live rules.
    Cons: execution counts churn in every diff, and they record what one kernel once did, not that a fresh kernel runs the notebook: an edited cell that was never re-run keeps its old count.
    Every edit would need a full re-run on the source file before committing, and an editor that writes no counts trips the rules.
  - **Option C: scan the repository root and run the repository rules.**
    Pros: the literal reading of "notebook and repository QA".
    Cons: `repository-not-versioned` looks for a `.git` directory and fires in every git worktree, which this project uses for parallel work; locally the scan walks `.venv` and other worktrees and lints other branches' stale notebook copies; `test-coverage-data-not-available` depends on whether the tests ran in that checkout.
    Each repository rule is about a property something else here already enforces more strongly, which the table says rule by rule.
  - **Option D: scan the `notebooks/` directory.**
    Pros: one command, no file list.
    Cons: three structural false positives to exclude, untracked scratch files get linted, and a notebook committed anywhere else escapes the scan.
  - **Option E: the policy in a commented `.pynblint` file.**
    Pros: Pynblint's own configuration format.
    Cons: it is read only from the current directory and is overridden by environment variables, its reasons are comments nothing checks, and a misspelt rule name is accepted silently, so an exclusion with a typo excludes nothing and says nothing.
  - **Option F: raise `cell-too-long` or `notebook-too-long` so the first notebook passes.**
    Pros: no restructuring.
    Cons: it is the threshold adjusted to its result that EDN-62 rejects; the reason would be the notebook, not the rule.
  - **Option G: report the excluded rules as non-blocking warnings.**
    Pros: nothing is hidden.
    Cons: Pynblint has no severities, and a warning tier is exactly the "findings nobody acts on" the issue warns about.
  - **Option H: execute the notebook in CI, with the source file cached under its MD5.**
    Pros: the execution evidence that options A and B argue about would be automated, and because the summary cell fails on a mismatch, every pull request would check the dataset card against the data.
    Cons: a cold cache downloads 548 MB from Zenodo, so a third party's server would decide whether our CI is green, which is a flaky test by construction; the job needs the full project environment, about a minute and 2.8 GB of RAM; and GitHub evicts a cache unused for seven days, so the download recurs.
    It stays open: if the raw data ever comes from our own DVC remote instead, most of the cons go away.
- **Rationale:** A rule is enforced where it can fire on what we commit, and excluded where it cannot, with the reason written next to it.
  That principle is what keeps the gate honest in both directions: nothing it reports is noise, and nothing it lists as checked is unchecked.
  It also bit on the first run: the dataset card's profiling notebook, as first drafted, had five code cells over 30 lines, and they were restructured into shorter cells with precomputed values rather than the threshold being raised, so the notebook stayed within the 50-cell limit as well.
  On the stripped copy that is committed, the only rule that fired was `non-executed-notebook`, which is the measurement behind excluding the execution rules.
  Making the documentation table the configuration follows the pattern of the requirement matrix, which reads the requirements and the specification rather than a copy of them: the reasons cannot drift from what the gate does, and a rule a future Pynblint adds or renames stops the build until someone decides about it.
  The table names every rule, enforced or excluded, and the gate refuses one that does not classify exactly the installed rules, enforces nothing, or enforces a repository rule that would never run, so a misspelt slug, a rule a new release adds or an empty policy stops the gate rather than silently changing what is checked.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI read the Pynblint 0.1.6 source and tested every rule on purpose-built notebooks, stripped and unstripped, which is how the interaction with nbstripout was found rule by rule rather than assumed; it also found the blind spots the table now states as caveats.
  It proposed the policy, the docs-as-configuration design and the location rule, and laid out options A to G; the owner chose A for the execution rules and the docs table as the policy.
  The notebook itself was drafted by AI, and its first Pynblint run found five code cells over 30 lines; they were restructured rather than the threshold raised, so the policy applied to AI-written code as to any other.
  An AI review of the documents then found two holes in the first version of this policy, both closed before the pull request was opened: nothing stopped a `keep_output` flag from carrying outputs past nbstripout, and `make notebook-run`, the stand-in for the excluded execution rules, passed even when a dataset card figure did not reproduce.
- **AI interaction evidence:** Claude Code session on 2026-10-01 implementing issue #41: a rule-by-rule experiment on stripped and unstripped sample notebooks, an architecture pass that produced the policy table and options A to G, the owner's choice among them, and a document review that added option H and the two fixes above.
- **Other evidence:** [Notebooks](../docs/docs/notebooks.md), the rule table; [`notebooks/1.0-lh-dataset-card-profiling.ipynb`](../notebooks/1.0-lh-dataset-card-profiling.ipynb); `tests/test_notebook_lint.py`; [specification](../docs/docs/specification.md) NFR-07; [EDN-64](#edn-64-pynblint-runs-from-its-own-locked-environment-and-a-wrapper-that-reads-its-json-report-is-the-gate); [issue #41](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/41), [PR #67](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/67).
- **In LaTeX:** yes

### EDN-66: NFR-06 and NFR-09 are each split in two, and each ID keeps the half its citations mean

- **Date:** 2026-10-05
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Requirements Engineering, Testing Strategy
- **Participants:** @lukas2510, decided by the repository owner
- **Decision:** NFR-06 and NFR-09 each promised two things that come due in different milestones and are verified by different evidence, so each is split in two, and every half has its own ID, acceptance criterion, verification route and milestone.
  The existing ID is narrowed to the half its citations mean, and the other half gets the next free ID.
  **NFR-06** keeps reproducibility: from a clean clone, `dvc pull` and `dvc repro` reproduce the splits and metrics within ±0.1 percentage points; **[manual]**, the clean-clone drill of [The DVC pipeline](../docs/docs/pipeline.md), due M3.
  **NFR-14** takes run provenance: every training run records its commit, its data version and its parameters; **[automated]**, by the tracking tests and a `train` test that reads a real run back, due M3.
  **NFR-09** keeps the running system's security: a read-only public surface, the body-size limit, no TLS, the API key from the environment, non-root containers and the vulnerability scan; **[automated]**, due M4.
  **NFR-15** takes credential hygiene: no secret in any commit of the history, a scan before every merge, and a leaked secret revoked rather than only removed; **[manual]**, the CI secret scan of EDN-71, due M3.
- **Alternatives considered:**
  - **Option A (chosen): narrow the existing ID to one half, and give the other half a new ID.**
    Pros: on `main` before the split, 66 lines cited NFR-06, 15 of them `req` markers, and 18 cited NFR-09.
    Apart from the requirement rows themselves and the passages that tell NFR-06's history, eleven of NFR-06's citations mean provenance - the tracking seam and its tests, two documentation pages and EDN-54 - and of NFR-09's only EDN-46's evidence pointer means secret hygiene.
    Every other citation, the report's included, means the half the ID keeps and stays correct without an edit, and the twelve tests marked NFR-06 for determinism and the download pin keep their marker.
    No ID is reused for a different requirement: a narrowed ID promises a subset of what it promised before and nothing new.
    Cons: a reader holding an older text can still read NFR-06 as including provenance, so the citations of the moved halves had to be found and repointed.
    Earlier entries written under the combined meaning, EDN-45, EDN-55 and EDN-56, are records of their date and are not rewritten; EDN-56 in fact already argued the two halves of NFR-06 separately, which is the strongest sign they were two requirements.
  - **Option B: retire NFR-06 and NFR-09 and issue four new IDs.**
    Pros: no ID ever changes meaning, not even by narrowing.
    Cons: every one of those 84 citations becomes a pointer to a requirement that no longer exists, including three sections of the LaTeX report; the fifteen NFR-06 markers fail the per-test check until each is rewritten; and the matrix needs every requirement to have a table row, so a retired ID could only survive in prose, where nothing checks it.
  - **Option C: suffixes, NFR-06a and NFR-06b.**
    Pros: the family stays visible in the name.
    Cons: `tools/requirement_matrix.py` accepts `(?:FR|NFR)-\d{2,}` and nothing else, so the tool would have to change first; and a bare "NFR-06" in an existing citation would become ambiguous between the parent and its halves, which is the ambiguity the split exists to remove.
  - **Option D: keep one ID each, and let the "Verified by" cell name both routes.**
    Pros: no ID change at all.
    Cons: it is the state issue #60 describes.
    The parser requires exactly one of **[automated]** and **[manual]**, and the gate books an ID to one milestone, so NFR-09's hygiene half sat in M4 behind an API that does not exist yet while the token it protects was already on every contributor's disk, and NFR-06's tests could only ever make it "named by a test".
- **Rationale:** The constraint is that a requirement ID is never reused for a different requirement, because the report and the EDN cite them; between A and B the question is only which citations break.
  In A it is a bounded, listed set, the ones that mean the half that moved; in B it is all of them.
  The halves get their milestones from when their evidence can exist, not from the requirement they came from: NFR-14 is due in M3 because the tracking seam (PR #50) and the `train` stage's parameter logging (PR #62) are both merged, and NFR-15 is due in M3 because the risk has existed since issue #42 and PR #50 put a DagsHub token into every contributor's `.env` and `.dvc/config.local` (EDN-46).
  NFR-09 stays booked to M4 as decided with the split, although its image scan, its Dockerfile review and the check from outside the VM are M5 work; `tools/expected_coverage.yaml` says so, so the delivery review notices if they are still missing then.
  NFR-14 is **[automated]** because what a run records can be read back from a run; NFR-15 is **[manual]** because its evidence is a CI job and not a Pytest test, the route NFR-07 already uses for the CI run it names, and making it **[automated]** would have needed a test that only stands in for the scan.
  Giving NFR-14 a route of its own also showed that it was not met: `train.num_threads` is declared as a parameter of the `train` stage in `dvc.yaml` and changes the artefact (EDN-53), yet the run carried it only inside the `model.json` artefact, not as an MLflow parameter, so a comparison of runs by parameter could not see it, and the only test spot-checked two parameters.
  So a test now reads the keys `dvc.yaml` declares for each `train@<variant>` out of `dvc.yaml` itself and compares them with the parameters of a real run; it failed on the code before the change, on `num_threads` for all four variants, and passes since the stage logs it.
  `train.mlflow_experiment` is the one declared key recorded as what it is, the experiment the run belongs to, and the test checks it there rather than as a repeated parameter.
  `train.py` and `recommenditos/tracking.py`, whose three comments also move from NFR-06 to NFR-14, are dependencies of the four `train` stages and of `evaluate`, so the change reruns them on the real snapshot.
  The re-run on 2026-10-05 moved no number: `metrics.json`, the segment table and the masking sweep are unchanged, and inside each model bundle only `model.json` changed, by the provenance EDN-56 puts there, while every payload file kept its hash; the four new runs carry all of their stage's declared parameters, `num_threads` among them.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** The owner decided to split both requirements, which half comes due when, and that the hygiene half is verified by a secret scan; the issue and its comment are his.
  How to split the IDs he delegated, under the constraint that no ID is reused.
  AI counted every citation of both IDs and sorted them by the half each one means, which is the evidence that decides between A and B, read the matrix's ID pattern to rule out C, and chose A.
  It also found the unlogged `num_threads` while writing NFR-14's route, and its first recommendation was to leave that and the `tracking.py` comments to a later change that retrains anyway.
  An independent AI review of the pull request rejected that, because it left NFR-14 reported as verified by a test while its parameter clause was unmet, and the fix was made in this pull request, test first, which is the modification.
- **AI interaction evidence:** Claude Code session on 2026-10-05 implementing issue #60: `grep -rn "NFR-0[69]"` over the documents, the report, the tests, the tools and the package, each hit classified by the half it means, and `git grep` counts on `main` for the figures above; the requirement matrix rebuilt after the split, with NFR-14 verified by five tests and NFR-15 verified by hand; `test_a_run_records_every_parameter_its_stage_declares` run against the unchanged `train.py` first, failing for all four variants on `num_threads` alone.
- **Other evidence:** [issue #60](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/60) and its comment on NFR-06; [PR #70](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/70); [requirements](../docs/docs/requirements.md) and [specification](../docs/docs/specification.md) NFR-06, NFR-09, NFR-14 and NFR-15; `tools/expected_coverage.yaml`; `DUE_AT_M3` in `tests/test_requirement_matrix.py`; [EDN-27](#edn-27-separate-the-requirements-from-their-specification-keeping-one-set-of-frnfr-ids) for the one ID space, [EDN-44](#edn-44-the-traceability-gate-is-milestone-scoped-and-its-expected-coverage-set-is-a-validated-yaml-file), [EDN-45](#edn-45-a-req-marker-is-verification-only-where-the-specification-says-a-test-is-the-evidence), [EDN-46](#edn-46-dagshub-credentials-live-in-two-gitignored-stores-and-the-token-is-entered-twice), [EDN-56](#edn-56-the-model-bundle-carries-its-own-provenance-so-models-is-deliberately-not-byte-reproducible), [EDN-71](#edn-71-secrets-are-scanned-by-gitleaks-before-every-commit-and-over-the-whole-history-in-ci-with-a-rule-of-our-own-for-the-dagshub-token).
- **In LaTeX:** yes

### EDN-67: `features` applies the supported-make list to every frame, the `ES` holdout included, before the vocabulary is built

- **Date:** 2026-10-05
- **Milestone:** M3: Quality Assurance (feature engineering; the holdout feeds M6: Monitoring)
- **Activity / Topic:** Feature Encoding, Evaluation Protocol, Monitoring
- **Participants:** @lukas2510, decided by the repository owner
- **Decision:** The `features` stage restricts all five frames it reads - train, validation, calibration, test and the `ES` holdout - to the makes `data/processed/supported_makes.json` names, and does so before any training-derived vocabulary or statistic is computed.
  That discharges the first and second of the four guarantees EDN-48 left to the downstream stages, in one place.
  The holdout matrix therefore holds 5,979 of the 6,079 `ES` rows: the 100 it loses are exactly the listings the API answers with a 422 under FR-04, so the matrix is the holdout as the model receives it.
  It is not what M6 replays: a request cannot be rebuilt from a matrix, because the holdout's `country_code` is empty in every row (EDN-42) and `registration_date` has become `age_years`, both fields FR-01 requires.
  The replay reads `data/processed/holdout_es.parquet`, which `split` still writes with all 6,079 rows, and has to filter to the supported makes itself or count the 100 refused requests as FR-04's 422s.
  As a consequence `train` and `evaluate` no longer filter at all; they check the property with `check_served_makes_only` and fail when it does not hold.
- **Alternatives considered:**
  - **Option A (chosen): filter every frame in `features`, the holdout included, before the vocabulary.**
    Pros: the rule gets one owner, at the only point where it still decides the feature space, so the `make` levels (11 rather than 25), the `model` levels (437 rather than 561), the equipment threshold and the `model_version` floor are all shares of the population the product serves.
    Obligations 1 and 2 close together, and the UC2 conformal calibration, which has no ticket yet, inherits a served population instead of having to remember the filter.
    Any model-side use of the holdout matrix - scoring it offline, or comparing encoded inputs for drift - is on the 5,979 rows the API would pass to the model, the population EDN-14 measured NFR-11 on.
    Cons: the holdout matrix no longer holds every `ES` listing, so a model-side analysis that wants the refused rows has to start from `data/processed/holdout_es.parquet`, which `split` still writes losslessly.
    The vocabulary moves, so every model and every number in the model card had to be re-measured.
  - **Option B: restrict only the training rows the vocabulary is fitted on, and leave the holdout matrix unfiltered for the M6 ticket to decide.**
    Pros: the smallest change to the holdout matrix, which would keep every `ES` row.
    Cons: it leaves obligation 1 open for calibration and the holdout, so the guarantee stays spread over stages and tickets that do not exist yet, which is the cost EDN-48 accepted and this ticket exists to remove.
    And the raw holdout matrix would not even be raw: once the vocabulary is fitted on served rows, the 100 unserved rows are encoded with no `make` level at all, so the matrix would hold rows the model never receives, in an encoding that cannot say what they were.
    Nor would it keep anything for M6: the replay cannot be built from a matrix under either option, and the split frame keeps the refused rows under both.
- **Rationale:** The restriction belongs where the feature space is decided, and the holdout matrix belongs inside it rather than beside it.
  The matrix is how the model sees the holdout, and the API refuses an unsupported make with a 422 before the model is called (FR-04), so what the model can ever receive from the `ES` listings is the 5,979 rows, not the 6,079 - the population EDN-14 measured NFR-11 on.
  What M6 replays is a separate question this decision does not settle.
  The replay is built from the split frame, because a matrix cannot be turned back into requests, and it must either apply the same list or count FR-04's 422s for the 100 refused rows; either way the drift and error figures it reports are about the 5,979 the model answers.
  That is also why option B's apparent advantage, keeping refused requests replayable, is not one: they stay in `data/processed/holdout_es.parquet` under either option, so the choice made here stays reversible.

  **Interaction with EDN-42, checked on the real run.** The filter acts on rows by `make` and touches no other encoding, so the holdout's `country_code` is still missing in every one of its 5,979 rows, because `ES` is still not a training level, and the stage's warning about a column a split never fills still names `country_code` and nothing else.
  What changes is the rest of the holdout matrix: before, its 100 unserved rows were encoded with make levels the model was never meant to serve; now every holdout row is one of the 11 training makes and none has a missing make, so `country_code` is the only feature of the holdout matrix the model sees as unknown, which is the new-market signal EDN-42 chose to keep.

  **What it moved, measured on the real snapshot (2026-10-05, `reports/analysis/served_vocabulary.py` and the committed `dvc.lock`).** The `make` levels go from 25 to 11, the `model` levels from 561 to 437, and the `model_version` levels from 279 covering 85.9 % of the training rows to 278 covering 86.5 % of the served ones; the equipment vocabulary is the same 133 of 146 items either way, so neither matrix gains or loses a column.
  On the ladder, three of the four variants are unchanged to full precision: the `b0` and `b1` payloads are byte-identical, and `lgbm-basic` builds the same trees, with only the category codes inside its split bitsets renumbered.
  `lgbm-extended` is the one that moved, through the one `model_version` level it lost: MdAPE 6.318 % to 6.325 %, within 20 % 90.40 % to 90.29 %, SC-04's worst segment 17.78 % to 17.86 %, SC-06 1.584x to 1.583x, and its best iteration is now the 1,000th rather than the 996th, so early stopping fires for neither LightGBM variant inside the 1,000-tree budget.
  No criterion changed its verdict for any variant, and EDN-62's choice of candidate holds: `lgbm-extended` still fails SC-06, and its pooled advantage shrinks from 0.51 pp to 0.50 pp.

  **Consequences for the other stages.** `train` and `evaluate` used to filter their own rows; the filter is now a check, `check_served_makes_only`, which refuses a matrix holding a make outside the list, a row with no make level, or a `make` level the list does not name.
  A second filter would never remove a row from matrices `features` wrote, and on any other matrices it would quietly repair the rows while keeping a feature space decided over another population, so the rule keeps one owner and its consumers check a property rather than an intention.
  This changes behaviour only for inconsistent inputs - matrices from another run, or a make list rewritten without rebuilding them - which used to be filtered silently and now fail with a message naming the state.
  One state is not an inconsistency and gets a failure of its own in `features`: a supported make that no training row holds, which `split` can produce because it counts support over all four sets, so after a change to `seed` or to the threshold every seller of a make can land outside train.
  The API would accept that make while the model had never seen it, and only `features` can say why, so it refuses the state by name rather than leaving `train` to report matrices from another run.
  For the same reason the guard against an empty population moved into `features` and covers every frame: an empty matrix cannot even be read back, because Parquet keeps no levels for an empty categorical, so the downstream guards it replaces could no longer be reached.
- **AI involvement:** Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI generated options A and B with the holdout counts and the EDN-42 interaction, and recommended A; Lukas chose A before the implementation started.
  Implementing it test first turned up two things the options had not anticipated.
  Once the vocabulary is fitted on served rows, an unfiltered holdout would encode its unserved rows with no make level at all, which takes away most of what option B was meant to preserve; and an empty matrix cannot be read back through Parquet, which is why the empty-population guard moved from `train` and `evaluate` into `features`.
  AI measured the accuracy effect rather than assuming it, and established why three variants did not move rather than reporting the unchanged numbers as a coincidence.
  An independent review of the pull request, also by AI, then corrected two points without changing the decision.
  The first version of this entry argued that the filtered matrix is what M6 replays, which it cannot be, since a request cannot be rebuilt from a matrix; the replay reads the split frame, and the reasoning above says so.
  And a supported make with no training row was only warned about in `features`, so the failure surfaced in `train` with advice that looped; `features` now refuses it by name.
- **AI interaction evidence:** Claude Code session on 2026-10-05 implementing issue #63: the two options were laid out with the 5,979-of-6,079 holdout count and the EDN-42 interaction, and Lukas chose A.
  The independent review of PR #71 on the same day found the replay argument and the untrained-make diagnosis, and reproduced the second with a training split holding only BMW against the list BMW and Audi.
  The measurement behind the numbers is `reports/analysis/served_vocabulary_results.txt` and the ladder re-run the committed `dvc.lock` records, tracked as one MLflow run per variant.
- **Other evidence:** [issue #63](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/63); [PR #71](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/71); [`recommenditos/data/build_features.py`](../recommenditos/data/build_features.py) (`served_rows`, `check_served_makes_only`); `tests/test_features.py::test_every_frame_the_stage_writes_holds_only_supported_makes` and `::test_the_vocabulary_is_decided_by_the_served_training_rows_alone`; [`reports/analysis/served_vocabulary.py`](analysis/served_vocabulary.py) and its results file (2026-10-05); [pipeline docs](../docs/docs/pipeline.md), "The supported makes"; [model card](../docs/docs/model-card.md), "Feature space" and the ladder; [EDN-14](#edn-14-nfr-11s-drift-control-is-an-iid-sample-not-a-seller-grouped-one), [EDN-42](#edn-42-a-categorical-value-the-training-rows-never-saw-becomes-missing-and-the-holdouts-country-is-allowed-to-vanish), [EDN-48](#edn-48-split-records-the-supported-make-list-and-stays-a-lossless-partition-the-downstream-stages-apply-it).
- **In LaTeX:** yes

### EDN-69: CodeCarbon measures the fit alone, offline and per process, and its figure is reported as the estimate it is

- **Date:** 2026-10-05
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Energy Efficiency, Experiment Tracking, Evaluation Protocol
- **Participants:** @lukas2510, decided by the repository owner
- **Decision:** `train` measures exactly the `fit_variant` call with CodeCarbon 3.3.1's `OfflineEmissionsTracker`, in `process` mode, against the Spanish grid (`ESP`, 174.05 g CO2eq/kWh) and with `measure_power_secs: 15`, all three from `params.yaml`'s `train.energy`, which the `train` stages declare. CodeCarbon's row goes to `reports/emissions/<variant>.csv` and into the MLflow run that also receives the variant's test metrics, as metrics, with the CPU power method and the machine behind them as parameters. A `compare-energy` stage joins each model, its row and its metrics into `reports/figures/energy-vs-error.{csv,png}`. On every machine we train on the figure is CodeCarbon's estimate, not a reading: 10 W for the RAM plus the CPU's TDP times the share of its logical CPUs the fit kept busy. We keep that estimate in the pipeline, word every comparison as one of fit times expressed in energy, within one run on one machine, and checked the estimate once against the RAPL energy counter, with read access granted for that check only.
- **Alternatives considered:**
  - **How the figure is obtained, given that it is an estimate.**
    - **Option A: keep CodeCarbon's estimate and word it as what it is.**
      Pros: nothing changes on anyone's machine; the same mechanism and settings everywhere, hermetic, with no network call; it is the prescribed tool used as prescribed.
      Cons: the energy axis adds no information beyond the fit time it is computed from, and the absolute watt-hours are model output that depends on the machine and on its load.
    - **Option B: grant read access to RAPL on Lukas's laptop permanently,** with a udev or tmpfiles rule, so that CodeCarbon reads the package counter there.
      Pros: a hardware counter rather than a model.
      Cons: RAPL is root-only on purpose since kernel 5.10, because unprivileged power readings are a side channel that has been used to extract keys (Platypus, CVE-2020-8694), and this laptop has SGX. RAPL counts the whole package and CodeCarbon does not apportion it to the process, so on a laptop several agents share - the 1-minute load ranged from 10 to 31 on the day of the measurement - it would charge each fit for the others. Only Lukas's runs would read it, so a teammate's `dvc repro` would write a figure of another kind into the same files. And it is a standing change to a personal machine that somebody has to remember.
    - **Option C (chosen): A, plus a one-off check of the estimate against the counter.** Read access is granted with a `chmod` that does not survive a reboot, the four fits and an idle baseline are measured with both, and access is revoked again.
      Pros: the report can say what the estimate is worth instead of only what it is, at no standing cost to the laptop's security, and the pipeline stays as in A.
      Cons: it needs Lukas's `sudo` and an otherwise idle laptop, and it validates one machine on one day, not every run.
    - **Option D: compare the ladder on fit seconds and CPU-seconds, and keep CodeCarbon's figure only as NFR-10's log.**
      Pros: both are direct measurements.
      Cons: it drops the energy comparison issue #38 and the rubric ask for, and the comparison would show the same ratios under another name.
  - **`process` against `machine` mode.** `machine` charges the fit for the load of the whole machine: on the shared laptop, a process that only slept was charged 4.5 W of CPU in `machine` mode and 0.3 W in `process` mode. `process` reads the fit's own CPU time.
  - **The online tracker against the offline one with a pinned country.** Per CodeCarbon's source, the online tracker geolocates the machine through a call to geojs, writes its latitude and longitude into the CSV, and queries the AWS, Azure and GCP metadata endpoint, switching to the provider's regional factor when one answers, which on a GitHub runner, an Azure VM, one would. The emissions of an identical fit would then depend on where it ran, a git-tracked file would carry a contributor's location, and the test suite would make network calls. With every socket connection made to fail, the offline tracker made none.
  - **The demo's `measure_power_secs=1` against CodeCarbon's default of 15.** The power is sampled every second by a separate thread either way; this setting is how often the samples are integrated, and at 1 s that integration races the tracker's own shutdown. Against CodeCarbon's own model, 1 s gave 0.86x to 1.07x on a single-threaded LightGBM fit and up to 4.31x on a busy loop's RAM half, an interval counted several times over; 15 s gave 0.987x to 0.993x on the same fit.
- **Rationale:** Issue #38 was written on the assumption that CodeCarbon measures the energy of a fit, and the first thing done was to test that on the laptop the ladder is trained on, before any number was quoted. It does not. Since kernel 5.10 RAPL is root-only, so CodeCarbon falls back to its `cpu_load` model, with the i5-10210U's TDP of 15 W from its own table and a constant 10 W for the RAM, its floor of two DIMMs for any x86 machine. The energy of a fit is then 10 W times its wall time plus 15 W / 8 logical CPUs times its CPU time, and for a single-threaded fit that is its duration times 11.75 to 11.80 W, to within 1 % over repeated fits. The real fits are charged more, 13 to 18 W for every second in the ladder run of 2026-10-06, because they use more than one thread although `train.num_threads` is 1: the LightGBM variants' CPU time is 1.6 to 2.2 times their wall time. The RAM constant is 68 % of the `lgbm-basic` fit's figure and 77 % of the `lgbm-extended` fit's. On a GitHub runner, an AMD EPYC 7763 with four vCPUs and no powercap interface at all, the same model charges 80 W per second, because it divides the chip's 280 W TDP by the four vCPUs the VM sees. So the figure is a fit time multiplied by a machine-specific constant, and it moves with the machine's load as well as with the work: the same `lgbm-basic` fit took 33.0 s on a laptop shared with other jobs and 8.6 s on a quiet one, and was charged accordingly.

  That makes the comparison the issue asked for weaker than it reads, not wrong. Between two variants fitted in one run on one machine the ratio of their energies follows the ratio of their fit times closely (3.2 against 3.6 for `lgbm-extended` over `lgbm-basic`), which is a fair measure of their relative cost. What it cannot be is a statement in watt-hours that holds across machines, or evidence that a variant uses energy differently from how it uses time. So the pipeline keeps the estimate (option A), every place that shows the figure says what it is - the CSV, the `codecarbon.cpu_power_method` parameter of every run, the subtitle of the figure, the model card and the report - and the one check that can say how far the estimate is from the hardware was made once (option C) rather than built into every run (option B). NFR-10's specification says the same since this decision, because "CodeCarbon measures" was not true of any machine we have.

  The RAPL check, `reports/analysis/rapl_validation.py`, measured an idle baseline and three fits of each variant on an idle laptop with the package, core, uncore, DRAM and platform counters, CodeCarbon's forced `cpu_load` estimate and CodeCarbon's own RAPL figure, all around the same fit. On 2026-10-06, before hyperparameter tuning, the idle package draw was 6.6 W. The estimate's total is about 70 % of the counter's package energy for the three fitted variants and 112 % for `b0`, whose whole fit is about 1 J, and its CPU half is 27 % to 41 % of the energy above idle. The ranking and the ratios hold: `lgbm-extended` costs 2.97 times `lgbm-basic` by the counter and 2.97 times by the estimate. One variant's package energy varies by a factor of 1.3 to 2.1 between repeats, and in one fit other processes used 28.8 CPU-seconds despite the idle check, so only medians are quoted. The estimate is therefore good for ranking variants on one machine in one run, and about 30 % low in absolute terms.

  The rest of the design follows from treating the figure as evidence that has to be checkable. Only the fit is inside the tracker, as the demo does, which `tests/test_model.py::test_only_the_fit_is_measured` asserts by making the read and the write slow. Every CodeCarbon argument is passed explicitly, because it reads `~/.codecarbon.config`, `./.codecarbon.config` and any `CODECARBON_*` variable for every argument it is not given, so a contributor's configuration could switch on its API output or force a CPU power. CodeCarbon swallows every exception in its constructor, `start` and `stop`, so the module reads its row back by CodeCarbon's run id and by column name and fails the stage when it is missing, and refuses a country CodeCarbon would silently measure against the world average. The row is joined to its model by that run id, which `model.json` records, and `fit_cpu_seconds` is logged beside the energy because it is a measurement and the CPU half is proportional to it. DVC deletes a stage's outputs before running it, so the CSV CodeCarbon appends to holds exactly one row after `dvc repro` (measured: one row after three `dvc repro --force`, three with `persist: true`).

  What the comparison concluded, on the ladder of 2026-10-06 (pull request #68, before hyperparameter tuning, so the numbers are replaced when the ladder is re-run): `b0` 0.09 s, 0.0003 Wh, 0.06 mg CO2eq, MdAPE 12.2 %; `b1` 1.34 s, 0.0066 Wh, 1.15 mg, 9.5 %; `lgbm-basic` 8.42 s, 0.0342 Wh, 5.96 mg, 6.8 %; `lgbm-extended` 30.02 s, 0.1082 Wh, 18.82 mg, 6.3 %. `lgbm-extended` costs 3.2 times the energy of `lgbm-basic` for 0.55 percentage points of MdAPE, 0.07 Wh more in absolute terms, so the extra energy is not what decides between them. What does is that, given only the fields FR-01 requires, it is worse (10.3 % against 8.3 %): it earns its place only when the user supplies the extra inputs. The whole ladder is 0.149 Wh and 40 s of fitting, against NFR-10's 15-minute budget.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI found the limitation the decision is about, by reading CodeCarbon's source and then measuring it on the laptop and on a GitHub runner, before writing the module; the issue, the course demo and the report draft all assumed a measurement. It laid out the four options with the security and the shared-machine arguments against B, recommended A with C, and Lukas chose that. It also chose the tracker's settings on measurements rather than on the demo: `process` mode, the offline tracker, and 15 s rather than the demo's 1 s once it was shown to race the tracker's shutdown. It did not grant itself RAPL access by other means, although Lukas's `docker` group would have allowed it, because that would have bypassed the permission boundary this decision is partly about.
- **AI interaction evidence:** Claude Code session on 2026-10-05 implementing issue #38: probes of CodeCarbon 3.3.1 with every socket blocked, on four workloads in both modes and at both integration intervals, a DVC repository testing the append, and a temporary CI step on pull request #68 that printed what CodeCarbon does on a GitHub runner; then the four options above with a recommendation, which Lukas decided as option C.
- **Other evidence:** [`recommenditos/modeling/energy.py`](../recommenditos/modeling/energy.py), whose docstring states what the figure is, and [`recommenditos/modeling/compare_energy.py`](../recommenditos/modeling/compare_energy.py); `train.energy` in [`params.yaml`](../params.yaml); [`tests/test_energy.py`](../tests/test_energy.py) and `tests/test_model.py::test_every_training_run_records_its_emissions_beside_its_accuracy` and `::test_only_the_fit_is_measured`; [`reports/analysis/codecarbon_validity.py`](analysis/codecarbon_validity.py) with its results, [`codecarbon_github_runner.txt`](analysis/codecarbon_github_runner.txt) and [`rapl_validation.py`](analysis/rapl_validation.py) with its results; `reports/figures/energy-vs-error.png`; [specification](../docs/docs/specification.md) NFR-10; [model card](../docs/docs/model-card.md), Environmental Impact; [issue #38](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/38), [pull request #68](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/68).
- **In LaTeX:** yes

### EDN-70: `n_estimators` is a ceiling of 5,000 that early stopping stays under, not a tree count

- **Date:** 2026-10-05
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Modelling Approach, Hyperparameters, Energy Efficiency
- **Participants:** @lukas2510, decided by the repository owner
- **Decision:** Both LightGBM variants get `n_estimators: 5000`, with `early_stopping_rounds: 50` unchanged, so early stopping on the validation split decides how many trees each model keeps: round 1,243 for `lgbm-basic` and 1,647 for `lgbm-extended` on the real snapshot.
  At 1,000 the ceiling had ended both fits before early stopping could.
  `train` now records in `model.json` and in the MLflow run whether early stopping or the ceiling ended each fit (`early_stopped`, beside `boosting_rounds`, the rounds the fit ran), and logs a warning when it was the ceiling.
  `learning_rate` and `num_leaves` are not changed by this entry.
- **Alternatives considered:**
  - **Option A: keep 1,000.**
    Pros: no retrain, and the report's numbers stay where they are.
    Cons: the parameter, not the validation curve, decides the tree count of both variants, so every later comparison - of the feature sets, of `learning_rate`, of `num_leaves` - is judged between models cut off by a budget.
    And it hides itself: `lgbm-extended` once recorded a best iteration of 996 out of 1,000, which reads like early stopping and was not.
  - **Option B: 2,000, just above where early stopping lands.**
    Pros: the smallest worst case of every cost that grows with the trees.
    Cons: `lgbm-extended` ran to round 1,697, so 2,000 leaves 303 rounds of slack.
    A modest change to the data - a new snapshot, or the Great Expectations work on the stages before `split` - or a lower learning rate would bind it again, and only the new warning would say so.
  - **Option C (chosen): 5,000.**
    Pros: three times the later best round and four times the earlier, so it does not bind under any change measured so far: in a 12-point grid of `learning_rate` and `num_leaves` around the committed values, the slowest point (`learning_rate` 0.025, `num_leaves` 31) ran 4,116 rounds.
    And its worst case stays inside every budget the trees drive: a fit that used all 5,000 trees would take about 80 s for `lgbm-extended` against NFR-10's 15 minutes, write a booster of about 33 MB against NFR-04's 1 GB image, and spend about 105 ms on one row's native SHAP export against NFR-02's 200 ms for the whole of `/predict`.
    Cons: it is a value with a stated rationale, not a measured optimum, and the latency half of that rationale is a laptop measurement while NFR-02 is judged on the course VM, so its margin is less certain than the fit-time one.
  - **Option D: effectively unbounded, 20,000 or more, as the exploratory fits used.**
    Pros: it could never bind in any plausible configuration.
    Cons: the ceiling would stop being a guard.
    A configuration whose validation curve keeps creeping down - a lower learning rate, more data - would run to a model whose SHAP export alone exceeds NFR-02 (about 210 ms at 10,000 trees on the same laptop), with a fit time, an energy figure and an artefact size nobody chose.
  - **Option E: fix the tree count at the round early stopping chose (`n_estimators: 1243`, early stopping off).**
    Pros: the same model today, and a fit that no longer reads the validation split.
    Cons: it bakes one snapshot's answer into a parameter.
    The next retrain, which EDN-12 makes a response to drift, would get it wrong in either direction, where early stopping decides again on every retrain.
- **Rationale:** The value was found by measurement rather than guessed, in two steps that never open the test split.
  `reports/analysis/early_stopping_ceiling.py` fitted each variant once at a ceiling of 20,000 with every other setting as committed, through the stage's own `fit_variant`, on the `train` and `validation` matrices only: early stopping ended both fits, at best rounds 1,243 and 1,647 after 1,293 and 1,697 rounds.
  The ceiling was then chosen from those numbers and the per-tree costs the same script measured, and only the committed `dvc repro` read the test split, through `evaluate`.

  **What the extra trees are worth, which is less than issue #64 expected.** The issue read the 1,000-tree numbers as a lower bound on what the hyperparameters could reach.
  They were very nearly the bound itself.
  On validation, from 1,000 trees to the round early stopping chose, L1 in log space falls by 0.2 % and 0.6 % while MdAPE stays flat (6.947 % to 6.956 %, 6.448 % to 6.453 %): L1 is a mean and MdAPE a median, so the last trees improve the tail of the errors and not the typical one.
  On the test split the committed run moves MdAPE from 6.827 % to 6.808 % for `lgbm-basic` and from 6.325 % to 6.256 % for `lgbm-extended`, within 20 % from 89.68 % to 89.81 % and from 90.29 % to 90.40 %, SC-03 from 43.9 % to 44.0 % and from 48.0 % to 48.6 %.
  SC-04's worst segment, cars over 20 years, moves from 17.12 % to 16.77 % for `lgbm-basic` and from 17.86 % to 18.45 % for `lgbm-extended`; SC-06 from 1.208x to 1.213x and from 1.583x to 1.646x.
  No criterion changes its verdict for any variant, `b0` and `b1` are unchanged, and EDN-62's choice of candidate holds more firmly than before: `lgbm-extended`'s pooled advantage grows from 0.50 pp to 0.55 pp, and its disadvantage on a request carrying only the required fields from 1.77 pp to 2.04 pp (10.30 % against 8.26 %).
  The training L1 falls much further than the validation L1 - 0.0688 to 0.0655 against 0.1005 to 0.1003 for the candidate - so the added trees mostly fit the training rows more closely, and on `lgbm-extended` that comes with a stronger dependence on its optional fields.

  **What they cost, which is linear in the trees.** 1.29x and 1.70x the boosting rounds, so the same factor of fit time and of the energy the fit draws; the boosters grow from 6.2 to 7.6 MB and from 6.8 to 10.9 MB; one row's native SHAP export, the FR-08 path, from 18.9 to 26.1 ms and from 22.3 to 35.4 ms on the laptop that measured it.
  The committed run fits the candidate in 8.3 s and the whole ladder in 37.6 s, far inside NFR-10's 15 minutes; fits of the identical trees on 2026-10-05 and 2026-10-06 ranged from 8.0 s to 10.0 s and from 26.4 s to 32.9 s with machine load, so the rounds, not the seconds, are the measurement.
  Bought at that price, 0.02 pp of MdAPE on the candidate is not much; it is accepted because the alternative is a model whose size is set by an arbitrary parameter rather than by its data, which is what makes every other comparison in the ladder trustworthy.

  **The issue's "Care needed" claims, checked in the code.** Early stopping reads the validation split and nothing else: `LightGBMModel.fit` passes `validation` as the only evaluation set, under `first_metric_only`, and `train.read_matrices` reads only `train` and `validation`; the test split is read by `evaluate` alone, after training.
  One claim needs a qualification rather than a correction: `early_stopping_rounds` bounds how many rounds the fit runs past its best one, not the optimism of choosing that best round on the same split it is scored on.
  That optimism is what the untouched test split measures, and here test and validation moved together and by little.

  **A record that can say whether the ceiling bound.** Investigating the issue showed that `best_iteration` alone cannot: lightgbm 4.7 reports the best round inside the budget whether or not early stopping fired, so `best_iteration_` is never 0 at the ceiling, as a code comment in `model.py` claimed, and the 996 of `lgbm-extended` was a fit the ceiling ended.
  `model.json` therefore records the rounds run as well, and `early_stopped` is true when the fit ran the full patience past its best round.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** The method was set in the brief: raise the ceiling until early stopping fires, find where with one generous exploratory fit per variant, and commit a value with clear headroom.
  AI ran the measurement, chose 5,000 by the worst-case argument above rather than by a multiple of the observed best round alone, and laid out options A to E.
  Lukas confirmed 5,000 after seeing the exploratory fits, the per-tree costs and the committed run's numbers.
  Two of its findings went beyond the brief.
  It established that the issue's premise, that the 1,000-tree models were materially short of what the hyperparameters could reach, does not hold: the extra trees are worth 0.02 pp and 0.07 pp of test MdAPE.
  And it found, by probing lightgbm 4.7's early-stopping callback, that the code comment explaining the `best_iteration_ or num_trees()` fallback was wrong and that the record could not tell a binding ceiling from early stopping, which is why `early_stopped` and `boosting_rounds` were added with tests at the `train` seam.
- **AI interaction evidence:** Claude Code session on 2026-10-05 implementing issue #64: the exploratory fits are `reports/analysis/early_stopping_ceiling.py` and its results file; the committed run is the `dvc.lock` of PR #77, tracked as one MLflow run per variant in `recommenditos-price` with `best_iteration`, `boosting_rounds` and `early_stopped` logged.
- **Other evidence:** [issue #64](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/64); [PR #77](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/77); [`params.yaml`](../params.yaml), `train.variants`; [`recommenditos/modeling/model.py`](../recommenditos/modeling/model.py), `LightGBMModel.fit`; `tests/test_model.py::test_a_lightgbm_record_says_early_stopping_ended_the_fit`, `::test_early_stopped_is_decided_by_the_patience_not_by_the_best_round` and `::test_a_budget_early_stopping_never_reaches_is_recorded_as_a_budget_that_bound`; [`reports/analysis/early_stopping_ceiling.py`](analysis/early_stopping_ceiling.py) and [`tuning_sweep.py`](analysis/tuning_sweep.py) with their results files (2026-10-05); [model card](../docs/docs/model-card.md), Training and the ladder; [EDN-53](#edn-53-trainnum_threads-is-pinned-to-1-trading-fit-speed-for-a-machine-independent-artefact), [EDN-62](#edn-62-lgbm-basic-is-the-candidate-and-sc-04s-miss-on-cars-over-20-years-is-accepted-rather-than-worked-around).
- **In LaTeX:** yes

### EDN-72: The mileage range is a tolerant check on both frames, not a preprocessing filter

- **Date:** 2026-10-05
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Data Validation, Data Preparation
- **Participants:** @lukas2510, decided by the repository owner
- **Decision:** The FR-03 mileage range, 0 to 1,000,000 km, is asserted on the raw and the interim frame alike as a tolerant check, `validate.mileage_mostly = 0.999`, and preprocessing does not filter on it. The three listings above it stay in the cleaned frame and are reported by both suites.
- **Alternatives considered:**
  - **Option A: drop them in preprocessing, as EDN-22 does for late registrations, and make the interim rule hard.**
    Pros: the cleaned frame becomes FR-03's domain exactly, so no training row is one the API refuses; the values are evidently entry errors - a 2007 Volvo C30 at 1,560,000 km, a 2023 BYD Atto 3 at 2,424,696 km, a 2002 BMW 320 at 2,570,000 km - and the interim suite would be hard on every bounded rule.
    Cons: it changes the training rows of every model and every published count (105,405 cleaned rows become 105,402), and it is a scope decision rather than part of putting #25's rules into Great Expectations. Measured with the real downstream stages (`reports/analysis/mileage_scope.py`): one of the three is a training row of a supported make, one a validation row and none a test row, and refitting without them moves `lgbm-basic`'s MdAPE from 6.83 % to 6.79 % and `b1`'s from 9.52 % to 9.60 %, leaves `lgbm-extended` and `b0` unchanged, and changes no gate verdict.
  - **Option B (chosen): a tolerant check on both frames.**
    Pros: no data or model changes; it matches how `params.yaml` already classed the range ("Bounds the Great Expectations suites assert", while the price range "lives under `preprocess`, because there it is a scope filter"); 0.999 tolerates more than thirty times today's count and still fails a scrape in another unit or with a shifted decimal.
    Cons: three rows the API would refuse stay in training, and the interim suite has one tolerant rule where the others are hard.
  - **Option C: a hard rule on the interim frame and no filter.**
    Pros: none beyond simplicity.
    Cons: it fails the stage on the current data, so the pipeline could not run at all.
- **Rationale:** The measurement settled what was at stake: none of the three rows is a test row, and the model changes it causes are within a few hundredths of a percentage point with no verdict moving, so dropping them buys almost nothing that is measurable. What it would cost is a change to the training scope and to every count the documents and the report publish, a week before the first delivery, for an effect smaller than those documents round to. Keeping the range a check also keeps a systematic break visible as a failed rule instead of silently cleaning it away, the same reasoning EDN-22 gives for the raw suite's tolerance.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI found the conflict by running the suites on the real snapshot, where a hard interim rule failed the stage on the three rows, and laid out the options instead of choosing one inside the Great Expectations work. It measured option A with the real stages rather than estimating it, which is where the MdAPE deltas come from, and recommended B for this PR. Lukas chose B.
- **AI interaction evidence:** Claude Code session on 2026-10-05 implementing issue #25: the failing run, the measurement and the options were presented as a decision question in the session's report; Lukas answered "keep the tolerant check".
- **Other evidence:** [`reports/analysis/mileage_scope.py`](analysis/mileage_scope.py) and its results; [`reports/data-validation/summary.json`](data-validation/summary.json); [`recommenditos/data/gx_context_configuration.py`](../recommenditos/data/gx_context_configuration.py), `_mileage_in_range`; [EDN-22](#edn-22-drop-listings-registered-after-the-age-reference-date), [EDN-68](#edn-68-the-great-expectations-store-is-a-cached-build-product-the-suites-are-generated-from-the-contract-and-each-rule-sits-on-the-frame-where-it-can-hold); [issue #25](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/25), [PR #76](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/76).
- **In LaTeX:** yes

### EDN-74: A run is tagged with the hashes of its own stage's inputs, and `git_dirty` ignores what DVC writes itself

- **Date:** 2026-10-05
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Experiment Tracking, Reproducibility
- **Participants:** @lukas2510, decided by the repository owner
- **Decision:** Every MLflow run a pipeline stage opens or appends to is tagged with the hash of each dependency `dvc.yaml` declares for that stage, one `<stage>.deps.<path>` tag each, computed through DVC's own API when the stage starts; it is the value `dvc.lock` records under `stages.<stage>.deps` once the stage has finished.
  `git_dirty` is computed while ignoring the files `dvc repro` writes itself: each pipeline's lock file and every output declared `cache: false`, read off the pipeline rather than listed.
  `dvc_lock_md5`, the MD5 of the whole lock, is dropped.
  `evaluate`, which appends to the run `train` opened (EDN-55), records its own `evaluate.git_commit`, `evaluate.git_dirty` and `evaluate.deps.<path>` beside the fit's `git_commit`, `git_dirty` and `train.deps.<path>`.
  Input hashes are recorded only when DVC runs the stage, which DVC signals by setting `DVC_STAGE`.
- **Alternatives considered:**
  - **Option A (chosen): the hashes of the stage's own inputs, taken when it starts, and `git_dirty` without DVC's writes.**
    Pros: every tag names something a commit or the DVC remote reproduces: for the runs of a committed `dvc repro`, each input tag equals the committed lock line for line, and a clean commit is reported clean.
    The tags exist from the moment the run does, so a stage that fails half-way still says what it read.
    A data dependency's hash is its address in the DVC cache and on the remote, so `dvc pull` restores that input from it and it identifies the input whatever happens to the commits, a squash merge or a rebase included; a code file is restored by git at the commit, where a clean tree guarantees it has the recorded hash.
    And one tag per input lets MLflow find every run that read a given matrix with an exact filter.
    Cons: a run carries 24 provenance tags instead of three, ten input hashes and a commit and a dirty flag for each of its two stages; the stage opens the DVC repository in-process, which costs about 0.3 s per run, measured at 0.28 to 0.41 s on the laptop against fits of 0.1 to 19 s; the code depends on DVC's Python API (`Index.stages`, `Output.get_hash`), which DVC does not promise to keep stable, so a DVC upgrade can break it - the test that runs a real `dvc repro` and compares with the lock is what would say so; and a hand edit of an uncached output, `metrics.json` say, no longer marks a run dirty, which costs nothing today, because no stage that opens or appends to a run depends on an uncached output; a stage that did would hash it among its inputs, so the edit would show there.
  - **Option B: keep the whole-lock MD5, and document what it does not identify.**
    Pros: no code change; the model card already warned that the four runs of one ladder carry four different digests.
    Cons: the tag identifies no committed state, so documenting it would only document that NFR-14's data-version clause is unmet.
    Measured on the runs `main`'s models pointed at before this change: four `train` runs, four different digests, none equal to the committed lock; and the first stage's digest equals the lock from before the re-run.
    It also leaves `git_dirty` true for every stage after the first of a clean `dvc repro`, three of those four runs, only because `dvc.lock` had changed under them, so the report could not say its runs were made from a committed state either.
  - **Option C: tag after the run instead of before it.**
    Either at the end of the stage's own process, or by a separate step after `dvc repro` that reads the committed lock and writes the tags onto the runs.
    Pros: the separate step reads the very lock it is compared with, so the tags would equal it by construction; and the stage-end variant needs no change to how the run is opened.
    Cons: at the end of its own process a stage still cannot read its lock entry, because DVC writes it only after the command has exited, so the stage-end variant would have to compute the same hashes A computes, later and with less of the run covered: a stage that fails half-way would carry none.
    The separate step is a second command somebody has to remember after every `dvc repro`, and one forgotten leaves the runs without provenance and nothing failing; it would also describe the tree when the step ran, not when the stage did, and attribute anything changed in between to the run.
    And it equals the lock by copying it, so it could not detect a run whose inputs differ from what the lock says, which is the one thing the check is for.
- **Rationale:** NFR-14 promises that a run can be traced back to what produced it, and only a committed state can be traced back to.
  A whole-lock digest taken during a `dvc repro` is never the lock committed with that run, so option B keeps a tag that cannot do its job, and C either recomputes A's hashes later or copies the lock instead of checking against it.
  A is the only option whose tags are evidence: two programs, the stage at its start and DVC at its end, write the same values independently, and the check is that they agree.

  Three implementation choices are part of the decision.
  The hashes are computed through DVC's API rather than read from upstream stages' outputs in `dvc.lock`, because only some dependencies are another stage's output: the code files are no stage's output, and `evaluate`'s `models` and `data/processed/features` are the parents of several outputs and the output of none, so their `.dir` hashes are in the lock only as `evaluate`'s own dependencies, written after it has run.
  Reimplementing DVC's directory hash with `hashlib` would be one more thing that can disagree with the lock; calling the code that writes the lock cannot, and `tests/test_provenance.py` proves the two equal on a three-stage pipeline DVC really runs, on a directory, a file, a `foreach` stage and an uncached output that a later stage depends on.
  The tags are recorded only under DVC because every test calls the stages with paths under a temporary directory: the hashes of the files `dvc.yaml` declares would then describe files the run never opened.
  And `evaluate` records its own commit beside the fit's because the two need not run at the same commit: a change to `evaluate.py` alone reruns only `evaluate`, against the runs of an earlier `train`, and overwriting `git_commit` would then claim the fit came from code it did not come from.

  The implementation found more than the issue described.
  DVC deletes a stage's outputs before running it, so a stage with an uncached output starts with that git-tracked file deleted: on a three-stage pipeline reproduced from a clean commit with the old code, even the first stage was tagged dirty, and only the first stage's digest matched any committed lock, the one from before the run.
  And DVC adds a line to the `.gitignore` beside a cached output the first time it produces it, which for `gx/` is the repository root's hand-written `.gitignore`; ignoring that file would hide every edit to it, so the first run of a new cached output is reported dirty until its line is committed, which `pipeline.md` says.
  And DVC addresses a stage relative to the directory `dvc repro` was started in, `../dvc.yaml:train@b0` from a subdirectory, so the stage is looked up by its name alone; an independent review of the pull request found that the first version resolved the path against the repository root, and that from a subdirectory every run would then have lost its tracking, under a warning that blamed the credentials.

  Checked on the real snapshot rather than only in a test: the four `train` stages and `evaluate` were re-run from a clean commit with `RECOMMENDITOS_REQUIRE_TRACKING=1`, and `reports/analysis/run_provenance_check.py` read the four runs back from DagsHub.
  Before the re-run, the four runs `main`'s committed models pointed at carried none of the input tags, three of them were tagged dirty, and each `dvc_lock_md5` differed from the digest of the committed lock.
  After it, each of the four carries ten `train.deps` and ten `evaluate.deps` tags, every one equal to the committed lock, and `git_dirty` and `evaluate.git_dirty` are `false` for all four, naming the clean commit the re-run started from (`reports/analysis/run_provenance_check_results.txt`).
  The re-run, 1 min 48 s at a peak RSS of 693 MB, moved no number: `metrics.json`, the segment table and the masking sweep are byte-identical, each per-variant record changed only in its MLflow run id, and inside each model bundle only `model.json` changed, by the provenance EDN-56 puts there, while every payload file kept its hash.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI found the defect in an independent review of PR #70, by reading the runs back from DagsHub instead of trusting the tests, which checked that the tags exist and that they change when `dvc.lock` changes, not that they identify a committed state.
  It measured the four digests against the committed lock and the dirty flags against the files that had changed, generated options A to C and assessed them, and recommended A, fixed now in a pull request of its own; Lukas chose that, before the first delivery, so that the report's provenance claim is true and the CodeCarbon run of #38 retrains only once afterwards.
  The implementation details were AI's, as were the two further findings, the deleted uncached output and the `.gitignore` line, both reproduced before they were acted on.
  An independent AI review of the pull request reproduced the evidence and found the subdirectory regression, two overstated claims - that every hash retrieves its input from the remote, and that a hand edit of the validation summary would show in a run - and three smaller gaps: a dependency path MLflow refuses as a tag key, a failed `git status` read as a clean tree, and a test pipeline without a parameter; each was fixed, test first where there was code, before merging, which is the modification.
- **AI interaction evidence:** Claude Code sessions on 2026-10-05: the independent review of PR #70, whose measurement is the body of issue #79, and the implementation of issue #79, in which the issue's scenario was reproduced on a three-stage pipeline with `main`'s `run_provenance` (every stage tagged dirty, the digests of the lock before the run and of two mid-run locks), the new tests were confirmed to fail on `main`, where the module they test does not exist, and the check script was run against DagsHub before and after the re-run.
- **Other evidence:** [issue #79](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/79); [PR #80](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/80); [`recommenditos/provenance.py`](../recommenditos/provenance.py) and `recommenditos/tracking.py`; [`tests/test_provenance.py`](../tests/test_provenance.py) and the NFR-14 tests of `tests/test_tracking.py`; [`reports/analysis/run_provenance_check.py`](analysis/run_provenance_check.py) and [its results](analysis/run_provenance_check_results.txt); [The DVC pipeline](../docs/docs/pipeline.md), From a run to its inputs; [specification](../docs/docs/specification.md) NFR-14; [EDN-55](#edn-55-evaluate-resumes-the-run-train-created-instead-of-opening-its-own), [EDN-66](#edn-66-nfr-06-and-nfr-09-are-each-split-in-two-and-each-id-keeps-the-half-its-citations-mean).
- **In LaTeX:** yes

### EDN-77: The raw frame's measured fill rates are asserted by the Great Expectations suites, not promised by the raw contract

- **Date:** 2026-10-06
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Data Quality, Data Validation
- **Participants:** @lukas2510, decided by the repository owner
- **Decision:** `make`, `body_type` and the four `equipment_*` lists are nullable in `RAW_SCHEMA`, as they already were in `INTERIM_SCHEMA`, so the contracts hold only structural nullability.
  Both Great Expectations suites assert those six columns filled up to one tolerance, `validate.required_filled_mostly: 0.999`, in `params.yaml`.
  The interim rule, which was hard, becomes the same tolerant rule.
- **Alternatives considered:**
  - **Option A (chosen): nullable in the raw contract, tolerant fill-rate rule in both suites, one shared tolerance.**
    Pros: a gap in a scrape is reported by `validate-data` as a data-quality finding, with the rule named and the row count, instead of as a contract error in `download` that reads like a bug in our code; the raw contract keeps only what the file is structurally; one semantics on both frames, so a gap the raw suite tolerates does not then fail the interim suite.
    Cons: a tolerated gap travels through `preprocess` as a null, which the interim contract already allowed; the interim rule no longer fails on a single value that preprocessing empties, only past 0.1 % of the rows.
  - **Option B: keep the raw contract as it is and correct the wording only.**
    Pros: no code change and no risk to the pipeline.
    Cons: the pipeline stops in the wrong stage with the wrong kind of message, and the fill-rate rule keeps documenting a path no scrape can reach.
  - **Option C: tolerant on raw, the interim rule stays hard.**
    Pros: catches every value preprocessing empties.
    Cons: contradicts the raw rule, because a gap the raw suite tolerates would then fail the interim suite, so the tolerance would never take effect.
  - **Option D: tolerant on raw, a separate and stricter tolerance on interim.**
    Pros: mirrors the two roles of the frames.
    Cons: a second parameter for a distinction that nothing in the data currently needs.
- **Rationale:** A contract answers "is this our bug", a suite answers "did the data change", and a fill rate is the second kind of question.
  The raw contract answered it on the data's behalf, because the snapshot happens to fill these columns in every row, so the first gap in a future scrape would have stopped the pipeline in `download` and never reached the rule written for it.
  Option A makes both frames say the same thing.
  The tolerance of 0.999 follows `mileage_mostly`: far tighter than the 0.99 of `filled_mostly`, because these columns are filled in 100 % of the snapshot, yet it tolerates a few gaps as a finding and still fails a scrape that has lost a column.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI read the contract, both suites and every consumer of the raw frame, generated options A to D with their pros and cons, recommended A with the 0.999 tolerance, and implemented it with tests that fail on a missing `body_type` through `download`'s contract and through the `validate-data` stage.
  Lukas chose A as recommended and asked for the decision to be recorded.
  Issue #78 had already named the problem and the two open questions, the tolerance and the fate of the interim rule, which the options settle.
- **AI interaction evidence:** Claude Code session of 2026-10-06: the options were put to Lukas with a recommendation before any code changed, and he chose "1A + 2A with 0.999" and asked for an EDN entry.
- **Other evidence:** [issue #78](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/78), found in the review of [#76](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/76); [`recommenditos/schema.py`](../recommenditos/schema.py); [`recommenditos/data/gx_context_configuration.py`](../recommenditos/data/gx_context_configuration.py), `REQUIRED_FILLED_COLUMNS`; [`tests/test_validate_data.py`](../tests/test_validate_data.py); [pipeline docs](../docs/docs/pipeline.md), Data validation; EDN-68 is the entry for #25, whose suites this changes.
- **In LaTeX:** yes

### EDN-78: The hyperparameter search extends EDN-73's protocol to both LightGBM variants and `min_child_samples`, and measures and tracks every fit

- **Date:** 2026-10-06
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Hyperparameter Tuning, Experiment Tracking, Energy Efficiency, Requirements
- **Participants:** @kadameit, who decided, with AI assistance
- **Decision:** The LightGBM hyperparameters are searched by `recommenditos/modeling/tune.py`, outside `dvc repro`, under the protocol EDN-73 wrote down, extended in four ways.
  Both variants are swept, `lgbm-basic` and `lgbm-extended`, on EDN-73's grid: `learning_rate` 0.025, 0.05 and 0.1 against `num_leaves` 31, 63, 127 and 255, every point at a ceiling of 20,000 trees so early stopping decides it.
  A second grid follows for each variant, over `min_child_samples` 5, 10, 20, 50 and 100, with `learning_rate` and `num_leaves` fixed at the first grid's winner, which is the committed point when nothing was adopted; `min_child_samples` becomes an optional LightGBM key at LightGBM's own default of 20, the value every committed model already used, so `params.yaml` names it only if a search adopts another value.
  The adoption rule is EDN-73's, unchanged: a point replaces the reference only when the simultaneous 95 % interval of its validation L1 difference, from the paired bootstrap resampled by seller group and widened by the max-statistic bootstrap over all the grid's points, lies entirely below zero. The search never reads the test or calibration split.
  Every fit is measured with CodeCarbon and logged as its own MLflow run in `recommenditos-price-tuning`, tagged with its sweep, `sweep_role=point` and its variant, carrying its validation metrics, fit time and energy; one `sweep_role=summary` run per sweep carries the totals and the verdict.
  A sweep's record - its results as JSON, the same as a readable table, and CodeCarbon's row for every fit as a CSV - is kept in MLflow only, as artifacts under `results/` of the sweep's summary run, `sweep.json`, `sweep.txt` and `emissions.csv` for a sweep run from now on; nothing of a search is committed.
  The four sweeps of #88 ran before this, so their summary runs carry the JSON and the table under the names the module then used, `results/<sweep id>.json` and `.txt`, and their CodeCarbon CSV, until then committed, was uploaded to the same run as `results/emissions.csv`.
  The search's energy is reported as its own total there, never in `reports/emissions/<variant>.csv`, which belongs to the fit of the final configuration.
  NFR-10 is reworded to match: the 15-minute budget applies to fitting the final chosen configuration, and a search runs separately, with its energy recorded per fit and reported apart from the final fit's.
- **Alternatives considered:**
  - **Option A (chosen): extend EDN-73's protocol, and build its decision step into the package.**
    Pros: the first grid for `lgbm-basic` repeats EDN-73's twelve fits exactly, so the new tool is checked against recorded evidence before anything new is read from it; the simultaneous interval still guards against the winner's curse within each grid; the cost is bounded and small, about 34 fits in all, and EDN-73's twelve `lgbm-basic` fits took 127 s of fitting; and issue #88's acceptance criteria - a tagged run per configuration with its metric, fit time and energy, the search's emissions apart, NFR-10 reworded - are met by code that runs every time rather than by hand once.
    Cons: on `lgbm-basic` the likely outcome is EDN-73's again, "keep", so the model card may still have to say the model is untuned; one validation split stays the only source of selection noise; the two grids are run one after the other, so an interaction between `min_child_samples` and the first grid's knobs is not searched; nothing corrects across the two grids or the two variants, because the max-statistic interval is taken per sweep, and the second grid is chosen after the first on the same validation split; and `feature_fraction` and L2 regularisation are still not parameters.
  - **Option B: the issue as written - grid search, and the best validation point goes into `params.yaml`.**
    Pros: the least machinery, it matches the issue text, and the model can be called tuned.
    Cons: it adopts noise. EDN-73 measured exactly this case: the lowest validation L1 of its grid, `(0.025, 63)`, is indistinguishable from the committed point even point by point, and the two metrics disagree on which point is best. It would also overturn the adoption rule EDN-73 decided, move every number in the report for no measurable gain, and with the issue's 0.03 the grid would no longer be comparable with EDN-73's measured 0.025.
  - **Option C: random or Optuna search over more knobs, selected by seller-grouped k-fold cross-validation on train plus validation.**
    Pros: cross-validation is a less noisy selection than one split, and a larger space might hold a real gain.
    Cons: a new dependency and new code; about five times the fits per point, so five times the energy, for a region EDN-73 found flat; early stopping then needs a validation fold inside every fold; and it departs from the problem specification's "the validation set is used for early stopping and tuning". EDN-73 left it for the day a criterion depends on a fraction of a point, and none does.
  - **Option D: close #88 as covered by EDN-73.**
    Pros: no work and no compute.
    Cons: `lgbm-extended` would never have been swept and `min_child_samples` never examined, although many `model` values occur only once; EDN-73's screening runs carry no energy figure per point; and NFR-10 would still say "no hyperparameter search" with no rule for where a search's energy goes. None of #88's acceptance criteria would be met.
  - **Where a sweep's record is stored (decided 2026-10-09, after the search had run).** The pipeline's own `reports/emissions/<variant>.csv` stays git-tracked whichever is chosen, because the `compare-energy` stage reads it; this is about the search's record only.
    - **Option A: the results and CodeCarbon's CSV committed under `reports/tuning/`, as well as the runs in MLflow.**
      Pros: the numbers can be checked from the repository without access to the DagsHub tracking server, and it follows the convention of `reports/analysis/`, whose results files are committed beside their scripts.
      Cons: the same data in two places, which can disagree; and a commit is needed between sweeps, because a sweep's uncommitted files make every later run `git_dirty` (EDN-74).
    - **Option B (chosen): MLflow only, with CodeCarbon's CSV uploaded as an artifact of the summary run.**
      Pros: one source of truth; the closest to the course demo, which logs CodeCarbon's row to MLflow as metrics and parameters and commits no emissions file; no commit between sweeps; and the raw CSV keeps how each figure was produced - CodeCarbon's version, the CPU and RAM power columns, the operating system - which matters because EDN-69 calls the figure an estimate, and an estimate is checked by its method.
      Cons: the numbers this entry and the model card quote can be checked only with access to the DagsHub tracking server.
    - **Option C: B without the upload, exactly as the course demo does, with the CSV discarded.**
      Pros: the least to store, and nothing beyond what the demo does.
      Cons: rejected, because it loses the columns that say how each figure was produced, which the per-run metrics and parameters carry only in part.
- **Rationale:** EDN-73 already answered how a value may be adopted; what #88 added was scope and record-keeping, so the protocol was extended rather than replaced.
  Two parts of the issue were stale against EDN-73, which landed after it was written: it proposed `learning_rate` 0.03 where EDN-73 measured 0.025, and it said the winning values go into `params.yaml`, where EDN-73 adopts only a value whose simultaneous interval lies below zero.
  0.025 was kept because a repeat of EDN-73's grid is the one check of the new code against numbers already on record, and because 0.03 lies inside the 0.025 to 0.05 range EDN-73 found flat.
  `min_child_samples` is the knob the issue named as optional, and the data gives it a reason: a leaf of fewer rows can fit one listing of a `model` seen once.
  Making it optional at LightGBM's default rather than writing `min_child_samples: 20` into `params.yaml` keeps every committed parameter value as it is, and a test holds that leaving it out and setting it to 20 fit the same booster.
  The second grid holds the first grid's winner fixed because a joint grid would be 60 fits per variant instead of 5, for an interaction nothing so far suggests.

  The decision step became a module of the package, run from the command line, rather than staying an analysis script, because #88 asks for every searched configuration to be a tracked run with its energy, and that is what `train` already does for one fit: the search reuses its `read_matrices`, `fit_variant`, `energy.measure` and run logging, so a search run and a pipeline run are measured and named alike.
  It is not a DVC stage, because a search's result is a verdict a person acts on through a reviewed change to `params.yaml`, not an artefact a later stage consumes, and its cost scales with the grid rather than with the product.
  For the same reason NFR-10's 15 minutes apply to the final fit: the budget exists to keep retraining the product cheap, and a search is run once per decision, not once per retraining.
  The search's energy is not written beside the variants' CSVs, because each of those is the record of the fit that produced its model, and because `compare-energy` declares the whole `reports/emissions` directory as a dependency, so a search writing there would make that stage stale.
  Where it is kept instead is the sub-decision above: MLflow alone, as artifacts of each sweep's summary run, so a sweep refuses to start when tracking is not configured, opens its runs so that a server it cannot reach fails it rather than silently dropping them, and refuses a sweep id the experiment already holds runs for.
  Queued `dvc exp` screening stays available as EDN-73 describes, and is now optional, because the decision step measures and logs every fit itself.
  EDN-73's bootstrap moved into the module unchanged: `tuning_sweep.py` now imports it and computes bit-identical draws and multipliers, so EDN-73's recorded evidence still reproduces.

  **Results (2026-10-06).** All four sweeps kept their reference, `learning_rate` 0.05, `num_leaves` 63 and `min_child_samples` 20, so nothing is adopted: `params.yaml` and `dvc.lock` are unchanged, and the committed test metrics and gate verdict, computed once by the committed pipeline run, are those of the configuration the search kept; the search never read the test split.
  The search ran from clean commits on a 13th Gen Intel Core i7-1360P laptop (Windows, CodeCarbon 3.3.1's estimate, `num_threads` 1), into the DagsHub MLflow experiment `recommenditos-price-tuning`; every one of the 34 fits was ended by early stopping, not by the 20,000-tree ceiling.
  - `88-lgbm-basic-lr-leaves` reproduces EDN-73's twelve fits exactly: the same best round, validation L1 and validation MdAPE at every point, and the same per-point and simultaneous intervals, so the new module is checked against the recorded evidence, and only the fit times differ, because the machine does.
    Its lowest validation L1 is again `(0.025, 63)`, 0.00031 below the reference with a simultaneous interval of -0.00109 to +0.00047, and only `(0.1, 127)` and `(0.1, 255)` are measurably worse; validation MdAPE spans 6.88 % to 7.04 %.
  - `88-lgbm-extended-lr-leaves`: the lowest validation L1 is `(0.025, 127)`, 0.00040 below the reference with a simultaneous interval of -0.00150 to +0.00070.
    `(0.025, 255)` is the case the rule exists for: it is 0.20 pp better in validation MdAPE, 6.25 % against 6.45 %, with a per-point interval of -0.385 % to -0.060 %, entirely below zero, and a search that took the best validation MdAPE would have adopted it.
    But MdAPE is not the selection metric, its L1 difference, -0.00039, has a per-point interval of -0.00146 to +0.00069 that the split cannot resolve, and even its MdAPE difference, widened to hold for all eleven points at once, spans zero at -0.427 % to +0.028 %: the best of eleven noisy comparisons looks better than it is, which is the winner's curse.
    Five points are measurably worse under the simultaneous L1 interval: all four at `learning_rate` 0.1, whose simultaneous intervals start at +0.00037 or higher, and `(0.05, 31)`, at +0.00003 to +0.00243; validation MdAPE spans 6.25 % to 6.69 %.
  - `88-lgbm-basic-min-child-samples`: the lowest validation L1 is `min_child_samples` 10, 0.00043 below the reference with a simultaneous interval of -0.00124 to +0.00038, and 100 is measurably worse, at +0.00010 to +0.00210; validation MdAPE spans 6.92 % to 7.03 %.
  - `88-lgbm-extended-min-child-samples`: the reference, 20, has the lowest validation L1 of the five, no point is resolved either way under the simultaneous interval, and validation MdAPE spans only 6.45 % to 6.49 %.

  The search cost 34 fits, 391.3 s of fitting (1,069.0 CPU-s), 3.6071 Wh and 627.81 mg CO2eq: 17 fits, 112.3 s, 1.0864 Wh and 189.09 mg for `lgbm-basic`, and 17 fits, 279.1 s, 2.5207 Wh and 438.73 mg for `lgbm-extended`, each sweep's totals on the "search totals" line of its table.
  On the search's own machine that is about 20 times the energy of one fit of `lgbm-basic`'s committed configuration (0.0549 Wh) and 16 times one of `lgbm-extended`'s (0.1584 Wh); its fit times are not comparable with the ladder's energy table in the model card, which was measured on another laptop, an i5-10210U.
  Because nothing was adopted, the final configuration is the committed one, and its fit time against NFR-10's 15 minutes is the one already on record: 8.3 s for `lgbm-basic` and 27.9 s for `lgbm-extended` in the tracked run of 2026-10-06; the search itself sits outside that budget, as the reworded NFR-10 says.
  As option A's cons anticipated, the verdict on `lgbm-basic` is EDN-73's again, and the model card now says the values were searched and kept rather than that they are untuned.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation
- **Response to AI:** Accepted with modifications
- **Assessment of the AI contribution:** AI compared issue #88 with EDN-73 and the code, and pointed out where the issue had become stale: it proposed 0.03 where EDN-73 measured 0.025, and it said the winning values go into `params.yaml` where EDN-73 adopts only a value its rule resolves.
  It laid out options A to D with their pros and cons and recommended option A, a second grid over `min_child_samples` with the first grid's winner fixed, recording the method decision as a new entry extending EDN-73, and `learning_rate` 0.025.
  Kevin weighed the options and accepted all four recommendations before the search was implemented or run.
  The NFR-10 rewording and reporting the search's energy apart were not separate choices: they follow from the scope of issue #88.
  AI also pointed out that the search's emissions could not go under `reports/emissions/` without making the `compare-energy` stage stale.
  On where a sweep's record should live, AI first recommended option A of that sub-decision, committing it under `reports/tuning/` beside MLflow, so the numbers could be checked without an account; on Kevin's question it checked the course demo repository and found that the demo keeps its emissions in MLflow only.
  Kevin chose MLflow only, with CodeCarbon's CSV uploaded as an artifact, which is the modification.
- **AI interaction evidence:** Claude Code session of 2026-10-06 on issue #88: the comparison with EDN-73 and options A to D with a recommendation, put to Kevin before the search was implemented; Claude Code session of 2026-10-09: the options for where a sweep's record is stored, with a recommendation for option A, the check of the course demo repository Kevin asked for, and his choice of option B.
- **Other evidence:** [issue #88](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/88); [`recommenditos/modeling/tune.py`](../recommenditos/modeling/tune.py) and [`tests/test_tune.py`](../tests/test_tune.py); [`reports/analysis/tuning_sweep.py`](analysis/tuning_sweep.py), now importing the module's statistics; the search's results in the [DagsHub MLflow server](https://dagshub.com/recommenditos/MLOPS_Recommenditos.mlflow), experiment `recommenditos-price-tuning`: the sweeps `88-lgbm-basic-lr-leaves`, `88-lgbm-extended-lr-leaves`, `88-lgbm-basic-min-child-samples` and `88-lgbm-extended-min-child-samples`, each found with the filter `tags.sweep = "<id>"` and its record under `results/` of the run with `tags.sweep_role = "summary"`, compared with EDN-73's [`tuning_sweep_results.txt`](analysis/tuning_sweep_results.txt); [pipeline docs](../docs/docs/pipeline.md), "Tuning a hyperparameter"; [specification](../docs/docs/specification.md) and [requirements](../docs/docs/requirements.md) NFR-10; [EDN-73](#edn-73-learning_rate-and-num_leaves-are-not-tuned-and-any-later-tuning-follows-a-written-protocol), [EDN-69](#edn-69-codecarbon-measures-the-fit-alone-offline-and-per-process-and-its-figure-is-reported-as-the-estimate-it-is), [EDN-70](#edn-70-n_estimators-is-a-ceiling-of-5000-that-early-stopping-stays-under-not-a-tree-count).
- **In LaTeX:** yes

### EDN-79: The seller-grouped hash split stays, and its figures are reported with their spread over other splits

- **Date:** 2026-10-12
- **Milestone:** M3: Quality Assurance
- **Activity / Topic:** Evaluation Protocol, Data Splitting
- **Participants:** @lukas2510, decided by the repository owner
- **Decision:** The split stays as it is: whole sellers assigned by a hash of the seed and the seller key, 60/10/10/20, unstratified, with the committed seed.
  What changes is the reporting: the model card states the distribution of every set, and a figure of the committed split is given with its spread over 20 other splits.
  An adversarial-validation check is recorded once and not made a standing check, and the missing temporal split is a stated limitation.
- **Alternatives considered:**
  - **Option A (chosen): keep the grouped hash split and report the set distributions.**
    Pros: no rerun and no number moves a week before the delivery; adding sellers moves no seller already there, measured at 0.00 % of the rows when the `ES` listings join the pool, which is what FR-15's retrain comparison rests on; the committed draw is better balanced than 72 % of 1,000 seeds.
    Cons: the balance is the luck of this draw, not a property of the design, and Mercedes-Benz stays at 8.67 % of the calibration set against 13.54 % of the pool.
  - **Option B: a stratified-group split on `make`.**
    Pros: the largest gap of any make's share in any set to its pool share falls from 4.86 pp to 2.04 pp, and calibration reaches 11.5 % Mercedes-Benz.
    Cons: the assignment depends on all sellers at once, so 52.6 % of the existing rows change set when the `ES` listings join, about as many as a new seed moves, which confounds FR-15's before-and-after comparison; a full pipeline rerun and new split code, with every reported number changing; and it cannot balance the make either, because one dealer holds 25.2 % of the Mercedes-Benz listings and has to go somewhere whole.
  - **Option C (chosen, with A): repeated grouped splits, reporting the spread beside the single figure.**
    Pros: it answers what was never measured, how much a figure depends on which sellers landed where, in about 15 minutes and without touching the pipeline.
    Cons: an analysis script rather than a stage, so it has to be re-run by hand when the candidate or the data changes.
  - **Option D (recorded once): adversarial validation.**
    Pros: one classifier, AUC 0.495 on held-out sellers, so train and test are indistinguishable.
    Cons: nearly a tautology for a random assignment of sellers; run row-wise it reports 0.599, which is the dealer signature EDN-14 found and not a difference between the sets, so as a standing check it would mostly raise false alarms.
  - **Option E: choose a seed for a better mix.**
    Pros: none that survives a question.
    Cons: a new split selected by its outcome; and the committed seed already gives the most favourable SC-04 figure of 21 draws, so a reseed would either worsen that number or look chosen for it.
  - **Option F: a temporal split.**
    Pros: the strongest test for a price model.
    Cons: impossible, the snapshot has no listing dates; it is stated as a limitation.
- **Rationale:** The question was whether the default split deserved to stay, and three measurements could have said no; none did.
  The imbalance that prompted the issue has one cause, and it is not what the issue assumed: the largest seller group is a single dealer with 3,383 listings, 3,382 of them Mercedes-Benz, that landed in train, so train is raised (14.95 %, 9.97 % without it) rather than calibration being short, and the calibration share is at the 21st percentile of its own distribution over 1,000 seeds.
  The verdict on the pooled criteria does not depend on the draw: over 20 other splits the candidate's test MdAPE is 6.73 % on average with a standard deviation of 0.27 pp, against 6.81 % committed, and SC-01 and SC-02 hold on all 20.
  And SC-04's miss is not an artefact of it: cars older than 20 years are over the bound on 20 of 20 splits, at 18.3 % on average, so the committed 16.77 % understates the miss.
  Against that, option B buys 2.8 pp of make balance for the one property the split was built for.
  The same run puts EDN-73 in context: the validation MdAPE moves by 0.42 pp (standard deviation) between splits, more than the 0.16 pp the whole tuning grid spans on one.
  The 20 splits share one pool of listings, so the spread describes how a figure moves with the assignment and is not a confidence interval for new data.
  The issue's criterion that the UC2 interval coverage be checked per make is deferred to the work that builds the intervals: no model has an interval method and SC-05 is not measured, so there is nothing to compute a coverage from.
- **AI involvement:** Information seeking, Alternative generation, Alternative assessment, Recommendation, Solution generation
- **Response to AI:** Accepted
- **Assessment of the AI contribution:** AI wrote and ran the evidence script against the stages' own split, feature and fit code, checked first that it reproduces the committed `metrics.json` exactly, and ran 20 seeds instead of the ten asked for after timing one.
  It corrected the issue's premise, which Lukas had written: the largest dealer is the cause of the Mercedes-Benz gap, but by raising train, not by leaving calibration short.
  It laid out the six options with their measured costs, recommended A with C, and stopped for the decision; Lukas chose A with C as recommended, with D recorded once and F as a limitation.
  It also reported that the interval-coverage criterion could not be met and proposed deferring it rather than producing a number.
  The recommendation was accepted without change because every claim in it is a line of the results file that can be re-run.
- **AI interaction evidence:** Claude Code session on 2026-10-12 implementing issue #89: the agent measured, reported the options with the numbers of `reports/analysis/split_strategy_results.txt` and its recommendation, and Lukas decided on that report.
- **Other evidence:** [issue #89](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/89); [PR #107](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/pull/107); [`reports/analysis/split_strategy.py`](analysis/split_strategy.py) and [its results file](analysis/split_strategy_results.txt) (2026-10-12); [model card](../docs/docs/model-card.md), "How comparable the sets are, and how much a figure moves with the split"; [pipeline docs](../docs/docs/pipeline.md), "The split"; [EDN-14](#edn-14-nfr-11s-drift-control-is-an-iid-sample-not-a-seller-grouped-one), [EDN-73](#edn-73-learning_rate-and-num_leaves-are-not-tuned-and-any-later-tuning-follows-a-written-protocol).
- **In LaTeX:** yes
