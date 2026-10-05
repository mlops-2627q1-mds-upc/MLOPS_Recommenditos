The DVC pipeline
================

How the pipeline is laid out, what each stage promises the next one, and how to work on a stage without colliding with anyone else.
For the DVC remote, tracking granularity and the day-to-day `dvc pull` / `dvc push` rules, see [Data versioning](data-versioning.md).

Running it
----------

```bash
uv sync
uv run dvc repro          # or: make repro
uv run dvc metrics show   # or: make metrics
```

`download.source` in `params.yaml` is `zenodo`, so `dvc repro` builds the pipeline from the real snapshot: the `download` stage fetches the pinned Zenodo file itself, checks it against `download.md5` and converts it to Parquet.
The committed `dvc.lock` records that run, and `dvc pull` gets every artefact it names, so a clean clone has the whole pipeline up to date without running a stage.
Setting the parameter back to `synthetic` swaps in a generated stand-in, which needs neither the network nor DagsHub access; that is what the test suite uses and what a contributor without credentials can still run.

Because the source is a parameter, `dvc.lock` records which one every artefact was built from, so a run on synthetic data cannot pass unnoticed: `dvc status` reports a workspace whose `download.source` disagrees with the lock.
`dvc params diff` does not show this, because it compares the params files of two Git revisions rather than the lock against the workspace.
`metrics.json` carries the resolved value as its `data_source` leaf, so a number read off `dvc metrics show` says which data it was measured on and no stand-in result can be quoted as a model result.

What `download.source: zenodo` costs, measured on the real file: about 1 GB of RAM (peak RSS 929 MB, because the frame is read and validated whole) and about 1 GB of disk, being the 548 MB CSV cached in `data/external/`, the 215 MB Parquet and another 215 MB for its copy in the DVC cache.
The stage takes about 90 s with a cold cache and about 66 s with a warm one, because the CSV is downloaded once and reused from there, so a rerun costs a 25-second re-read rather than a download (see [Data versioning](data-versioning.md)).
A clean clone pays none of that, because `dvc pull` brings the Parquet down and the stage is already up to date.
CI never runs the stage; the test suite covers it against fixtures instead.

That is measured, not assumed, and it is the drill NFR-06 names.
On a scratch clone of commit `3e05487` (#66, 2026-10-01), before the data-validation stages had outputs of their own, `dvc pull` fetched 34 files and added 32, and `dvc repro` then reported all twelve stages as `didn't change, skipping` and finished with `Data and pipelines are up to date.`
Nothing came from Zenodo: the clone has no `data/external/` directory at all, and its `metrics.json` and `data/raw/listings.parquet` are byte-identical to the ones this repository produced.

The tests never run the pipeline through DVC.
They call the same stage functions in the same order against a synthetic fixture in a temporary directory, so `pytest` stays fast and hermetic:

```bash
make test
```

Stages
------

One module per stage, which is what lets a `deps` entry name exactly the code that stage runs ([EDN-28](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).

| Stage | Module | Writes |
|---|---|---|
| `download` | `recommenditos/data/download_raw_dataset.py` | `data/raw/listings.parquet` |
| `preprocess` | `recommenditos/data/preprocess.py` | `data/interim/listings.parquet` |
| `configure_gx` | `recommenditos/data/gx_context_configuration.py` | `gx/`, the Great Expectations context |
| `validate-data` | `recommenditos/data/validate_data.py` | `reports/data-validation/`: the summary, the validation results and the Data Docs; a failed expectation fails the pipeline |
| `split` | `recommenditos/data/split_data.py` | `data/processed/{train,validation,calibration,test,holdout_es}.parquet`, `supported_makes.json` |
| `features` | `recommenditos/data/build_features.py` | `data/processed/features/<set>/` |
| `train` | `recommenditos/modeling/train.py` | `models/<variant>/` |
| `evaluate` | `recommenditos/modeling/evaluate.py` | `metrics.json`, `reports/metrics/<variant>.json`, `reports/metrics/{segments,masked-inputs}.csv` |

`features` and `evaluate` have no equivalent in the [course demo](https://github.com/mlops-2627q1-mds-upc/MLOps-2627q1-demos).
Its text model needs no feature engineering, and it asserts its metric threshold inside `tests/test_model.py` instead of producing a metrics artefact, which is not enough for the gate [NFR-01](specification.md) describes.
`metrics.json` is deliberately narrow: `dvc metrics show` renders one column per JSON leaf, so its 29 leaves (28 of them rendered while `deployable_variant` is null) are read with `dvc metrics show --md` and the per-variant record, the per-segment table and the masking sweep go to `reports/metrics/` instead.
Five of those leaves are not measurements but provenance - `data_source` and each variant's `estimator` - because a number is only readable as a result once the file says which data produced it and which model answered.
`dvc metrics diff` is long-format and lists only the leaves that changed, which is what compares two model versions across a merge.

Two stages iterate a mapping in `params.yaml` with `foreach`, so they are named after the item rather than its position: `features@basic`, `train@lgbm-extended`.
Each declares only its own slice under `params:`, so changing one variant's hyperparameters retrains that variant and no other.

The contract
------------

`recommenditos/schema.py` is what the stages code against instead of against each other.
It describes **structure**: which columns exist, their dtype and whether they may be null.
Value rules - date and price bounds, ranges, fill rates - are **data quality** and live in the Great Expectations suites (see [Data validation](#data-validation)), because the two answer different questions.
A structural break is a bug in our code; a value break is a change in the data.

```python
from recommenditos.pipeline import read_frame, write_frame
from recommenditos.schema import INTERIM_SCHEMA

frame = read_frame(path, INTERIM_SCHEMA)  # validates after reading
write_frame(frame, out, INTERIM_SCHEMA)  # conforms before writing
```

`conform` selects the contract's columns in its order and casts them.
Because it selects, a column the contract does not name cannot reach an artefact: the PII columns [NFR-08](specification.md) forbids are kept out structurally, not by every stage remembering to drop them.

That guarantee starts at the interim frame, not before it.
The raw layer deliberately keeps the published file's PII columns, because preprocessing hashes `seller_company_name` and the location into the split's group key before dropping them, and it is pushed to the remote like any other stage output (EDN-34).
The pushed Parquet therefore holds the published file's personal data, which is the exposure EDN-34 weighed.

Both halves exist because Parquet preserves whatever dtype it is handed rather than normalising it.
Without an explicit cast at each boundary the frames drift apart stage by stage, and the stage that notices is never the stage that caused it.

Three contracts are defined:

- `RAW_SCHEMA`, the 75 columns of the AutoScout24 snapshot as published.
- `INTERIM_SCHEMA`, what `preprocess` writes: row-filtered, PII-free, deduplicated, carrying `log_price` and the hashed `seller_group_id`, but not yet feature-engineered.
- `PROCESSED_SCHEMA`, what `split` writes once per set. The split moves rows between files; it does not change columns.

Feature matrices depend on `features.sets` in `params.yaml`, so they come from `feature_schema(columns, name=...)` rather than from a constant.
A feature set naming a column no stage produces fails there, with the offending name, instead of producing a matrix that is quietly missing a column.

They also depend on the **data**, which a params-derived contract cannot express: which equipment items cleared the frequency threshold, and which levels each categorical has.
So the `features` stage writes what it actually built as `feature_space.json` beside its matrices, and every stage that reads a matrix loads that rather than rebuilding it.
A categorical carries its level list, because a code is a level's position: two frames over the same levels in a different order hold different numbers under the same dtype name.
`Schema.conform` therefore casts to the **declared** levels rather than inferring them from the frame in front of it, which is what lets a stage encode a subset, a rebuilt frame or a one-row request and get the numbering the model was fitted on.
`build_features.FeatureSpace.load(directory, name=...)` is the one entry point for a consumer that needs the contract, the vocabulary or a frame cast against them.

The supported makes
-------------------

The model is fitted, and the SC-01 to SC-06 gate measured, only on the makes the API serves ([EDN-48](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
Three stages share that rule, and each has one job in it:

- `split` **records** the list and writes it as `data/processed/supported_makes.json`, but removes no row.
  It stays a lossless partition, so EDN-05's threshold can be revisited without re-running the split, and no row disappears without an artefact saying where it went.
- `features` **applies** it, in `build_features.served_rows`, and is the only stage that does ([EDN-67](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
  It restricts all five frames - train, validation, calibration, test and the `ES` holdout - and it does so before the vocabulary is built, so the `make` levels, the equipment columns and the `model_version` levels are decided by served training rows alone.
  It declares the artefact as a dependency, so changing `split.min_listings_per_make` reruns the matrices, the models and the metrics.
  It also refuses a supported make that no training row holds, naming it: `split` counts support over all four sets, so after a change to `seed` or to the threshold every seller of a make can land outside train, and the API would then accept a make the model has never seen.
- `train` and `evaluate` **check** it, with `build_features.check_served_makes_only`, rather than filtering a second time.
  Each refuses a matrix holding a make outside the list, a row with no make level, or a `make` level the list does not name, and says that the matrices and the list come from different runs.
  A second filter would never remove a row from matrices `features` wrote; on any other matrices it would quietly repair the rows while keeping a feature space decided over another population, which is the one outcome worse than a failure.
  `evaluate` takes the list out of the model's own metadata, so the evaluated population is checked against the trained one, and `check_make_levels_are_served` then refuses a per-make SC-04 segment outside it.

EDN-48 named four guarantees this arrangement owes, and each has an owner and a test:

1. The filter is applied to train, validation, calibration, test **and** the `ES` holdout.
   **`features`**, asserted by `tests/test_features.py::test_every_frame_the_stage_writes_holds_only_supported_makes`.
2. The filter is applied **before** any training-derived vocabulary or statistic is computed - the equipment multi-hot columns, a category encoding, a mean, a quantile.
   **`features`**, asserted by `tests/test_features.py::test_the_vocabulary_is_decided_by_the_served_training_rows_alone`, whose training rows are built so that the `make` levels, the equipment columns and the `model_version` levels each come out differently if a single unserved row is counted.
3. `evaluate`'s per-make segments are restricted to the listed makes, or SC-04 reports a segment the API answers with a 422.
   **`evaluate`**, through `check_make_levels_are_served`.
4. A test asserts that the frames reaching a model contain only supported makes, so the guarantee is checked rather than intended.
   The two tests above, plus `tests/test_model.py::test_the_model_is_fitted_on_the_supported_makes_only` and `tests/test_pipeline.py::test_the_reported_population_is_the_served_population`.
   Every one of them first asserts that its input holds an unsupported make, and the last two count against the frames `split` wrote rather than against the matrices, so none can pass by the filter being unnecessary.

Measured on the real snapshot (2026-10-05), the filter removes 1,437 of the 99,326 split rows and moves every realised split share by at most 0.08 pp, so it does not disturb the split proportions.
Per frame: train keeps 60,378 of 61,180 rows, validation 8,859 of 8,992, calibration 8,987 of 9,169 and test 19,665 of 19,985.
Of the 6,079 `ES` holdout rows the holdout matrix keeps 5,979, which is EDN-14's count of the `ES` rows the API accepts under FR-04; the 100 it drops are requests the API refuses before the model sees them.
The matrix is the holdout as the model receives it, and it is not what M6 replays: a request cannot be rebuilt from it, because its `country_code` is empty throughout (see below) and `registration_date` has become `age_years`, both fields FR-01 requires.
The replay reads `data/processed/holdout_es.parquet`, which keeps all 6,079 rows, and has to filter to the supported makes itself or count the 100 refused requests as FR-04's 422s.
The holdout's `country_code` stays empty in every one of those rows, because `ES` is still not a training level ([EDN-42](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)); the filter removes rows and leaves that encoding as it was.
The 1,437 out-of-scope rows would **not** inflate the reported metrics - rare makes are harder, so a pooled figure computed over them is if anything pessimistic.
The problem they pose is a different one: the reported population would not be the served population.

Data validation
---------------

`validate-data` is the gate between preprocessing and the split, and it checks the raw and the interim frame in two layers ([EDN-68](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
Each frame is first read against its contract, then run through the Great Expectations suite `configure_gx` built for it.
A failure in either fails the stage, and because `split` depends on the stage's summary, the pipeline stops there.
A frame that breaks its contract is not handed to its suite, whose expectations assume the contract holds.
A suite with no expectations, or a run that returns fewer results than its suite holds, fails the stage too, so no suite can pass by checking nothing.

| Rule | Raw frame | Interim frame | Bounds from |
|---|---|---|---|
| Column list, dtypes and non-null columns | asserted | asserted | `schema.py`, generated |
| No PII column (NFR-08) and no `is_used` (EDN-23) | - | asserted, through the column list | `schema.py` |
| `registration_date` at or before the reference date (EDN-22) | tolerated up to `validate.raw_mostly` | hard | `reference_date` |
| `price` inside the training range | - | hard | `preprocess.price_min_eur`, `preprocess.price_max_eur` |
| `mileage_km_raw` inside the range FR-03 accepts | tolerated up to `validate.mileage_mostly` | tolerated up to `validate.mileage_mostly` | `validate.mileage_min_km`, `validate.mileage_max_km` |
| Filled wherever the raw contract declares the column filled | through the contract, which fails `download` on a gap | hard: a check on preprocessing, since a scrape gap never gets this far ([#78](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/78)) | `schema.py` |
| At least `validate.min_rows` rows, and `registration_date` and `mileage_km_raw` filled up to `validate.filled_mostly` | asserted | asserted | `params.yaml` |
| `has_full_service_history`, `non_smoking`, `is_rental` boolean and non-null (EDN-23) | through the contract | through the contract | `schema.py` |

The raw suite sees the published file with one change: `registration_date` is parsed by the interim contract's own cast, because Great Expectations 1.23 cannot bound a text date.
It has no price rule, because the published file runs from 1 EUR to 13.5M EUR and the range is a filter preprocessing applies.
The mileage range is the opposite case: a check and not a filter, so the published file's three readings above 1,000,000 km reach the cleaned frame, and the rule tolerates a few on both frames while still failing on a systematic break such as a scrape in another unit.
Two rules are deliberately absent.
That `is_used` agrees with `offer_type` would fail the published file by design, because a `False` there means "not asserted" (EDN-23).
The supported-make list is computed by `split`, after this stage, and applied by `features` (see [The supported makes](#the-supported-makes)), so no frame this stage sees is meant to satisfy it.
The docstring of `recommenditos/data/gx_context_configuration.py` gives each rule's reason.

What the stage writes, all under `reports/data-validation/`:

- `summary.json`, git-tracked: per frame whether it passed, its row count, how many contract expectations ran and which failed, and every rule with its bounds and what it found. It carries no run id and no timestamp, so its diff in a pull request is what the data did.
- `data-docs/`, the Data Docs rendered from the run. Open `data-docs/index.html` in a browser; `dvc pull reports/data-validation/data-docs` brings them to a clone.
- `results/`, the full validation results the Data Docs are rendered from.

`gx/` itself is `configure_gx`'s output, cached and pushed rather than committed.
Great Expectations writes a fresh UUID into every object it saves and ignores one passed in, so two builds from the same code differ in six of the nine files a context holds, and a committed store would be a diff on every rebuild.
The definition of the suites a reviewer reads is the module that builds them, and the store is rebuilt from scratch on every run, so it cannot keep a rule the module no longer defines.
To rebuild it by hand, run `uv run dvc repro configure_gx`.

On the real snapshot the stage takes 11 s and peaks at 2.2 GB of RAM, most of it the raw frame itself.
Great Expectations would take that to 14 s and 4.4 GB by hashing the whole frame into a fingerprint for each run's markers, so `validate-data` switches the fingerprint off (`reports/analysis/gx_fingerprint_cost.py`).

Parameters
----------

Everything configurable lives in `params.yaml`: the seed, the reference date, the scope bounds, the split ratios, the feature sets, the model variants and the thresholds behind SC-01 to SC-06.
Paths do not: they stay in `recommenditos/config.py`, derived from the repository root, so nothing in the project depends on a machine-specific absolute path.

This is a deliberate improvement on the demo, which keeps the same settings as Python constants in `config.py`.
A parameter change reruns the stages that read it, `dvc params diff` shows what changed between two commits, and `dvc exp` can sweep them.

From a run to its inputs
------------------------

Every MLflow run the pipeline makes names the committed state that produced it ([NFR-14](specification.md), EDN-74).
`train@<variant>` opens the variant's run and `evaluate` appends its metrics to it (EDN-55), and the tracking seam, `recommenditos/provenance.py`, tags it with what each stage was produced from:

| Tag | Set when | What it identifies |
|---|---|---|
| `git_commit`, `git_dirty` | `train` opens the run | The commit the fit ran at, and whether the working tree had changed since. |
| `train.deps.<path>` | `train` opens the run | Each dependency of `train@<variant>`, by the hash `dvc.lock` records for it under that stage. |
| `evaluate.git_commit`, `evaluate.git_dirty` | `evaluate` resumes the run | The same for the evaluation, which can run at a later commit than the fit: a change to `evaluate.py` alone reruns only `evaluate`. |
| `evaluate.deps.<path>` | `evaluate` resumes the run | Each dependency of `evaluate`, by the hash `dvc.lock` records for it under that stage. |

`git_dirty` does not count what `dvc repro` writes itself: `dvc.lock`, which DVC rewrites after every stage, and the outputs it does not cache, which git tracks instead, here `metrics.json`, `reports/metrics/` and `reports/data-validation/summary.json`.
So the runs of a clean commit reproduced with `dvc repro` are tagged clean, and `true` means a change somebody made, an untracked file included; `unknown` means git named the commit but could not read the tree's status.
The one write of DVC's that still counts is the line it adds to a `.gitignore` the first time a new cached output is produced, because that file is also the hand-written `.gitignore` of the repository root; commit the line, and the next run is clean.

The input hashes are computed through DVC's own API when the stage starts, and DVC records the same values in `dvc.lock` once the stage has finished: an MD5 for a file and an MD5 with a `.dir` suffix for a directory.
So for the runs of a committed `dvc repro`, every `train.deps.<path>` equals the `md5` the committed lock lists for `<path>` under `train@<variant>`, and every `evaluate.deps.<path>` the one under `evaluate`; `reports/analysis/run_provenance_check.py` checks exactly that against the tracking server, for the runs the committed models point at.
A run made any other way - by a test, or by calling a stage module by hand - carries the commit and no input hashes, because its arguments may point it at other files than `dvc.yaml` declares and no lock records what it read.
MLflow cannot remove a tag by setting the others, so when `evaluate` resumes a run a second time after a dependency was dropped from its `deps`, the dropped dependency's `evaluate.deps.<path>` tag stays on the run from the first time; the committed lock is the list to compare against, not the run's tags.

To get from a run back to what produced it:

1. **Code and parameters:** `git checkout` the run's `git_commit`, or `evaluate.git_commit` for the evaluation.
   With `git_dirty: false` the code there is what ran, and every code file has the hash its `deps` tag records, which is the plain `md5sum` of its bytes; `params.yaml` at that commit is what the stage read.
   The commit is the one the pipeline ran from, on the branch of its pull request, so after the squash merge it is reachable through the pull request (`git fetch origin pull/<n>/head`) rather than from `main`.
2. **Data:** a data dependency's hash is its address in the DVC cache and on the remote, so it identifies the input whatever happened to the commits.
   `git log --reverse --format=%H -S <hash> -- dvc.lock` lists the commits that added or removed the hash in the lock, oldest first; the first one added it, and its lock records the input.
   `git checkout` that commit and `dvc pull <path>` restores the file or directory.

Runs made before 2026-10-05 carry a `dvc_lock_md5` tag instead of the input hashes: the MD5 of `dvc.lock` as it stood when that stage started, which for the first stage of a `dvc repro` is the previous run's lock and for every later one a lock that only existed mid-run, so it is never the lock that records the run (EDN-74).

The test fixture
----------------

`recommenditos/data/synthetic.py` generates a schema-valid stand-in for the raw file.
It is **generated, never sampled**: the catalogues it draws from are the public make and model names and category value sets already published in the [dataset card](dataset-card.md).
The raw file is DVC-tracked and available (EDN-25), so this is not about availability; it is that a fixture is committed to Git, and a sample of real listings would put personal data in the repository rather than behind a DVC pointer (NFR-08).

Every edge case the pipeline rules exist for is guaranteed present whatever the row count, so a stage test can assert against it rather than hope for it: a listing registered after the snapshot, a duplicate on the deduplication key, an `ES` listing, a make far below the support threshold, prices outside the training range at both ends, a new car, a pre-registered car, a transporter, a missing registration date, a private seller, and a used car whose `is_used` flag says otherwise.

It also keeps the real make skew and gives each seller several listings, because a seller-grouped split on a flat distribution would be indistinguishable from a random one.

The tests build the frame in memory, so `pytest` writes nothing and no file can go stale.
When something needs the fixture as an actual file - a notebook is easier to poke at with one - `make fixture` writes it to `data/fixture/listings_fixture.parquet`.
That path is gitignored: the generator is the artefact, the Parquet is a convenience, and both come from the same seeded function.

Working on a stage
------------------

1. Read the contract your stage writes in `schema.py`, and `params.yaml` for the values your stage may not hard-code.
2. Replace the stub body. Keep the module's public functions, because other stages and, later, the API import them: `hash_seller_group`, `assign_split`, `age_years`, `point_metrics`.
   The API image installs only the serving runtime, so a module the API imports may only import what that runtime provides; [Contributing](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/CONTRIBUTING.md#dependencies) lists the modules held to it.
   `point_metrics` is not in one yet: `evaluate.py` imports MLflow through `tracking.py`, so it has to move before the API can use it.
3. If your stage changes a column's type or replaces a column with something derived from it, say so on the schema rather than editing `schema.py`: `schema.with_dtype("weight_kg", "float64")` for a parsed column, `schema.drop([...]).extend([...])` for the equipment multi-hot columns. That keeps the change inside your module.
4. Every stage takes its parameters file as an argument, so a test can vary a parameter without patching anything.
5. Add tests against the fixture: a module of its own once the stage is more than a stub, as `preprocess` has `tests/test_preprocess.py`, otherwise `tests/test_data.py` or `tests/test_model.py`. Mark the requirement IDs a test verifies with `@pytest.mark.req("NFR-08")`.
6. Run `make lint`, `make test` and `dvc repro`, then commit `dvc.lock` and run `dvc push`. `dvc repro` runs on the real snapshot, so run `dvc pull` first and the stages your change does not touch stay up to date; a stage you do touch pays the budget above. Set `download.source` to `synthetic` while iterating if you have no credentials, and set it back before you commit a lock, because a lock built from the stand-in is not the one the project ships.

`dvc.yaml` is hand-edited on purpose.
`dvc stage add` rewrites the whole file, re-indents every list and line-wraps long commands, which would destroy the comments that explain why each stage is wired the way it is.

Known gaps
----------

- A stage's `deps` and `params` cannot be conditional, so `download` declares both sources' inputs whichever one is selected. Under `download.source: zenodo`, editing `recommenditos/data/synthetic.py` or `download.rows` therefore reruns the stage as a 25-second re-read of the cached CSV that produces an identical Parquet. Dropping either would be worse, because a synthetic run would then not notice that its own generator or row count changed.
- The fixture's make distribution is the real one, but scaled down: at 2,000 rows only three makes clear the 300-listing support threshold, and at 20,000 rows seven do. A test about supported makes should set the threshold it wants rather than relying on the project's.
- Neither source's output is byte-stable across a toolchain bump, so `dvc.lock` is only reproducible within one. The project's own source, `zenodo`, is pinned hard on the input and not all the way on the output: `download.md5` pins the *input* CSV, while the Parquet the stage writes embeds the pyarrow version and the pandas type metadata, so a pyarrow or pandas bump changes the output hash and invalidates the lock for everyone even though the data is identical. The `synthetic` source is reproducible within a fixed toolchain too, but it depends on one thing more: NumPy makes no promise that `default_rng` produces the same stream across releases, so a NumPy upgrade changes what it generates. Within one toolchain version both writes are byte-stable, which is what NFR-06 is measured against, and `dvc pull` sidesteps the question entirely for everyone who does not rerun the stage.
