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
On a scratch clone of this commit, `dvc pull` fetched 34 files and added 32, and `dvc repro` then reported all twelve stages as `didn't change, skipping` and finished with `Data and pipelines are up to date.`
`configure_gx` is skipped along with the rest, although it declares no `outs`: its `deps` and its lock entry are enough for DVC to answer, so the missing output costs ordering guarantees (see [Known gaps](#known-gaps)) and not an unconditional rerun.
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
| `configure_gx` | `recommenditos/data/gx_context_configuration.py` | the Great Expectations context |
| `validate-data` | `recommenditos/data/validate_data.py` | nothing; it fails the pipeline instead |
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
Value rules - price ranges, fill rates, the supported make list - are **data quality** and live in the Great Expectations suites, because the two answer different questions.
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
Of the 6,079 `ES` holdout rows it keeps 5,979, which is EDN-14's count of the `ES` rows the API accepts under FR-04, so the M6 replay is exactly the traffic NFR-11 was measured on; the 100 it drops are requests the API refuses before the model sees them.
The holdout's `country_code` stays empty in every one of those rows, because `ES` is still not a training level ([EDN-42](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)); the filter removes rows and leaves that encoding as it was.
The 1,437 out-of-scope rows would **not** inflate the reported metrics - rare makes are harder, so a pooled figure computed over them is if anything pessimistic.
The problem they pose is a different one: the reported population would not be the served population.

Parameters
----------

Everything configurable lives in `params.yaml`: the seed, the reference date, the scope bounds, the split ratios, the feature sets, the model variants and the thresholds behind SC-01 to SC-06.
Paths do not: they stay in `recommenditos/config.py`, derived from the repository root, so nothing in the project depends on a machine-specific absolute path.

This is a deliberate improvement on the demo, which keeps the same settings as Python constants in `config.py`.
A parameter change reruns the stages that read it, `dvc params diff` shows what changed between two commits, and `dvc exp` can sweep them.

The test fixture
----------------

`recommenditos/data/synthetic.py` generates a schema-valid stand-in for the raw file.
It is **generated, never sampled**: the catalogues it draws from are the public make and model names and category value sets already published in the [dataset card](dataset-card.md).
The raw file is DVC-tracked and available (EDN-25), so this is not about availability; it is that a fixture is committed to Git, and a sample of real listings would put personal data in the repository rather than behind a DVC pointer (NFR-08).

Every edge case the pipeline rules exist for is guaranteed present whatever the row count, so a stage test can assert against it rather than hope for it: a listing registered after the snapshot, a duplicate on the deduplication key, an `ES` listing, a make far below the support threshold, prices outside the training range at both ends, a new car, a pre-registered car, a transporter, a missing registration date, a private seller, and a used car whose `is_used` flag says otherwise.

It also keeps the real make skew and gives each seller several listings, because a seller-grouped split on a flat distribution would be indistinguishable from a random one.

The tests build the frame in memory, so `pytest` writes nothing and no file can go stale.
When something needs the fixture as an actual file - a Great Expectations asset points at a path, and a notebook is easier to poke at with one - `make fixture` writes it to `data/fixture/listings_fixture.parquet`.
That path is gitignored: the generator is the artefact, the Parquet is a convenience, and both come from the same seeded function.

Working on a stage
------------------

1. Read the contract your stage writes in `schema.py`, and `params.yaml` for the values your stage may not hard-code.
2. Replace the stub body. Keep the module's public functions, because other stages and, later, the API import them: `hash_seller_group`, `assign_split`, `age_years`, `point_metrics`.
3. If your stage changes a column's type or replaces a column with something derived from it, say so on the schema rather than editing `schema.py`: `schema.with_dtype("weight_kg", "float64")` for a parsed column, `schema.drop([...]).extend([...])` for the equipment multi-hot columns. That keeps the change inside your module.
4. Every stage takes its parameters file as an argument, so a test can vary a parameter without patching anything.
5. Add tests against the fixture: a module of its own once the stage is more than a stub, as `preprocess` has `tests/test_preprocess.py`, otherwise `tests/test_data.py` or `tests/test_model.py`. Mark the requirement IDs a test verifies with `@pytest.mark.req("NFR-08")`.
6. Run `make lint`, `make test` and `dvc repro`, then commit `dvc.lock` and run `dvc push`. `dvc repro` runs on the real snapshot, so run `dvc pull` first and the stages your change does not touch stay up to date; a stage you do touch pays the budget above. Set `download.source` to `synthetic` while iterating if you have no credentials, and set it back before you commit a lock, because a lock built from the stand-in is not the one the project ships.

`dvc.yaml` is hand-edited on purpose.
`dvc stage add` rewrites the whole file, re-indents every list and line-wraps long commands, which would destroy the comments that explain why each stage is wired the way it is.

Known gaps
----------

- `configure_gx` declares no `outs`, so it is a disconnected node in the graph and nothing forces it to run before `validate-data`. The demo has the same wart. Whoever implements the Great Expectations context should give the stage an output and make `validate-data` depend on it, the way `split` now depends on `validate-data`.
- A stage's `deps` and `params` cannot be conditional, so `download` declares both sources' inputs whichever one is selected. Under `download.source: zenodo`, editing `recommenditos/data/synthetic.py` or `download.rows` therefore reruns the stage as a 25-second re-read of the cached CSV that produces an identical Parquet. Dropping either would be worse, because a synthetic run would then not notice that its own generator or row count changed.
- The fixture's make distribution is the real one, but scaled down: at 2,000 rows only three makes clear the 300-listing support threshold, and at 20,000 rows seven do. A test about supported makes should set the threshold it wants rather than relying on the project's.
- Neither source's output is byte-stable across a toolchain bump, so `dvc.lock` is only reproducible within one. The project's own source, `zenodo`, is pinned hard on the input and not all the way on the output: `download.md5` pins the *input* CSV, while the Parquet the stage writes embeds the pyarrow version and the pandas type metadata, so a pyarrow or pandas bump changes the output hash and invalidates the lock for everyone even though the data is identical. The `synthetic` source is reproducible within a fixed toolchain too, but it depends on one thing more: NumPy makes no promise that `default_rng` produces the same stream across releases, so a NumPy upgrade changes what it generates. Within one toolchain version both writes are byte-stable, which is what NFR-06 is measured against, and `dvc pull` sidesteps the question entirely for everyone who does not rerun the stage.
- `configure_gx` and `validate-data` are still stubs: `validate-data` enforces the structural contract from `recommenditos/schema.py` and the value expectations of issue #25 are not built yet. Each module's docstring names the issue that implements it and what that issue still owes.
