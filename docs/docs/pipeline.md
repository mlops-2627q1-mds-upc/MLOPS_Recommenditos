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

`download.source` in `params.yaml` is `synthetic`, so `dvc repro` builds the pipeline from a generated stand-in and needs neither the network nor DagsHub access.
Setting it to `zenodo` swaps in the real snapshot: the `download` stage fetches the pinned Zenodo file itself, checks it against `download.md5` and converts it to Parquet.
**[decided]**, not yet done: `zenodo` becomes the default together with the `dvc.lock` refresh and the `dvc push` at the end of the stage chain ([#57](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/57)), because one consolidation records one real end-to-end run instead of four half-real ones.

Because the source is a parameter, `dvc.lock` records which one every artefact was built from, so a run on synthetic data cannot pass unnoticed: `dvc status` reports a workspace whose `download.source` disagrees with the lock.
`dvc params diff` does not show this, because it compares the params files of two Git revisions rather than the lock against the workspace.

What `download.source: zenodo` costs, measured on the real file: about 1 GB of RAM (peak RSS 929 MB, because the frame is read and validated whole) and about 1 GB of disk, being the 548 MB CSV cached in `data/external/`, the 215 MB Parquet and another 215 MB for its copy in the DVC cache.
The stage takes about 90 s with a cold cache and about 66 s with a warm one, because the CSV is downloaded once and reused from there, so a rerun costs a 25-second re-read rather than a download (see [Data versioning](data-versioning.md)).
CI never runs the stage; the test suite covers it against fixtures instead.

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
| `evaluate` | `recommenditos/modeling/evaluate.py` | `metrics.json`, `reports/metrics/<variant>.json` |

`features` and `evaluate` have no equivalent in the [course demo](https://github.com/mlops-2627q1-mds-upc/MLOps-2627q1-demos).
Its text model needs no feature engineering, and it asserts its metric threshold inside `tests/test_model.py` instead of producing a metrics artefact, which is not enough for the gate [NFR-01](specification.md) describes.

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
While `download.source` is `synthetic` the pushed Parquet is the generated stand-in and holds no real personal data; from the flip in [#57](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/57) onward it holds the published file's, which is the exposure EDN-34 weighed.

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
6. Run `make lint`, `make test` and `dvc repro`, then commit `dvc.lock` and run `dvc push`. `dvc repro` runs on the synthetic stand-in, so it needs no credentials and downloads nothing; switch `download.source` to `zenodo` only when you mean to pay the budget above.

`dvc.yaml` is hand-edited on purpose.
`dvc stage add` rewrites the whole file, re-indents every list and line-wraps long commands, which would destroy the comments that explain why each stage is wired the way it is.

Known gaps
----------

- `configure_gx` declares no `outs`, so it is a disconnected node in the graph and nothing forces it to run before `validate-data`. The demo has the same wart. Whoever implements the Great Expectations context should give the stage an output and make `validate-data` depend on it, the way `split` now depends on `validate-data`.
- `data/processed/supported_makes.json` is read by `features` but **not applied** by it yet.
  The stage declares it as a dependency and reads it before the vocabulary is built, which is the only point at which the restriction could still decide the level set, so changing `split.min_listings_per_make` reruns the matrices, the models and the metrics.
  What is still missing is the restriction itself, in all three of `features`, `train` and `evaluate` ([EDN-48](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md)).
  `split` deliberately records the scope instead of enforcing it: it stays a lossless partition, so EDN-05's threshold can be revisited without re-running the split, and no rows disappear without an artefact saying where they went.
  Four things the stages that turn a set into model input (#36, #37, #39) have to guarantee, none of them optional:

    1. The filter is applied to train, validation, calibration, test **and** the `ES` holdout.
       The holdout is the one people forget, and it is the set NFR-11 and FR-15 rest on.
    2. The filter is applied **before** any training-derived vocabulary or statistic is computed - the equipment multi-hot columns, a category encoding, a mean, a quantile.
       Fitting on rows the API will refuse puts makes the product cannot serve into the model's own inputs.
    3. `evaluate`'s per-make segments are restricted to the listed makes, or SC-04 reports a segment the API answers with a 422.
    4. A test asserts that the filtered frames contain only supported makes, so the guarantee is checked rather than intended.

    Measured on the real snapshot: the filter moves every realised share by at most 0.08 pp (99,326 rows to 97,889), so it does not disturb the split proportions.
    Of the 6,079 holdout rows, 5,979 are a supported make and 100 are not, which matches EDN-14's count of the `ES` rows the API would accept under FR-04.
    The 1,437 out-of-scope rows would **not** inflate the reported metrics - rare makes are harder, so a pooled figure computed over them is if anything pessimistic.
    The problem is a different one: the reported population would not be the served population, and it would not be comparable to the reference values in problem-spec section 8, which are all post-filter.
- A stage's `deps` and `params` cannot be conditional, so `download` declares both sources' inputs whichever one is selected. Under `download.source: zenodo`, editing `recommenditos/data/synthetic.py` or `download.rows` therefore reruns the stage as a 25-second re-read of the cached CSV that produces an identical Parquet. Dropping either would be worse, because a synthetic run would then not notice that its own generator or row count changed.
- The fixture's make distribution is the real one, but scaled down: at 2,000 rows only three makes clear the 300-listing support threshold, and at 20,000 rows seven do. A test about supported makes should set the threshold it wants rather than relying on the project's.
- Neither source's output is byte-stable across a toolchain bump, so `dvc.lock` is only reproducible within one. The synthetic data is reproducible within a fixed toolchain, but NumPy makes no promise that `default_rng` produces the same stream across releases, so a NumPy upgrade changes what `download.source: synthetic` generates. `zenodo` is pinned harder but not all the way: `download.md5` pins the *input* CSV, while the Parquet the stage writes embeds the pyarrow version and the pandas type metadata, so a pyarrow or pandas bump changes the output hash and invalidates the lock for everyone even though the data is identical. Within one toolchain version the write is byte-stable, which is what NFR-06 is measured against.
- The stage bodies are stubs apart from `download` and `preprocess`. Each module's docstring names the issue that implements it and what that issue still owes.
