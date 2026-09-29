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

`dvc repro` needs no credentials and no download while `download.source` in `params.yaml` is `synthetic`.
That is deliberate: the stage tickets are built in parallel, and each one needs a local oracle it can run without waiting for the 548 MB raw file or for DagsHub access.
Once the download stage is implemented, the parameter flips to `zenodo` and the pipeline runs on the real snapshot.

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

Both halves exist because Parquet preserves whatever dtype it is handed rather than normalising it.
Without an explicit cast at each boundary the frames drift apart stage by stage, and the stage that notices is never the stage that caused it.

Three contracts are defined:

- `RAW_SCHEMA`, the 75 columns of the AutoScout24 snapshot as published.
- `INTERIM_SCHEMA`, what `preprocess` writes: row-filtered, PII-free, deduplicated, carrying `log_price` and the hashed `seller_group_id`, but not yet feature-engineered.
- `PROCESSED_SCHEMA`, what `split` writes once per set. The split moves rows between files; it does not change columns.

Feature matrices depend on `features.sets` in `params.yaml`, so they come from `feature_schema(columns, name=...)` rather than from a constant.
A feature set naming a column no stage produces fails there, with the offending name, instead of producing a matrix that is quietly missing a column.

Parameters
----------

Everything configurable lives in `params.yaml`: the seed, the reference date, the scope bounds, the split ratios, the feature sets, the model variants and the thresholds behind SC-01 to SC-06.
Paths do not: they stay in `recommenditos/config.py`, derived from the repository root, so nothing in the project depends on a machine-specific absolute path.

This is a deliberate improvement on the demo, which keeps the same settings as Python constants in `config.py`.
A parameter change reruns the stages that read it, `dvc params diff` shows what changed between two commits, and `dvc exp` can sweep them.

The test fixture
----------------

`recommenditos/data/synthetic.py` generates a schema-valid stand-in for the raw file.
It is **generated, never sampled**: the raw file carries PII, and the catalogues it draws from are the public make and model names and category value sets already published in the [dataset card](dataset-card.md).

Every edge case the pipeline rules exist for is guaranteed present whatever the row count, so a stage test can assert against it rather than hope for it: a listing registered after the snapshot, a duplicate on the deduplication key, an `ES` listing, a make far below the support threshold, prices outside the training range at both ends, a new car, a pre-registered car, a transporter, a missing registration date, a private seller, and a used car whose `is_used` flag says otherwise.

It also keeps the real make skew and gives each seller several listings, because a seller-grouped split on a flat distribution would be indistinguishable from a random one.

The tests build the frame in memory, so `pytest` writes nothing and no file can go stale.
When something needs the fixture as an actual file - a Great Expectations asset points at a path, and a notebook is easier to poke at with one - `make fixture` writes it to `data/fixture/listings_fixture.parquet`.
That path is gitignored: the generator is the artefact, the Parquet is a convenience, and both come from the same seeded function.

Working on a stage
------------------

1. Read the contract your stage writes in `schema.py`, and `params.yaml` for the values your stage may not hard-code.
2. Replace the stub body. Keep the module's public functions, because other stages and, later, the API import them: `hash_seller_group`, `assign_split`, `age_years`, `point_metrics`.
3. Every stage takes its parameters file as an argument, so a test can vary a parameter without patching anything.
4. Add tests to `tests/test_data.py` or `tests/test_model.py` against the fixture. Mark the requirement IDs a test verifies with `@pytest.mark.req("NFR-08")`.
5. Run `make lint`, `make test` and `dvc repro`, then commit `dvc.lock` and run `dvc push`.

`dvc.yaml` is hand-edited on purpose.
`dvc stage add` rewrites the whole file, re-indents every list and line-wraps long commands, which would destroy the comments that explain why each stage is wired the way it is.

Known gaps
----------

- `configure_gx` declares no `outs`, so it is a disconnected node in the graph and DVC does not guarantee it runs before `validate-data`. The demo has the same wart. Whoever implements the Great Expectations context should give the stage an output and make `validate-data` depend on it.
- `data/raw/autoscout24_dataset_20251108.csv.dvc` is still a manual `dvc add` pointer. It is replaced by the `download` stage once that stage fetches the real file, as [Data versioning](data-versioning.md) describes.
- The stage bodies are stubs. Each module's docstring names the issue that implements it and what that issue still owes.
