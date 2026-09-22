Data versioning with DVC
=========================

This applies to every contributor to this repo, human or AI agent.

For the general DVC tutorial (install, pipeline stages, commands), follow the instructor-recommended
[DVC demo](https://github.com/mlops-2627q1-mds-upc/MLOps-2627q1-demos/blob/main/docs/dvc-demo.md).
This page only documents the conventions we've settled on for this project, on top of that demo.

## Remote

We use DagsHub Storage as the DVC remote, configured over **HTTP** (not S3), as the demo prescribes.
Setup status is tracked in [issue #10](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/issues/10).

Credentials never go into a commit.
Each person configures their own token locally:

```bash
dvc remote modify origin --local access_key_id <token>
dvc remote modify origin --local secret_access_key <token>
```

This writes to `.dvc/config.local`, which is gitignored by DVC itself.

## Tracking granularity

Track individual files, or a self-contained dataset directory made up of many small files (for example
thousands of images that always belong together).
Do not `dvc add` the whole `data/` tree, and do not `dvc add` a whole subfolder like `data/raw` unless
it truly is one indivisible dataset.

```bash
# Good
dvc add data/raw/cars.csv

# Avoid
dvc add data
dvc add data/raw
```

Why:

- `data/raw`, `data/interim`, `data/processed`, and `data/external` have different lifecycles and
  provenance. A single pointer covering all of them hides which artifact actually changed.
- DVC directories are represented as one hash for the whole tree. Any change inside means the pointer
  changes and needs to be re-added and re-pushed, even though DVC still hashes and caches each file
  individually under the hood, so no efficiency is gained by tracking coarsely.
- It leaves room for the pipeline (see below) to own `interim`/`processed` without conflicting with a
  manually tracked directory. DVC refuses to track a directory that already contains a tracked path,
  and vice versa.

## Pipeline ownership

Once a `dvc.yaml` pipeline exists (separate issue, follows once the training code does), stage outputs
are tracked automatically by the pipeline, not by a manual `dvc add`.
In practice that means:

- `data/raw` (or whatever the download stage produces): tracked manually now via `dvc add`, or later
  produced by a `download` stage, per the demo.
- `data/interim`, `data/processed`, `models/`: once a stage declares them as `-o` outputs, don't
  `dvc add` them separately. Let `dvc repro` manage them.

## `.gitignore`

Don't add blanket rules like `/data/` to `.gitignore`. DVC generates its own `.gitignore` entries next
to each tracked path, and Git still needs to see the `.dvc` pointer files themselves.

## Day to day

```bash
git pull && dvc pull   # after pulling, get the data that matches this commit
# ... make changes ...
dvc push && git push   # push data before (or together with) the commit that references it
```

### Commit the pointer, push the data - always both

`git push` alone is not enough.
After `dvc add` (creates/updates a `.dvc` file) or `dvc repro` (creates/updates `dvc.lock`):

1. Commit the pointer file (`*.dvc` or `dvc.lock`) to Git.
2. Run `dvc push` so the actual data reaches the DagsHub remote.

If you skip step 2, the commit still looks fine in GitHub, but the data was never uploaded.
Anyone else running `git pull && dvc pull` gets a "failed to pull data" error and is stuck, because the
hash the pointer references does not exist on the remote yet.
If you skip step 1, `dvc push` has nothing tracked to upload, and nobody sees a new pointer to pull in
the first place.
Neither step alone is enough to hand data off to the rest of the team.
