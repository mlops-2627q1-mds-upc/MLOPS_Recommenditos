Data versioning with DVC
=========================

This applies to every contributor to this repo, human or AI agent.

For the general DVC tutorial (install, pipeline stages, commands), follow the instructor-recommended
[DVC demo](https://github.com/mlops-2627q1-mds-upc/MLOps-2627q1-demos/blob/main/docs/dvc-demo.md).
This page only documents the conventions we've settled on for this project, on top of that demo.

## Remote

We use DagsHub Storage as the DVC remote, configured over **HTTP** (not S3), as the demo prescribes.
The DagsHub repo is <https://dagshub.com/recommenditos/MLOPS_Recommenditos>.
It belongs to the `recommenditos` DagsHub organisation, which the team owns, so the remote does not depend on any one person's private account.
It is connected through DagsHub's GitHub integration, not as a plain git mirror, which is why issues and pull requests show up on DagsHub and why it syncs by webhook instead of polling.

The shared part of the configuration is committed in `.dvc/config`:

```ini
[core]
    remote = origin
['remote "origin"']
    url = https://dagshub.com/recommenditos/MLOPS_Recommenditos.dvc
    auth = basic
```

### First-time setup

1. Create a [DagsHub](https://dagshub.com) account and ask @lukas2510 to add you to the `recommenditos`
   organisation with write access, otherwise `dvc push` is rejected.
   The repository is public (EDN-20), so `dvc pull` works for any signed-in DagsHub user; only pushing needs membership.
2. Install the project environment, which includes DVC:

    ```bash
    uv sync
    ```

3. Copy your token from DagsHub (profile picture → **Settings → Tokens**) and store your credentials locally:

    ```bash
    uv run dvc remote modify origin --local user <your-dagshub-username>
    uv run dvc remote modify origin --local password <your-dagshub-token>
    ```

4. Check that it works:

    ```bash
    uv run dvc pull
    ```

    This downloads the tracked data (see [Tracked data](#tracked-data)) into `data/`.
    A 401 or 403 error means the username, the token or the collaborator access is wrong.

5. If you will push large files, raise the HTTP timeouts, otherwise the upload fails with
   "Timeout on reading data from socket":

    ```bash
    uv run dvc remote modify origin --local read_timeout 1800
    uv run dvc remote modify origin --local connect_timeout 120
    ```

Credentials never go into a commit.
The commands in step 3 write to `.dvc/config.local`, which DVC's own `.dvc/.gitignore` keeps out of Git.
Never run them without `--local`: that would write the token into the committed `.dvc/config`.

## Tracked data

| File | Source | MD5 | Tracked by |
|------|--------|-----|------------|
| `data/raw/autoscout24_dataset_20251108.csv` (548.6 MB) | [Zenodo record 17643343](https://zenodo.org/records/17643343), DOI `10.5281/zenodo.17643343`, v1.0.0, MIT in the structured field, though the record's own prose reads narrower | `b23a122cc51baf7de39f449193ff0d28` | `dvc add` (EDN-25) |

The MD5 of the raw file equals the checksum Zenodo publishes, so anyone can verify that our copy is the original.

## Raw data

**[decided]**, [EDN-07](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md) as amended by EDN-25.
The raw AutoScout24 file contains PII (street, zip, exact coordinates, seller company name for private sellers, see [Requirements](requirements.md) NFR-08), which preprocessing removes before anything else is tracked.
The file is downloaded once from its Zenodo DOI, tracked with `dvc add` and pushed to our remote, so a teammate gets it with `dvc pull` like any other input.

EDN-07 had proposed `dvc import-url` with `push: false`, to keep the raw PII off our own remote.
EDN-25 replaced that: the course demo teaches `dvc add`, and `dvc import-url` derives its change detection from an `ETag` or `Content-MD5` header that the Zenodo URL does not send, so the mechanism was never verified against this source.

The exact commands are set by the work on issue #3; this page only records which mechanism applies and why.
A job that needs only some outputs, like the CI build that bakes the model into the API image (FR-12), pulls them by target (`dvc pull <target>`) and never needs the raw file.

## Tracking granularity

Track individual files, or a self-contained dataset directory made up of many small files (for example
thousands of images that always belong together).
Do not `dvc add` the whole `data/` tree, and do not `dvc add` a whole subfolder like `data/raw` unless
it truly is one indivisible dataset.

```bash
# Good
dvc add data/raw/autoscout24_dataset_20251108.csv

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

The `dvc.yaml` pipeline exists (see [The DVC pipeline](pipeline.md)), so stage outputs are tracked
automatically by the pipeline, not by a manual `dvc add`.
In practice that means:

- `data/raw`: the `download` stage owns `data/raw/listings.parquet`. The raw CSV is still tracked
  manually with `dvc add` (see [Raw data](#raw-data)); that pointer is replaced by the stage once
  the stage fetches the real file rather than generating a synthetic stand-in.
- `data/interim`, `data/processed`, `models/`: once a stage declares them as `-o` outputs, don't
  `dvc add` them separately. Let `dvc repro` manage them.

## `.gitignore`

Keep `data/` ignored, but let the pointers through. The root `.gitignore` does this:

```gitignore
/data/**
!/data/**/
!/data/**/*.dvc
!/data/**/.gitignore
```

A bare `/data/` rule is not enough, because it would also hide the `.dvc` pointer files Git has to see,
and Git cannot re-include a file inside an ignored directory - hence the `**` form plus the directory
negation. The last line keeps the `.gitignore` files DVC generates next to each tracked path
committable; without it DVC's own entries would be ignored and could never reach a commit.

The rule matters most before `dvc add` runs. A raw file dropped into `data/raw/` to look at it is
covered from the moment it lands, so a stray `git add .` cannot put it - or its personal data - into a
repository the whole cohort can read.

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
