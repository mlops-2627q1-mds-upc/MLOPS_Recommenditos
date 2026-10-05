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
The repository is public (EDN-20), so `dvc pull` works for any signed-in DagsHub account; `dvc push` needs write access on it, which @lukas2510 grants.

The shared part of the configuration is committed in `.dvc/config`:

```ini
[core]
    remote = origin
['remote "origin"']
    url = https://dagshub.com/recommenditos/MLOPS_Recommenditos.dvc
    auth = basic
```

### First-time setup

[Getting started](getting-started.md) walks a fresh clone through the whole path: the DagsHub token,
`.env`, the `dvc remote modify --local` commands and the first `dvc pull`.
It is not repeated here, so that there is one place to fix when it changes.
Two DVC-specific points belong with the conventions rather than with the walkthrough.

**Never run `dvc remote modify` without `--local`.**
With `--local` the credentials land in `.dvc/config.local`, which DVC's own `.dvc/.gitignore` keeps
out of Git. Without it, they go into the committed `.dvc/config` and the next `git push` publishes
your token.

**Raise the HTTP timeouts before you push a large file**, otherwise the upload fails with
"Timeout on reading data from socket":

```bash
uv run dvc remote modify origin --local read_timeout 1800
uv run dvc remote modify origin --local connect_timeout 120
```

## Tracked data

Everything DVC tracks in this repository is a pipeline output.
There is no manual `dvc add` pointer left.

| Artefact | Written by | Held where |
|------|--------|-----|
| `data/raw/listings.parquet` | `download` | DVC cache, pushed |
| `data/interim/`, `data/processed/`, `models/` | the stages that declare them as outputs | DVC cache, pushed |
| `gx/`, the Great Expectations context | `configure_gx` | DVC cache, pushed; gitignored, because every rebuild writes fresh UUIDs into it (EDN-68) |
| `reports/data-validation/results/`, `reports/data-validation/data-docs/` | `validate-data` | DVC cache, pushed; both carry the run's timestamp |
| `reports/data-validation/summary.json`, `reports/metrics/`, `metrics.json` | `validate-data` and `evaluate` | `cache: false`, so Git, and they show up in a pull request's diff |

What the committed `dvc.lock` names for `data/raw/listings.parquet` is the real snapshot, 215,309,993 bytes over 118,382 rows, cached and pushed like any other stage output (**[decided]**, EDN-34).
It keeps the PII columns of the published file, because preprocessing needs `seller_company_name` and the location to build the split's group key before dropping them.
So the remote holds that personal data in the Parquet, even though we no longer re-host the CSV.
The alternatives were weighed in EDN-34: keeping the Parquet local with `push: false` would have moved the pipeline's first artefact from pulled to locally regenerated, and stripping the PII inside `download` would have made the raw layer stop being a faithful copy of the published file.

## Raw data

**[decided]**, [EDN-07](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/reports/edn.md), amended by EDN-25 and amended again by EDN-35 and EDN-36 for the `download` stage of issue #33.
The raw AutoScout24 file contains PII (street, zip, exact coordinates, seller company name for private sellers, see [Requirements](requirements.md) NFR-08), which preprocessing removes before anything else is tracked.

Under `download.source: zenodo` the stage acquires it, so acquisition is part of the pipeline rather
than a manual step beside it:

| | |
|---|---|
| Source | [Zenodo record 17643343](https://zenodo.org/records/17643343), DOI `10.5281/zenodo.17643343`, v1.0.0, file `autoscout24_dataset_20251108.csv`, 548.6 MB |
| Licence | MIT in the structured field, though the record's own prose reads narrower |
| Pinned by | `download.md5` in `params.yaml`, `b23a122cc51baf7de39f449193ff0d28`, which is the checksum Zenodo publishes |
| Kept at | `data/external/autoscout24_dataset_20251108.csv`, a local cache: gitignored, not tracked, not pushed |

`download.source` is `zenodo`, so the stage is the acquisition.
A clean clone fetches nothing all the same: `dvc pull` brings the derived Parquet down and leaves the stage up to date, so the CSV is only fetched by someone who reruns `download` itself.
Setting the parameter to `synthetic` generates a stand-in instead, which is what lets the test suite and a clone without credentials run.

The CSV is neither a `dep` nor an `out` of the stage (**[decided]**, EDN-35).
DVC deletes a stage's outputs before running it, so a plain `out` would re-download 548 MB on every `dvc repro download`, and a `dep` would make every `dvc status` hash 548 MB before it can answer.
A third shape does exist and EDN-35 weighs it: an `out` with `persist: true` and `cache: false` survives the run that produces it and never reaches the remote, which would make `dvc status` honest about the CSV, but it pays exactly the same 548 MB hash the `dep` pays.
`download.md5` does the job instead: the stage refuses to read bytes that hash to anything else, and because the MD5 is a *parameter*, re-pinning the file is a change DVC sees and reruns on.
A local copy that no longer matches is reported as such rather than used, so a changed upstream file fails the stage instead of quietly retraining the model.

The cache lives under `data/external/` because that is the third-party slot of the project layout, which leaves `data/raw/` to the pipeline alone.
Delete the file to force a fresh download; nothing else depends on it being there.
Two things to know when a fetch goes wrong:

- A failed transfer takes its own `<filename>.part` file with it, because nothing here resumes a
  transfer. A process killed outright cannot do that, so a stray `.part` file beside the cache is read
  by nothing and is safe to delete.
- A cached file that does not match `download.md5` fails the stage on every run until somebody deletes
  it. That is on purpose: it may be a damaged local copy or a genuinely changed upstream file, the two
  need different answers, and the error message names the choice rather than throwing data away on a
  guess.

EDN-07 had proposed `dvc import-url` with `push: false`, to keep the raw PII off our own remote.
EDN-25 replaced that with `dvc add` plus a push to our remote, because the course demo teaches `dvc add` and `dvc import-url` derives its change detection from an `ETag` or `Content-MD5` header that the Zenodo URL does not send.
That second point still holds, and the stage measured it: Zenodo serves this file chunked and sends no `Content-Length` either.
What EDN-36 changed is who acquires the file.
The stage does, so the `dvc add` pointer is gone and the CSV is no longer tracked or pushed by us; the Parquet derived from it is, which is what EDN-34 weighed.
The blob the old pointer referenced is left on the remote untouched, so checking out an earlier commit and running `dvc pull` still works.
No `.dvc` file at the tip of `main` references it any more, though, so recovering that exact snapshot without Zenodo means reading the hash out of the Git history of the deleted pointer file, and it survives only as long as nobody runs the wrong `dvc gc` (see [Never run `dvc gc` without `--all-commits`](#never-run-dvc-gc-without-all-commits)).

A job that needs only some outputs, like the CI build that bakes the model into the API image (FR-12), pulls them by target (`dvc pull <target>`) and never needs the raw file at all.

## Tracking granularity

Track individual files, or a self-contained dataset directory made up of many small files (for example
thousands of images that always belong together).
Do not `dvc add` the whole `data/` tree, and do not `dvc add` a whole subfolder like `data/raw` unless
it truly is one indivisible dataset.

```bash
# Good
dvc add data/external/some_third_party_table.csv

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

- `data/raw`: the `download` stage owns it, output and acquisition both. It writes
  `data/raw/listings.parquet` and fetches the CSV it derives that from (see
  [Raw data](#raw-data)), so nothing here is tracked by hand any more.
- `data/interim`, `data/processed`, `models/`: once a stage declares them as `-o` outputs, don't
  `dvc add` them separately. Let `dvc repro` manage them.
- `data/external`: not tracked at all today. It holds the `download` stage's local copy of the
  published CSV, which is reproducible from the DOI and the pinned MD5, so there is nothing for DVC
  to version.

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

### Never run `dvc gc` without `--all-commits`

`dvc gc` prunes the local cache and `dvc gc --cloud` prunes the DagsHub remote.
Both default to **workspace** scope: they keep only what the checked-out `dvc.lock` and `.dvc` files
reference and delete everything else, including every artefact an older commit's lock still points at.

That is not a theoretical loss here. The 548.6 MB raw CSV is still on the remote and in the local cache
from when a `.dvc` pointer tracked it, and nothing at the tip of `main` references it any more (see
[Raw data](#raw-data)). A workspace-scoped `dvc gc` deletes it, and with it the only copy that does not
depend on Zenodo staying online. So if the cache has to be pruned at all:

```bash
uv run dvc gc --all-commits            # local cache
uv run dvc gc --all-commits --cloud    # the DagsHub remote
```

`--all-commits` keeps whatever any commit in the repository references, which is the only scope that is
safe while history still has to be reproducible.
There is no reason to prune at all yet: the whole remote is under a gigabyte.
