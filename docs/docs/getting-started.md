Getting started
===============

This page takes a fresh clone to two things you can see: the project's data on your disk, and one of your own runs in the MLflow UI on DagsHub.
Do it once per machine.
Everything after that is [Contributing](https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos/blob/main/CONTRIBUTING.md) for the workflow and [Data versioning](data-versioning.md) for the day-to-day DVC conventions.

You need [uv](https://docs.astral.sh/uv/getting-started/installation/), Git, and a free [DagsHub](https://dagshub.com) account.
uv brings its own Python, so you do not have to install 3.12 yourself.

## 1. Clone and install

```bash
git clone https://github.com/mlops-2627q1-mds-upc/MLOPS_Recommenditos.git
cd MLOPS_Recommenditos
uv sync
uv run pre-commit install
```

`uv sync` creates `.venv` from `uv.lock`, so everyone gets byte-identical versions.
Run project commands through `uv run <command>` rather than activating the environment, which is how CI runs them too.

## 2. Get a DagsHub access token

Our DVC remote and our MLflow tracking server are the same DagsHub repository, <https://dagshub.com/recommenditos/MLOPS_Recommenditos> (EDN-19).
Both authenticate with your DagsHub username and an access token.

1. Sign in to DagsHub.
2. Click your avatar in the top right → **Your Settings** → **Tokens**.
3. Copy the **Default Access Token**, or create a new one.

The repository is public (EDN-20), so reading works for any signed-in DagsHub account.
Writing - `dvc push`, and logging an MLflow run - needs write access on the repository, which @lukas2510 grants.

## 3. Fill in `.env`

```bash
cp .env.template .env
```

Open `.env` and paste your DagsHub username into `MLFLOW_TRACKING_USERNAME` and your token into `MLFLOW_TRACKING_PASSWORD`.
`MLFLOW_TRACKING_URI` is already filled in, because it is the same for everyone.
Those three are all the project reads from `.env`, and DVC is not one of its readers; step 4 is where DVC gets the same token.
`.env` is gitignored and must stay that way.
Never paste a token into an issue, a pull request, a commit or a chat - revoke it on the Tokens page instead if it ever leaks.

`.env` is read from this clone only: `recommenditos/config.py` loads `<repo>/.env` by name rather than searching upwards, so a checkout inside another checkout cannot pick up the other one's credentials.

## 4. Give DVC the same credentials

```bash
uv run dvc remote modify origin --local user <your-dagshub-username>
uv run dvc remote modify origin --local password <your-dagshub-token>
```

**Never without `--local`.**
Without it, DVC writes the token into `.dvc/config`, which is committed, and the next `git push` publishes it.
With it, the token lands in `.dvc/config.local`, which DVC's own `.dvc/.gitignore` keeps out of Git.

**Then clear that line from your shell history**, because the token was typed as an argument: `history -d <number>` in bash, or delete the line from `~/.zsh_history` in zsh.
While the command runs it is also visible in `ps` to anyone else on the machine.
That is the cost of DVC's HTTP remote, which takes a password from its config file and from nowhere else.
DVC does offer `dvc remote modify origin --local ask_password true`, which stores nothing and prompts instead; it is the better choice on a machine you share, and it is not our default because `getpass` needs a terminal, so it would prompt on every `dvc pull` and break any unattended `dvc repro` or CI job.

### Why the credentials are entered twice

Because neither tool can read the other's store, and we verified that rather than assuming it:

- **MLflow reads only environment variables.** It looks up `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` itself, and `.env` is how they get there.
- **DVC reads only its own config files.** An HTTP remote with `auth = basic` accepts `user`, `password` and `ask_password` and nothing else; there is no environment variable that supplies them, so `.env` alone leaves `dvc pull` failing with "HTTP 'basic' authentication require both 'user' and 'password'". `DAGSHUB_USERNAME` and `DAGSHUB_USER_TOKEN` are not in `.env.template` for exactly this reason: exporting them changes nothing, and listing them would suggest otherwise.

Both stores are gitignored, so both keep the token out of the repository, which is the property that actually matters.
The alternative - one place, and a script that copies it into the other - would trade a one-off paste for a piece of machinery that has to be maintained and that writes credentials to disk on its own, so we did not build it.

## 5. Pull the data

```bash
uv run dvc pull
```

This downloads every artefact `dvc.lock` references into `data/` and `models/`: the raw Parquet the `download` stage writes, the interim and processed frames, the feature matrices and the models.
How big that is depends on what the lock records, so read what `dvc pull` prints rather than expecting a number.
The lock records the real snapshot, so the raw Parquet alone is 215 MB.

The 548 MB source CSV is **not** among them.
The `download` stage fetches that from its pinned Zenodo DOI into `data/external/`, where it is a gitignored local cache rather than something on our remote, so no `dvc pull` brings it down (see [Data versioning](data-versioning.md)).

To prove the credentials work without downloading anything at all:

```bash
uv run dvc status -c   # compares your disk against the remote; transfers no data
```

A 401 or 403 there means the username or the token is wrong; on a `dvc push` it means the token is fine but the account has no write access yet.
`HTTP 'basic' authentication require both 'user' and 'password'` means step 4 did not run, or ran in a different clone.

## 6. Log an MLflow run

```bash
uv run python -m recommenditos.tracking
```

This logs one throwaway run - a single parameter and a single metric - to the `setup-check` experiment and prints its URL.
Open <https://dagshub.com/recommenditos/MLOPS_Recommenditos/experiments> and you should see your run there within seconds.
It is deliberately not written to the pipeline's experiment: a credential check is not an experiment result and does not belong next to the runs the report cites.
The run also carries your git commit, whether your working tree was clean and the MD5 of `dvc.lock`, because [NFR-06](specification.md) wants every run traceable to the code and data it saw.

If a variable is missing, the command prints one line naming which ones and stops there, rather than failing later with an HTTP 401 halfway through a training run:

```text
ERROR | MLflow is not configured: MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD not set.
        Copy .env.template to .env and fill in your DagsHub credentials (see docs/docs/getting-started.md).
```

All of that lives in `recommenditos/tracking.py`, which is the one place the project configures MLflow, so the `train` stage (#37) reaches DagsHub through exactly the setup you just proved.
Note the consequence: with no tracking URI configured, nothing is logged anywhere.
MLflow's own fallback would be a SQLite database in whatever directory you started in, and `tracking.py` refuses that on purpose, so a run that seems to have worked without credentials has not quietly gone somewhere local.

## 7. Run the pipeline

```bash
uv run dvc repro
```

`dvc repro` runs each stage whose command, `deps` or `params` no longer match what `dvc.lock` recorded, and skips the rest with `Stage '<name>' didn't change, skipping`.
A stage with no `outs`, like `configure_gx`, is compared the same way and skipped the same way; its missing output is a gap in the graph's ordering, not in its change detection.

So how much runs depends on whether the commit you cloned was reproduced with the `dvc.lock` it carries.
When it was, DVC touches nothing and ends with `Data and pipelines are up to date.`
When it was not - `params.yaml` moved on without a relock, say - the first run rebuilds from wherever the mismatch starts, and if that reaches `download` it includes fetching the 548 MB Zenodo file.
Either is fine; read the stage list it prints rather than expecting silence.

See [The DVC pipeline](pipeline.md) for what the stages do and [Data versioning](data-versioning.md) for what to do after you change data.

## Notebooks

Notebooks run in the project environment, through `uv run jupyter lab`, and the `pre-commit install` of step 1 is what strips their outputs before a commit.
Before a pull request that touches a notebook, run `make notebook-lint`; its first run builds Pynblint's own environment, which takes a few seconds.
[Notebooks](notebooks.md) has the naming convention, the rules the lint enforces and how to run a notebook against the real data.
