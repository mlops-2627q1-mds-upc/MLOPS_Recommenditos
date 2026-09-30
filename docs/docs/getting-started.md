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

Open `.env` and paste your username and token into the four empty variables.
The token goes in twice, once as `DAGSHUB_USER_TOKEN` and once as `MLFLOW_TRACKING_PASSWORD`; the template explains why.
`.env` is gitignored and must stay that way.
Never paste a token into an issue, a pull request, a commit or a chat - revoke it on the Tokens page instead if it ever leaks.

## 4. Give DVC the same credentials

```bash
uv run dvc remote modify origin --local user <your-dagshub-username>
uv run dvc remote modify origin --local password <your-dagshub-token>
```

**Never without `--local`.**
Without it, DVC writes the token into `.dvc/config`, which is committed, and the next `git push` publishes it.
With it, the token lands in `.dvc/config.local`, which DVC's own `.dvc/.gitignore` keeps out of Git.

### Why the credentials are entered twice

Because neither tool can read the other's store, and we verified that rather than assuming it:

- **MLflow reads only environment variables.** It looks up `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` itself, and `.env` is how they get there.
- **DVC reads only its own config files.** An HTTP remote with `auth = basic` accepts `user` and `password` and nothing else; there is no environment variable that supplies them, so `.env` alone leaves `dvc pull` failing with "HTTP 'basic' authentication require both 'user' and 'password'".

Both stores are gitignored, so both keep the token out of the repository, which is the property that actually matters.
The alternative - one place, and a script that copies it into the other - would trade a one-off paste for a piece of machinery that has to be maintained and that writes credentials to disk on its own, so we did not build it.

## 5. Pull the data

```bash
uv run dvc pull
```

This downloads everything the pipeline references into `data/` and `models/`, including the 548 MB raw CSV.
To check the connection without waiting for that, either list what is missing or pull one small file:

```bash
uv run dvc status -c                       # what differs between your disk and the remote
uv run dvc pull data/raw/listings.parquet  # 1.6 MB, enough to prove the credentials work
```

A 401 or 403 on a pull means the username or the token is wrong; on a `dvc push` it means the token is fine but the account has no write access yet.
`HTTP 'basic' authentication require both 'user' and 'password'` means step 4 did not run, or ran in a different clone.

## 6. Log an MLflow run

```bash
uv run python -m recommenditos.tracking
```

This logs one throwaway run - a single parameter and a single metric - to the `setup-check` experiment and prints its URL.
Open <https://dagshub.com/recommenditos/MLOPS_Recommenditos/experiments> and you should see your run there within seconds.
It is deliberately not written to the pipeline's experiment: a credential check is not an experiment result and does not belong next to the runs the report cites.

If a variable is missing, the command says which one before it tries to connect, rather than failing later with an HTTP 401 halfway through a training run.
That check lives in `recommenditos/tracking.py`, which is the one place the project configures MLflow, so the `train` stage (#37) reaches DagsHub through exactly the setup you just proved.

## 7. Run the pipeline

```bash
uv run dvc repro
```

With the data pulled, this should report every stage as up to date.
See [The DVC pipeline](pipeline.md) for what the stages do and [Data versioning](data-versioning.md) for what to do after you change data.
