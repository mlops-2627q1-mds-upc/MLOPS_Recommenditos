"""The CI secret scan of NFR-15, and the canary that proves it can see a secret.

    python3 tools/secret_scan.py canary [--gitleaks PATH]
    python3 tools/secret_scan.py history [--gitleaks PATH]

`history` scans every commit of every ref this checkout has with gitleaks, which
in the CI job is every commit GitHub publishes for the repository (EDN-71).
`canary` runs first and fails unless the same scan, with the same configuration,
finds a fake DagsHub token planted in every place a real one has been or could
plausibly be written. Both build their gitleaks command in `gitleaks_git`, so the
canary cannot pass while the scan it vouches for is run differently.

Why there is a canary at all. NFR-15's evidence is a scan that reports nothing,
and a scan that reports nothing is only evidence if it could have reported
something. That is not a given for the one secret every contributor holds: a
DagsHub token is 40 lowercase hex characters with no prefix, gitleaks has no rule
for it, and its generic rule misses one token in twenty even in `.env` and six of
the thirteen places below outright (reports/analysis/secret_scan_dagshub_token.py).
The `dagshub-token` rule of `.gitleaks.toml` closes that, and an edit could open
it again without a word: a regex that matches nothing, an allowlist over `.env`,
a `.gitleaksignore` entry without a commit, or a gitleaks upgrade that reads the
file differently.

So the canary builds a throwaway repository with the history the real scan would
read: each place at its real path in the repository, one commit per token, and
two merge commits that each bring in a token of their own, one in a conflict
resolution and one added to a merge that had no conflict. It scans that
repository in git mode, with this repository's `.gitleaks.toml` and
`.gitleaksignore`, and fails unless every token is reported for the commit and
the file it was planted in. Planting in a directory and scanning it as files
would miss what only a history scan sees: paths relative to the repository, which
an allowlist anchored at `^\\.env$` matches, and commits, which a merge can carry
without any of its parents doing so.

Each place gets two tokens: an ordinary one, and one from the low-entropy tail
that gitleaks' generic rule drops, because that tail is the case the custom rule
exists for and an ordinary token would pass with the defaults alone. All of them
are drawn from a seeded generator, so a failure reproduces exactly, and none is a
real credential or appears in this source.

Standard library only, so the CI job runs it with the runner's Python and no
project environment. It needs git 2.36 or newer for `--remerge-diff`; the runner
has it, and an older git is refused with a message rather than misread.
"""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile

PROJ_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJ_ROOT / ".gitleaks.toml"

#: What `git log` is asked for, which is what the scan covers. gitleaks' own
#: default is the first three; giving `--log-opts` replaces it, so they are
#: repeated here. `--remerge-diff` is the addition: without it `git log -p` shows
#: a merge commit no diff at all, so a secret written into a conflict resolution,
#: or added to a merge, is in the published history and in no scanned diff. It
#: shows what the merge itself changed against an automatic remerge of its
#: parents, and not what it inherited from them, so a secret that already has a
#: finding in the commit that introduced it is not reported again for every merge
#: that carries it (`--diff-merges=first-parent` would, and on a branch that
#: merges `main` that is every secret `main` ever had, each needing its own
#: `.gitleaksignore` entry).
LOG_OPTS = "--all --full-history --diff-filter=tuxdb --remerge-diff"
MINIMUM_GIT = (2, 36)

#: gitleaks' generic-api-key rule drops a match below this many bits per character.
GENERIC_ENTROPY = 3.5
#: The floor of the low-entropy tokens planted here. About 1 real token in 100,000
#: falls below it, so a token that uniform is not a case worth planting, and it
#: stays clear of the 2.5 floor of `.gitleaks.toml`'s own rule.
LOWEST_PLANTED_ENTROPY = 3.0
SEED = 15

#: Where a token has been or could plausibly be written, as the file it would be
#: committed in and its content. The first three are the stores getting-started.md
#: tells a contributor to fill in; the rest are the ways a token escapes from them.
PLACES = {
    ".env, DVC-side name": (".env", "DAGSHUB_USER_TOKEN={token}\n"),
    ".env, MLflow-side name": (".env", "MLFLOW_TRACKING_PASSWORD={token}\n"),
    ".dvc/config, without --local": (
        ".dvc/config",
        "['remote \"origin\"']\n    auth = basic\n    user = someone\n    password = {token}\n",
    ),
    "Python constant": ("recommenditos/config.py", 'DAGSHUB_TOKEN = "{token}"\n'),
    "Python keyword argument": (
        "recommenditos/config.py",
        'dagshub.auth.add_app_token(token="{token}")\n',
    ),
    "Python positional argument": (
        "recommenditos/config.py",
        'dagshub.auth.add_app_token("{token}")\n',
    ),
    "Python os.environ key": (
        "recommenditos/config.py",
        'os.environ["MLFLOW_TRACKING_PASSWORD"] = "{token}"\n',
    ),
    "YAML value": ("params.yaml", "mlflow:\n  password: {token}\n"),
    "shell export": ("Makefile", "export MLFLOW_TRACKING_PASSWORD={token}\n"),
    "dvc remote modify, as documented": (
        "docs/docs/getting-started.md",
        "uv run dvc remote modify origin --local password {token}\n",
    ),
    "dagshub login": ("README.md", "dagshub login --token {token}\n"),
    "URL user part, DVC remote": (
        ".dvc/config",
        (
            "['remote \"origin\"']\n"
            "    url = https://someone:{token}@dagshub.com/recommenditos/MLOPS_Recommenditos.dvc\n"
        ),
    ),
    "URL user part, MLflow URI": (
        ".env",
        (
            "MLFLOW_TRACKING_URI=https://someone:{token}@dagshub.com/recommenditos/"
            "MLOPS_Recommenditos.mlflow\n"
        ),
    ),
}

#: A repository of its own: no hook, signing key or template of the machine it
#: runs on may change what gets committed.
_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "canary",
    "GIT_AUTHOR_EMAIL": "canary@example.invalid",
    "GIT_COMMITTER_NAME": "canary",
    "GIT_COMMITTER_EMAIL": "canary@example.invalid",
}


def gitleaks_git(gitleaks: str, source: Path, *extra: str) -> list[str]:
    """The one gitleaks command both the canary and the history scan run.

    The configuration and the ignore file are passed explicitly rather than left
    to gitleaks' lookup, which reads `.gitleaks.toml` from the scanned directory
    and `.gitleaksignore` from the working directory: the canary scans another
    directory and has to be judged by this repository's rules all the same.
    """
    return [
        gitleaks,
        "git",
        str(source),
        "--config",
        str(CONFIG),
        "--gitleaks-ignore-path",
        str(PROJ_ROOT),
        f"--log-opts={LOG_OPTS}",
        "--no-banner",
        *extra,
    ]


def _require_git() -> None:
    version = subprocess.run(["git", "--version"], check=True, capture_output=True, text=True)
    found = tuple(int(part) for part in re.findall(r"\d+", version.stdout)[:2])
    if found < MINIMUM_GIT:
        sys.exit(
            f"{version.stdout.strip()} cannot show what a merge commit changed "
            f"(`--remerge-diff` needs git {'.'.join(map(str, MINIMUM_GIT))}), so the scan "
            "would not see a secret introduced in one; CI's runner has a newer git"
        )


def entropy(text: str) -> float:
    """Shannon entropy in bits per character, the measure gitleaks thresholds on."""
    counts = Counter(text)
    return -sum(n / len(text) * math.log2(n / len(text)) for n in counts.values())


def fake_token(rng: random.Random, *, low_entropy: bool = False) -> str:
    """A token of DagsHub's shape, from the generic rule's blind spot if asked.

    About 5 % of real tokens fall below the generic threshold, so a few dozen
    draws find one.
    """
    while True:
        token = f"{rng.getrandbits(160):040x}"
        if not low_entropy or LOWEST_PLANTED_ENTROPY <= entropy(token) < GENERIC_ENTROPY:
            return token


class _Repository:
    """A throwaway repository the canary writes its history into."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.git("init", "--quiet", "--initial-branch=main")
        self.commit_file("README.md", "A history planted for the secret scan's canary.\n", "base")

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(self.root), "-c", "commit.gpgsign=false", *args],
            check=False,
            capture_output=True,
            text=True,
            env=_GIT_ENV,
        )
        # A merge that stops on a conflict is the point of one case, so the
        # caller decides; anything else failing is a broken canary.
        if result.returncode and args[0] != "merge":
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    def write(self, path: str, content: str) -> None:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def commit_file(self, path: str, content: str, message: str) -> str:
        self.write(path, content)
        self.git("add", "--all")
        self.git("commit", "--quiet", "--no-verify", "--message", message)
        return self.git("rev-parse", "HEAD")


def plant(root: Path) -> dict[tuple[str, str, str], str]:
    """Write the canary's history below `root`, keyed by (commit, file, token)."""
    rng = random.Random(SEED)
    repository = _Repository(root)
    planted = {}
    for place, (path, content) in PLACES.items():
        for kind in ("ordinary", "low-entropy"):
            token = fake_token(rng, low_entropy=kind == "low-entropy")
            commit = repository.commit_file(path, content.format(token=token), place)
            planted[(commit, path, token)] = f"{place} ({kind})"

    # A token that exists only in a conflict resolution: both sides change the
    # same line, neither carries a secret, and the merge commit writes one.
    repository.commit_file("notes.md", "the remote is\n", "notes")
    repository.git("checkout", "--quiet", "-b", "side")
    repository.commit_file("notes.md", "the remote is on DagsHub\n", "side")
    repository.git("checkout", "--quiet", "main")
    repository.commit_file("notes.md", "the remote is the DVC one\n", "main")
    repository.git("merge", "--quiet", "--no-edit", "side")
    token = fake_token(rng)
    repository.write("notes.md", f"the remote is on DagsHub, password {token}\n")
    repository.git("add", "--all")
    repository.git("commit", "--quiet", "--no-verify", "--no-edit")
    planted[(repository.git("rev-parse", "HEAD"), "notes.md", token)] = (
        "merge commit, conflict resolution"
    )

    # A token added to a merge that had no conflict to resolve.
    repository.git("checkout", "--quiet", "-b", "other")
    repository.commit_file("docs/other.md", "unrelated\n", "other")
    repository.git("checkout", "--quiet", "main")
    repository.git("merge", "--quiet", "--no-ff", "--no-commit", "other")
    token = fake_token(rng)
    repository.write(".env", f"MLFLOW_TRACKING_PASSWORD={token}\n")
    repository.git("add", "--all")
    repository.git("commit", "--quiet", "--no-verify", "--message", "merge other")
    planted[(repository.git("rev-parse", "HEAD"), ".env", token)] = (
        "merge commit, added without a conflict"
    )
    return planted


def canary(gitleaks: str) -> int:
    _require_git()
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch, "planted")
        root.mkdir()
        planted = plant(root)
        report = Path(scratch, "report.json")
        # Not --redact, because the check is that the whole planted token was
        # found; every token here is fake. --exit-code 0, because findings are the
        # point and the report, not the exit code, says which.
        subprocess.run(
            gitleaks_git(
                gitleaks,
                root,
                "--exit-code",
                "0",
                "--report-format",
                "json",
                "--report-path",
                str(report),
            ),
            check=True,
            capture_output=True,
        )
        findings = json.loads(report.read_text(encoding="utf-8"))

    # Guards against a canary that plants nothing and so finds everything.
    expected = 2 * len(PLACES) + 2
    if len(planted) != expected:
        print(f"the canary planted {len(planted)} tokens instead of {expected}")
        return 1
    # gitleaks 8.30.1 reports a file of a conflicted merge as `b/<path>`: the
    # "remerge CONFLICT" line `--remerge-diff` puts into that diff's header keeps
    # its parser from stripping git's prefix. The finding is real and complete, so
    # the comparison allows for the prefix rather than missing the case. A path
    # allowlist anchored at the root does not match it either, which errs towards
    # reporting.
    found = {
        (item["Commit"], item["File"].removeprefix("b/"), item["Secret"]) for item in findings
    }
    missed = [place for key, place in planted.items() if key not in found]
    if missed:
        print(
            f"the secret scan misses a DagsHub token in {len(missed)} of {len(planted)} "
            "places it was planted:",
            *(f"  - {place}" for place in missed),
            "A scan of the history that finds nothing is no evidence while this fails.",
            sep="\n",
        )
        return 1
    print(
        f"the secret scan finds a DagsHub token in all {len(planted)} places it was planted, "
        "each for the commit and the file it was planted in."
    )
    return 0


def history(gitleaks: str) -> int:
    """Scan every commit of every ref this checkout has; gitleaks' exit code."""
    _require_git()
    command = gitleaks_git(gitleaks, PROJ_ROOT, "--redact", "--verbose")
    # check=False: a finding is exit code 1, which is the scan's answer, not an error.
    return subprocess.run(command, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("canary", "history"))
    parser.add_argument("--gitleaks", default="gitleaks", help="the gitleaks binary")
    arguments = parser.parse_args(argv)
    return {"canary": canary, "history": history}[arguments.command](arguments.gitleaks)


if __name__ == "__main__":
    sys.exit(main())
