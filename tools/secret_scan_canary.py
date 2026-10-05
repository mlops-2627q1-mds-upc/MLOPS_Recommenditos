"""Fail unless the secret scan can see a DagsHub token, before it is trusted to find none.

NFR-15's evidence is a scan of the whole history that reports nothing, and a scan
that reports nothing is only evidence if it could have reported something. That is
not a given for the one secret every contributor holds: a DagsHub token is 40
lowercase hex characters with no prefix, gitleaks has no rule for it, and its
generic rule misses one token in twenty even in `.env` and six of the thirteen
places below outright (EDN-71, reports/analysis/secret_scan_dagshub_token.py). The
`dagshub-token` rule of `.gitleaks.toml` is what closes that, and a regex edited
into matching nothing, an allowlist widened over `.env`, or a gitleaks upgrade
that reads the file differently would all turn the scan green without a word.

So this writes a fake token into every place a real one has been or could
plausibly be written in this project, scans them with the binary and the
configuration the CI job is about to scan the history with, and exits 1 unless
every one is found, as the token that was planted and not a piece of it.

Each place gets two tokens: an ordinary one, and one from the low-entropy tail
that gitleaks' generic rule drops, because that tail is the case the custom rule
exists for and an ordinary token would pass with the defaults alone. Both are
drawn from a seeded generator, so a failure reproduces exactly, and neither is a
real credential or appears in the source.

Standard library only, so the CI job runs it with the runner's Python and no
project environment:

    python3 tools/secret_scan_canary.py [path/to/gitleaks]
"""

from collections import Counter
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile

PROJ_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJ_ROOT / ".gitleaks.toml"

#: gitleaks' generic-api-key rule drops a match below this many bits per character.
GENERIC_ENTROPY = 3.5
#: The floor of the low-entropy tokens planted here. About 1 real token in 100,000
#: falls below it, so a token that uniform is not a case worth planting, and it
#: stays clear of the 2.5 floor of `.gitleaks.toml`'s own rule.
LOWEST_PLANTED_ENTROPY = 3.0
SEED = 15

#: Where a token has been or could plausibly be written, as a file name and a line.
#: The first three are the stores getting-started.md tells a contributor to fill
#: in; the rest are the ways a token escapes from them into something committed.
PLACES = {
    ".env, DVC-side name": (".env", "DAGSHUB_USER_TOKEN={token}\n"),
    ".env, MLflow-side name": (".env", "MLFLOW_TRACKING_PASSWORD={token}\n"),
    ".dvc/config, without --local": (
        ".dvc/config",
        "['remote \"origin\"']\n    auth = basic\n    user = someone\n    password = {token}\n",
    ),
    "Python constant": ("config.py", 'DAGSHUB_TOKEN = "{token}"\n'),
    "Python keyword argument": ("config.py", 'dagshub.auth.add_app_token(token="{token}")\n'),
    "Python positional argument": ("config.py", 'dagshub.auth.add_app_token("{token}")\n'),
    "Python os.environ key": ("config.py", 'os.environ["MLFLOW_TRACKING_PASSWORD"] = "{token}"\n'),
    "YAML value": ("config.yaml", "mlflow:\n  password: {token}\n"),
    "shell export": ("setup.sh", "export MLFLOW_TRACKING_PASSWORD={token}\n"),
    "dvc remote modify, as documented": (
        "setup.sh",
        "uv run dvc remote modify origin --local password {token}\n",
    ),
    "dagshub login": ("setup.sh", "dagshub login --token {token}\n"),
    "URL user part, DVC remote": (
        "notes.md",
        "https://someone:{token}@dagshub.com/recommenditos/MLOPS_Recommenditos.dvc\n",
    ),
    "URL user part, MLflow URI": (
        ".env",
        (
            "MLFLOW_TRACKING_URI=https://someone:{token}@dagshub.com/recommenditos/"
            "MLOPS_Recommenditos.mlflow\n"
        ),
    ),
}


def entropy(text: str) -> float:
    """Shannon entropy in bits per character, the measure gitleaks thresholds on."""
    counts = Counter(text)
    return -sum(n / len(text) * math.log2(n / len(text)) for n in counts.values())


def fake_token(rng: random.Random, *, low_entropy: bool) -> str:
    """A token of DagsHub's shape, from the generic rule's blind spot if asked.

    About 5 % of real tokens fall below the generic threshold, so a few dozen
    draws find one.
    """
    while True:
        token = f"{rng.getrandbits(160):040x}"
        if not low_entropy or LOWEST_PLANTED_ENTROPY <= entropy(token) < GENERIC_ENTROPY:
            return token


def plant(root: Path) -> dict[tuple[str, str], str]:
    """Write every place's two tokens below `root`, keyed by (file, token)."""
    rng = random.Random(SEED)
    planted = {}
    for index, (place, (name, line)) in enumerate(PLACES.items()):
        for kind in ("ordinary", "low-entropy"):
            token = fake_token(rng, low_entropy=kind == "low-entropy")
            path = root / f"{index:02d}-{kind}" / name
            path.parent.mkdir(parents=True)
            path.write_text(line.format(token=token), encoding="utf-8")
            planted[(str(path.relative_to(root)), token)] = f"{place} ({kind})"
    return planted


def scan(gitleaks: str, root: Path, report: Path) -> set[tuple[str, str]]:
    """What gitleaks finds below `root`, as (file relative to `root`, secret)."""
    # Not --redact, because the check is that the whole planted token was found;
    # every token here is fake. --exit-code 0, because findings are the point and
    # the report, not the exit code, says which.
    subprocess.run(
        [
            gitleaks,
            "dir",
            str(root),
            "--config",
            str(CONFIG),
            "--no-banner",
            "--exit-code",
            "0",
            "--report-format",
            "json",
            "--report-path",
            str(report),
        ],
        check=True,
        capture_output=True,
    )
    findings = json.loads(report.read_text(encoding="utf-8"))
    return {
        (str(Path(item["File"]).resolve().relative_to(root)), item["Secret"]) for item in findings
    }


def main(gitleaks: str = "gitleaks") -> int:
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch, "planted").resolve()
        root.mkdir()
        planted = plant(root)
        found = scan(gitleaks, root, Path(scratch, "report.json"))

    missed = [place for key, place in planted.items() if key not in found]
    if missed:
        print(
            f"{CONFIG.name} misses a DagsHub token in {len(missed)} of {len(planted)} places:",
            *(f"  - {place}" for place in missed),
            "A scan of the history that finds nothing is no evidence while this fails.",
            sep="\n",
        )
        return 1
    print(f"{CONFIG.name} finds a DagsHub token in all {len(planted)} places it was planted.")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
