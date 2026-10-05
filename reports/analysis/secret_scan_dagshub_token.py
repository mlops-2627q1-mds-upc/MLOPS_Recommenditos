"""Which secret scanner sees a DagsHub token where ours could end up, and what else does it flag?

Question (EDN-71, NFR-15): a DagsHub access token is 40 lowercase hex characters with no prefix,
read off the shape of a real one in a gitignored `.env` without printing it. Neither scanner has a
rule for it. How often do gitleaks' default rules, gitleaks with `.gitleaks.toml`, and
detect-secrets' default plugins catch such a token, in each place one has been or could plausibly
be written in this project? What does each flag that is not a secret, on the committed tree and,
for gitleaks, which can read it, on the whole history of every branch?

Every token here is generated with `secrets.token_hex(20)`; none is real and none is written
anywhere but a temporary directory. Run from the repository root with the gitleaks binary the CI
job pins on PATH, or its path as the first argument; detect-secrets is fetched by `uvx`:

    python reports/analysis/secret_scan_dagshub_token.py [path/to/gitleaks]
"""

import collections
import json
import math
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile

GITLEAKS = sys.argv[1] if len(sys.argv) > 1 else "gitleaks"
DETECT_SECRETS = ["uvx", "--from", "detect-secrets==1.5.0", "detect-secrets"]
CONFIG = Path(".gitleaks.toml").resolve()
TOKENS_PER_CONTEXT = 200
ENTROPY_SAMPLES = 200_000
GENERIC_ENTROPY = 3.5  # generic-api-key in gitleaks' default config

# Where a token has been or could plausibly be written. The first three are the stores
# getting-started.md tells a contributor to fill in; the rest are the ways one leaks from them.
CONTEXTS = {
    ".env, DVC-side name": (".env", "DAGSHUB_USER_TOKEN={t}\n"),
    ".env, MLflow-side name": (".env", "MLFLOW_TRACKING_PASSWORD={t}\n"),
    ".dvc/config without --local": (
        ".dvc/config",
        "['remote \"origin\"']\n    auth = basic\n    user = someone\n    password = {t}\n",
    ),
    "Python constant": ("config.py", 'DAGSHUB_TOKEN = "{t}"\n'),
    "Python keyword argument": ("config.py", 'dagshub.auth.add_app_token(token="{t}")\n'),
    "Python positional argument": ("config.py", 'dagshub.auth.add_app_token("{t}")\n'),
    "Python os.environ key": ("config.py", 'os.environ["MLFLOW_TRACKING_PASSWORD"] = "{t}"\n'),
    "YAML value": ("config.yaml", "mlflow:\n  password: {t}\n"),
    "shell export": ("setup.sh", "export MLFLOW_TRACKING_PASSWORD={t}\n"),
    "dvc remote modify, as documented": (
        "setup.sh",
        "uv run dvc remote modify origin --local password {t}\n",
    ),
    "dagshub login": ("setup.sh", "dagshub login --token {t}\n"),
    "URL user part, DVC remote": (
        "notes.md",
        "https://someone:{t}@dagshub.com/recommenditos/MLOPS_Recommenditos.dvc\n",
    ),
    "URL user part, MLflow URI": (
        ".env",
        "MLFLOW_TRACKING_URI=https://someone:{t}@dagshub.com/recommenditos/"
        "MLOPS_Recommenditos.mlflow\n",
    ),
}


def entropy(text):
    counts = collections.Counter(text)
    return -sum(n / len(text) * math.log2(n / len(text)) for n in counts.values())


def gitleaks(arguments, root, config):
    # Always an explicit config: without one, gitleaks reads `.gitleaks.toml` from the scanned
    # directory, so a "defaults" scan of this repository would silently use ours.
    if config is None:
        config = Path(root) / "defaults.toml"
        config.write_text("[extend]\nuseDefault = true\n")
    report = Path(root) / "report.json"
    command = [GITLEAKS, *arguments, "--no-banner", "--exit-code", "0", "--config", str(config)]
    command += ["--report-format", "json", "--report-path", str(report)]
    subprocess.run(command, check=True, capture_output=True)
    return json.loads(report.read_text())


def detect_secrets(tree):
    """detect-secrets' findings below `tree`, as {relative path: [types]}."""
    out = subprocess.run(
        [*DETECT_SECRETS, "scan", "--all-files", "."],
        cwd=tree,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {path: [item["type"] for item in items] for path, items in json.loads(out)["results"].items()}


def plant(tree):
    planted = {}
    for index, (context, (name, template)) in enumerate(CONTEXTS.items()):
        for case in range(TOKENS_PER_CONTEXT):
            path = Path(tree, f"{index:02d}", f"{case:03d}", name)
            path.parent.mkdir(parents=True, exist_ok=True)
            token = secrets.token_hex(20)
            path.write_text(template.format(t=token))
            planted[str(path.relative_to(tree))] = (context, token)
    return planted


def detection():
    """Share of planted tokens each scanner finds, per context."""
    rates = collections.defaultdict(dict)
    with tempfile.TemporaryDirectory() as root:
        tree = Path(root, "tree")
        planted = plant(tree)
        for label, config in (("gitleaks defaults", None), ("gitleaks + ours", CONFIG)):
            findings = gitleaks(["dir", str(tree)], root, config)
            found = {(item["File"].split("tree/")[-1], item["Secret"]) for item in findings}
            for context in CONTEXTS:
                cases = [(f, t) for f, (c, t) in planted.items() if c == context]
                rates[label][context] = sum(case in found for case in cases) / len(cases)
        # detect-secrets reports a line, not the secret, and every planted file holds one token.
        flagged = detect_secrets(tree)
        for context in CONTEXTS:
            cases = [f for f, (c, _) in planted.items() if c == context]
            rates["detect-secrets"][context] = sum(f in flagged for f in cases) / len(cases)
    return rates


def main():
    version = subprocess.run([GITLEAKS, "version"], check=True, capture_output=True, text=True)
    scanner = subprocess.run(
        [*DETECT_SECRETS, "--version"], check=True, capture_output=True, text=True
    )
    print(
        f"gitleaks {version.stdout.strip()}, detect-secrets {scanner.stdout.strip()}, "
        f"{TOKENS_PER_CONTEXT} fake tokens per context\n"
    )

    rates = detection()
    labels = list(rates)
    print(f"{'context':36s}" + "".join(f"{label:>20s}" for label in labels))
    for context in CONTEXTS:
        print(f"{context:36s}" + "".join(f"{rates[label][context]:20.1%}" for label in labels))

    samples = [entropy(secrets.token_hex(20)) for _ in range(ENTROPY_SAMPLES)]
    print(f"\nShannon entropy of {ENTROPY_SAMPLES:,} fake tokens, bits per character:")
    print(f"  mean {sum(samples) / len(samples):.3f}, lowest {min(samples):.3f}")
    for threshold in (2.5, 3.0, 3.3, 3.4, GENERIC_ENTROPY):
        share = sum(value < threshold for value in samples) / len(samples)
        print(f"  below {threshold}: {share:.3%}")

    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    with tempfile.TemporaryDirectory() as root:
        archive = subprocess.run(["git", "archive", "HEAD"], check=True, capture_output=True)
        subprocess.run(["tar", "-x", "-C", root], input=archive.stdout, check=True)
        print(f"\nthe committed tree at {head}:")
        for label, config in (("gitleaks defaults", None), ("gitleaks + ours", CONFIG)):
            with tempfile.TemporaryDirectory() as scratch:
                findings = gitleaks(["dir", "--redact", root], scratch, config)
            print(f"  {label}: {len(findings)} finding(s)")
        flagged = detect_secrets(root)
        print(f"  detect-secrets: {sum(map(len, flagged.values()))} finding(s)")
        for path, types in sorted(flagged.items()):
            print(f"    {path}: {', '.join(types)}")

    commits = subprocess.run(
        ["git", "rev-list", "--all", "--no-merges", "--count"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"\nthe whole history, every branch, {commits} non-merge commits (gitleaks only):")
    for label, config in (("gitleaks defaults", None), ("gitleaks + ours", CONFIG)):
        with tempfile.TemporaryDirectory() as root:
            findings = gitleaks(["git", "--redact", "--log-opts=--all", "."], root, config)
        print(f"  {label}: {len(findings)} finding(s)")
        for item in findings:
            print(f"    {item['RuleID']} {item['File']}:{item['StartLine']} {item['Commit'][:8]}")


if __name__ == "__main__":
    main()
