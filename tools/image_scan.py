"""The image scan of NFR-09, and the canary that proves it can fail.

    python3 tools/image_scan.py scan IMAGE [IMAGE ...] [--trivy PATH]
    python3 tools/image_scan.py canary [--trivy PATH]

`scan` runs trivy over each image and fails when one of them has a HIGH or
CRITICAL vulnerability with a fix available. A finding without a fix does not
fail it: NFR-09 keeps an unfixable CVE of the base image from blocking every
deploy, and the summary still counts those findings. `canary` runs first and
fails unless the same scan finds fixable findings in an image known to have
them, so that a scan reporting nothing is evidence, as the canary of
tools/secret_scan.py is for gitleaks. Both build their trivy command in
`trivy_command`, so the canary cannot pass while the scan it vouches for runs
differently.

Accepted findings go into `.trivyignore`, each with the reason in a comment
line above it and an expiry date (`exp:YYYY-MM-DD`), after which trivy reports
the finding again. An entry without either is refused before anything is
scanned.

The vulnerability database is downloaded on every run rather than kept in the
Actions cache: the download takes about 5 s in CI, and the database is rebuilt
every few hours, so a cached copy would be stale by the next run and trivy would
download it again anyway. A rate limit (HTTP 429) or a server error on
one registry does not fail the scan: trivy then tries the next of
`DB_REPOSITORIES` (`shouldTryOtherRepo` in trivy's pkg/oci/artifact.go).

Both commands write their result as Markdown to the job summary when they run
in GitHub Actions, and to standard output otherwise. Standard library only, so
the CI job runs it with the runner's Python and no project environment.
"""

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

PROJ_ROOT = Path(__file__).resolve().parents[1]
IGNORE_FILE = PROJ_ROOT / ".trivyignore"

SEVERITIES = ("CRITICAL", "HIGH")

# In order of priority. mirror.gcr.io and ghcr.io are trivy's defaults; the
# copy on Amazon ECR Public is a third registry with limits of its own.
DB_REPOSITORIES = (
    "mirror.gcr.io/aquasec/trivy-db:2",
    "public.ecr.aws/aquasecurity/trivy-db:2",
    "ghcr.io/aquasecurity/trivy-db:2",
)

# python:3.12.0-slim-bookworm as published in October 2023, pinned by digest so
# it never changes: Debian 12.2 with dozens of HIGH and CRITICAL vulnerabilities
# that later point releases fixed (93 with trivy 0.76.1 on 2026-10-22). Fixes
# only accumulate for an image that never changes, so it stays a positive.
# The same digest from two registries in turn, so that one registry's anonymous
# pull limit does not fail the job: ECR Public answered a CI run with
# TOOMANYREQUESTS on 2026-10-22. (mirror.gcr.io serves the manifest but no
# longer the layers of an image this old.) Only an error moves on to the next
# registry; a scan that finds nothing fails the canary at once.
CANARY_DIGEST = "sha256:19a6235339a74eca01227b03629f63b6f5020abc21142436eced6ec3a9839a76"
CANARY_IMAGES = tuple(
    f"{registry}/python:3.12.0-slim-bookworm@{CANARY_DIGEST}"
    for registry in ("public.ecr.aws/docker/library", "docker.io/library")
)

# trivy's status for a vulnerability with a released fix; `--ignore-unfixed`
# keeps exactly these.
FIXED = "fixed"

EXPIRY = re.compile(r"(?:^|\s)exp:(\d{4}-\d{2}-\d{2})(?:\s|$)")


class ScanError(Exception):
    """trivy could not scan, or `.trivyignore` is malformed."""


@dataclass
class Finding:
    vulnerability: str
    package: str
    installed: str
    fixed_in: str
    severity: str
    target: str


@dataclass
class ImageResult:
    image: str
    fixable: list[Finding] = field(default_factory=list)
    unfixed: int = 0
    suppressed: int = 0

    @property
    def passed(self) -> bool:
        return not self.fixable


def ignore_file_problems(text: str) -> list[str]:
    """What is wrong with the entries of a `.trivyignore`: each needs a reason
    in the comment line directly above it and an expiry date."""
    problems = []
    previous = ""
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if line and not line.startswith("#"):
            if not EXPIRY.search(line):
                problems.append(f"line {number}: `{line}` has no expiry (exp:YYYY-MM-DD)")
            if not previous.startswith("#") or not previous.lstrip("#").strip():
                problems.append(
                    f"line {number}: `{line}` has no reason in a comment line above it"
                )
        previous = line
    return problems


def trivy_command(trivy: str, image: str, output: Path) -> list[str]:
    return [
        trivy,
        "image",
        "--scanners",
        "vuln",
        "--severity",
        ",".join(SEVERITIES),
        "--db-repository",
        ",".join(DB_REPOSITORIES),
        "--ignorefile",
        str(IGNORE_FILE),
        "--show-suppressed",
        "--no-progress",
        "--format",
        "json",
        "--output",
        str(output),
        image,
    ]


def parse_report(image: str, report: dict) -> ImageResult:
    result = ImageResult(image)
    for target in report.get("Results") or []:
        for vulnerability in target.get("Vulnerabilities") or []:
            if vulnerability.get("Status") == FIXED:
                result.fixable.append(
                    Finding(
                        vulnerability=vulnerability["VulnerabilityID"],
                        package=vulnerability["PkgName"],
                        installed=vulnerability.get("InstalledVersion", ""),
                        fixed_in=vulnerability.get("FixedVersion", ""),
                        severity=vulnerability.get("Severity", ""),
                        target=target.get("Target", ""),
                    )
                )
            else:
                result.unfixed += 1
        result.suppressed += len(target.get("ExperimentalModifiedFindings") or [])
    return result


def scan_image(trivy: str, image: str) -> ImageResult:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "report.json"
        # S603: trivy with an argument list this module builds, no shell.
        completed = subprocess.run(trivy_command(trivy, image, output), check=False)  # noqa: S603
        if completed.returncode != 0 or not output.is_file():
            raise ScanError(f"trivy could not scan {image} (exit code {completed.returncode})")
        report = json.loads(output.read_text(encoding="utf-8"))
    return parse_report(image, report)


def trivy_version(trivy: str) -> str:
    """trivy's version and the age of the database it scanned with."""
    completed = subprocess.run(  # noqa: S603 - as in scan_image
        [trivy, "version", "--format", "json"], capture_output=True, text=True, check=False
    )
    try:
        info = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return "trivy (version unknown)"
    updated = (info.get("VulnerabilityDB") or {}).get("UpdatedAt", "unknown")
    updated = updated[:16].replace("T", " ")
    return (
        f"trivy {info.get('Version', 'unknown')} with the vulnerability database of {updated} UTC"
    )


def render_scan(results: list[ImageResult], version: str) -> str:
    lines = [
        "## Image scan (trivy)",
        "",
        (
            f"{version}. A HIGH or CRITICAL vulnerability with a fix available fails the job;"
            " one without a fix is counted and does not (NFR-09)."
        ),
        "",
        "| Image | Fixable | Without a fix | Accepted in `.trivyignore` | Verdict |",
        "|---|---:|---:|---:|---|",
    ]
    for result in results:
        verdict = "pass" if result.passed else "**fail**"
        lines.append(
            f"| `{result.image}` | {len(result.fixable)} | {result.unfixed}"
            f" | {result.suppressed} | {verdict} |"
        )
    failing = [result for result in results if not result.passed]
    if failing:
        lines += [
            "",
            (
                "Update the base image or the package, or accept the finding in `.trivyignore`"
                " with a reason and an expiry date."
            ),
            "",
            "| Image | Vulnerability | Severity | Package | Installed | Fixed in | Target |",
            "|---|---|---|---|---|---|---|",
        ]
        for result in failing:
            for f in result.fixable:
                lines.append(
                    f"| `{result.image}` | {f.vulnerability} | {f.severity} | {f.package}"
                    f" | {f.installed} | {f.fixed_in} | {f.target} |"
                )
    return "\n".join(lines) + "\n"


def publish(markdown: str) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(markdown + "\n")
    print(markdown)


def check_ignore_file() -> None:
    if IGNORE_FILE.is_file():
        problems = ignore_file_problems(IGNORE_FILE.read_text(encoding="utf-8"))
        if problems:
            raise ScanError(".trivyignore: " + "; ".join(problems))


def cmd_scan(trivy: str, images: list[str]) -> int:
    check_ignore_file()
    results = [scan_image(trivy, image) for image in images]
    publish(render_scan(results, trivy_version(trivy)))
    return 0 if all(result.passed for result in results) else 1


def cmd_canary(trivy: str) -> int:
    check_ignore_file()
    errors = []
    for image in CANARY_IMAGES:
        try:
            result = scan_image(trivy, image)
        except ScanError as error:
            errors.append(str(error))
            continue
        if result.passed:
            publish(
                "## Image scan canary\n\n**Failed**: the scan found no fixable HIGH or CRITICAL"
                f" vulnerability in `{image}`, which has dozens, so a passing scan proves"
                " nothing. Check the trivy flags in tools/image_scan.py and `.trivyignore`.\n"
            )
            return 1
        publish(
            f"## Image scan canary\n\nThe scan fails `{image}`, an image from October 2023,"
            f" with {len(result.fixable)} fixable HIGH or CRITICAL findings, so a scan that"
            " passes is evidence.\n"
        )
        return 0
    raise ScanError("; ".join(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trivy", default=None, help="the trivy binary (default: from PATH)")
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="scan images and fail on fixable findings")
    scan.add_argument("images", nargs="+")
    commands.add_parser("canary", help="check that the scan fails a known-vulnerable image")
    args = parser.parse_args(argv)

    trivy = args.trivy or shutil.which("trivy")
    if not trivy:
        print("trivy is not installed or not on PATH; pass --trivy", file=sys.stderr)
        return 2
    try:
        if args.command == "scan":
            return cmd_scan(trivy, args.images)
        return cmd_canary(trivy)
    except ScanError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
