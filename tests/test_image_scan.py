"""The image scan of NFR-09 (tools/image_scan.py) and the pins it relies on.

The scan itself runs in CI, where trivy is installed: the integration job runs
the canary, which fails unless the scan fails a known-vulnerable image, and then
scans the stack's images. These tests cover what decides a verdict without
trivy: which findings fail the job, what an entry of `.trivyignore` must carry,
and that every image the system runs from is pinned by digest, so a rebuild
starts from what the last scan saw.
"""

from pathlib import Path
import re

import pytest
from tools import image_scan
from tools.image_scan import Finding, ImageResult, ignore_file_problems, parse_report
import yaml

PROJ_ROOT = Path(__file__).resolve().parents[1]
DIGEST = re.compile(r"^[^@\s]+:[^@\s]+@sha256:[0-9a-f]{64}$")


def vulnerability(vuln_id: str, status: str, severity: str = "HIGH") -> dict:
    return {
        "VulnerabilityID": vuln_id,
        "PkgName": "openssl",
        "InstalledVersion": "3.0.11-1",
        "FixedVersion": "3.0.13-1" if status == "fixed" else "",
        "Status": status,
        "Severity": severity,
    }


REPORT = {
    "Results": [
        {
            "Target": "image (debian 12.2)",
            "Vulnerabilities": [
                vulnerability("CVE-2024-0001", "fixed", "CRITICAL"),
                vulnerability("CVE-2024-0002", "affected"),
                vulnerability("CVE-2024-0003", "will_not_fix"),
                vulnerability("CVE-2024-0004", "fix_deferred"),
            ],
            "ExperimentalModifiedFindings": [{"Status": "ignored"}],
        },
        # A target trivy found nothing in has no `Vulnerabilities` key at all.
        {"Target": "Python"},
    ]
}


@pytest.mark.req("NFR-09")
def test_only_a_finding_with_a_fix_fails_the_scan():
    result = parse_report("api:local", REPORT)

    assert [f.vulnerability for f in result.fixable] == ["CVE-2024-0001"]
    assert result.unfixed == 3
    assert result.suppressed == 1
    assert not result.passed
    assert parse_report("api:local", {"Results": [REPORT["Results"][1]]}).passed


@pytest.mark.req("NFR-09")
def test_scan_exits_1_and_lists_the_fixable_findings(monkeypatch, capsys):
    finding = Finding("CVE-2024-0001", "openssl", "3.0.11-1", "3.0.13-1", "CRITICAL", "debian")
    results = {
        "api:local": ImageResult("api:local", [finding], unfixed=4),
        "proxy": ImageResult("proxy"),
    }
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(image_scan, "scan_image", lambda trivy, image: results[image])
    monkeypatch.setattr(image_scan, "trivy_version", lambda trivy: "trivy 0.0.0")

    assert image_scan.main(["--trivy", "trivy", "scan", "api:local", "proxy"]) == 1
    summary = capsys.readouterr().out
    assert "| `api:local` | 1 | 4 | 0 | **fail** |" in summary
    assert "| `proxy` | 0 | 0 | 0 | pass |" in summary
    assert (
        "| `api:local` | CVE-2024-0001 | CRITICAL | openssl | 3.0.11-1 | 3.0.13-1 | debian |"
        in summary
    )

    results["api:local"] = ImageResult("api:local", unfixed=4)
    assert image_scan.main(["--trivy", "trivy", "scan", "api:local", "proxy"]) == 0


@pytest.mark.req("NFR-09")
def test_trivy_scans_for_high_and_critical_with_the_ignore_file(tmp_path):
    output = tmp_path / "report.json"
    command = image_scan.trivy_command("trivy", "api:local", output)

    def value(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert command[:2] == ["trivy", "image"]
    assert command[-1] == "api:local"
    assert set(value("--severity").split(",")) == {"CRITICAL", "HIGH"}
    assert value("--scanners") == "vuln"
    assert Path(value("--ignorefile")) == PROJ_ROOT / ".trivyignore"
    assert (value("--format"), Path(value("--output"))) == ("json", output)
    # Findings without a fix are counted in the summary, so trivy has to report them.
    assert "--ignore-unfixed" not in command


@pytest.mark.req("NFR-09")
def test_canary_tries_the_next_registry_only_after_an_error(monkeypatch, capsys):
    scanned = []

    def scan(trivy, image):
        scanned.append(image)
        if len(scanned) == 1:
            raise image_scan.ScanError("TOOMANYREQUESTS")
        return ImageResult(image, [Finding("CVE-1", "openssl", "1", "2", "HIGH", "debian")])

    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(image_scan, "scan_image", scan)
    assert image_scan.main(["--trivy", "trivy", "canary"]) == 0
    assert scanned == list(image_scan.CANARY_IMAGES[:2])
    assert "with 1 fixable" in capsys.readouterr().out


@pytest.mark.req("NFR-09")
def test_canary_fails_when_no_registry_can_be_scanned(monkeypatch):
    def scan(trivy, image):
        raise image_scan.ScanError(f"trivy could not scan {image}")

    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(image_scan, "scan_image", scan)
    assert image_scan.main(["--trivy", "trivy", "canary"]) == 2


@pytest.mark.req("NFR-09")
def test_canary_fails_when_the_scan_finds_nothing(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    monkeypatch.setattr(
        image_scan, "scan_image", lambda trivy, image: ImageResult(image, unfixed=9)
    )
    assert image_scan.main(["--trivy", "trivy", "canary"]) == 1
    assert "**Failed**" in capsys.readouterr().out


@pytest.mark.req("NFR-09")
@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("# Only in the builder stage.\nCVE-2026-1 exp:2026-12-31\n", None),
        ("# Only in the builder stage.\nCVE-2026-1\n", "no expiry"),
        ("CVE-2026-1 exp:2026-12-31\n", "no reason"),
        ("#\nCVE-2026-1 exp:2026-12-31\n", "no reason"),
        ("# A reason.\n\nCVE-2026-1 exp:2026-12-31\n", "no reason"),
        ("# Only comments, as in the repository today.\n#   CVE-2026-1 exp:2026-12-31\n", None),
    ],
)
def test_an_accepted_finding_needs_a_reason_and_an_expiry(text, problem):
    problems = ignore_file_problems(text)
    if problem is None:
        assert problems == []
    else:
        assert len(problems) == 1 and problem in problems[0]


@pytest.mark.req("NFR-09")
def test_the_committed_trivyignore_is_well_formed():
    assert ignore_file_problems((PROJ_ROOT / ".trivyignore").read_text(encoding="utf-8")) == []


def dockerfile_images() -> list[str]:
    images = []
    for dockerfile in (PROJ_ROOT / "Dockerfile", PROJ_ROOT / "reports" / "latex" / "Dockerfile"):
        text = dockerfile.read_text(encoding="utf-8")
        images += [line.split()[1] for line in text.splitlines() if line.startswith("FROM ")]
    return images


def compose_images() -> list[str]:
    compose = yaml.safe_load((PROJ_ROOT / "deploy" / "compose.yaml").read_text(encoding="utf-8"))
    # The API's own image is built from the Dockerfile above and has no digest.
    return [s["image"] for s in compose["services"].values() if "build" not in s]


@pytest.mark.req("NFR-09")
def test_every_image_is_pinned_by_tag_and_digest():
    images = dockerfile_images() + compose_images() + list(image_scan.CANARY_IMAGES)
    assert len(images) >= 8
    assert [image for image in images if not DIGEST.match(image)] == []
