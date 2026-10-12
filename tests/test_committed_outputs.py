"""The pipeline outputs Git holds are the ones `dvc.lock` records.

A stage output with `cache: false` is committed to Git instead of pushed to the
DVC remote, and `dvc.lock` records it by the MD5 of its bytes. So Git has to hand
back exactly the bytes the stage wrote, on every platform. It does not by default:
`.gitattributes` makes every text file LF, and the `csv` module ends its rows with
CRLF, so `reports/figures/energy-vs-error.csv` was committed with other bytes than
the lock recorded and `dvc status` reported `compare-energy` as changed on every
fresh checkout.

These tests run against the repository itself, not a fixture, because what they
guard is the committed state. On a contributor's machine the second one can pass
while the bug is there, since the working tree still holds what the stage wrote;
a fresh checkout, which is what CI tests on Linux and on Windows, is what gives
it teeth, and the first one fails everywhere.
"""

from pathlib import Path
import subprocess

from dvc.repo import Repo
import pytest

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> list[str]:
    # `core.quotepath=off` so a path with a non-ASCII character comes back as itself.
    return subprocess.run(
        ["git", "-C", str(ROOT), "-c", "core.quotepath=off", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout.splitlines()


@pytest.fixture(scope="module")
def uncached_outputs():
    """Every output `dvc.yaml` declares with `cache: false`, `foreach` resolved."""
    with Repo(str(ROOT)) as repo:
        outputs = [out for out in repo.index.outs if not out.use_cache]
        assert outputs, "dvc.yaml declares no uncached output, so these tests check nothing"
        yield outputs


def test_git_leaves_the_line_endings_of_every_uncached_output_alone(uncached_outputs):
    paths = [Path(out.fs_path).relative_to(ROOT).as_posix() for out in uncached_outputs]
    tracked = git("ls-files", "--", *paths)
    assert tracked, paths

    converted = [
        line.split(": text: ")[0]
        for line in git("check-attr", "text", "--", *tracked)
        if not line.endswith(": text: unset")
    ]

    assert not converted, (
        "Git may rewrite the line endings of these pipeline outputs, so a checkout can hold "
        f"other bytes than dvc.lock records. Mark them `-text` in .gitattributes: {converted}"
    )


def test_every_uncached_output_in_the_checkout_is_the_one_the_lock_records(uncached_outputs):
    changed = {}
    for out in uncached_outputs:
        changed.update(out.workspace_status())

    assert not changed, (
        "dvc.lock records other bytes for these outputs than the checkout holds. Either a "
        f"stage was rerun without committing both, or Git converted the file: {changed}"
    )
