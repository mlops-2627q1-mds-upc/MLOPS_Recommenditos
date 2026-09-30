"""The `download` stage's real source: the pinned Zenodo file.

Nothing here reaches beyond the loopback interface, and the autouse guard below
makes that structural rather than a convention. The stage takes its cache
directory as an argument, so a test can put a small, contract-shaped CSV there
and exercise the whole `zenodo` path - the MD5 check, the column check and the
conversion - on a file it wrote itself. The tests that have to cover the fetch
install a fake downloader, which is also how the URL the stage would really
request is pinned.

`_download` itself is covered against a throwaway HTTP server on 127.0.0.1
rather than a mock of `requests`, because what it promises - an error status
fails, a stalled socket fails, and a transfer that does not finish leaves no
file - is behaviour of the socket and not of our call into the library.

The synthetic source is tested in `test_pipeline.py`, next to the stage wiring.
"""

from dataclasses import dataclass, field
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import inspect
from pathlib import Path
import threading
import time

import pandas as pd
import pytest
import requests
import yaml

from recommenditos.config import EXTERNAL_DATA_DIR, RAW_DATA_DIR
from recommenditos.data import download_raw_dataset
from recommenditos.data.download_raw_dataset import ZENODO, SourceChangedError
from recommenditos.schema import RAW_SCHEMA

#: The `download.md5` a test uses when it wants the check to fail. A well-formed
#: digest rather than a word, so the fixture reads like the failure a changed
#: upstream file would produce. The check itself compares the two strings, so it
#: would reject a word just as flatly; nothing here rests on the shape.
WRONG_MD5 = "0" * 32

#: What `download.filename` pins, which is what the cached file is called.
CSV_NAME = "autoscout24_dataset_20251108.csv"

#: The real `_download`, bound before the autouse guard below replaces the
#: module attribute, so the tests that cover the fetch run the code the stage
#: runs instead of the guard's refusal.
_REAL_DOWNLOAD = download_raw_dataset._download

#: How long a test waits for a thread or a file it expects within milliseconds.
#: Generous, because it is only ever reached when the assertion is about to fail
#: anyway, and a loaded CI machine should not decide the outcome.
_PATIENCE_SECONDS = 5.0


@pytest.fixture(autouse=True)
def no_download(monkeypatch):
    """Refuse to download anything, for every test in this module.

    A wrong turn in the cache logic would otherwise pull 548 MB from Zenodo in
    the middle of a unit test run. A test that needs a fetch installs a fake of
    its own over this one.
    """

    def _refuse(url: str, destination: Path) -> None:
        raise AssertionError(f"a test tried to download {url}")

    monkeypatch.setattr(download_raw_dataset, "_download", _refuse)


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    """Where the stage keeps the downloaded CSV between runs."""
    path = tmp_path / "external"
    path.mkdir()
    return path


@pytest.fixture
def pinned_csv(cache_dir: Path, raw_frame: pd.DataFrame) -> Path:
    """A CSV in the cache, shaped like the published one.

    The synthetic frame is written out and read back through the stage, so the
    columns the contract keeps as text - `mileage_km`, `ratings_average`,
    `weight_kg`, `registration_date` - make the round trip the real file makes.
    """
    path = cache_dir / CSV_NAME
    raw_frame.to_csv(path, index=False)
    return path


def test_a_cached_csv_is_converted_without_being_downloaded_again(
    tmp_path, params, cache_dir, pinned_csv, raw_frame
):
    # 548 MB per `dvc repro` is what the cache exists to avoid; `no_download`
    # above is what fails this test if the stage fetches it anyway.
    output = tmp_path / "listings.parquet"

    download_raw_dataset.main(output, _params(tmp_path, params, md5=_md5(pinned_csv)), cache_dir)

    written = pd.read_parquet(output)
    RAW_SCHEMA.validate(written)
    assert len(written) == len(raw_frame)


@pytest.mark.req("NFR-06")
def test_a_changed_upstream_file_fails_the_stage(
    tmp_path, params, cache_dir, raw_frame, monkeypatch
):
    # The scenario the MD5 exists for: Zenodo serves something other than the
    # file this pipeline is pinned to. Retraining on it in silence would be the
    # worst outcome, so nothing is written.
    def _serve_something_else(url: str, destination: Path) -> None:
        raw_frame.head(10).to_csv(destination, index=False)

    monkeypatch.setattr(download_raw_dataset, "_download", _serve_something_else)
    output = tmp_path / "listings.parquet"

    with pytest.raises(SourceChangedError, match=WRONG_MD5):
        download_raw_dataset.main(output, _params(tmp_path, params, md5=WRONG_MD5), cache_dir)

    assert not output.exists()


@pytest.mark.req("NFR-06")
def test_a_cached_file_that_does_not_match_the_pin_is_not_read(
    tmp_path, params, cache_dir, pinned_csv
):
    # The same check one run later: a cached copy is re-hashed rather than
    # trusted, so a truncated or hand-edited local file is caught as well.
    with pytest.raises(SourceChangedError, match=CSV_NAME):
        download_raw_dataset.main(
            tmp_path / "listings.parquet", _params(tmp_path, params, md5=WRONG_MD5), cache_dir
        )


@pytest.mark.req("NFR-06")
def test_a_missing_file_is_fetched_from_the_url_the_record_pins(
    tmp_path, params, cache_dir, raw_frame, monkeypatch
):
    served = tmp_path / "served.csv"
    raw_frame.to_csv(served, index=False)
    requested: list[str] = []

    def _fake_download(url: str, destination: Path) -> None:
        requested.append(url)
        destination.write_bytes(served.read_bytes())

    monkeypatch.setattr(download_raw_dataset, "_download", _fake_download)

    download_raw_dataset.main(
        tmp_path / "listings.parquet", _params(tmp_path, params, md5=_md5(served)), cache_dir
    )

    download = params["download"]
    assert requested == [
        f"https://zenodo.org/records/{download['zenodo_record']}/files/{download['filename']}"
    ]


#: An upstream column change, as (what it does to the frame, the name the
#: failure has to mention). One parameter rather than two, so the test still
#: takes the five arguments a Pylint rule of this project allows.
COLUMN_CHANGES = [
    pytest.param(
        (lambda frame: frame.rename(columns={"power_kw": "power_kilowatt"}), "power_kw"),
        id="renamed",
    ),
    pytest.param(
        (lambda frame: frame.assign(dealer_rating_v2=1.0), "dealer_rating_v2"), id="added"
    ),
]


@pytest.mark.req("NFR-06")
@pytest.mark.parametrize("change", COLUMN_CHANGES)
def test_a_column_the_contract_does_not_describe_fails_the_stage(
    tmp_path, params, cache_dir, raw_frame, change
):
    # `Schema.conform` selects, so an added column would be dropped in silence
    # and a renamed one would only surface as a missing feature stages later.
    mutation, reported = change
    path = cache_dir / CSV_NAME
    mutation(raw_frame).to_csv(path, index=False)

    with pytest.raises(SourceChangedError, match=reported):
        download_raw_dataset.main(
            tmp_path / "listings.parquet", _params(tmp_path, params, md5=_md5(path)), cache_dir
        )


def _params(tmp_path: Path, params: dict, **download) -> Path:
    """A copy of params.yaml with keys of its `download` block changed."""
    changed = {**params, "download": {**params["download"], "source": ZENODO, **download}}
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    return path


def _md5(path: Path) -> str:
    # `usedforsecurity=False` for the same reason the stage passes it: this is a
    # content checksum, and without the flag the call raises on a FIPS-mode
    # Python.
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


@dataclass
class _Reply:
    """What the throwaway server answers with, so a test states its own case."""

    status: HTTPStatus = HTTPStatus.OK
    body: bytes = b"id,price\n1,1000\n"
    #: What to put in `Content-Length`, when it should not be the real length.
    #: A larger value cuts the response short, which is the interrupted transfer.
    announced_length: int | None = None
    #: How long to leave the request unanswered, to exercise the read timeout.
    stall_seconds: float = 0.0
    #: Where to break off mid-body and wait for `resume`, so a test can look at
    #: the filesystem while the transfer is still open.
    pause_after_bytes: int | None = None
    paused: threading.Event = field(default_factory=threading.Event)
    resume: threading.Event = field(default_factory=threading.Event)


@pytest.fixture
def serve():
    """Start a loopback HTTP server with a given reply and return its URL.

    Loopback only: no name is resolved and nothing leaves the machine, so the
    suite still runs on a clean clone with no network.
    """
    released = threading.Event()
    servers: list[ThreadingHTTPServer] = []

    def _start(reply: _Reply) -> str:
        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if reply.stall_seconds:
                    # `wait` rather than `sleep`, so teardown does not have to
                    # outlast the stall this test never means to finish.
                    released.wait(reply.stall_seconds)
                if reply.status != HTTPStatus.OK:
                    self.send_error(reply.status)
                    return
                self.send_response(HTTPStatus.OK)
                length = reply.announced_length or len(reply.body)
                self.send_header("Content-Length", str(length))
                self.end_headers()
                if reply.pause_after_bytes is None:
                    self.wfile.write(reply.body)
                    return
                self.wfile.write(reply.body[: reply.pause_after_bytes])
                self.wfile.flush()
                reply.paused.set()
                reply.resume.wait(_PATIENCE_SECONDS)
                self.wfile.write(reply.body[reply.pause_after_bytes :])

            def log_message(self, *args):
                """Keep the request log out of the test output."""

        class _Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                """A client that hangs up mid-response is the point, not a fault."""

        server = _Server(("127.0.0.1", 0), _Handler)
        servers.append(server)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return f"http://127.0.0.1:{server.server_port}/records/1/files/{CSV_NAME}"

    yield _start

    released.set()
    for server in servers:
        server.shutdown()
        server.server_close()


def test_download_writes_the_served_bytes_and_leaves_no_part_file(tmp_path, serve):
    reply = _Reply()
    destination = tmp_path / "nested" / CSV_NAME

    _REAL_DOWNLOAD(serve(reply), destination)

    assert destination.read_bytes() == reply.body
    assert not _part_of(destination).exists()


def test_download_fails_on_an_error_status_instead_of_saving_the_error_page(tmp_path, serve):
    # A Zenodo record that has been withdrawn or renamed answers with HTML and a
    # 4xx. Saving that as the dataset would turn a clear failure into an MD5
    # mismatch two steps later, blaming the pinned file for a missing one.
    destination = tmp_path / CSV_NAME

    with pytest.raises(requests.exceptions.HTTPError):
        _REAL_DOWNLOAD(serve(_Reply(status=HTTPStatus.NOT_FOUND)), destination)

    assert not destination.exists()
    assert not _part_of(destination).exists()


@pytest.mark.req("NFR-06")
def test_the_destination_stays_absent_while_bytes_are_still_arriving(tmp_path, serve, monkeypatch):
    # The atomic rename the module docstring promises, checked where it matters:
    # mid-transfer. Writing straight to the destination would put a truncated CSV
    # exactly where the next run looks for a cached copy, and a machine that
    # loses power or is killed has no chance to tidy up afterwards, so that run
    # would report a changed upstream file instead of a failed download.
    # The chunk size is cut down so a body of this size arrives in several
    # pieces, and the pause is past the file object's own buffer, so the bytes
    # the assertion looks for have really reached the disk.
    monkeypatch.setattr(download_raw_dataset, "_CHUNK_BYTES", 4096)
    reply = _Reply(body=b"id,price\n" + b"1,1000\n" * 5000, pause_after_bytes=16384)
    destination = tmp_path / CSV_NAME
    worker = threading.Thread(target=_REAL_DOWNLOAD, args=(serve(reply), destination), daemon=True)

    worker.start()
    try:
        assert reply.paused.wait(_PATIENCE_SECONDS), "the server never reached the pause"
        _wait_until(lambda: _first_bytes_landed(destination))
        assert not destination.exists()
    finally:
        reply.resume.set()
        worker.join(_PATIENCE_SECONDS)

    assert destination.read_bytes() == reply.body
    assert not _part_of(destination).exists()


@pytest.mark.req("NFR-06")
def test_an_interrupted_transfer_leaves_no_file_where_the_stage_looks(tmp_path, serve):
    # A response that stops short of what it announced is what a dropped
    # connection looks like. Nothing here resumes a transfer, so the `.part` file
    # goes with the failure rather than waiting on disk for somebody to find it.
    reply = _Reply(announced_length=len(_Reply.body) + 500)
    destination = tmp_path / CSV_NAME

    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        _REAL_DOWNLOAD(serve(reply), destination)

    assert not destination.exists()
    assert not _part_of(destination).exists()


def test_a_stalled_server_fails_the_download_instead_of_hanging(tmp_path, serve, monkeypatch):
    # Without a timeout on the request a silent server holds the whole pipeline
    # open for as long as it likes. The real 60 s is cut down here so the test
    # costs a fraction of a second; the stall outlasts it either way.
    monkeypatch.setattr(download_raw_dataset, "_TIMEOUT_SECONDS", 0.5)
    destination = tmp_path / CSV_NAME

    with pytest.raises(requests.exceptions.Timeout):
        _REAL_DOWNLOAD(serve(_Reply(stall_seconds=3.0)), destination)

    assert not destination.exists()
    assert not _part_of(destination).exists()


def test_the_cache_dir_defaults_to_the_third_party_slot(params):
    # Every other test in this module passes `cache_dir` explicitly, so nothing
    # else notices where the stage puts the file when nobody says. That default
    # is what the ownership rule rests on: `data/raw/` holds pipeline outputs
    # alone, and the published CSV is a third-party copy under `data/external/`.
    # Asserted on the signature rather than by running the stage, because a run
    # with the default would read whatever 548 MB file the machine has cached.
    default = inspect.signature(download_raw_dataset.main).parameters["cache_dir"].default

    assert default == EXTERNAL_DATA_DIR
    assert RAW_DATA_DIR not in (default, *default.parents)
    assert (default / params["download"]["filename"]).name == CSV_NAME


def _part_of(destination: Path) -> Path:
    return destination.with_name(destination.name + ".part")


def _first_bytes_landed(destination: Path) -> bool:
    """Whether the transfer has written anywhere the stage could be writing.

    The destination is checked first, so a stage that writes straight to it is
    reported as soon as it does rather than after the wait below runs out.
    """
    return destination.exists() or _part_of(destination).stat().st_size > 0


def _wait_until(condition) -> None:
    """Block until `condition` holds, so a test waits on a state, not on a sleep.

    A missing file raises out of the condition and counts as "not yet": the point
    of waiting is that the writer has not caught up.
    """
    deadline = time.monotonic() + _PATIENCE_SECONDS
    while time.monotonic() < deadline:
        try:
            if condition():
                return
        except OSError:
            pass
        time.sleep(0.01)
    raise AssertionError("timed out waiting for the transfer to write its first bytes")
