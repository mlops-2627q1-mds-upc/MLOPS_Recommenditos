"""Temporary: start the API with 2 workers as test_logs does, call it, stop it, and look for non-JSON lines."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.getcwd())
from tests.api.helpers import free_port, stop_service

bundle, listing, out, n = (
    sys.argv[1],
    json.loads(Path(sys.argv[2]).read_text()),
    Path(sys.argv[3]),
    int(sys.argv[4]),
)
wait_all = "--wait-all" in sys.argv
out.mkdir(parents=True, exist_ok=True)
bad = 0
for i in range(n):
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    log = out / f"run{i}.log"
    env = {
        **os.environ,
        "RECOMMENDITOS_LOG_FORMAT": "json",
        "RECOMMENDITOS_MODEL_DIR": bundle,
        "MLFLOW_TRACKING_URI": "",
    }
    t0 = time.monotonic()
    with log.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "recommenditos.api",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--workers",
                "2",
            ],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        try:
            while True:
                try:
                    urllib.request.urlopen(f"{base}/health", timeout=30).read()
                    break
                except (urllib.error.URLError, ConnectionError):
                    time.sleep(0.5)
            t_ready = time.monotonic() - t0
            if wait_all:
                while log.read_text(encoding="utf-8").count("Application startup complete.") < 2:
                    time.sleep(0.1)
            req = urllib.request.Request(
                f"{base}/v1/predict",
                method="POST",
                data=json.dumps(listing).encode(),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=30).read()
            urllib.request.urlopen(
                urllib.request.Request(f"{base}/health", method="HEAD"), timeout=30
            ).read()
        finally:
            stop_service(p)
    text = log.read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if l.strip()]
    nonjson = [k for k, l in enumerate(lines) if not l.startswith("{")]
    print(
        f"run {i}: first ready {t_ready:.1f}s, startups {text.count('Application startup complete.')}, lines {len(lines)}, non-JSON {len(nonjson)}",
        flush=True,
    )
    if nonjson:
        bad += 1
        print("----- full log -----\n" + text + "--------------------", flush=True)
print(f"BAD {bad} of {n}")
