"""Whether `b1`'s Ridge coefficients depend on the number of OpenBLAS threads (issue #113).

`fit_cpu_threads.py` found that the training and validation L1 of `b1` differ
between a fit at OpenBLAS's default of one thread per core and one at
`OPENBLAS_NUM_THREADS=1`. This script fits `b1` as `train` does, in a process of
its own per thread count (the pool is sized when numpy loads), twice at 8, and
prints the md5 of the pickled pipeline and of the coefficients, LSQR's iteration
count and the validation L1, then the md5 of the coefficients of the committed
`models/b1/pipeline.joblib`.

Nothing is written into the repository. Run from the repository root, with the
matrices and `models/b1` pulled:

    uv run python reports/analysis/ridge_blas_threads.py > reports/analysis/ridge_blas_threads_results.txt
"""

from datetime import datetime
import hashlib
import io
import os
import subprocess
import sys

THREAD_COUNTS = ("1", "2", "4", "8", "8")


def child() -> None:
    import joblib

    from recommenditos.config import PARAMS_FILE, PROCESSED_DATA_DIR
    from recommenditos.data.build_features import read_supported_makes
    from recommenditos.modeling.model import RequestContract, fit_variant
    from recommenditos.modeling.train import read_matrices
    from recommenditos.pipeline import load_params

    params = load_params(PARAMS_FILE)
    settings = params["train"]["variants"]["b1"]
    directory = PROCESSED_DATA_DIR / "features"
    data = read_matrices(
        directory,
        settings["feature_set"],
        read_supported_makes(directory.parent),
        request=RequestContract.from_params(params),
    )
    model = fit_variant(
        "b1", settings, data, seed=params["seed"], num_threads=params["train"]["num_threads"]
    )
    buffer = io.BytesIO()
    joblib.dump(model.pipeline, buffer)
    ridge = model.pipeline[-1]
    print(
        f"OPENBLAS_NUM_THREADS={os.environ['OPENBLAS_NUM_THREADS']}: "
        f"pipeline md5 {hashlib.md5(buffer.getvalue()).hexdigest()}, "
        f"coef md5 {hashlib.md5(ridge.coef_.tobytes()).hexdigest()}, "
        f"LSQR iterations {ridge.n_iter_[0]}, "
        f"validation L1 {model.metadata['training']['validation_l1_log_price']!r}"
    )


def main() -> None:
    if sys.argv[1:] == ["--child"]:
        child()
        return
    print(f"{datetime.now():%Y-%m-%d %H:%M}, {os.cpu_count()} logical CPUs")
    base = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    for count in THREAD_COUNTS:
        out = subprocess.run(
            [sys.executable, __file__, "--child"],
            env={**base, "OPENBLAS_NUM_THREADS": count},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        print(out.strip().splitlines()[-1], flush=True)

    import joblib

    committed = joblib.load("models/b1/pipeline.joblib")[-1]
    print(
        f"committed models/b1: coef md5 {hashlib.md5(committed.coef_.tobytes()).hexdigest()}, "
        f"LSQR iterations {committed.n_iter_[0]}"
    )


if __name__ == "__main__":
    main()
