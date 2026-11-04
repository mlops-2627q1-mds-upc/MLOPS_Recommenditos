"""Price predictions for used-car listings.

Every pipeline stage runs as `python3 -m recommenditos.<module>`, so this file
runs before any stage module, and before anything they import, loads numpy.
"""

import importlib
import os
import platform
import sys
import warnings

#: numpy's AVX-512 dispatch targets on x86-64 (numpy 2.5's names, the ones
#: `numpy._core._multiarray_umath.__cpu_dispatch__` lists besides X86_V3).
#: Their kernels round differently: on an AMD EPYC 9V74 numpy's AVX-512 `log`
#: put `log_price` one unit in the last place off for 2 listings, which moved
#: `b1` (#259). Without them numpy's float64 `log` and `exp` are the C
#: library's, so on Linux x86-64 every CPU computes glibc's values, which the
#: laptop's i5-10210U, without AVX-512, computes anyway.
NUMPY_DISABLED_CPU_FEATURES = "X86_V4 AVX512_ICL AVX512_SPR"

# numpy reads this variable once, when it is first imported, so it comes before
# `config` and everything else. Naming a target the CPU lacks is silent, naming
# one the build does not dispatch is an ImportWarning, so it is set only where
# numpy has these targets: on x86-64, but not on Windows, whose numpy 2.5.4
# wheel dispatches nothing beyond X86_V3.
# A value the caller set wins.
if platform.machine().lower() in {"x86_64", "amd64"} and sys.platform != "win32":
    os.environ.setdefault("NPY_DISABLE_CPU_FEATURES", NUMPY_DISABLED_CPU_FEATURES)

from recommenditos import config  # noqa: E402, F401 - after the variable above

#: Set to 1 to compute pipeline outputs while numpy's AVX-512 kernels are on.
ANY_NUMPY_KERNEL_VARIABLE = "RECOMMENDITOS_ANY_NUMPY_KERNEL"


def check_numpy_kernels() -> None:
    """Refuse to compute pipeline outputs on numpy's AVX-512 kernels (NFR-06, #259).

    Setting NPY_DISABLE_CPU_FEATURES above takes effect only in a process that
    imports `recommenditos` before numpy, as every stage does. A notebook or a
    script that imports pandas first, or a caller that set the variable to
    something else, runs the AVX-512 kernels on a CPU that has them, and its
    `log_price` would differ from the pipeline's; this fails there instead.
    `ANY_NUMPY_KERNEL_VARIABLE` computes anyway, with a warning, for the test
    suite, whose process loads numpy before any test runs. On a CPU without
    AVX-512 the targets are never on, so this passes whatever the import order.
    """
    multiarray = importlib.import_module("numpy._core._multiarray_umath")
    features = vars(multiarray)["__cpu_features__"]
    on = [name for name in NUMPY_DISABLED_CPU_FEATURES.split() if features.get(name)]
    if not on:
        return
    message = (
        f"numpy runs its AVX-512 kernels ({', '.join(on)}) in this process, which round "
        "some logarithms differently from the C library's log that the pipeline computes "
        "everywhere else (#259). numpy reads NPY_DISABLE_CPU_FEATURES once, when it is "
        "first imported: run the stage through `dvc repro` or `python -m recommenditos...`, "
        "import recommenditos before pandas or numpy, or start Python with "
        f'NPY_DISABLE_CPU_FEATURES="{NUMPY_DISABLED_CPU_FEATURES}".'
    )
    if os.environ.get(ANY_NUMPY_KERNEL_VARIABLE) == "1":
        warnings.warn(
            f"{message} Computing anyway ({ANY_NUMPY_KERNEL_VARIABLE}=1): the results "
            "differ from those of a pipeline run.",
            stacklevel=2,
        )
        return
    raise RuntimeError(
        f"{message} Set {ANY_NUMPY_KERNEL_VARIABLE}=1 to compute anyway; the results then "
        "differ from those of a pipeline run."
    )
