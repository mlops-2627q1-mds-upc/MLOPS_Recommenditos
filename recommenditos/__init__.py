"""Price predictions for used-car listings.

Every pipeline stage runs as `python3 -m recommenditos.<module>`, so this file
runs before any stage module, and before anything they import, loads numpy.
"""

import os
import platform
import sys

#: numpy's AVX-512 dispatch targets on x86-64 (numpy 2.5's names, the ones
#: `numpy._core._multiarray_umath.__cpu_dispatch__` lists besides X86_V3).
#: Their kernels round differently: on an AMD EPYC 9V74 numpy's AVX-512 `log`
#: put `log_price` one unit in the last place off for 2 listings, which moved
#: `b1` (#259). Without them a CPU with AVX-512 runs the AVX2 kernels the
#: laptop's i5-10210U runs, which every x86-64 CPU we run the pipeline on has
#: anyway (`b1`'s OpenBLAS Haswell kernel needs AVX2 too).
NUMPY_DISABLED_CPU_FEATURES = "X86_V4 AVX512_ICL AVX512_SPR"

# numpy reads this variable once, when it is first imported, so it comes before
# `config` and everything else. Naming a target the CPU lacks is silent, naming
# one the build does not dispatch is an ImportWarning, so it is set only where
# numpy has these targets: on x86-64, but not on Windows, whose numpy 2.5.4
# wheel dispatches nothing beyond X86_V3 and so runs the AVX2 kernels anyway.
# A value the caller set wins.
if platform.machine().lower() in {"x86_64", "amd64"} and sys.platform != "win32":
    os.environ.setdefault("NPY_DISABLE_CPU_FEATURES", NUMPY_DISABLED_CPU_FEATURES)

from recommenditos import config  # noqa: E402, F401 - after the variable above
