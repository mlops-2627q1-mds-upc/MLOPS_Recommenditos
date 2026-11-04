"""The monitoring jobs compute the drift features with the code the reference was built with.

The `monitoring_samples` stage writes the drift reference with
`recommenditos/data/drift_features.py`; the replay (#227) and the drift job
(#228) run in their own uv project (`monitoring/`, numpy 1.26 and pandas 2 for
Alibi Detect), which cannot install this package, so they carry a copy. A copy
that drifted would shift the registration dates or compute the features of the
log differently from the reference's, and every window would be compared with
something else. This test is in the main suite, so the required Tests job runs it.
"""

from recommenditos.config import PROJ_ROOT

ORIGINAL = PROJ_ROOT / "recommenditos" / "data" / "drift_features.py"
COPY = PROJ_ROOT / "monitoring" / "recommenditos_monitoring" / "drift_features.py"


def test_the_monitoring_package_carries_drift_features_byte_for_byte():
    assert COPY.read_bytes() == ORIGINAL.read_bytes(), (
        f"{COPY.relative_to(PROJ_ROOT)} differs from {ORIGINAL.relative_to(PROJ_ROOT)}: "
        "copy the original over it (cp recommenditos/data/drift_features.py "
        "monitoring/recommenditos_monitoring/)"
    )
