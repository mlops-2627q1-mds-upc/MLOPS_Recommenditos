"""The input-drift test of NFR-11 on one window of logged requests, with Alibi Detect (#228).

Each of the 23 features of `drift_features.DRIFT_FEATURES` gets a detector of its
own, fitted once on the 10,000-listing reference of the `monitoring_samples`
stage (`data/monitoring/reference.parquet`):

- a numeric feature, `KSDrift` on its non-missing values, of the reference and of
  the window alike, because a missing value is not a number to compare. How often
  a feature is missing is its missing indicator's test;
- a categorical feature or a missing indicator, `ChiSquareDrift` on its values,
  `__MISSING__` and `missing` being values like any other. The chi-square runs on
  the categories of the reference and of the window together, so a category the
  reference never saw counts as drift rather than being dropped.

`TabularDrift`, which runs the same tests over a whole table, cannot leave out
the missing values of one column without dropping the row for every other, so
the job runs one detector per feature and applies the Bonferroni correction
itself: a feature has drifted when its p-value is below `THRESHOLD`, 0.005 / 23
(EDN-14), and the window is flagged when any feature has. That is the setup
`reports/analysis/nfr11_check.py` validated NFR-11 with in scipy, which Alibi
Detect runs underneath (`ks_2samp(mode="asymp")` and `chi2_contingency`).

`age_years__missing`, `mileage_km_raw__missing` and `power_kw__missing` come from
fields FR-01 requires, so they are `present` in every row of the reference and of
every window. `ChiSquareDrift` accepts such a one-category feature: its table has
no degree of freedom, and the p-value is 1, never a flag. They stay in the 23,
so the threshold is the one EDN-14 validated.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from recommenditos_monitoring.drift_features import (
    CATEGORICAL,
    DRIFT_FEATURES,
    MISSING_INDICATORS,
    NUMERIC,
)

#: NFR-11's family-wise significance level (EDN-14).
ALPHA = 0.005
#: The per-feature level after the Bonferroni correction over the 23 features.
THRESHOLD = ALPHA / len(DRIFT_FEATURES)

#: The names of the two tests in the store and the report.
KS, CHI_SQUARE = "ks", "chi2"

#: The fewest non-missing values, of the reference and of a window, a KS test is
#: run on; fewer give the p-value 1 (no evidence), as the validated emulation did.
MIN_KS_VALUES = 2


class Detector(Protocol):
    def predict(self, x: np.ndarray, *args: Any, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FeatureResult:
    """One feature's test on one window."""

    feature: str
    test: str
    p_value: float
    statistic: float
    #: The values the test compared: the non-missing ones for a KS test.
    n_values: int

    @property
    def drifted(self) -> bool:
        return self.p_value < THRESHOLD


class SamplesError(Exception):
    """The reference does not hold what the drift test compares."""


def read_reference(data_dir: Path) -> tuple[pd.DataFrame, str]:
    """The drift reference and its `reference_date`, from the samples of `monitoring_samples`.

    The manifest has to name the same features as this package's copy of
    `drift_features`, so that a reference of another feature set fails here
    rather than comparing different columns.
    """
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    features = manifest["features"]
    if (
        tuple(features["numeric"]) != NUMERIC
        or tuple(features["categorical"]) != CATEGORICAL
        or tuple(features["missing_indicators"]) != MISSING_INDICATORS
    ):
        raise SamplesError(
            f"{data_dir / 'manifest.json'} names other drift features than this job compares"
        )
    reference = pd.read_parquet(data_dir / "reference.parquet")
    absent = [name for name in DRIFT_FEATURES if name not in reference]
    if absent:
        raise SamplesError(f"the reference has no column {', '.join(absent)}")
    return reference, str(manifest["reference_date"])


def _numeric(values: pd.Series) -> np.ndarray:
    array = values.to_numpy(dtype="float64")
    return array[~np.isnan(array)]


def _categorical(values: pd.Series) -> np.ndarray:
    return values.astype("str").to_numpy(dtype=object).reshape(-1, 1)


class DriftTest:
    """The 23 detectors of NFR-11, fitted on the reference."""

    def __init__(self, reference: pd.DataFrame) -> None:
        # Imported here, not at the top: Alibi Detect takes seconds and about
        # 200 MiB to import, which the replay and the report do not need.
        from alibi_detect.cd import ChiSquareDrift, KSDrift  # noqa: PLC0415 - see above

        self.reference_rows = len(reference)
        self._reference_values: dict[str, int] = {}
        self._detectors: dict[str, Detector | None] = {}
        for name in NUMERIC:
            values = _numeric(reference[name])
            self._reference_values[name] = len(values)
            self._detectors[name] = (
                KSDrift(values, p_val=THRESHOLD) if len(values) >= MIN_KS_VALUES else None
            )
        for name in CATEGORICAL + MISSING_INDICATORS:
            self._detectors[name] = ChiSquareDrift(_categorical(reference[name]), p_val=THRESHOLD)

    def run(self, window: pd.DataFrame) -> list[FeatureResult]:
        """Each feature's result on `window`, in the order of `DRIFT_FEATURES`."""
        return [self._feature(name, window[name]) for name in DRIFT_FEATURES]

    def _feature(self, name: str, values: pd.Series) -> FeatureResult:
        detector = self._detectors[name]
        if name in NUMERIC:
            sample = _numeric(values)
            if detector is None or len(sample) < MIN_KS_VALUES:
                return FeatureResult(name, KS, 1.0, 0.0, len(sample))
            return _result(name, KS, detector.predict(sample), len(sample))
        if detector is None:
            raise SamplesError(f"{name} has no chi-square detector")
        return _result(name, CHI_SQUARE, detector.predict(_categorical(values)), len(values))


def _result(name: str, test: str, prediction: Mapping[str, Any], n_values: int) -> FeatureResult:
    data = prediction["data"]
    # One feature per detector, so each array holds one value; float32 in Alibi.
    return FeatureResult(
        feature=name,
        test=test,
        p_value=float(np.asarray(data["p_val"]).reshape(-1)[0]),
        statistic=float(np.asarray(data["distance"]).reshape(-1)[0]),
        n_values=n_values,
    )
