"""Drift detection on the occupancy distribution.

The model was trained on a particular distribution. If today's drifts away, the
probabilities stop being calibrated even when the model still ranks well, and
in this product calibration **is** the deliverable.

The metric is **PSI** (Population Stability Index), the operational standard
for this: it compares two histograms over the same cut points and summarises
the difference in one number.

    PSI = sum_i (actual_i - reference_i) * ln(actual_i / reference_i)

Thresholds widely used in industry, adopted here as-is:

| PSI | Reading |
|---|---|
| < 0.10 | stable |
| 0.10 to 0.25 | moderate shift, keep watching |
| > 0.25 | large shift, review the model |

They are convention, not statistical truth: there is no hypothesis test behind
them. They exist to trigger a review, not to conclude anything on their own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

#: Occupancy histogram cut points. Fixed rather than quantiles of the current
#: sample: with moving cuts, two genuinely different distributions could score
#: PSI zero because each one defines its own bands.
OCCUPANCY_BINS = (0.0, 0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95, 1.0001)

STABLE_BELOW = 0.10
INVESTIGATE_ABOVE = 0.25

#: Avoids log(0) when a band comes out empty in one of the two samples.
_EPSILON = 1e-6


@dataclass(frozen=True)
class DriftResult:
    """Comparison of two distributions."""

    psi: float
    reference_n: int
    current_n: int
    bins: pl.DataFrame

    @property
    def level(self) -> str:
        if self.psi < STABLE_BELOW:
            return "stable"
        if self.psi < INVESTIGATE_ABOVE:
            return "moderate"
        return "large"

    @property
    def needs_attention(self) -> bool:
        return self.psi >= STABLE_BELOW


def _histogram(values: np.ndarray, bins: tuple[float, ...]) -> np.ndarray:
    counts, _ = np.histogram(values, bins=list(bins))
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), _EPSILON)
    return np.maximum(counts / total, _EPSILON)


def population_stability_index(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: tuple[float, ...] = OCCUPANCY_BINS,
) -> DriftResult:
    """PSI between a reference sample and a current one.

    `reference` is usually the training distribution and `current` the recent
    window.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]

    reference_share = _histogram(reference, bins)
    current_share = _histogram(current, bins)
    contributions = (current_share - reference_share) * np.log(current_share / reference_share)

    detail = pl.DataFrame(
        {
            "bin_low": list(bins[:-1]),
            "bin_high": [min(edge, 1.0) for edge in bins[1:]],
            "reference_pct": (reference_share * 100).round(2),
            "current_pct": (current_share * 100).round(2),
            "contribution": contributions.round(5),
        }
    )
    return DriftResult(
        psi=float(contributions.sum()),
        reference_n=int(reference.size),
        current_n=int(current.size),
        bins=detail,
    )


def occupancy_drift(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    *,
    column: str = "occupancy_rate",
) -> DriftResult:
    """Drift of the occupancy rate between two datasets.

    **Occupancy** is compared rather than the target because it is the signal
    that dominates the model: 90% of the gain comes from the current state and
    its recent mean (see `docs/results.md` §5). If that distribution moves, the
    model has lost its footing.
    """
    for frame, name in ((reference, "reference"), (current, "current")):
        if column not in frame.columns:
            raise ValueError(f"missing column {column!r} in the {name} dataset")
    return population_stability_index(
        reference[column].drop_nulls().to_numpy(),
        current[column].drop_nulls().to_numpy(),
    )
