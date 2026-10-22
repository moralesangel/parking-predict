"""Tests for `parking.models.registry` and `parking.monitoring`.

What matters most to protect:

- **A model that does not beat persistence is never promoted.** It is the only
  safeguard between a bad automated retrain and production.
- Rejections are recorded just like promotions.
- The backtest does not evaluate over rows the gap-filling invented.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from parking.models.registry import (
    REGRESSION_TOLERANCE,
    ModelRegistry,
    decide,
)
from parking.monitoring.backtest import backtest, coverage_trend, realised_targets
from parking.monitoring.drift import (
    INVESTIGATE_ABOVE,
    occupancy_drift,
    population_stability_index,
)

T0 = dt.datetime(2026, 10, 22, 12, 0)
SEGMENT = "A|B|C"


# --- promotion gate ---------------------------------------------------------


def test_model_that_loses_to_persistence_is_never_promoted() -> None:
    """The central safeguard of the whole automated retraining loop."""
    decision = decide(candidate_brier=0.05, persistence_brier=0.04, production_brier=None)

    assert decision.promoted is False
    assert "does not beat persistence" in decision.reason


def test_tying_with_persistence_is_not_enough() -> None:
    """A tie means the ML pipeline adds nothing."""
    assert decide(0.04, 0.04, None).promoted is False


def test_first_model_only_needs_to_beat_persistence() -> None:
    decision = decide(candidate_brier=0.03, persistence_brier=0.04, production_brier=None)

    assert decision.promoted is True
    assert "first model" in decision.reason


def test_clear_improvement_over_production_is_promoted() -> None:
    decision = decide(candidate_brier=0.02, persistence_brier=0.04, production_brier=0.03)

    assert decision.promoted is True
    assert "improves" in decision.reason


def test_regression_beyond_tolerance_is_rejected() -> None:
    """Retraining must not be a step backwards."""
    production = 0.020
    decision = decide(production * 1.5, persistence_brier=0.09, production_brier=production)

    assert decision.promoted is False
    assert "regresses" in decision.reason


def test_small_regression_within_tolerance_is_accepted() -> None:
    """Without a margin the model would change over differences that mean nothing."""
    production = 0.020
    candidate = production * (1 + REGRESSION_TOLERANCE / 2)

    assert decide(candidate, 0.09, production).promoted is True


def test_decision_reports_whether_it_beat_persistence() -> None:
    assert decide(0.03, 0.04, None).beats_persistence is True
    assert decide(0.05, 0.04, None).beats_persistence is False


# --- registry ---------------------------------------------------------------


def test_history_records_rejections_too(tmp_path: Path) -> None:
    """Knowing it has gone weeks without promoting is operational information."""
    registry = ModelRegistry(tmp_path)

    registry.record(decide(0.05, 0.04, None), version="v1")
    registry.record(decide(0.03, 0.04, None), version="v2")

    history = registry.history()
    assert [entry["promoted"] for entry in history] == [False, True]
    assert [entry["version"] for entry in history] == ["v1", "v2"]


def test_history_is_empty_before_any_run(tmp_path: Path) -> None:
    assert ModelRegistry(tmp_path).history() == []


def test_no_production_metrics_before_promotion(tmp_path: Path) -> None:
    assert ModelRegistry(tmp_path).production_metrics() is None


def test_promote_archives_and_points_production(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "model.txt").write_text("trees", encoding="utf-8")
    registry = ModelRegistry(tmp_path / "registry")

    registry.promote(candidate, version="v1", metrics={"brier": 0.03})

    assert (registry.production_dir / "model.txt").read_text(encoding="utf-8") == "trees"
    assert (registry.versions_dir / "v1" / "model.txt").exists()
    assert registry.production_metrics() == {"brier": 0.03}


def test_promote_replaces_the_previous_production(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    for version, content in (("v1", "old"), ("v2", "new")):
        candidate = tmp_path / version
        candidate.mkdir()
        (candidate / "model.txt").write_text(content, encoding="utf-8")
        registry.promote(candidate, version=version, metrics={"brier": 0.03})

    assert (registry.production_dir / "model.txt").read_text(encoding="utf-8") == "new"
    # The previous version stays archived: rolling back is possible.
    assert (registry.versions_dir / "v1" / "model.txt").read_text(encoding="utf-8") == "old"


# --- drift ------------------------------------------------------------------


def test_identical_distributions_have_no_drift() -> None:
    values = np.repeat(np.linspace(0, 1, 11), 50)

    result = population_stability_index(values, values.copy())

    assert result.psi == pytest.approx(0.0, abs=1e-9)
    assert result.level == "stable"
    assert result.needs_attention is False


def test_shifted_distribution_is_flagged() -> None:
    """A half-empty city against a half-full one has to fire."""
    empty = np.random.default_rng(0).uniform(0.0, 0.3, 5000)
    full = np.random.default_rng(1).uniform(0.7, 1.0, 5000)

    result = population_stability_index(empty, full)

    assert result.psi > INVESTIGATE_ABOVE
    assert result.level == "large"
    assert result.needs_attention is True


def test_psi_is_finite_when_a_bin_is_empty() -> None:
    """Without the epsilon an empty band would give log(0) and an infinite PSI."""
    reference = np.full(100, 0.05)
    current = np.full(100, 0.95)

    assert np.isfinite(population_stability_index(reference, current).psi)


def test_drift_bins_sum_to_one_hundred_percent() -> None:
    result = population_stability_index(
        np.random.default_rng(0).uniform(0, 1, 1000),
        np.random.default_rng(1).uniform(0, 1, 1000),
    )

    assert result.bins["reference_pct"].sum() == pytest.approx(100, abs=0.5)
    assert result.bins["current_pct"].sum() == pytest.approx(100, abs=0.5)


def test_occupancy_drift_needs_the_column() -> None:
    with pytest.raises(ValueError, match="occupancy_rate"):
        occupancy_drift(pl.DataFrame({"x": [1.0]}), pl.DataFrame({"x": [1.0]}))


def test_occupancy_drift_ignores_nulls() -> None:
    reference = pl.DataFrame({"occupancy_rate": [0.5, 0.5, None]})
    current = pl.DataFrame({"occupancy_rate": [0.5, 0.5]})

    result = occupancy_drift(reference, current)

    assert result.reference_n == 2
    assert result.psi == pytest.approx(0.0, abs=1e-9)


# --- backtest ---------------------------------------------------------------


def live_frame(free: list[int], *, interpolated: list[bool] | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "segment_id": [SEGMENT] * len(free),
            "timestamp": [T0 + dt.timedelta(minutes=5 * i) for i in range(len(free))],
            "free_bays": free,
            "is_interpolated": interpolated or [False] * len(free),
        }
    )


def test_realised_target_looks_forward() -> None:
    result = realised_targets(live_frame([2, 1, 0, 3]), horizon_minutes=5)

    assert result["realised_free_bays"].to_list() == [1, 0, 3, None]
    assert result["realised_target"].to_list() == [True, False, True, None]


def test_backtest_compares_against_persistence() -> None:
    frame = live_frame([3, 0, 3, 0, 3, 0])
    perfect = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])

    result = backtest(frame, perfect, horizon_minutes=5)

    assert result is not None
    assert result.model_brier == 0.0
    # Persistence is wrong every time over an alternating series.
    assert result.persistence_brier == 1.0
    assert result.is_degraded is False


def test_backtest_flags_a_model_worse_than_persistence() -> None:
    """The alarm is `is_degraded`, and it does not depend on the skill score.

    Over a constant series persistence is always right (Brier 0) and the skill
    is undefined: a division by zero. `is_degraded` is still correct, and that
    is what raises the alert.
    """
    frame = live_frame([3, 3, 3, 3])
    pessimistic = np.zeros(4)

    result = backtest(frame, pessimistic, horizon_minutes=5)

    assert result is not None
    assert result.is_degraded is True
    assert np.isnan(result.skill_vs_persistence)


def test_skill_is_negative_when_the_model_loses_to_an_imperfect_reference() -> None:
    """Persistence is wrong two times out of five; the model, four."""
    frame = live_frame([3, 3, 3, 3, 0, 3])
    always_full = np.zeros(6)

    result = backtest(frame, always_full, horizon_minutes=5)

    assert result is not None
    assert result.persistence_brier == pytest.approx(0.4)
    assert result.model_brier == pytest.approx(0.8)
    assert result.skill_vs_persistence == pytest.approx(-1.0)


def test_backtest_excludes_interpolated_rows() -> None:
    """Evaluating them would measure the gap-filling, not the model."""
    frame = live_frame([3, 3, 3, 3], interpolated=[False, True, True, False])
    probabilities = np.full(4, 0.9)

    result = backtest(frame, probabilities, horizon_minutes=5)

    assert result is not None
    assert result.n == 1  # only the first: the last has no observed future


def test_backtest_can_include_interpolated_rows() -> None:
    frame = live_frame([3, 3, 3, 3], interpolated=[False, True, True, False])

    result = backtest(frame, np.full(4, 0.9), horizon_minutes=5, exclude_interpolated=False)

    assert result is not None
    assert result.n == 3


def test_backtest_returns_none_without_realised_outcomes() -> None:
    """Better nothing than metrics over zero rows."""
    frame = live_frame([3, 3])

    assert backtest(frame, np.full(2, 0.9), horizon_minutes=30) is None


def test_backtest_result_serialises() -> None:
    result = backtest(live_frame([3, 0, 3]), np.array([0.1, 0.9, 0.1]), horizon_minutes=5)

    assert result is not None
    payload = result.as_dict()
    assert payload["horizon_minutes"] == 5
    assert "skill_vs_persistence" in payload
    assert isinstance(payload["is_degraded"], bool)


# --- coverage over time -----------------------------------------------------


def test_coverage_trend_buckets_by_hour() -> None:
    rows = []
    for index, reliable in enumerate([True, True, False, False]):
        rows.append(
            {
                "feed_lastupdated": T0 + dt.timedelta(hours=index // 2, minutes=index),
                "is_reliable": reliable,
            }
        )

    trend = coverage_trend(pl.DataFrame(rows))

    assert trend.height == 2
    assert trend["coverage_pct"].to_list() == [100.0, 0.0]


def test_coverage_trend_on_empty_input() -> None:
    assert coverage_trend(pl.DataFrame()).is_empty()
