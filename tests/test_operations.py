"""Tests de `parking.models.registry` y `parking.monitoring`.

Lo que más importa proteger:

- **Un modelo que no bate a la persistencia no se promociona jamás.** Es el
  único salvaguarda entre un reentrenamiento automático malo y producción.
- Los rechazos se registran igual que las promociones.
- El backtest no evalúa sobre filas inventadas por el relleno de huecos.
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


# --- puerta de promocion ----------------------------------------------------


def test_model_that_loses_to_persistence_is_never_promoted() -> None:
    """El salvaguarda central de todo el reentrenamiento automático."""
    decision = decide(candidate_brier=0.05, persistence_brier=0.04, production_brier=None)

    assert decision.promoted is False
    assert "no bate a la persistencia" in decision.reason


def test_tying_with_persistence_is_not_enough() -> None:
    """Empatar significa que el pipeline de ML no aporta nada."""
    assert decide(0.04, 0.04, None).promoted is False


def test_first_model_only_needs_to_beat_persistence() -> None:
    decision = decide(candidate_brier=0.03, persistence_brier=0.04, production_brier=None)

    assert decision.promoted is True
    assert "primer modelo" in decision.reason


def test_clear_improvement_over_production_is_promoted() -> None:
    decision = decide(candidate_brier=0.02, persistence_brier=0.04, production_brier=0.03)

    assert decision.promoted is True
    assert "mejora" in decision.reason


def test_regression_beyond_tolerance_is_rejected() -> None:
    """Reentrenar no debe ser un paso atrás."""
    production = 0.020
    decision = decide(production * 1.5, persistence_brier=0.09, production_brier=production)

    assert decision.promoted is False
    assert "empeora" in decision.reason


def test_small_regression_within_tolerance_is_accepted() -> None:
    """Sin margen, el modelo cambiaría por diferencias que no significan nada."""
    production = 0.020
    candidate = production * (1 + REGRESSION_TOLERANCE / 2)

    assert decide(candidate, 0.09, production).promoted is True


def test_decision_reports_whether_it_beat_persistence() -> None:
    assert decide(0.03, 0.04, None).beats_persistence is True
    assert decide(0.05, 0.04, None).beats_persistence is False


# --- registro ---------------------------------------------------------------


def test_history_records_rejections_too(tmp_path: Path) -> None:
    """Saber que lleva semanas sin promocionar es información operativa."""
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
    (candidate / "model.txt").write_text("arboles", encoding="utf-8")
    registry = ModelRegistry(tmp_path / "registry")

    registry.promote(candidate, version="v1", metrics={"brier": 0.03})

    assert (registry.production_dir / "model.txt").read_text(encoding="utf-8") == "arboles"
    assert (registry.versions_dir / "v1" / "model.txt").exists()
    assert registry.production_metrics() == {"brier": 0.03}


def test_promote_replaces_the_previous_production(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "registry")
    for version, content in (("v1", "viejo"), ("v2", "nuevo")):
        candidate = tmp_path / version
        candidate.mkdir()
        (candidate / "model.txt").write_text(content, encoding="utf-8")
        registry.promote(candidate, version=version, metrics={"brier": 0.03})

    assert (registry.production_dir / "model.txt").read_text(encoding="utf-8") == "nuevo"
    # La version anterior sigue archivada: se puede volver atras.
    assert (registry.versions_dir / "v1" / "model.txt").read_text(encoding="utf-8") == "viejo"


# --- drift ------------------------------------------------------------------


def test_identical_distributions_have_no_drift() -> None:
    values = np.repeat(np.linspace(0, 1, 11), 50)

    result = population_stability_index(values, values.copy())

    assert result.psi == pytest.approx(0.0, abs=1e-9)
    assert result.level == "estable"
    assert result.needs_attention is False


def test_shifted_distribution_is_flagged() -> None:
    """Una ciudad medio vacía frente a una medio llena tiene que saltar."""
    empty = np.random.default_rng(0).uniform(0.0, 0.3, 5000)
    full = np.random.default_rng(1).uniform(0.7, 1.0, 5000)

    result = population_stability_index(empty, full)

    assert result.psi > INVESTIGATE_ABOVE
    assert result.level == "fuerte"
    assert result.needs_attention is True


def test_psi_is_finite_when_a_bin_is_empty() -> None:
    """Sin el epsilon, un tramo vacío daría log(0) y el PSI sería infinito."""
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
    # La persistencia se equivoca siempre en una serie que alterna.
    assert result.persistence_brier == 1.0
    assert result.is_degraded is False


def test_backtest_flags_a_model_worse_than_persistence() -> None:
    """La alarma es `is_degraded`, y no depende del skill.

    Con una serie constante la persistencia acierta siempre (Brier 0) y el
    skill queda indefinido: dividir por cero. `is_degraded` sigue siendo
    correcto, que es lo que dispara el aviso.
    """
    frame = live_frame([3, 3, 3, 3])
    pessimistic = np.zeros(4)

    result = backtest(frame, pessimistic, horizon_minutes=5)

    assert result is not None
    assert result.is_degraded is True
    assert np.isnan(result.skill_vs_persistence)


def test_skill_is_negative_when_the_model_loses_to_an_imperfect_reference() -> None:
    """La persistencia falla dos de cinco veces; el modelo, cuatro."""
    frame = live_frame([3, 3, 3, 3, 0, 3])
    always_full = np.zeros(6)

    result = backtest(frame, always_full, horizon_minutes=5)

    assert result is not None
    assert result.persistence_brier == pytest.approx(0.4)
    assert result.model_brier == pytest.approx(0.8)
    assert result.skill_vs_persistence == pytest.approx(-1.0)


def test_backtest_excludes_interpolated_rows() -> None:
    """Evaluarlas mediría la calidad del relleno, no la del modelo."""
    frame = live_frame([3, 3, 3, 3], interpolated=[False, True, True, False])
    probabilities = np.full(4, 0.9)

    result = backtest(frame, probabilities, horizon_minutes=5)

    assert result is not None
    assert result.n == 1  # solo la primera: la ultima no tiene futuro observado


def test_backtest_can_include_interpolated_rows() -> None:
    frame = live_frame([3, 3, 3, 3], interpolated=[False, True, True, False])

    result = backtest(frame, np.full(4, 0.9), horizon_minutes=5, exclude_interpolated=False)

    assert result is not None
    assert result.n == 3


def test_backtest_returns_none_without_realised_outcomes() -> None:
    """Mejor nada que métricas sobre cero filas."""
    frame = live_frame([3, 3])

    assert backtest(frame, np.full(2, 0.9), horizon_minutes=30) is None


def test_backtest_result_serialises() -> None:
    result = backtest(live_frame([3, 0, 3]), np.array([0.1, 0.9, 0.1]), horizon_minutes=5)

    assert result is not None
    payload = result.as_dict()
    assert payload["horizon_minutes"] == 5
    assert "skill_vs_persistence" in payload
    assert isinstance(payload["is_degraded"], bool)


# --- cobertura en el tiempo -------------------------------------------------


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
