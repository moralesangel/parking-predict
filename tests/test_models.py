"""Tests de `parking.models`.

Lo que más importa proteger:

- La persistencia es la referencia del plan y tiene que ser exactamente eso:
  el estado actual, sin aprender nada.
- Las métricas tienen que castigar lo que debe castigarse. Un modelo que
  contesta siempre "sí" acierta el 95 % de las veces, así que si la métrica no
  lo penaliza, no sirve.
- Ningún modelo puede mirar al futuro al entrenarse.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from parking.models.baselines import (
    CalibratedPersistence,
    HistoricalMeanBaseline,
    PersistenceBaseline,
    clip_probabilities,
)
from parking.models.evaluation import (
    breakdown,
    brier_score,
    calibration_curve,
    comparison_table,
    evaluate,
    log_loss,
    occupancy_bucket,
    skill_score,
)
from parking.models.gbm import DEFAULT_PARAMS, GradientBoostingModel

#: `min_data_in_leaf` vale 200 en produccion, que es lo correcto con 5 M de
#: filas pero impide cualquier division con los pocos cientos de un test: el
#: modelo no partiria ni una vez y predeciria una constante.
TINY_PARAMS = dict(DEFAULT_PARAMS) | {"min_data_in_leaf": 5, "num_leaves": 7}

T0 = dt.datetime(2019, 6, 1)


def frame(
    free: list[int],
    target: list[bool | None],
    *,
    segment: str = "A|B|C",
    day: int = 1,
    minute: int = 600,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "segment_id": [segment] * len(free),
            "timestamp": [T0 + dt.timedelta(minutes=5 * i) for i in range(len(free))],
            "free_bays": free,
            "total_bays": [6] * len(free),
            "occupancy_rate": [(6 - value) / 6 for value in free],
            "day_of_week": [day] * len(free),
            "minute_of_day": [minute] * len(free),
            "hour": [minute // 60] * len(free),
            "target_h5": target,
        }
    )


# --- persistencia -----------------------------------------------------------


def test_persistence_returns_the_current_state() -> None:
    model = PersistenceBaseline().fit(frame([2, 0, 1], [True, True, True]), "target_h5")

    assert model.predict_proba(frame([2, 0, 1], [None] * 3)).tolist() == [1.0, 0.0, 1.0]


def test_persistence_learns_nothing() -> None:
    """Es determinista a propósito: es la referencia, no un modelo."""
    optimistic = PersistenceBaseline().fit(frame([0, 0], [True, True]), "target_h5")
    pessimistic = PersistenceBaseline().fit(frame([0, 0], [False, False]), "target_h5")

    probe = frame([0], [None])
    assert optimistic.predict_proba(probe) == pessimistic.predict_proba(probe)


def test_calibrated_persistence_learns_the_survival_rate() -> None:
    """De las veces con sitio, ¿en qué fracción seguía habiéndolo?"""
    train = frame([3, 3, 3, 3, 0, 0], [True, True, True, False, True, False])

    model = CalibratedPersistence().fit(train, "target_h5")

    assert model.predict_proba(frame([3], [None]))[0] == pytest.approx(0.75)
    assert model.predict_proba(frame([0], [None]))[0] == pytest.approx(0.5)


def test_calibrated_persistence_ignores_null_targets() -> None:
    """Las últimas filas de cada tramo no tienen futuro observado."""
    train = frame([3, 3, 3], [True, True, None])

    model = CalibratedPersistence().fit(train, "target_h5")

    assert model.predict_proba(frame([3], [None]))[0] == pytest.approx(1.0)


def test_calibrated_persistence_falls_back_on_unseen_states() -> None:
    train = frame([3, 3], [True, False])

    model = CalibratedPersistence().fit(train, "target_h5")

    # Nunca vio un tramo lleno: usa la tasa global en vez de inventar un 0,5.
    assert model.predict_proba(frame([0], [None]))[0] == pytest.approx(0.5)


# --- media histórica --------------------------------------------------------


def test_historical_mean_uses_segment_day_and_time() -> None:
    morning = frame([3] * 4, [True, True, True, False], minute=600)
    evening = frame([3] * 4, [False, False, False, True], minute=1200)

    model = HistoricalMeanBaseline().fit(pl.concat([morning, evening]), "target_h5")

    assert model.predict_proba(frame([3], [None], minute=600))[0] == pytest.approx(0.75)
    assert model.predict_proba(frame([3], [None], minute=1200))[0] == pytest.approx(0.25)


def test_historical_mean_ignores_the_current_state() -> None:
    """Es el contrapunto de la persistencia: solo rutina, nada de situación."""
    train = frame([3] * 4, [True, True, True, True])

    model = HistoricalMeanBaseline().fit(train, "target_h5")

    full = model.predict_proba(frame([0], [None]))[0]
    empty = model.predict_proba(frame([6], [None]))[0]
    assert full == empty


def test_historical_mean_falls_back_through_the_hierarchy() -> None:
    """Un tramo que siempre está lleno no tiene un 50 % solo porque falte su martes."""
    train = frame([3] * 2, [False, False], day=1, minute=600)

    model = HistoricalMeanBaseline().fit(train, "target_h5")

    # Mismo tramo y franja, otro día: cae a la media del tramo+franja.
    assert model.predict_proba(frame([3], [None], day=5, minute=600))[0] == pytest.approx(0.0)


def test_historical_mean_falls_back_to_global_for_unknown_segments() -> None:
    train = frame([3] * 2, [True, False], segment="A|B|C")

    model = HistoricalMeanBaseline().fit(train, "target_h5")

    assert model.predict_proba(frame([3], [None], segment="X|Y|Z"))[0] == pytest.approx(0.5)


# --- métricas ---------------------------------------------------------------


def test_brier_is_zero_for_perfect_predictions() -> None:
    assert brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0


def test_brier_punishes_confident_mistakes_hardest() -> None:
    truth = np.array([1.0])
    assert brier_score(truth, np.array([0.0])) > brier_score(truth, np.array([0.4]))


def test_log_loss_is_finite_for_a_wrong_certainty() -> None:
    """Sin recorte sería infinito, y comparar infinitos no informa de nada."""
    value = log_loss(np.array([1.0]), np.array([0.0]))

    assert np.isfinite(value)
    assert value > 10


def test_clip_keeps_probabilities_inside_the_open_interval() -> None:
    clipped = clip_probabilities(np.array([0.0, 0.5, 1.0]))

    assert clipped[0] > 0
    assert clipped[2] < 1
    assert clipped[1] == 0.5


def test_accuracy_would_hide_a_useless_model_but_brier_does_not() -> None:
    """Con 95 % de positivos, contestar siempre 'sí' acierta el 95 %."""
    truth = np.array([1.0] * 95 + [0.0] * 5)
    always_yes = np.full(100, 1.0)
    honest = np.full(100, 0.95)

    assert (always_yes.round() == truth).mean() == 0.95
    assert brier_score(truth, honest) < brier_score(truth, always_yes)


def test_skill_score_is_positive_when_better_than_the_reference() -> None:
    assert skill_score(0.5, 1.0) == pytest.approx(0.5)
    assert skill_score(1.0, 1.0) == 0.0
    assert skill_score(2.0, 1.0) < 0


def test_evaluate_reports_the_base_rate() -> None:
    metrics = evaluate(np.array([1.0, 1.0, 0.0, 1.0]), np.array([0.9] * 4))

    assert metrics["base_rate"] == pytest.approx(0.75)
    assert metrics["n"] == 4


# --- calibración ------------------------------------------------------------


def test_calibration_curve_detects_overconfidence() -> None:
    truth = np.array([1.0] * 5 + [0.0] * 5)
    overconfident = np.full(10, 0.95)

    curve = calibration_curve(truth, overconfident)

    assert curve.height == 1
    assert curve["observed"][0] == pytest.approx(0.5)
    assert curve["gap"][0] < 0


def test_calibration_curve_is_flat_for_a_calibrated_model() -> None:
    truth = np.array([1.0] * 8 + [0.0] * 2)
    calibrated = np.full(10, 0.8)

    curve = calibration_curve(truth, calibrated)

    assert curve["gap"][0] == pytest.approx(0.0)


def test_calibration_curve_puts_probability_one_in_the_last_bin() -> None:
    curve = calibration_curve(np.array([1.0]), np.array([1.0]), bins=10)

    assert curve["bin_low"][0] == pytest.approx(0.9)


def test_calibration_curve_omits_empty_bins() -> None:
    """Un baseline emite dos valores: mostrar ocho filas vacías confundiría."""
    truth = np.array([1.0, 0.0])
    two_valued = np.array([1.0, 0.0])

    assert calibration_curve(truth, two_valued).height == 2


# --- desgloses --------------------------------------------------------------


def test_breakdown_splits_by_group() -> None:
    data = frame([3, 3, 0, 0], [True, True, False, False]).with_columns(
        hour=pl.Series([9, 9, 18, 18])
    )
    truth = np.array([1.0, 1.0, 0.0, 0.0])
    predictions = {"perfect": truth.copy(), "always_yes": np.ones(4)}

    result = breakdown(data, truth, predictions, by="hour").sort("hour")

    assert result["hour"].to_list() == [9, 18]
    assert result["perfect"].to_list() == [0.0, 0.0]
    assert result["always_yes"].to_list() == [0.0, 1.0]


def test_occupancy_bucket_separates_full_from_nearly_full() -> None:
    data = frame([0, 1, 2, 4, 9], [None] * 5).with_columns(occupancy_bucket())

    assert data["occupancy_bucket"].to_list() == [
        "0 libres (lleno)",
        "1 libre",
        "2 libres",
        "3-5 libres",
        "6+ libres",
    ]


def test_comparison_table_is_sorted_and_scores_skill() -> None:
    results = {
        "persistence": {"brier": 0.04, "log_loss": 0.4},
        "lightgbm": {"brier": 0.02, "log_loss": 0.1},
    }

    table = comparison_table(results)

    assert table["model"].to_list() == ["lightgbm", "persistence"]
    assert table["brier_skill_vs_persistence"][0] == pytest.approx(0.5)


# --- LightGBM ---------------------------------------------------------------


def test_gbm_learns_a_separable_pattern() -> None:
    free = [0, 0, 0, 0, 5, 5, 5, 5] * 40
    target = [False, False, False, False, True, True, True, True] * 40
    train = frame(free, target)

    model = GradientBoostingModel(
        features=["free_bays", "occupancy_rate"], params=TINY_PARAMS, num_boost_round=30
    ).fit(train, "target_h5")

    predictions = model.predict_proba(frame([0, 5], [None, None]))
    assert predictions[0] < 0.5 < predictions[1]


def test_gbm_requires_fitting_first() -> None:
    model = GradientBoostingModel(features=["free_bays"])

    with pytest.raises(RuntimeError, match="fit"):
        model.predict_proba(frame([1], [None]))


def test_gbm_ignores_rows_without_a_target() -> None:
    free = [0, 5] * 60
    target: list[bool | None] = [False, True] * 55 + [None, None] * 5
    train = frame(free, target)

    model = GradientBoostingModel(
        features=["free_bays"], params=TINY_PARAMS, num_boost_round=20
    ).fit(train, "target_h5")

    assert np.isfinite(model.predict_proba(frame([0], [None]))).all()


def test_gbm_importances_sum_to_one_hundred_percent() -> None:
    free = [0, 5] * 60
    train = frame(free, [False, True] * 60)

    model = GradientBoostingModel(
        features=["free_bays", "occupancy_rate"], params=TINY_PARAMS, num_boost_round=20
    ).fit(train, "target_h5")

    assert model.importances()["share_pct"].sum() == pytest.approx(100.0, abs=0.1)
