"""Tests de `parking.quality`.

El invariante que más importa: ninguna función borra filas. Un sensor malo se
marca y sigue en la tabla, porque si desaparece no se puede medir la cobertura.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import pytest

from parking.quality.coverage import coverage_by_hour, coverage_by_zone, flapping_sensors
from parking.quality.flags import (
    QUALITY_VALUES,
    annotate_quality,
    occupancy,
    quality_summary,
)
from parking.quality.gaps import (
    continuity_report,
    find_gaps,
    load_series,
    snapshot_timestamps,
)

BATCH = dt.datetime(2026, 10, 22, 12, 0, tzinfo=dt.UTC)


def reading(
    kerbsideid: int,
    *,
    occupied: bool = False,
    live: bool = True,
    status_age_hours: float = 1.0,
    zone: int | None = 7219,
    batch: dt.datetime = BATCH,
) -> dict:
    return {
        "kerbsideid": kerbsideid,
        "is_occupied": occupied,
        "status_timestamp": batch - dt.timedelta(hours=status_age_hours),
        "lastupdated": batch if live else batch - dt.timedelta(days=30),
        "feed_lastupdated": batch,
        "is_live": live,
        "zone_number": zone,
    }


def frame(*readings: dict) -> pl.DataFrame:
    return pl.DataFrame(list(readings))


# --- annotate_quality -------------------------------------------------------


def test_healthy_sensor_is_ok() -> None:
    result = annotate_quality(frame(reading(1, status_age_hours=2)))

    assert result["quality"].to_list() == ["ok"]
    assert result["is_reliable"].to_list() == [True]


def test_sensor_absent_from_the_batch_is_offline() -> None:
    result = annotate_quality(frame(reading(1, live=False)))

    assert result["quality"].to_list() == ["offline"]
    assert result["is_reliable"].to_list() == [False]


def test_reporting_but_frozen_sensor_is_stuck() -> None:
    """El fallo más numeroso: el relé lo reporta y el sensor está congelado."""
    result = annotate_quality(frame(reading(1, status_age_hours=150 * 24)))

    assert result["quality"].to_list() == ["stuck"]
    assert result["is_reliable"].to_list() == [False]


def test_offline_wins_over_stuck() -> None:
    """Si además está fuera del lote, la causa raíz es el relé: se marca offline."""
    result = annotate_quality(frame(reading(1, live=False, status_age_hours=500)))

    assert result["quality"].to_list() == ["offline"]


def test_threshold_boundary_is_exclusive() -> None:
    """Justo en el umbral todavía es fiable; un pelo por encima, no."""
    result = annotate_quality(
        frame(reading(1, status_age_hours=24.0), reading(2, status_age_hours=24.1)),
        stuck_after_hours=24.0,
    )

    assert result["quality"].to_list() == ["ok", "stuck"]


def test_threshold_is_configurable() -> None:
    readings = frame(reading(1, status_age_hours=10))

    assert annotate_quality(readings, stuck_after_hours=24)["quality"].to_list() == ["ok"]
    assert annotate_quality(readings, stuck_after_hours=6)["quality"].to_list() == ["stuck"]


def test_age_is_measured_against_the_batch_not_now() -> None:
    """Medir contra `now()` haría irreproducible el reprocesado de datos viejos."""
    old_batch = dt.datetime(2020, 1, 1, tzinfo=dt.UTC)
    result = annotate_quality(frame(reading(1, status_age_hours=3, batch=old_batch)))

    assert result["status_age_hours"].to_list() == [3.0]
    assert result["quality"].to_list() == ["ok"]


def test_nothing_is_ever_dropped() -> None:
    """Regla del plan: nunca borrar en silencio."""
    readings = frame(
        reading(1),
        reading(2, live=False),
        reading(3, status_age_hours=9999),
    )

    result = annotate_quality(readings)

    assert result.height == 3
    assert sorted(result["kerbsideid"].to_list()) == [1, 2, 3]


def test_quality_is_a_closed_enum() -> None:
    """El conjunto de estados es cerrado: no se puede escribir uno inventado.

    Ojo con lo que el Enum NO protege: comparar la columna con un valor
    inexistente devuelve 0 filas en silencio, no lanza. Contra eso solo valen
    las constantes de `QUALITY_VALUES`.
    """
    result = annotate_quality(frame(reading(1)))

    assert result.schema["quality"] == pl.Enum(QUALITY_VALUES)
    assert result.schema["quality"].categories.to_list() == list(QUALITY_VALUES)
    with pytest.raises(pl.exceptions.InvalidOperationError):
        result.with_columns(pl.lit("inventado").cast(result.schema["quality"]))


def test_missing_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="faltan columnas"):
        annotate_quality(pl.DataFrame({"kerbsideid": [1]}))


# --- resumen y ocupación ----------------------------------------------------


def test_quality_summary_counts_every_state() -> None:
    result = annotate_quality(
        frame(reading(1), reading(2), reading(3, live=False), reading(4, status_age_hours=900))
    )

    summary = quality_summary(result)

    counts = dict(zip(summary["quality"].to_list(), summary["bays"].to_list(), strict=True))
    assert counts == {"ok": 2, "offline": 1, "stuck": 1}
    assert summary.filter(pl.col("quality") == "ok")["share_pct"].item() == 50.0


def test_occupancy_excludes_unreliable_readings() -> None:
    """En el feed real la diferencia llega a 7,6 puntos en hora punta."""
    result = annotate_quality(
        frame(
            reading(1, occupied=False),
            reading(2, occupied=False),
            reading(3, occupied=True, live=False),
            reading(4, occupied=True, status_age_hours=900),
        )
    )

    assert occupancy(result) == 0.0
    assert occupancy(result, reliable_only=False) == 0.5


def test_occupancy_without_reliable_readings_is_none() -> None:
    """`None` significa 'no se sabe'; un 0.0 significaría 'todo libre'."""
    result = annotate_quality(frame(reading(1, live=False)))

    assert occupancy(result) is None


def test_summary_requires_annotation() -> None:
    with pytest.raises(ValueError, match="annotate_quality"):
        quality_summary(frame(reading(1)))


# --- cobertura --------------------------------------------------------------


def test_coverage_by_zone() -> None:
    result = annotate_quality(
        frame(
            reading(1, zone=7001),
            reading(2, zone=7001, live=False),
            reading(3, zone=7002),
            reading(4, zone=7002),
        )
    )

    coverage = coverage_by_zone(result)

    by_zone = dict(
        zip(coverage["zone_number"].to_list(), coverage["coverage_pct"].to_list(), strict=True)
    )
    assert by_zone == {7001: 50.0, 7002: 100.0}


def test_coverage_by_zone_keeps_bays_without_a_zone() -> None:
    """520 de 6324 plazas reales no tienen zona; descartarlas ocultaría el problema."""
    result = annotate_quality(frame(reading(1, zone=None), reading(2, zone=7001)))

    coverage = coverage_by_zone(result)

    assert None in coverage["zone_number"].to_list()


def test_coverage_by_zone_separates_bays_from_readings() -> None:
    """Con varios snapshots, 1 plaza leída 3 veces es 1 plaza, no 3."""
    batches = [BATCH + dt.timedelta(minutes=2 * i) for i in range(3)]
    result = annotate_quality(frame(*(reading(1, zone=7001, batch=b) for b in batches)))

    coverage = coverage_by_zone(result)

    assert coverage["bays"].to_list() == [1]
    assert coverage["readings"].to_list() == [3]


def test_coverage_by_zone_min_bays_filters_the_report() -> None:
    result = annotate_quality(
        frame(reading(1, zone=7001), reading(2, zone=7002), reading(3, zone=7002))
    )

    assert coverage_by_zone(result, min_bays=2)["zone_number"].to_list() == [7002]


def test_coverage_by_hour_groups_by_batch_hour() -> None:
    noon = annotate_quality(frame(reading(1, batch=BATCH), reading(2, batch=BATCH)))
    later = annotate_quality(frame(reading(1, batch=BATCH + dt.timedelta(hours=1), live=False)))

    coverage = coverage_by_hour(pl.concat([noon, later]))

    assert coverage["hour_utc"].to_list() == [12, 13]
    assert coverage["coverage_pct"].to_list() == [100.0, 0.0]


def test_coverage_requires_annotation() -> None:
    with pytest.raises(ValueError, match="annotate_quality"):
        coverage_by_zone(frame(reading(1)))


# --- sensores intermitentes -------------------------------------------------


def test_flapping_sensor_is_detected() -> None:
    """Un sensor que entra y sale mueve el denominador de su zona sin que se mueva un coche."""
    batches = [BATCH + dt.timedelta(minutes=2 * i) for i in range(4)]
    readings = [
        reading(1, batch=batches[0], live=True),
        reading(1, batch=batches[1], live=False),
        reading(1, batch=batches[2], live=True),
        reading(1, batch=batches[3], live=False),
        *(reading(2, batch=b, live=True) for b in batches),
    ]

    result = flapping_sensors(annotate_quality(frame(*readings)))

    assert result["kerbsideid"].to_list() == [1], "el sensor estable no debe aparecer"
    assert result["transitions"].item() == 3


def test_flapping_ignores_short_series() -> None:
    readings = [
        reading(1, live=True),
        reading(1, batch=BATCH + dt.timedelta(minutes=2), live=False),
    ]

    result = flapping_sensors(annotate_quality(frame(*readings)), min_snapshots=3)

    assert result.height == 0


# --- huecos -----------------------------------------------------------------


def stamps(*minutes: int) -> list[dt.datetime]:
    return [BATCH + dt.timedelta(minutes=m) for m in minutes]


def test_regular_series_has_no_gaps() -> None:
    assert find_gaps(stamps(0, 2, 4, 6)) == []


def test_single_cycle_delay_is_tolerated() -> None:
    """El feed se retrasa un ciclo de vez en cuando sin que falte nada."""
    assert find_gaps(stamps(0, 2, 6)) == []


def test_gap_is_detected_and_measured() -> None:
    gaps = find_gaps(stamps(0, 2, 22))

    assert len(gaps) == 1
    assert gaps[0].duration == dt.timedelta(minutes=20)
    assert gaps[0].missing_cycles == 9


def test_empty_and_single_element_series() -> None:
    assert find_gaps([]) == []
    assert find_gaps(stamps(0)) == []


def test_continuity_report_on_a_complete_series() -> None:
    report = continuity_report(stamps(0, 2, 4, 6))

    assert report["snapshots"] == 4
    assert report["expected"] == 4
    assert report["completeness_pct"] == 100.0
    assert report["gaps"] == []


def test_continuity_report_detects_missing_batches() -> None:
    report = continuity_report(stamps(0, 2, 20))

    assert report["snapshots"] == 3
    assert report["expected"] == 11
    assert report["completeness_pct"] == 27.3


def test_continuity_report_on_empty_series() -> None:
    report = continuity_report([])

    assert report["snapshots"] == 0
    assert report["completeness_pct"] is None


# --- lectura de disco -------------------------------------------------------


def test_snapshot_timestamps_are_read_from_filenames(tmp_path: Path) -> None:
    day = tmp_path / "date=2026-10-22"
    day.mkdir()
    for name in ("snapshot_20261022T120249Z", "snapshot_20261022T120049Z"):
        (day / f"{name}.parquet").touch()

    assert snapshot_timestamps(tmp_path) == [
        dt.datetime(2026, 10, 22, 12, 0, 49, tzinfo=dt.UTC),
        dt.datetime(2026, 10, 22, 12, 2, 49, tzinfo=dt.UTC),
    ]


def test_snapshot_timestamps_ignores_foreign_files(tmp_path: Path) -> None:
    """Un fichero ajeno no debe romper el informe de continuidad."""
    (tmp_path / "snapshot_no-es-una-fecha.parquet").touch()

    assert snapshot_timestamps(tmp_path) == []


def test_load_series_on_empty_dir_returns_empty_frame(tmp_path: Path) -> None:
    assert load_series(tmp_path).height == 0


def test_load_series_concatenates_snapshots(tmp_path: Path) -> None:
    day = tmp_path / "date=2026-10-22"
    day.mkdir()
    for index, minute in enumerate((0, 2)):
        batch = BATCH + dt.timedelta(minutes=minute)
        frame(reading(index, batch=batch)).write_parquet(
            day / f"snapshot_{batch:%Y%m%dT%H%M%SZ}.parquet"
        )

    assert load_series(tmp_path).height == 2
