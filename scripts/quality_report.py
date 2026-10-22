"""Genera las cifras del informe de calidad de `docs/data_quality.md`.

    uv run python scripts/quality_report.py

Todo sale de los snapshots en disco, así que las cifras del documento son
comprobables y se pueden recalcular cuando el poller lleve más tiempo.
"""

from __future__ import annotations

import argparse
import sys

import polars as pl

from parking.config import get_settings
from parking.quality import (
    annotate_quality,
    continuity_report,
    coverage_by_hour,
    coverage_by_zone,
    flapping_sensors,
    load_series,
    occupancy,
    quality_summary,
    snapshot_timestamps,
)


def section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stuck-after-hours",
        type=float,
        default=None,
        help="umbral de sensor atascado (por defecto, el de la configuración)",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    threshold = args.stuck_after_hours or settings.stuck_after_hours

    stamps = snapshot_timestamps(settings.realtime_raw_dir)
    if not stamps:
        print("no hay snapshots en disco; arranca el poller primero", file=sys.stderr)
        return 1

    series = annotate_quality(load_series(settings.realtime_raw_dir), stuck_after_hours=threshold)
    latest_batch = series["feed_lastupdated"].max()
    latest = series.filter(pl.col("feed_lastupdated") == latest_batch)

    section("1. Continuidad de la serie capturada")
    report = continuity_report(
        stamps,
        expected_interval_seconds=settings.poll_interval_seconds,
        tolerance_seconds=settings.gap_after_seconds,
    )
    print(f"  snapshots capturados : {report['snapshots']}")
    print(f"  lotes esperados      : {report['expected']}")
    print(f"  completitud          : {report['completeness_pct']} %")
    print(f"  ventana              : {report['first']} -> {report['last']}")
    print(f"  huecos detectados    : {len(report['gaps'])}")
    for gap in report["gaps"][:10]:
        print(
            f"    {gap.previous} -> {gap.following}  ({gap.duration}, {gap.missing_cycles} lotes)"
        )

    section(f"2. Calidad del último snapshot ({latest_batch})")
    print(quality_summary(latest))
    reliable = occupancy(latest)
    unfiltered = occupancy(latest, reliable_only=False)
    print(f"\n  plazas nominales            : {latest.height}")
    print(f"  en linea (is_live)          : {int(latest['is_live'].sum())}")
    print(f"  fiables (is_reliable)       : {int(latest['is_reliable'].sum())}")
    print(f"\n  ocupacion solo fiables      : {reliable * 100:.1f} %")
    print(f"  ocupacion sin filtrar       : {unfiltered * 100:.1f} %")
    print(f"  sesgo por no filtrar        : {(unfiltered - reliable) * 100:+.1f} puntos")

    section("3. Sensibilidad al umbral de 'atascado'")
    print(f"  {'umbral':>10}  {'fiables':>8}  {'cobertura':>10}  {'ocupacion':>10}")
    for hours in (6, 12, 18, 24, 48, 168, 720):
        annotated = annotate_quality(
            latest.drop(["status_age_hours", "quality", "is_reliable"]), stuck_after_hours=hours
        )
        n = int(annotated["is_reliable"].sum())
        occ = occupancy(annotated)
        print(f"  {hours:>8} h  {n:>8}  {n / annotated.height * 100:>9.1f} %  {occ * 100:>9.1f} %")

    section("4. Cobertura por zona (zonas con >= 10 plazas)")
    by_zone = coverage_by_zone(latest, min_bays=10)
    print(f"  zonas con datos    : {by_zone.height}")
    print(f"  cobertura mediana  : {by_zone['coverage_pct'].median():.1f} %")
    dead = by_zone.filter(pl.col("coverage_pct") == 0)
    print(f"  zonas al 0 %       : {dead.height}")
    print("\n  las 10 peores:")
    print(by_zone.head(10))
    print("\n  las 5 mejores:")
    print(by_zone.sort("coverage_pct", descending=True).head(5))

    section("5. Cobertura por hora UTC")
    print(coverage_by_hour(series))

    section("6. Sensores intermitentes")
    flapping = flapping_sensors(series)
    print(f"  plazas que alternan fiable/no fiable: {flapping.height}")
    if flapping.height:
        print(flapping.head(10))

    section("7. Zonas enteras caidas en bloque")
    # Si el fallo fuera por sensor, una zona tendria una mezcla de estados.
    # Que haya zonas al 100% de un mismo fallo apunta al rele, no al sensor.
    uniform = (
        latest.group_by("zone_number")
        .agg(
            bays=pl.len(),
            offline=(pl.col("quality") == "offline").sum(),
            stuck=(pl.col("quality") == "stuck").sum(),
        )
        .filter(pl.col("bays") >= 5)
        .with_columns(
            all_offline=pl.col("offline") == pl.col("bays"),
            all_stuck=pl.col("stuck") == pl.col("bays"),
        )
    )
    print(f"  zonas de >=5 plazas          : {uniform.height}")
    print(f"  enteramente fuera de linea   : {int(uniform['all_offline'].sum())}")
    print(f"  enteramente atascadas        : {int(uniform['all_stuck'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
