"""Produce the figures behind the quality report in `docs/data_quality.md`.

    uv run python scripts/quality_report.py

Everything comes from the snapshots on disk, so the document's numbers are
checkable and can be recomputed once the poller has been running longer.
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
        help="stuck-sensor threshold (defaults to the configured value)",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    threshold = args.stuck_after_hours or settings.stuck_after_hours

    stamps = snapshot_timestamps(settings.realtime_raw_dir)
    if not stamps:
        print("no snapshots on disk; start the poller first", file=sys.stderr)
        return 1

    series = annotate_quality(load_series(settings.realtime_raw_dir), stuck_after_hours=threshold)
    latest_batch = series["feed_lastupdated"].max()
    latest = series.filter(pl.col("feed_lastupdated") == latest_batch)

    section("1. Continuity of the captured series")
    report = continuity_report(
        stamps,
        expected_interval_seconds=settings.poll_interval_seconds,
        tolerance_seconds=settings.gap_after_seconds,
    )
    print(f"  snapshots captured : {report['snapshots']}")
    print(f"  batches expected   : {report['expected']}")
    print(f"  completeness       : {report['completeness_pct']} %")
    print(f"  window             : {report['first']} -> {report['last']}")
    print(f"  gaps detected      : {len(report['gaps'])}")
    for gap in report["gaps"][:10]:
        print(
            f"    {gap.previous} -> {gap.following}  ({gap.duration}, {gap.missing_cycles} batches)"
        )

    section(f"2. Quality of the latest snapshot ({latest_batch})")
    print(quality_summary(latest))
    reliable = occupancy(latest)
    unfiltered = occupancy(latest, reliable_only=False)
    print(f"\n  nominal bays               : {latest.height}")
    print(f"  online (is_live)           : {int(latest['is_live'].sum())}")
    print(f"  reliable (is_reliable)     : {int(latest['is_reliable'].sum())}")
    print(f"\n  occupancy, reliable only   : {reliable * 100:.1f} %")
    print(f"  occupancy, unfiltered      : {unfiltered * 100:.1f} %")
    print(f"  bias from not filtering    : {(unfiltered - reliable) * 100:+.1f} points")

    section("3. Sensitivity to the 'stuck' threshold")
    print(f"  {'threshold':>10}  {'reliable':>8}  {'coverage':>10}  {'occupancy':>10}")
    for hours in (6, 12, 18, 24, 48, 168, 720):
        annotated = annotate_quality(
            latest.drop(["status_age_hours", "quality", "is_reliable"]), stuck_after_hours=hours
        )
        n = int(annotated["is_reliable"].sum())
        occ = occupancy(annotated)
        print(f"  {hours:>8} h  {n:>8}  {n / annotated.height * 100:>9.1f} %  {occ * 100:>9.1f} %")

    section("4. Coverage per zone (zones with >= 10 bays)")
    by_zone = coverage_by_zone(latest, min_bays=10)
    print(f"  zones with data    : {by_zone.height}")
    print(f"  median coverage    : {by_zone['coverage_pct'].median():.1f} %")
    dead = by_zone.filter(pl.col("coverage_pct") == 0)
    print(f"  zones at 0 %       : {dead.height}")
    print("\n  worst 10:")
    print(by_zone.head(10))
    print("\n  best 5:")
    print(by_zone.sort("coverage_pct", descending=True).head(5))

    section("5. Coverage per UTC hour")
    print(coverage_by_hour(series))

    section("6. Flapping sensors")
    flapping = flapping_sensors(series)
    print(f"  bays alternating reliable/unreliable: {flapping.height}")
    if flapping.height:
        print(flapping.head(10))

    section("7. Zones failing as a block")
    # If the failure were per sensor, a zone would show a mix of states. Zones
    # that are 100% one failure point at the relay, not the sensor.
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
    print(f"  zones with >= 5 bays    : {uniform.height}")
    print(f"  entirely offline        : {int(uniform['all_offline'].sum())}")
    print(f"  entirely stuck          : {int(uniform['all_stuck'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
