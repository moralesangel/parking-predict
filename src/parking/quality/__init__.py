"""Detection of offline sensors, stuck sensors and network gaps.

Structural rule: no function in this package drops rows. They all add columns
or return aggregated reports, so that coverage stays measurable (see
docs/data_quality.md).
"""

from parking.quality.coverage import coverage_by_hour, coverage_by_zone, flapping_sensors
from parking.quality.flags import (
    QUALITY_VALUES,
    annotate_quality,
    occupancy,
    quality_summary,
)
from parking.quality.gaps import continuity_report, find_gaps, load_series, snapshot_timestamps

__all__ = [
    "QUALITY_VALUES",
    "annotate_quality",
    "continuity_report",
    "coverage_by_hour",
    "coverage_by_zone",
    "find_gaps",
    "flapping_sensors",
    "load_series",
    "occupancy",
    "quality_summary",
    "snapshot_timestamps",
]
