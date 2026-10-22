"""API response models.

Declared explicitly rather than returning plain dictionaries so the OpenAPI
schema documents the semantics - in particular that of `quality`, which is the
part a consumer is most likely to misread.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum

from pydantic import BaseModel, Field


class Quality(StrEnum):
    """Reliability of a bay's reading.

    The three values come from `parking.quality.flags`. They are exposed as-is
    because the client needs to tell them apart: `offline` and `stuck` both
    mean "no reliable data", but their cause differs and it is worth showing.
    """

    OK = "ok"
    STUCK = "stuck"
    OFFLINE = "offline"


class BayState(StrEnum):
    """A bay's state as the map paints it.

    `UNKNOWN` is a first-class state, not a hole: 2767 of the feed's 6324 bays
    give no reliable reading, and painting them as free would invent bays that
    may not exist.
    """

    FREE = "free"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"


class Bay(BaseModel):
    """One bay in the current snapshot."""

    kerbsideid: int = Field(description="Bay identifier in the source feed")
    state: BayState = Field(description="free, occupied or unknown (no reliable reading)")
    quality: Quality = Field(description="Why the reading is or is not reliable")
    is_reliable: bool = Field(description="True only when quality == ok")
    latitude: float
    longitude: float
    zone_number: int | None = Field(
        default=None, description="Restriction zone; null for 520 of the 6324 bays"
    )
    status_age_hours: float = Field(
        description="Hours since the last state change, measured against the batch stamp"
    )


class QualityBreakdown(BaseModel):
    """Bay counts per quality state."""

    ok: int
    stuck: int
    offline: int


class Summary(BaseModel):
    """Snapshot summary. This is what makes the correct figure readable at a glance."""

    total_bays: int = Field(description="Bays published by the feed")
    reliable_bays: int = Field(description="Bays with a usable reading")
    free_bays: int = Field(description="Free bays among the reliable ones")
    occupied_bays: int = Field(description="Occupied bays among the reliable ones")
    occupancy_pct: float | None = Field(
        description="Occupancy over reliable bays. Null when there are none: "
        "a 0 would mean 'everything free', which is the opposite of 'unknown'"
    )
    coverage_pct: float = Field(description="Share of bays with a reliable reading")
    by_quality: QualityBreakdown


class SnapshotInfo(BaseModel):
    """How old the data being served is."""

    feed_lastupdated: dt.datetime = Field(description="Feed batch stamp, in UTC")
    served_at: dt.datetime
    age_seconds: float = Field(description="Age of the batch at serving time")
    is_stale: bool = Field(
        description="True when the batch is older than tolerable: the poller may be down"
    )


class LiveResponse(BaseModel):
    """Response of `GET /bays/live`."""

    snapshot: SnapshotInfo
    summary: Summary
    bays: list[Bay]
    attribution: str


class ZoneCoverage(BaseModel):
    """Sensor coverage of one zone."""

    zone_number: int | None
    bays: int
    reliable: int
    stuck: int
    offline: int
    coverage_pct: float


class ZonesResponse(BaseModel):
    """Response of `GET /zones/coverage`."""

    snapshot: SnapshotInfo
    zones_total: int
    zones_without_data: int = Field(description="Zones without a single reliable reading")
    zones: list[ZoneCoverage]
    attribution: str


class HealthResponse(BaseModel):
    """Response of `GET /health`."""

    status: str = Field(description="ok, stale or no_data")
    detail: str
    snapshot: SnapshotInfo | None = None


class ZonePrediction(BaseModel):
    """Probability of finding a space in one street segment."""

    segment_id: str = Field(description="STREET|CROSS_A|CROSS_B, the two bounds sorted")
    label: str = Field(description="The same segment in readable text")
    probability: float = Field(
        ge=0, le=1, description="P(at least one free bay on arrival), calibrated"
    )
    free_bays: int = Field(description="Free bays RIGHT NOW among the reliable ones")
    total_bays: int = Field(description="Bays with a reliable reading in the segment")
    latitude: float
    longitude: float
    distance_m: float = Field(description="Distance from the destination queried")
    is_interpolated: bool = Field(
        description="True when the last state was carried over from an earlier batch across a gap"
    )


class PredictResponse(BaseModel):
    """Response of `GET /predict`."""

    snapshot: SnapshotInfo
    eta_minutes: float = Field(description="What the client asked for")
    horizon_minutes: int = Field(
        description="Horizon actually used: snapped to the nearest trained model"
    )
    zones: list[ZonePrediction] = Field(
        description="Nearby segments ordered by distance, not by probability: "
        "ordering by probability would send everyone to the same spot"
    )
    model_caveat: str = Field(description="Why this is a demonstration and not a service")
    attribution: str
