"""Tests for `parking.features.live` and `parking.api.prediction`.

What matters most to protect:

- The live features must carry **exactly** the columns the model was trained
  on. Missing one makes LightGBM fail; an extra one or a different order makes
  it score badly in silence.
- Only reliable readings count, same as phase 2.
- Gaps in the series are filled and flagged: a lag computed over a grid with
  holes would lie without warning.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from parking.api.prediction import (
    AVAILABLE_HORIZONS,
    PredictionService,
    PredictionUnavailableError,
    haversine_m,
    snap_horizon,
)
from parking.config import Settings
from parking.features.live import (
    LIVE_GRID_MINUTES,
    build_live_features,
    densify,
    latest_features,
    load_bay_coordinates,
    load_bay_segments,
    segment_occupancy,
)

BATCH = dt.datetime(2026, 10, 22, 12, 0, tzinfo=dt.UTC)
SEGMENT = "COLLINS STREET|EXHIBITION STREET|RUSSELL STREET"
DESCRIPTION = "Collins Street between Russell Street and Exhibition Street"


def reading(
    kerbsideid: int, *, occupied: bool, reliable: bool = True, batch: dt.datetime = BATCH
) -> dict:
    return {
        "kerbsideid": kerbsideid,
        "is_occupied": occupied,
        "feed_lastupdated": batch,
        "is_reliable": reliable,
    }


def series(*rows: dict) -> pl.DataFrame:
    return pl.DataFrame(list(rows))


def bays_json(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


# --- bay to segment map -----------------------------------------------------


def test_bay_segments_maps_kerbsideid_to_segment(tmp_path: Path) -> None:
    path = bays_json(
        tmp_path / "bays.json",
        [
            {
                "kerbsideid": "123",
                "roadsegmentdescription": DESCRIPTION,
                "latitude": -37.81,
                "longitude": 144.97,
            }
        ],
    )

    assert load_bay_segments(path) == {123: SEGMENT}


def test_bays_without_a_parseable_segment_are_skipped(tmp_path: Path) -> None:
    """Putting them in an approximate segment would contaminate its counts."""
    path = bays_json(
        tmp_path / "bays.json",
        [
            {"kerbsideid": "1", "roadsegmentdescription": "Queen Victoria Market car park"},
            {"kerbsideid": "", "roadsegmentdescription": "A Street between B Street and C Street"},
        ],
    )

    assert load_bay_segments(path) == {}


def test_coordinates_average_the_bays_of_a_segment(tmp_path: Path) -> None:
    path = bays_json(
        tmp_path / "bays.json",
        [
            {
                "kerbsideid": "1",
                "roadsegmentdescription": DESCRIPTION,
                "latitude": -37.80,
                "longitude": 144.96,
            },
            {
                "kerbsideid": "2",
                "roadsegmentdescription": DESCRIPTION,
                "latitude": -37.82,
                "longitude": 144.98,
            },
        ],
    )

    centroids = load_bay_coordinates(path)

    assert centroids["latitude"][0] == pytest.approx(-37.81)
    assert centroids["longitude"][0] == pytest.approx(144.97)
    assert centroids["bays"][0] == 2


# --- aggregation by segment -------------------------------------------------


def test_occupancy_counts_only_reliable_readings() -> None:
    """Phase 2 rule: a frozen sensor is not a free bay."""
    frame = series(
        reading(1, occupied=False),
        reading(2, occupied=True),
        reading(3, occupied=False, reliable=False),
    )

    result = segment_occupancy(frame, {1: SEGMENT, 2: SEGMENT, 3: SEGMENT})

    assert result["total_bays"].to_list() == [2]
    assert result["occupied_bays"].to_list() == [1]
    assert result["free_bays"].to_list() == [1]


def test_bays_without_a_segment_are_excluded() -> None:
    frame = series(reading(1, occupied=False), reading(99, occupied=True))

    result = segment_occupancy(frame, {1: SEGMENT})

    assert result["total_bays"].to_list() == [1]


def test_occupancy_takes_the_last_batch_within_a_bucket() -> None:
    """Two feed batches in the same 5-minute window: the final state wins."""
    later = BATCH + dt.timedelta(minutes=2)
    frame = series(
        reading(1, occupied=True, batch=BATCH),
        reading(1, occupied=False, batch=later),
    )

    result = segment_occupancy(frame, {1: SEGMENT})

    assert result.height == 1
    assert result["occupied_bays"].to_list() == [0]


def test_occupancy_on_an_empty_series() -> None:
    assert segment_occupancy(pl.DataFrame(), {1: SEGMENT}).is_empty()


def test_occupancy_without_any_mappable_bay() -> None:
    assert segment_occupancy(series(reading(9, occupied=True)), {1: SEGMENT}).is_empty()


# --- densification ----------------------------------------------------------


def occupancy_frame(timestamps: list[dt.datetime], free: list[int]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "segment_id": [SEGMENT] * len(free),
            "timestamp": timestamps,
            "total_bays": [5] * len(free),
            "occupied_bays": [5 - value for value in free],
            "free_bays": free,
        }
    )


def test_densify_fills_missing_instants() -> None:
    """Without this, `shift(1)` would jump to an arbitrary instant and the lag would lie."""
    stamps = [BATCH, BATCH + dt.timedelta(minutes=15)]

    result = densify(occupancy_frame(stamps, [3, 1]))

    assert result.height == 4  # 0, 5, 10, 15
    assert result["free_bays"].to_list() == [3, 3, 3, 1]


def test_densify_marks_the_filled_rows() -> None:
    stamps = [BATCH, BATCH + dt.timedelta(minutes=10)]

    result = densify(occupancy_frame(stamps, [3, 2]))

    assert result["is_interpolated"].to_list() == [False, True, False]


def test_densify_leaves_a_complete_series_untouched() -> None:
    stamps = [BATCH + dt.timedelta(minutes=LIVE_GRID_MINUTES * i) for i in range(3)]

    result = densify(occupancy_frame(stamps, [3, 2, 1]))

    assert result["is_interpolated"].to_list() == [False, False, False]
    assert result["free_bays"].to_list() == [3, 2, 1]


def test_densify_on_an_empty_frame() -> None:
    assert densify(pl.DataFrame()).is_empty()


# --- features ---------------------------------------------------------------


def test_live_features_match_the_training_columns() -> None:
    """This is the property that makes scoring with the trained model possible."""
    from parking.features.dataset import build_dataset

    stamps = [BATCH + dt.timedelta(minutes=5 * i) for i in range(20)]
    occupancy = densify(occupancy_frame(stamps, [3] * 20))

    live = build_live_features(occupancy)
    trained = build_dataset(occupancy.drop("is_interpolated"), {SEGMENT: frozenset()})

    expected = {c for c in trained.columns if not c.startswith(("target_h", "free_bays_h"))}
    assert expected - set(live.columns) == set()


def test_live_features_have_no_targets() -> None:
    """Live, the future does not exist yet."""
    stamps = [BATCH + dt.timedelta(minutes=5 * i) for i in range(12)]

    live = build_live_features(densify(occupancy_frame(stamps, [3] * 12)))

    assert not [c for c in live.columns if c.startswith("target_h")]


def test_latest_features_returns_one_row_per_segment() -> None:
    stamps = [BATCH + dt.timedelta(minutes=5 * i) for i in range(6)]
    live = build_live_features(densify(occupancy_frame(stamps, [5, 4, 3, 2, 1, 0])))

    latest = latest_features(live)

    assert latest.height == 1
    assert latest["free_bays"].to_list() == [0]


# --- horizon and distance ---------------------------------------------------


@pytest.mark.parametrize(
    ("eta", "expected"),
    [(0, 5), (3, 5), (7, 5), (8, 10), (12, 10), (14, 15), (22, 15), (30, 30), (90, 30)],
)
def test_snap_horizon_picks_the_nearest_trained_model(eta: float, expected: int) -> None:
    assert snap_horizon(eta) == expected


def test_snap_horizon_only_returns_trained_horizons() -> None:
    """Interpolating between two models would give an uncalibrated probability."""
    assert all(snap_horizon(eta) in AVAILABLE_HORIZONS for eta in range(0, 121, 7))


def test_haversine_is_zero_at_the_same_point() -> None:
    distance = haversine_m(-37.81, 144.96, np.array([-37.81]), np.array([144.96]))

    assert distance[0] == pytest.approx(0.0, abs=1e-6)


def test_haversine_matches_a_known_distance() -> None:
    """One degree of latitude is about 111 km."""
    distance = haversine_m(-37.0, 144.96, np.array([-38.0]), np.array([144.96]))

    assert distance[0] == pytest.approx(111_195, rel=0.01)


def test_haversine_is_vectorised() -> None:
    distances = haversine_m(-37.81, 144.96, np.array([-37.81, -37.82]), np.array([144.96, 144.97]))

    assert distances.shape == (2,)
    assert distances[1] > distances[0]


# --- service ----------------------------------------------------------------


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, data_dir=tmp_path)


def test_service_without_reference_data_is_unavailable(settings: Settings) -> None:
    with pytest.raises(PredictionUnavailableError, match="fetch_reference"):
        PredictionService(settings).bay_segments()


def test_service_without_a_model_is_unavailable(settings: Settings) -> None:
    with pytest.raises(PredictionUnavailableError, match="train_evaluate"):
        PredictionService(settings).model(15)


def _reference_file(settings: Settings) -> Path:
    directory = settings.data_dir / "reference"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "parking_bays.json"


def test_service_without_snapshots_is_unavailable(settings: Settings) -> None:
    """With geometry but no poller data, the message must point at the poller."""
    bays_json(
        _reference_file(settings),
        [
            {
                "kerbsideid": "1",
                "roadsegmentdescription": DESCRIPTION,
                "latitude": -37.81,
                "longitude": 144.97,
            }
        ],
    )

    with pytest.raises(PredictionUnavailableError, match="poller"):
        PredictionService(settings).features()
