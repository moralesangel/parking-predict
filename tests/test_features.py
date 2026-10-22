"""Tests de `parking.features`.

Lo que más importa proteger:

- `VehiclePresent` separa intervalos ocupados de intervalos libres. Tratarlos
  igual triplica la ocupación y aplana el ciclo diario, sin que nada falle.
- Ninguna feature puede mirar al futuro.
- El split es temporal, nunca aleatorio.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from parking.features.dataset import (
    add_lag_features,
    add_neighbour_features,
    add_targets,
    add_time_features,
    build_dataset,
    feature_columns,
    temporal_split,
)
from parking.features.occupancy import (
    clean_events,
    occupancy_series,
    parse_events,
    time_grid,
    with_segment_id,
)
from parking.features.segments import (
    build_adjacency,
    intersections,
    parse_segment_id,
    segment_id,
)
from parking.ingest.crosswalk import SegmentKey

T0 = dt.datetime(2019, 3, 1, 0, 0)


def key(street: str, first: str, second: str) -> SegmentKey:
    pair = tuple(sorted((first, second)))
    return SegmentKey(street, (pair[0], pair[1]))


# --- segments ---------------------------------------------------------------


def test_segment_has_two_intersections() -> None:
    nodes = intersections(key("COLLINS STREET", "RUSSELL STREET", "EXHIBITION STREET"))

    assert set(nodes) == {
        frozenset({"COLLINS STREET", "RUSSELL STREET"}),
        frozenset({"COLLINS STREET", "EXHIBITION STREET"}),
    }


def test_consecutive_blocks_are_neighbours() -> None:
    a = key("COLLINS STREET", "RUSSELL STREET", "EXHIBITION STREET")
    b = key("COLLINS STREET", "EXHIBITION STREET", "SPRING STREET")

    adjacency = build_adjacency([a, b])

    assert adjacency[a] == frozenset({b})


def test_perpendicular_segments_at_a_corner_are_neighbours() -> None:
    a = key("COLLINS STREET", "RUSSELL STREET", "EXHIBITION STREET")
    corner = key("EXHIBITION STREET", "COLLINS STREET", "LITTLE COLLINS STREET")

    adjacency = build_adjacency([a, corner])

    assert adjacency[a] == frozenset({corner})


def test_distant_segments_of_the_same_street_are_not_neighbours() -> None:
    """Definir la vecindad por 'misma calle' uniría los dos extremos del CBD."""
    west = key("COLLINS STREET", "KING STREET", "WILLIAM STREET")
    east = key("COLLINS STREET", "RUSSELL STREET", "SPRING STREET")

    adjacency = build_adjacency([west, east])

    assert adjacency[west] == frozenset()


def test_segment_is_never_its_own_neighbour() -> None:
    a = key("COLLINS STREET", "RUSSELL STREET", "EXHIBITION STREET")

    assert a not in build_adjacency([a, a])[a]


def test_segment_id_roundtrips() -> None:
    a = key("COLLINS STREET", "RUSSELL STREET", "EXHIBITION STREET")

    assert parse_segment_id(segment_id(a)) == a


# --- occupancy --------------------------------------------------------------


def raw_row(
    bay: str,
    arrival: str,
    departure: str,
    *,
    present: bool = True,
    street: str = "COLLINS STREET",
    cross_a: str = "RUSSELL STREET",
    cross_b: str = "EXHIBITION STREET",
) -> dict:
    return {
        "StreetMarker": bay,
        "StreetName": street,
        "BetweenStreet1": cross_a,
        "BetweenStreet2": cross_b,
        "VehiclePresent": present,
        "ArrivalTime": arrival,
        "DepartureTime": departure,
    }


def events(*rows: dict) -> pl.LazyFrame:
    return with_segment_id(clean_events(parse_events(pl.DataFrame(list(rows)).lazy())))


def test_us_date_format_is_parsed_as_month_first() -> None:
    """`11/03/2019` es el 3 de noviembre. Una inferencia automática fallaría aquí."""
    parsed = parse_events(
        pl.DataFrame([raw_row("1A", "11/03/2019 03:00:56 AM", "11/03/2019 04:00:00 AM")]).lazy()
    ).collect()

    assert parsed["arrival"][0] == dt.datetime(2019, 11, 3, 3, 0, 56)


def test_pm_times_are_parsed_as_afternoon() -> None:
    parsed = parse_events(
        pl.DataFrame([raw_row("1A", "04/16/2019 02:14:47 PM", "04/16/2019 03:00:00 PM")]).lazy()
    ).collect()

    assert parsed["arrival"][0].hour == 14


def test_vacancy_intervals_do_not_count_as_occupancy() -> None:
    """El bug que aplanaba el ciclo diario: sumar ocupado y libre da el 100 %."""
    occupied = events(raw_row("1A", "03/01/2019 12:00:00 AM", "03/01/2019 01:00:00 AM"))
    vacant = events(
        raw_row("1A", "03/01/2019 12:00:00 AM", "03/01/2019 01:00:00 AM", present=False)
    )

    with_car = occupancy_series(occupied, T0, T0 + dt.timedelta(hours=1))
    without = occupancy_series(vacant, T0, T0 + dt.timedelta(hours=1))

    assert with_car["occupied_bays"].to_list() == [1] * 12
    assert without["occupied_bays"].to_list() == [0] * 12


def test_vacant_only_bays_still_count_in_the_denominator() -> None:
    """Una plaza libre toda la ventana existe: omitirla vaciaría el denominador."""
    frame = events(
        raw_row("1A", "03/01/2019 12:00:00 AM", "03/01/2019 01:00:00 AM"),
        raw_row("2A", "03/01/2019 12:00:00 AM", "03/01/2019 01:00:00 AM", present=False),
    )

    series = occupancy_series(frame, T0, T0 + dt.timedelta(hours=1))

    assert series["total_bays"].unique().to_list() == [2]
    assert series["free_bays"].unique().to_list() == [1]


def test_stay_started_before_the_window_still_occupies() -> None:
    """Filtrar por `arrival >= start` haría parecer el tramo más vacío."""
    frame = events(raw_row("1A", "02/28/2019 10:00:00 PM", "03/01/2019 02:00:00 AM"))

    series = occupancy_series(frame, T0, T0 + dt.timedelta(hours=1))

    assert series["occupied_bays"].to_list() == [1] * 12


def test_occupancy_follows_arrivals_and_departures() -> None:
    frame = events(
        raw_row("1A", "03/01/2019 12:00:00 AM", "03/01/2019 12:20:00 AM"),
        raw_row("2A", "03/01/2019 12:10:00 AM", "03/01/2019 12:40:00 AM"),
    )

    series = occupancy_series(frame, T0, T0 + dt.timedelta(minutes=45))

    # 00:00 +1, 00:10 +1, 00:20 -1, 00:40 -1
    assert series["occupied_bays"].to_list() == [1, 1, 2, 2, 1, 1, 1, 1, 0]


def test_reversed_and_zero_length_intervals_are_dropped() -> None:
    """166 y 21 046 filas reales; un intervalo negativo corrompe la suma acumulada."""
    frame = events(
        raw_row("1A", "03/01/2019 02:00:00 AM", "03/01/2019 01:00:00 AM"),
        raw_row("2A", "03/01/2019 01:00:00 AM", "03/01/2019 01:00:00 AM"),
    )

    assert frame.collect().height == 0


def test_segment_id_is_order_independent() -> None:
    forward = events(raw_row("1A", "03/01/2019 12:00:00 AM", "03/01/2019 01:00:00 AM"))
    backward = events(
        raw_row(
            "1A",
            "03/01/2019 12:00:00 AM",
            "03/01/2019 01:00:00 AM",
            cross_a="EXHIBITION STREET",
            cross_b="RUSSELL STREET",
        )
    )

    assert forward.collect()["segment_id"][0] == backward.collect()["segment_id"][0]


def test_segment_id_matches_the_crosswalk_rule() -> None:
    """La misma regla que usa el cruce con la geometría, en polars."""
    produced = events(raw_row("1A", "03/01/2019 12:00:00 AM", "03/01/2019 01:00:00 AM"))

    expected = segment_id(key("COLLINS STREET", "RUSSELL STREET", "EXHIBITION STREET"))
    assert produced.collect()["segment_id"][0] == expected


def test_time_grid_is_half_open() -> None:
    grid = time_grid(T0, T0 + dt.timedelta(minutes=15))

    assert grid["timestamp"].to_list() == [
        T0,
        T0 + dt.timedelta(minutes=5),
        T0 + dt.timedelta(minutes=10),
    ]


# --- dataset ----------------------------------------------------------------


def occupancy_frame(free: list[int], *, total: int = 5, segment: str = "A|B|C") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "segment_id": [segment] * len(free),
            "timestamp": [T0 + dt.timedelta(minutes=5 * i) for i in range(len(free))],
            "free_bays": free,
            "total_bays": [total] * len(free),
            "occupied_bays": [total - value for value in free],
        }
    )


def test_targets_look_forward_by_the_horizon() -> None:
    frame = add_targets(occupancy_frame([3, 2, 1, 0, 2]), horizons=(5, 10))

    assert frame["free_bays_h5"].to_list() == [2, 1, 0, 2, None]
    # El tercer instante tiene 0 plazas libres en t+5: el objetivo es False.
    assert frame["target_h5"].to_list() == [True, True, False, True, None]
    assert frame["free_bays_h10"].to_list() == [1, 0, 2, None, None]


def test_target_is_false_only_when_the_segment_is_full() -> None:
    frame = add_targets(occupancy_frame([1, 0, 1]), horizons=(5,))

    assert frame["target_h5"].to_list() == [False, True, None]


def test_targets_at_the_end_are_null_not_imputed() -> None:
    """Rellenarlos inventaría etiquetas."""
    frame = add_targets(occupancy_frame([3, 3]), horizons=(30,))

    assert frame["target_h30"].to_list() == [None, None]


def test_lags_look_backward(tmp_path: object) -> None:
    frame = occupancy_frame([5, 4, 3, 2]).with_columns(
        occupancy_rate=pl.col("occupied_bays") / pl.col("total_bays")
    )

    result = add_lag_features(frame, lags=(5, 10))

    assert result["free_bays_lag5"].to_list() == [None, 5, 4, 3]
    assert result["free_bays_lag10"].to_list() == [None, None, 5, 4]


def test_trends_capture_direction() -> None:
    """Dos tramos con la misma ocupación y futuros opuestos no son iguales."""
    emptying = occupancy_frame([1, 2, 3]).with_columns(
        occupancy_rate=pl.col("occupied_bays") / pl.col("total_bays")
    )

    result = add_lag_features(emptying, lags=(5,))

    assert result["free_bays_trend5"].to_list() == [None, 1, 1]


def test_lags_do_not_leak_across_segments() -> None:
    first = occupancy_frame([5, 4], segment="A|B|C")
    second = occupancy_frame([1, 0], segment="X|Y|Z")
    frame = pl.concat([first, second]).with_columns(
        occupancy_rate=pl.col("occupied_bays") / pl.col("total_bays")
    )

    result = add_lag_features(frame, lags=(5,))

    assert result["free_bays_lag5"].to_list() == [None, 5, None, 1]


def test_time_features_are_cyclical() -> None:
    frame = add_time_features(occupancy_frame([1]))

    assert frame["hour"].to_list() == [0]
    assert frame["minute_of_day"].to_list() == [0]
    assert frame["minute_cos"][0] == pytest.approx(1.0)


def test_weekend_and_holiday_flags() -> None:
    saturday = dt.datetime(2019, 3, 2)
    frame = pl.DataFrame({"segment_id": ["A|B|C"], "timestamp": [saturday]})

    result = add_time_features(frame, holidays=(dt.date(2019, 3, 2),))

    assert result["is_weekend"].to_list() == [True]
    assert result["is_holiday"].to_list() == [True]


def test_neighbour_features_aggregate_the_same_instant() -> None:
    frame = pl.concat(
        [occupancy_frame([2, 2], segment="A|B|C"), occupancy_frame([4, 4], segment="X|B|Z")]
    ).with_columns(occupancy_rate=pl.col("occupied_bays") / pl.col("total_bays"))

    result = add_neighbour_features(frame, {"A|B|C": frozenset({"X|B|Z"}), "X|B|Z": frozenset()})

    first = result.filter(pl.col("segment_id") == "A|B|C")
    assert first["neighbour_free_bays"].to_list() == [4, 4]
    assert first["neighbour_count"].to_list() == [1, 1]


def test_segments_without_neighbours_get_nulls_not_zeros() -> None:
    """Un cero diría 'los vecinos están llenos', que es una afirmación."""
    frame = occupancy_frame([2]).with_columns(
        occupancy_rate=pl.col("occupied_bays") / pl.col("total_bays")
    )

    result = add_neighbour_features(frame, {"A|B|C": frozenset()})

    assert result["neighbour_free_bays"].to_list() == [None]


def test_feature_columns_exclude_every_target() -> None:
    frame = build_dataset(occupancy_frame([3] * 10), {"A|B|C": frozenset()})

    features = feature_columns(frame)

    assert not [name for name in features if name.startswith(("target_h", "free_bays_h"))]
    assert "segment_id" not in features
    assert "timestamp" not in features
    assert "free_bays" in features


def test_build_dataset_keeps_targets_consistent_with_counts() -> None:
    frame = build_dataset(occupancy_frame([3, 0, 2, 1, 4, 0]), {"A|B|C": frozenset()})

    mismatched = frame.filter(
        pl.col("target_h5").is_not_null() & (pl.col("target_h5") != (pl.col("free_bays_h5") >= 1))
    )
    assert mismatched.height == 0


# --- split ------------------------------------------------------------------


def test_split_is_temporal_and_ordered() -> None:
    frame = build_dataset(occupancy_frame([3] * 24), {"A|B|C": frozenset()})

    split = temporal_split(
        frame,
        train_end=T0 + dt.timedelta(minutes=50),
        validation_end=T0 + dt.timedelta(minutes=80),
    )

    assert split.train["timestamp"].max() < split.validation["timestamp"].min()
    assert split.validation["timestamp"].max() < split.test["timestamp"].min()


def test_split_covers_every_row_exactly_once() -> None:
    frame = build_dataset(occupancy_frame([3] * 24), {"A|B|C": frozenset()})

    split = temporal_split(
        frame,
        train_end=T0 + dt.timedelta(minutes=50),
        validation_end=T0 + dt.timedelta(minutes=80),
    )

    assert split.train.height + split.validation.height + split.test.height == frame.height


def test_split_rejects_inverted_boundaries() -> None:
    frame = build_dataset(occupancy_frame([3] * 10), {"A|B|C": frozenset()})

    with pytest.raises(ValueError, match="posterior"):
        temporal_split(frame, train_end=T0 + dt.timedelta(hours=2), validation_end=T0)


def test_split_summary_reports_each_block() -> None:
    frame = build_dataset(occupancy_frame([3] * 24), {"A|B|C": frozenset()})
    split = temporal_split(
        frame,
        train_end=T0 + dt.timedelta(minutes=50),
        validation_end=T0 + dt.timedelta(minutes=80),
    )

    summary = split.summary()

    assert summary["split"].to_list() == ["train", "validation", "test"]
    assert summary["rows"].sum() == frame.height
