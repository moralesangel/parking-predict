"""Tests de `parking.ingest.crosswalk`.

Los casos usan nombres de calle reales de los dos datasets, con las diferencias
de formato que motivan la normalización: el histórico escribe en mayúsculas
(`BOURKE STREET`) y la geometría en capitalización normal (`Bourke Street`).
"""

from __future__ import annotations

from parking.ingest.crosswalk import (
    SegmentKey,
    normalise_street,
    segment_key,
    segment_key_from_description,
    segment_key_from_historical_row,
)


def test_normalise_collapses_case_and_whitespace() -> None:
    assert normalise_street("  Little   Collins Street ") == "LITTLE COLLINS STREET"


def test_normalise_handles_missing_values() -> None:
    assert normalise_street(None) == ""
    assert normalise_street("   ") == ""


def test_normalise_keeps_apostrophes() -> None:
    """`O'CONNELL STREET` existe en el histórico; quitar el apóstrofo rompería el cruce."""
    assert normalise_street("O'Connell Street") == "O'CONNELL STREET"


def test_cross_streets_are_order_independent() -> None:
    """Los dos datasets no listan las calles delimitadoras en el mismo orden."""
    forward = segment_key("Collins Street", "Russell Street", "Exhibition Street")
    backward = segment_key("Collins Street", "Exhibition Street", "Russell Street")

    assert forward == backward
    assert forward == SegmentKey("COLLINS STREET", ("EXHIBITION STREET", "RUSSELL STREET"))


def test_incomplete_segment_yields_none() -> None:
    """Sin las dos delimitadoras el tramo no es identificable: mejor nada que un cruce falso."""
    assert segment_key("Collins Street", "Russell Street", None) is None
    assert segment_key("Collins Street", None, None) is None
    assert segment_key(None, "A Street", "B Street") is None


def test_description_is_parsed() -> None:
    key = segment_key_from_description(
        "Exhibition Street between Little Collins Street and Bourke Street"
    )

    assert key == SegmentKey("EXHIBITION STREET", ("BOURKE STREET", "LITTLE COLLINS STREET"))


def test_description_that_does_not_match_the_pattern_yields_none() -> None:
    """1035 de las 29 053 filas reales no siguen el formato 'X between A and B'."""
    assert segment_key_from_description("Queen Victoria Market car park") is None
    assert segment_key_from_description("") is None
    assert segment_key_from_description(None) is None


def test_description_with_and_in_a_street_name() -> None:
    """El patrón es no voraz en la calle pero voraz al final: 'and' en la segunda calle no rompe."""
    key = segment_key_from_description("Flinders Street between Queen Street and Elizabeth Street")

    assert key == SegmentKey("FLINDERS STREET", ("ELIZABETH STREET", "QUEEN STREET"))


def test_historical_row_and_description_produce_the_same_key() -> None:
    """Es la propiedad de la que depende el cruce entero."""
    row = {
        "StreetName": "BOURKE STREET",
        "BetweenStreet1": "RUSSELL STREET",
        "BetweenStreet2": "EXHIBITION STREET",
    }

    from_history = segment_key_from_historical_row(row)
    from_geometry = segment_key_from_description(
        "Bourke Street between Exhibition Street and Russell Street"
    )

    assert from_history == from_geometry


def test_historical_row_without_cross_streets_yields_none() -> None:
    row = {"StreetName": "BOURKE STREET", "BetweenStreet1": "", "BetweenStreet2": ""}

    assert segment_key_from_historical_row(row) is None


def test_segment_key_is_hashable_and_printable() -> None:
    key = segment_key("Collins Street", "Russell Street", "Exhibition Street")

    assert {key, key} == {key}, "debe servir de clave de diccionario para el cruce"
    assert str(key) == "COLLINS STREET between EXHIBITION STREET and RUSSELL STREET"
