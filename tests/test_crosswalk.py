"""Tests for `parking.ingest.crosswalk`.

The cases use real street names from both datasets, with the formatting
differences that motivate the normalisation: the archive writes uppercase
(`BOURKE STREET`) and the geometry dataset writes title case (`Bourke Street`).
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
    """`O'CONNELL STREET` exists in the archive; stripping the apostrophe would break the join."""
    assert normalise_street("O'Connell Street") == "O'CONNELL STREET"


def test_cross_streets_are_order_independent() -> None:
    """The two datasets do not list the bounding streets in the same order."""
    forward = segment_key("Collins Street", "Russell Street", "Exhibition Street")
    backward = segment_key("Collins Street", "Exhibition Street", "Russell Street")

    assert forward == backward
    assert forward == SegmentKey("COLLINS STREET", ("EXHIBITION STREET", "RUSSELL STREET"))


def test_incomplete_segment_yields_none() -> None:
    """Without both bounds the segment is unidentifiable: better nothing than a false join."""
    assert segment_key("Collins Street", "Russell Street", None) is None
    assert segment_key("Collins Street", None, None) is None
    assert segment_key(None, "A Street", "B Street") is None


def test_description_is_parsed() -> None:
    key = segment_key_from_description(
        "Exhibition Street between Little Collins Street and Bourke Street"
    )

    assert key == SegmentKey("EXHIBITION STREET", ("BOURKE STREET", "LITTLE COLLINS STREET"))


def test_description_that_does_not_match_the_pattern_yields_none() -> None:
    """1035 of the 29,053 real rows do not follow the 'X between A and B' format."""
    assert segment_key_from_description("Queen Victoria Market car park") is None
    assert segment_key_from_description("") is None
    assert segment_key_from_description(None) is None


def test_description_with_and_in_a_street_name() -> None:
    """The pattern is lazy on the street and greedy at the end: a trailing 'and' is fine."""
    key = segment_key_from_description("Flinders Street between Queen Street and Elizabeth Street")

    assert key == SegmentKey("FLINDERS STREET", ("ELIZABETH STREET", "QUEEN STREET"))


def test_historical_row_and_description_produce_the_same_key() -> None:
    """This is the property the entire crosswalk depends on."""
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

    assert {key, key} == {key}, "must work as a dictionary key for the crosswalk"
    assert str(key) == "COLLINS STREET between EXHIBITION STREET and RUSSELL STREET"
