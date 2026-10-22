"""Crosswalk between the historical sensor archive and the current datasets.

The historical archive shares no identifier with the real-time feed and carries
no coordinates, so there is no way to join them at bay level. There is one at
**street segment** level, using the street name and the two cross streets that
bound it: it covers 88% of the historical events. The full report with the
measurements lives in `docs/id_crosswalk.md`.

This module implements the segment key that makes that join possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: `roadsegmentdescription` follows the format
#: "Exhibition Street between Little Collins Street and Bourke Street".
_SEGMENT_DESCRIPTION = re.compile(
    r"^(?P<street>.+?) between (?P<first>.+?) and (?P<second>.+)$", re.I
)

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, order=True)
class SegmentKey:
    """Canonical key for a street segment, comparable across datasets.

    The pair of cross streets is stored **alphabetically sorted**, because
    "Collins between Russell and Exhibition" and "Collins between Exhibition
    and Russell" are the same segment, and the two datasets do not agree on the
    order.
    """

    street: str
    cross_streets: tuple[str, str]

    def __str__(self) -> str:
        return f"{self.street} between {self.cross_streets[0]} and {self.cross_streets[1]}"


def normalise_street(name: str | None) -> str:
    """Normalise a street name so it can be compared across datasets.

    The archive writes `BOURKE STREET` and the geometry dataset writes
    `Bourke Street`, with inconsistent whitespace. Both are folded to uppercase
    with collapsed spaces. Apostrophes are left alone (`O'CONNELL STREET`) and
    `STREET` is not abbreviated: both datasets are consistent there, and
    "normalising" further would invent equivalences nobody has verified.
    """
    if not name:
        return ""
    return _WHITESPACE.sub(" ", name.strip()).upper()


def segment_key(street: str | None, first: str | None, second: str | None) -> SegmentKey | None:
    """Build the segment key, or `None` if any of the three names is missing.

    A segment without both cross streets is not identifiable, so it is dropped
    rather than turned into a partial key that would join against the wrong
    segments.
    """
    street_norm = normalise_street(street)
    first_norm = normalise_street(first)
    second_norm = normalise_street(second)
    if not (street_norm and first_norm and second_norm):
        return None
    pair = tuple(sorted((first_norm, second_norm)))
    return SegmentKey(street_norm, (pair[0], pair[1]))


def segment_key_from_description(description: str | None) -> SegmentKey | None:
    """Segment key from a `roadsegmentdescription`.

    Returns `None` when the description does not follow the "X between A and B"
    pattern. 1035 of the 29,053 rows in `on-street-parking-bays` do not.
    """
    if not description:
        return None
    match = _SEGMENT_DESCRIPTION.match(description.strip())
    if match is None:
        return None
    return segment_key(match["street"], match["first"], match["second"])


def segment_key_from_historical_row(row: dict[str, str]) -> SegmentKey | None:
    """Segment key from a row of the historical CSV.

    Uses `StreetName`, `BetweenStreet1` and `BetweenStreet2`.
    """
    return segment_key(row.get("StreetName"), row.get("BetweenStreet1"), row.get("BetweenStreet2"))
