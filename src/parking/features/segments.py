"""Spatial structure: street segments and their adjacency.

The project's spatial unit is the **street segment**, not an H3 cell. The
justification is measured in `docs/id_crosswalk.md` §5: it is the only level at
which the historical archive and the real-time feed reconcile, because the
archive carries no coordinates. It also matches the driver's actual decision
— "is there space on this block?" — better than an arbitrary hexagon.

This module derives **adjacency** between segments without any geometry, from
the street names themselves. That is what makes "neighbouring zone" features
(phase 5) computable over the historical archive, which has no latitude or
longitude.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from parking.ingest.crosswalk import SegmentKey

#: An intersection is the unordered pair of the two streets that cross there.
#: `frozenset` rather than a tuple because "Collins at Russell" and "Russell at
#: Collins" are the same corner.
Intersection = frozenset[str]


def intersections(segment: SegmentKey) -> tuple[Intersection, Intersection]:
    """The two ends of a segment, as intersections.

    `COLLINS STREET between EXHIBITION STREET and RUSSELL STREET` occupies the
    corners `{COLLINS, EXHIBITION}` and `{COLLINS, RUSSELL}`.
    """
    first, second = segment.cross_streets
    return (
        frozenset({segment.street, first}),
        frozenset({segment.street, second}),
    )


def build_adjacency(segments: Iterable[SegmentKey]) -> dict[SegmentKey, frozenset[SegmentKey]]:
    """Adjacency graph: two segments are neighbours if they share an intersection.

    This definition captures the two cases that matter while driving, and
    rejects the one that does not:

    - **Adjacent block**: `COLLINS between RUSSELL and EXHIBITION` and
      `COLLINS between EXHIBITION and SPRING` share the corner
      `{COLLINS, EXHIBITION}`.
    - **Corner**: `EXHIBITION between COLLINS and LITTLE COLLINS` shares that
      same corner with the first one, even though it runs perpendicular.
    - **Same street but far apart**: two Collins Street segments at opposite
      ends of the CBD share no corner and are **not** neighbours. Defining
      adjacency as "same street" would have joined them, which is precisely
      the mistake to avoid.

    A segment is never its own neighbour.
    """
    by_intersection: dict[Intersection, set[SegmentKey]] = defaultdict(set)
    segments = list(segments)
    for segment in segments:
        for node in intersections(segment):
            by_intersection[node].add(segment)

    adjacency: dict[SegmentKey, set[SegmentKey]] = {segment: set() for segment in segments}
    for sharing in by_intersection.values():
        for segment in sharing:
            adjacency[segment].update(sharing - {segment})

    return {segment: frozenset(neighbours) for segment, neighbours in adjacency.items()}


def segment_id(segment: SegmentKey) -> str:
    """Stable, human-readable identifier for a segment, used as a table key.

    Readable on purpose: a `segment_id` you can read means inspecting the
    dataset or debugging a prediction does not require another join.
    """
    first, second = segment.cross_streets
    return f"{segment.street}|{first}|{second}"


def parse_segment_id(value: str) -> SegmentKey:
    """Inverse of `segment_id`."""
    street, first, second = value.split("|")
    return SegmentKey(street, (first, second))
