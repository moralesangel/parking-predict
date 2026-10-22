"""Estructura espacial: tramos de calle y su vecindad.

La unidad espacial del proyecto es el **tramo de calle**, no la celda H3. La
justificación está medida en `docs/id_crosswalk.md` §5: es el único nivel en el
que el histórico y el tiempo real se reconcilian, porque el histórico no trae
coordenadas. Además coincide con la decisión real del conductor —"¿hay sitio en
esta manzana?"— mejor que un hexágono arbitrario.

Este módulo deriva la **vecindad** entre tramos sin usar geometría, a partir de
los propios nombres de calle. Es lo que permite calcular features de "zonas
vecinas" (Fase 5) sobre el histórico, que no tiene latitud ni longitud.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from parking.ingest.crosswalk import SegmentKey

#: Una intersección es el par no ordenado de las dos calles que se cruzan.
#: `frozenset` y no tupla porque "Collins con Russell" y "Russell con Collins"
#: son el mismo cruce.
Intersection = frozenset[str]


def intersections(segment: SegmentKey) -> tuple[Intersection, Intersection]:
    """Los dos extremos de un tramo, como intersecciones.

    `COLLINS STREET between EXHIBITION STREET and RUSSELL STREET` ocupa los
    cruces `{COLLINS, EXHIBITION}` y `{COLLINS, RUSSELL}`.
    """
    first, second = segment.cross_streets
    return (
        frozenset({segment.street, first}),
        frozenset({segment.street, second}),
    )


def build_adjacency(segments: Iterable[SegmentKey]) -> dict[SegmentKey, frozenset[SegmentKey]]:
    """Grafo de vecindad: dos tramos son vecinos si comparten una intersección.

    Esta definición captura los dos casos que importan al conducir y descarta el
    que no:

    - **Manzana contigua**: `COLLINS entre RUSSELL y EXHIBITION` y `COLLINS
      entre EXHIBITION y SPRING` comparten el cruce `{COLLINS, EXHIBITION}`.
    - **Esquina**: `EXHIBITION entre COLLINS y LITTLE COLLINS` comparte ese
      mismo cruce con el primero, aunque vaya en perpendicular.
    - **Misma calle pero lejos**: dos tramos de Collins en extremos opuestos del
      CBD **no** son vecinos, porque no comparten ningún cruce. Definir la
      vecindad por "misma calle" los habría unido, que es justo el error que se
      quiere evitar.

    Un tramo nunca es vecino de sí mismo.
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
    """Identificador estable y legible de un tramo, para usar como clave en tablas.

    Legible a propósito: un `segment_id` que se puede leer hace que inspeccionar
    el dataset o depurar una predicción no exija volver a cruzar tablas.
    """
    first, second = segment.cross_streets
    return f"{segment.street}|{first}|{second}"


def parse_segment_id(value: str) -> SegmentKey:
    """Inverso de `segment_id`."""
    street, first, second = value.split("|")
    return SegmentKey(street, (first, second))
