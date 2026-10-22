"""Cruce entre el histórico de sensores y los datasets actuales.

El histórico no comparte identificador con el tiempo real y no trae
coordenadas, así que no hay cruce posible a nivel de plaza. Sí lo hay a nivel de
**tramo de calle**, por el nombre de la calle y las dos calles que lo delimitan:
cubre el 88 % de los eventos del histórico. El informe completo con las
mediciones está en `docs/id_crosswalk.md`.

Este módulo implementa la clave de tramo que hace posible ese cruce.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: `roadsegmentdescription` sigue el formato
#: "Exhibition Street between Little Collins Street and Bourke Street".
_SEGMENT_DESCRIPTION = re.compile(
    r"^(?P<street>.+?) between (?P<first>.+?) and (?P<second>.+)$", re.I
)

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, order=True)
class SegmentKey:
    """Clave canónica de un tramo de calle, comparable entre datasets.

    El par de calles delimitadoras se guarda **ordenado alfabéticamente** porque
    "Collins between Russell and Exhibition" y "Collins between Exhibition and
    Russell" son el mismo tramo, y los dos datasets no usan el mismo orden.
    """

    street: str
    cross_streets: tuple[str, str]

    def __str__(self) -> str:
        return f"{self.street} between {self.cross_streets[0]} and {self.cross_streets[1]}"


def normalise_street(name: str | None) -> str:
    """Normaliza un nombre de calle para poder compararlo entre datasets.

    El histórico escribe `BOURKE STREET` y la geometría `Bourke Street`, con
    espacios inconsistentes. Se unifica a mayúsculas con espacios colapsados.
    No se tocan los apóstrofos (`O'CONNELL STREET`) ni se abrevia `STREET`:
    ambos datasets son consistentes en eso y "normalizar" más sería inventar
    equivalencias no verificadas.
    """
    if not name:
        return ""
    return _WHITESPACE.sub(" ", name.strip()).upper()


def segment_key(street: str | None, first: str | None, second: str | None) -> SegmentKey | None:
    """Construye la clave de tramo, o `None` si falta alguno de los tres nombres.

    Un tramo sin las dos calles delimitadoras no es identificable: se descarta
    en lugar de generar una clave parcial que cruzaría con tramos equivocados.
    """
    street_norm = normalise_street(street)
    first_norm = normalise_street(first)
    second_norm = normalise_street(second)
    if not (street_norm and first_norm and second_norm):
        return None
    pair = tuple(sorted((first_norm, second_norm)))
    return SegmentKey(street_norm, (pair[0], pair[1]))


def segment_key_from_description(description: str | None) -> SegmentKey | None:
    """Clave de tramo a partir de un `roadsegmentdescription`.

    Devuelve `None` si la descripción no sigue el patrón "X between A and B".
    1035 de las 29 053 filas de `on-street-parking-bays` no lo siguen.
    """
    if not description:
        return None
    match = _SEGMENT_DESCRIPTION.match(description.strip())
    if match is None:
        return None
    return segment_key(match["street"], match["first"], match["second"])


def segment_key_from_historical_row(row: dict[str, str]) -> SegmentKey | None:
    """Clave de tramo a partir de una fila del CSV histórico.

    Usa `StreetName`, `BetweenStreet1` y `BetweenStreet2`.
    """
    return segment_key(row.get("StreetName"), row.get("BetweenStreet1"), row.get("BetweenStreet2"))
