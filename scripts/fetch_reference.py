"""Descarga la geometria de bahias, que la API necesita para situar los tramos.

    uv run python scripts/fetch_reference.py

Se guarda en `data/reference/parking_bays.json`. Es un dataset de inventario:
cambia muy poco, asi que no hace falta refrescarlo con cada despliegue.
"""

from __future__ import annotations

import logging
import sys

import httpx

from parking.config import get_settings

DATASET = "on-street-parking-bays"
FIELDS = "kerbsideid,roadsegmentid,roadsegmentdescription,latitude,longitude"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    settings = get_settings()
    destination = settings.data_dir / "reference" / "parking_bays.json"
    destination.parent.mkdir(parents=True, exist_ok=True)

    url = f"{settings.api_base_url}/api/explore/v2.1/catalog/datasets/{DATASET}/exports/json"
    logging.info("descargando %s", DATASET)
    with httpx.Client(timeout=httpx.Timeout(60, read=None)) as client:
        response = client.get(url, params={"limit": -1, "select": FIELDS})
        response.raise_for_status()
        destination.write_bytes(response.content)

    logging.info("escrito %s (%.1f MB)", destination, destination.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
