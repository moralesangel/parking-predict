"""Download the bay geometry the API needs to locate street segments.

    uv run python scripts/fetch_reference.py

Stored in `data/reference/parking_bays.json`. It is an inventory dataset: it
changes very rarely, so there is no need to refresh it on every deployment.
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
    logging.info("downloading %s", DATASET)
    with httpx.Client(timeout=httpx.Timeout(60, read=None)) as client:
        response = client.get(url, params={"limit": -1, "select": FIELDS})
        response.raise_for_status()
        destination.write_bytes(response.content)

    logging.info("wrote %s (%.1f MB)", destination, destination.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
