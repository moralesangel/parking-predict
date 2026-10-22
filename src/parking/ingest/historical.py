"""Download of the yearly sensor archives.

The historical datasets are not queryable through the API: `has_records` is
`false` and the rows only exist as archive ZIPs on S3 (717 MB for 2019). See
`docs/data_schema.md` §3.

S3 answers with `Accept-Ranges: bytes`, so the download resumes instead of
restarting: at 717 MB, a dropped connection halfway through should not cost the
whole transfer.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# URL verified on 2026-10-22, extracted from the href inside the description of
# the `on-street-car-parking-sensor-data-2019` dataset (the API does not expose
# it as a field).
HISTORICAL_ARCHIVES: dict[str, str] = {
    "2019": "https://opendatasoft-s3.s3.amazonaws.com/downloads/archive/7pgd-bdf2.zip",
}

_CHUNK_BYTES = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of a download."""

    path: Path
    total_bytes: int
    downloaded_bytes: int
    resumed: bool

    @property
    def already_complete(self) -> bool:
        """True when nothing had to be requested because the file was whole."""
        return self.downloaded_bytes == 0


def remote_size(url: str, client: httpx.Client) -> int | None:
    """Size declared by the server, or `None` when it declares none."""
    head = client.head(url, follow_redirects=True)
    head.raise_for_status()
    length = head.headers.get("content-length")
    return int(length) if length is not None else None


def download_archive(
    url: str,
    destination: Path,
    client: httpx.Client,
    *,
    progress_every_bytes: int = 64 << 20,
) -> DownloadResult:
    """Download `url` to `destination`, resuming a partial transfer if present.

    Idempotent: if `destination` already has the size the server declares,
    nothing is requested. If it is shorter, only the missing bytes are fetched
    with a `Range` header. If it is longer than declared, it is discarded and
    restarted: an oversized file cannot be repaired by range.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(url, client)
    have = destination.stat().st_size if destination.exists() else 0

    if total is not None and have > total:
        logger.warning(
            "%s is larger than the remote file (%d > %d); discarding and restarting",
            destination.name,
            have,
            total,
        )
        destination.unlink()
        have = 0

    if total is not None and have == total:
        logger.info("%s is already complete (%d bytes)", destination.name, total)
        return DownloadResult(destination, total, 0, resumed=False)

    resumed = have > 0
    headers = {"Range": f"bytes={have}-"} if resumed else {}
    if resumed:
        logger.info("resuming %s at byte %d of %s", destination.name, have, total)

    written = 0
    next_report = progress_every_bytes
    with client.stream("GET", url, headers=headers, follow_redirects=True) as response:
        response.raise_for_status()
        # If a range was requested but the server answers 200, it is sending
        # the whole file: overwrite rather than append.
        append = resumed and response.status_code == 206
        if resumed and not append:
            logger.warning(
                "the server ignored the Range header (HTTP %d); restarting", response.status_code
            )
            have = 0
        with destination.open("ab" if append else "wb") as handle:
            for chunk in response.iter_bytes(_CHUNK_BYTES):
                handle.write(chunk)
                written += len(chunk)
                if written >= next_report:
                    logger.info("%s: %.0f MB downloaded", destination.name, (have + written) / 1e6)
                    next_report += progress_every_bytes

    final = destination.stat().st_size
    if total is not None and final != total:
        raise OSError(
            f"incomplete download of {destination.name}: {final} bytes out of {total} expected"
        )
    logger.info("%s downloaded (%d bytes)", destination.name, final)
    return DownloadResult(destination, total if total is not None else final, written, resumed)


def download_year(
    year: str,
    destination_dir: Path,
    *,
    client: httpx.Client | None = None,
    timeout_seconds: float = 60.0,
) -> DownloadResult:
    """Download the historical ZIP for `year` into `destination_dir`."""
    if year not in HISTORICAL_ARCHIVES:
        known = ", ".join(sorted(HISTORICAL_ARCHIVES))
        raise KeyError(f"year {year!r} has no verified URL; known years: {known}")

    url = HISTORICAL_ARCHIVES[year]
    destination = destination_dir / f"sensor-data-{year}.zip"
    owns_client = client is None
    # read=None: the stream arrives chunk by chunk and a 717 MB file can take
    # far longer than the connection timeout without being stalled.
    client = client or httpx.Client(timeout=httpx.Timeout(timeout_seconds, read=None))
    try:
        return download_archive(url, destination, client)
    finally:
        if owns_client:
            client.close()


def _is_real_entry(name: str) -> bool:
    """Drop the macOS metadata the source ZIP carries inside.

    The 2019 archive was packed on macOS and includes
    `__MACOSX/._On-street_Car_Parking_Sensor_Data_-_2019.csv`, a resource fork
    that is not a CSV even though the extension suggests otherwise.
    """
    return not name.startswith("__MACOSX/") and not Path(name).name.startswith("._")


def iter_csv_lines(zip_path: Path, *, max_bytes: int | None = None) -> Iterator[str]:
    """Iterate the lines of the single CSV inside a historical ZIP.

    Streams through `zipfile` without materialising the 7.5 GB the CSV occupies
    once decompressed. `max_bytes` caps the read for EDA sampling.

    Careful: the CSV is sorted by `DeviceId`, not by time, so a prefix is a
    subset of sensors rather than a temporal sample (see `docs/data_schema.md`
    §3).
    """
    import io
    import zipfile

    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv") and _is_real_entry(n)]
        if len(names) != 1:
            raise ValueError(f"expected 1 CSV in {zip_path.name}, found {len(names)}: {names}")
        read = 0
        with archive.open(names[0]) as raw:
            # utf-8-sig: the 2019 CSV starts with a BOM.
            for line in io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""):
                if max_bytes is not None and read >= max_bytes:
                    return
                read += len(line)
                yield line.rstrip("\r\n")
