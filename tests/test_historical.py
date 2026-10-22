"""Tests de `parking.ingest.historical`.

La reanudación se prueba con respx porque es la parte que importa: a 717 MB, un
corte de red no debe costar la descarga entera.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import pytest
import respx

from parking.ingest.historical import (
    HISTORICAL_ARCHIVES,
    download_archive,
    download_year,
    iter_csv_lines,
    remote_size,
)

URL = "https://example.test/archive.zip"
PAYLOAD = b"0123456789" * 10  # 100 bytes


def _head(size: int) -> httpx.Response:
    return httpx.Response(200, headers={"content-length": str(size)})


def test_known_year_has_a_verified_url() -> None:
    """La URL de 2019 se extrajo del href de la descripción del dataset real."""
    assert HISTORICAL_ARCHIVES["2019"].endswith("7pgd-bdf2.zip")


def test_unknown_year_is_rejected(tmp_path: Path) -> None:
    """No se inventan URLs para años sin verificar (regla 2 del plan)."""
    with pytest.raises(KeyError, match="2018"):
        download_year("2018", tmp_path)


@respx.mock
def test_remote_size_reads_content_length() -> None:
    respx.head(URL).respond(**{"headers": {"content-length": "717085379"}})

    with httpx.Client() as client:
        assert remote_size(URL, client) == 717085379


@respx.mock
def test_full_download(tmp_path: Path) -> None:
    respx.head(URL).mock(return_value=_head(len(PAYLOAD)))
    respx.get(URL).respond(content=PAYLOAD)
    destination = tmp_path / "a.zip"

    with httpx.Client() as client:
        result = download_archive(URL, destination, client)

    assert destination.read_bytes() == PAYLOAD
    assert (result.downloaded_bytes, result.resumed) == (len(PAYLOAD), False)


@respx.mock
def test_complete_file_is_not_redownloaded(tmp_path: Path) -> None:
    respx.head(URL).mock(return_value=_head(len(PAYLOAD)))
    route = respx.get(URL).respond(content=PAYLOAD)
    destination = tmp_path / "a.zip"
    destination.write_bytes(PAYLOAD)

    with httpx.Client() as client:
        result = download_archive(URL, destination, client)

    assert result.already_complete is True
    assert route.call_count == 0, "no debe pedir nada si ya está completo"


@respx.mock
def test_partial_file_is_resumed(tmp_path: Path) -> None:
    respx.head(URL).mock(return_value=_head(len(PAYLOAD)))
    route = respx.get(URL).respond(206, content=PAYLOAD[40:])
    destination = tmp_path / "a.zip"
    destination.write_bytes(PAYLOAD[:40])

    with httpx.Client() as client:
        result = download_archive(URL, destination, client)

    assert destination.read_bytes() == PAYLOAD
    assert result.resumed is True
    assert route.calls[0].request.headers["range"] == "bytes=40-"


@respx.mock
def test_server_ignoring_range_restarts_cleanly(tmp_path: Path) -> None:
    """Si responde 200 a un Range, manda el fichero entero: sobreescribir, no concatenar."""
    respx.head(URL).mock(return_value=_head(len(PAYLOAD)))
    respx.get(URL).respond(200, content=PAYLOAD)
    destination = tmp_path / "a.zip"
    destination.write_bytes(PAYLOAD[:40])

    with httpx.Client() as client:
        download_archive(URL, destination, client)

    assert destination.read_bytes() == PAYLOAD


@respx.mock
def test_oversized_file_is_discarded_and_restarted(tmp_path: Path) -> None:
    """Un fichero más largo que el remoto no se puede reparar por rango."""
    respx.head(URL).mock(return_value=_head(len(PAYLOAD)))
    respx.get(URL).respond(content=PAYLOAD)
    destination = tmp_path / "a.zip"
    destination.write_bytes(PAYLOAD + b"sobra")

    with httpx.Client() as client:
        download_archive(URL, destination, client)

    assert destination.read_bytes() == PAYLOAD


@respx.mock
def test_truncated_download_raises(tmp_path: Path) -> None:
    respx.head(URL).mock(return_value=_head(len(PAYLOAD)))
    respx.get(URL).respond(content=PAYLOAD[:50])

    with httpx.Client() as client, pytest.raises(OSError, match="descarga incompleta"):
        download_archive(URL, tmp_path / "a.zip", client)


# --- lectura del ZIP --------------------------------------------------------


def _make_zip(path: Path, *, name: str = "data.csv", body: str = "a,b\n1,2\n3,4\n") -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, body)


def test_iter_csv_lines_streams_rows(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    _make_zip(archive)

    assert list(iter_csv_lines(archive)) == ["a,b", "1,2", "3,4"]


def test_iter_csv_lines_strips_the_bom(tmp_path: Path) -> None:
    """El CSV de 2019 empieza con BOM; sin quitarlo la primera columna sale mal."""
    archive = tmp_path / "a.zip"
    _make_zip(archive, body="﻿DeviceId,BayId\n1,2\n")

    assert next(iter(iter_csv_lines(archive))) == "DeviceId,BayId"


def test_iter_csv_lines_honours_max_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    _make_zip(archive, body="aaaa\nbbbb\ncccc\ndddd\n")

    assert list(iter_csv_lines(archive, max_bytes=6)) == ["aaaa", "bbbb"]


def test_iter_csv_lines_ignores_macos_metadata(tmp_path: Path) -> None:
    """El ZIP real de 2019 trae un resource fork `__MACOSX/._...csv` junto al CSV."""
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("data.csv", "a,b\n1,2\n")
        handle.writestr("__MACOSX/._data.csv", "basura binaria")

    assert list(iter_csv_lines(archive)) == ["a,b", "1,2"]


def test_iter_csv_lines_rejects_ambiguous_archives(tmp_path: Path) -> None:
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("one.csv", "a\n")
        handle.writestr("two.csv", "b\n")

    with pytest.raises(ValueError, match="se esperaba 1 CSV"):
        list(iter_csv_lines(archive))


def test_iter_csv_lines_on_the_real_archive_if_present() -> None:
    """Si el histórico ya está descargado, comprueba su cabecera real.

    Se salta cuando no está, para que la suite no dependa de 717 MB en disco.
    """
    from parking.config import get_settings

    archive = get_settings().historical_raw_dir / "sensor-data-2019.zip"
    if not archive.exists():
        pytest.skip("histórico de 2019 no descargado")

    header = next(iter(iter_csv_lines(archive)))

    assert header.split(",")[:4] == ["DeviceId", "ArrivalTime", "DepartureTime", "DurationMinutes"]
    assert "StreetMarker" in header
    assert io  # el módulo se usa dentro de iter_csv_lines
