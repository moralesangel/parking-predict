"""Tests de `parking.config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from parking.config import Settings, get_settings


def test_defaults_are_usable_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PARKING_POLL_INTERVAL_SECONDS", raising=False)
    settings = Settings(_env_file=None)

    assert settings.poll_interval_seconds == 120
    assert settings.realtime_dataset == "on-street-parking-bay-sensors"
    assert settings.api_key is None


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARKING_POLL_INTERVAL_SECONDS", "600")
    monkeypatch.setenv("PARKING_API_BASE_URL", "https://example.test/")

    settings = Settings(_env_file=None)

    assert settings.poll_interval_seconds == 600
    # El validador quita la barra final para poder concatenar rutas sin dobles.
    assert settings.api_base_url == "https://example.test"


def test_poll_interval_below_floor_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sondear más rápido que 1/min abusaría de una API con límites no documentados."""
    monkeypatch.setenv("PARKING_POLL_INTERVAL_SECONDS", "5")

    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_relative_data_dir_becomes_absolute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARKING_DATA_DIR", "data")

    settings = Settings(_env_file=None)

    assert settings.data_dir.is_absolute()


def test_derived_paths_hang_off_data_dir(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)

    assert settings.raw_dir == tmp_path / "raw"
    assert settings.processed_dir == tmp_path / "processed"
    assert settings.realtime_raw_dir == tmp_path / "raw" / "realtime"
    assert settings.historical_raw_dir == tmp_path / "raw" / "historical"


def test_quality_thresholds_have_the_measured_defaults() -> None:
    """24 h sale del valle de la distribución real; ver docs/data_quality.md."""
    settings = Settings(_env_file=None)

    assert settings.stuck_after_hours == 24.0
    # 3 ciclos de 120 s: tolera el retraso ocasional del feed sin marcar hueco.
    assert settings.gap_after_seconds == 360.0


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("PARKING_STUCK_AFTER_HOURS", "0"),
        ("PARKING_STUCK_AFTER_HOURS", "-1"),
        ("PARKING_GAP_AFTER_SECONDS", "0"),
    ],
)
def test_non_positive_thresholds_are_rejected(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_quality_thresholds_are_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARKING_STUCK_AFTER_HOURS", "48")

    assert Settings(_env_file=None).stuck_after_hours == 48.0


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
