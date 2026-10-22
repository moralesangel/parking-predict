"""Configuración central del proyecto, leída de variables de entorno.

Todos los valores tienen un default razonable para desarrollo local, de modo que
el proyecto arranca sin `.env`. Ver `.env.example` para la lista completa.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Ajustes del proyecto.

    Las variables de entorno usan el prefijo `PARKING_`
    (p. ej. `PARKING_POLL_INTERVAL_SECONDS`).
    """

    model_config = SettingsConfigDict(
        env_prefix="PARKING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- API de origen ---
    api_base_url: str = "https://data.melbourne.vic.gov.au"
    realtime_dataset: str = "on-street-parking-bay-sensors"
    api_key: str | None = None

    # --- Sondeo ---
    # 5 min: intervalo prudente mientras los límites de la API no estén
    # documentados (regla 3 del plan).
    poll_interval_seconds: int = Field(default=300, ge=60)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=4, ge=0)

    # --- Almacenamiento ---
    data_dir: Path = _REPO_ROOT / "data"

    @field_validator("api_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("data_dir")
    @classmethod
    def _absolutise(cls, value: Path) -> Path:
        return value if value.is_absolute() else (_REPO_ROOT / value).resolve()

    # --- Rutas derivadas ---

    @property
    def raw_dir(self) -> Path:
        """Datos tal como llegan de la fuente, sin transformar."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Datos limpios y features, derivables de `raw_dir`."""
        return self.data_dir / "processed"

    @property
    def realtime_raw_dir(self) -> Path:
        """Snapshots del poller, particionados por fecha."""
        return self.raw_dir / "realtime"

    @property
    def historical_raw_dir(self) -> Path:
        """Descargas del histórico anual de sensores."""
        return self.raw_dir / "historical"


@lru_cache
def get_settings() -> Settings:
    """Devuelve los ajustes (cacheados: una sola lectura del entorno por proceso)."""
    return Settings()
