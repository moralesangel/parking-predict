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
    # 120 s: cadencia nativa del feed, medida el 2026-10-22 (los lotes salen
    # cada 2 min exactos). Sondear mas despacio pierde lotes enteros; mas
    # rapido solo relee el mismo. El suelo de 60 s protege la API de origen,
    # cuyos limites no estan documentados (regla 3 del plan).
    poll_interval_seconds: int = Field(default=120, ge=60)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=4, ge=0)

    # --- Calidad de datos ---
    # Una plaza que sigue en el lote del feed pero cuyo estado no cambia desde
    # hace mucho esta atascada: el rele la reporta y el sensor esta congelado.
    # 24 h sale del valle de la distribucion real, que es bimodal: el modo sano
    # se agota hacia las 12 h, la franja 18-24 h tiene 12 sensores de 5264, y el
    # modo congelado arranca pasados los 30 dias (mediana 150 dias). Ver
    # docs/data_quality.md.
    stuck_after_hours: float = Field(default=24.0, gt=0)
    # Hueco de red: el feed publica cada 120 s, asi que dos snapshots separados
    # por mas de 3 ciclos indican lotes perdidos (por el poller o por la fuente).
    gap_after_seconds: float = Field(default=360.0, gt=0)

    # --- Almacenamiento ---
    data_dir: Path = _REPO_ROOT / "data"
    # Frontend estático que sirve la API. Explícito y no deducido de `__file__`:
    # al instalar el paquete, el módulo vive en site-packages y la ruta relativa
    # dejaría de apuntar al repositorio. En Docker se fija a /app/web.
    web_dir: Path = _REPO_ROOT / "web"

    @field_validator("api_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("data_dir", "web_dir")
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
