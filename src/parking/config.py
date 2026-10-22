"""Central project configuration, read from environment variables.

Every value has a sensible default for local development, so the project runs
without a `.env` file. See `.env.example` for the full list.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Project settings.

    Environment variables use the `PARKING_` prefix
    (for example `PARKING_POLL_INTERVAL_SECONDS`).
    """

    model_config = SettingsConfigDict(
        env_prefix="PARKING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Source API ---
    api_base_url: str = "https://data.melbourne.vic.gov.au"
    realtime_dataset: str = "on-street-parking-bay-sensors"
    api_key: str | None = None

    # --- Polling ---
    # 120 s is the feed's native cadence, measured on 2026-10-22: batches land
    # exactly every 2 minutes. Polling slower loses whole batches; polling
    # faster just re-reads the same one. The 60 s floor protects the upstream
    # API, whose rate limits are undocumented (plan rule 3).
    poll_interval_seconds: int = Field(default=120, ge=60)
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=4, ge=0)

    # --- Data quality ---
    # A bay that is still in the feed's batch but whose state has not changed
    # for a long time is stuck: the relay reports it and the sensor is frozen.
    # 24 h comes from the valley of the real distribution, which is bimodal:
    # the healthy mode tails off around 12 h, the 18-24 h band holds 12 sensors
    # out of 5264, and the frozen mode starts past 30 days (median 150 days).
    # See docs/data_quality.md.
    stuck_after_hours: float = Field(default=24.0, gt=0)
    # Network gap: the feed publishes every 120 s, so two snapshots more than
    # three cycles apart mean batches were lost (by the poller or the source).
    gap_after_seconds: float = Field(default=360.0, gt=0)

    # --- Storage ---
    data_dir: Path = _REPO_ROOT / "data"
    # Static frontend served by the API. Explicit rather than derived from
    # `__file__`: once the package is installed the module lives in
    # site-packages and a relative path would stop pointing at the repository.
    # Docker sets it to /app/web.
    web_dir: Path = _REPO_ROOT / "web"

    @field_validator("api_base_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("data_dir", "web_dir")
    @classmethod
    def _absolutise(cls, value: Path) -> Path:
        return value if value.is_absolute() else (_REPO_ROOT / value).resolve()

    # --- Derived paths ---

    @property
    def raw_dir(self) -> Path:
        """Data exactly as it arrives from the source, untransformed."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Clean data and features, all derivable from `raw_dir`."""
        return self.data_dir / "processed"

    @property
    def realtime_raw_dir(self) -> Path:
        """Poller snapshots, partitioned by date."""
        return self.raw_dir / "realtime"

    @property
    def historical_raw_dir(self) -> Path:
        """Downloads of the yearly sensor archives."""
        return self.raw_dir / "historical"


@lru_cache
def get_settings() -> Settings:
    """Return the settings (cached: the environment is read once per process)."""
    return Settings()
