# Shared image for the poller and the API: same code, different command (see
# docker-compose.yml). The ML extras are not installed; neither service needs
# them yet and the image stays smaller.
FROM python:3.12-slim-bookworm

# uv from its official image: avoids a `pip install uv` and pins the version.
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Dependencies are installed before copying the code so that a change in
# `src/` does not invalidate the dependency layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev

COPY src/ ./src/
COPY web/ ./web/
COPY scripts/ ./scripts/
RUN uv sync --locked --no-dev

# Unprivileged user. The data volume is mounted with this UID (see
# docker-compose.yml) so the poller can write to it.
RUN useradd --create-home --uid 10001 parking \
    && mkdir -p /data \
    && chown -R parking:parking /data
USER parking

ENV PARKING_DATA_DIR=/data \
    PARKING_WEB_DIR=/app/web

# Checks the latest snapshot is not too old. Polling every 2 min, 20 minutes
# without a write means the poller is stuck even though the process is alive:
# network failures are logged and skipped, they do not kill the loop.
HEALTHCHECK --interval=5m --timeout=20s --start-period=2m --retries=2 \
    CMD ["python", "-c", "import sys,time,pathlib,os; \
d=pathlib.Path(os.environ['PARKING_DATA_DIR'])/'raw'/'realtime'; \
f=sorted(d.rglob('*.parquet'), key=lambda p: p.stat().st_mtime); \
sys.exit(0 if f and time.time()-f[-1].stat().st_mtime < 1200 else 1)"]

# CMD rather than ENTRYPOINT: docker-compose replaces this command with the
# uvicorn one in the `api` service. With an exec-form ENTRYPOINT, `command`
# would be appended as poller arguments instead of replacing it.
CMD ["python", "-m", "parking.ingest.poller"]
