# Imagen comun del poller y de la API: mismo codigo, distinto comando (ver
# docker-compose.yml). Las dependencias de ML (extra `ml`) no se instalan;
# ninguno de los dos servicios las necesita todavia y la imagen baja de tamano.
FROM python:3.12-slim-bookworm

# uv desde su imagen oficial: evita un `pip install uv` y fija la versión.
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Las dependencias se instalan antes de copiar el código para que un cambio en
# `src/` no invalide la capa de dependencias.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-dev

COPY src/ ./src/
COPY web/ ./web/
COPY scripts/ ./scripts/
RUN uv sync --locked --no-dev

# Usuario sin privilegios. El volumen de datos se monta con este UID (ver
# docker-compose.yml) para que el poller pueda escribir.
RUN useradd --create-home --uid 10001 parking \
    && mkdir -p /data \
    && chown -R parking:parking /data
USER parking

ENV PARKING_DATA_DIR=/data \
    PARKING_WEB_DIR=/app/web

# Comprueba que el último snapshot no es demasiado viejo. Con sondeo cada 2 min,
# 20 min sin escribir significa que el poller está atascado aunque el proceso
# siga vivo: los fallos de red se registran y se omiten, no matan el bucle.
HEALTHCHECK --interval=5m --timeout=20s --start-period=2m --retries=2 \
    CMD ["python", "-c", "import sys,time,pathlib,os; \
d=pathlib.Path(os.environ['PARKING_DATA_DIR'])/'raw'/'realtime'; \
f=sorted(d.rglob('*.parquet'), key=lambda p: p.stat().st_mtime); \
sys.exit(0 if f and time.time()-f[-1].stat().st_mtime < 1200 else 1)"]

# CMD y no ENTRYPOINT: docker-compose sustituye este comando por el de uvicorn
# en el servicio `api`. Con ENTRYPOINT en forma exec, `command` se concatenaria
# como argumentos del poller en vez de reemplazarlo.
CMD ["python", "-m", "parking.ingest.poller"]
