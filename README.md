# parking-predict

Ocupación de aparcamiento en calle en tiempo real en Melbourne y predicción de la
probabilidad de encontrar plaza en una zona **en el momento de llegada**.

La pregunta que el proyecto intenta responder es por qué predecir si ya existen
datos en tiempo real: cuando conduces 12 minutos hasta el destino, lo que importa
no es el estado actual de la zona sino su estado al llegar. La referencia a batir
es, precisamente, "usar el estado actual" (baseline de persistencia).

> Estado: **Fase 1 (ingesta) completada**. Ver [PLAN.md](PLAN.md) para el roadmap.

## Requisitos

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)

## Puesta en marcha

```bash
uv sync --all-groups        # entorno + dependencias de desarrollo
cp .env.example .env        # ajustar si hace falta; los defaults sirven
uv run pre-commit install   # hooks de lint/formato
```

Comprobaciones:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Uso

```bash
# Un snapshot del feed en tiempo real (comprueba que la ingesta funciona)
uv run python -m parking.ingest.poller --once

# Sondeo continuo (cada 5 min por defecto)
uv run python -m parking.ingest.poller

# Recalcular el informe de cruce de identificadores
uv run python scripts/crosswalk_report.py
```

Con Docker, el poller como servicio:

```bash
docker compose up -d poller
docker compose logs -f poller
```

## Documentacion

| Documento | Contenido |
|---|---|
| [`docs/data_schema.md`](docs/data_schema.md) | Esquema real de la API, verificado campo a campo |
| [`docs/id_crosswalk.md`](docs/id_crosswalk.md) | Por que el historico no cruza con el tiempo real, y que si funciona |

Dos hallazgos que condicionan el resto del proyecto:

- **`Present` significa plaza ocupada**, no libre: el campo describe el coche.
- **`lastupdated` es el sello del lote del feed**, no del sensor. Los 5264
  sensores vivos comparten un valor identico y los caidos se quedan congelados
  con el del ultimo lote en que aparecieron. De las 6324 plazas del dataset,
  **solo ~5264 reportan**; las otras 979 llevan mas de 30 dias caidas.

## Restricciones del entorno de desarrollo

En la maquina de desarrollo (Windows 11), la politica de **Application Control**
bloquea las extensiones nativas de `pandas`, `duckdb` y `pyarrow.parquet`. El
stack usa **polars**, cuyo motor Parquet si funciona, y `duckdb` queda como
extra opcional (`uv sync --extra sql`) para usarlo en el contenedor Linux. El
plan permitia "pandas/polars", asi que no es una desviacion.

## Estructura

```
src/parking/
├── ingest/     # sondeo en tiempo real + descarga del histórico
├── quality/    # sensores caídos, atascados y huecos de red
├── features/   # agregación espacial y features temporales
├── models/     # baselines y modelo
├── api/        # FastAPI
└── config.py   # configuración desde variables de entorno
web/            # frontend estático (mapa)
data/           # raw/ y processed/ — no se versiona
docs/           # esquema de datos, calidad, resultados
notebooks/      # solo EDA
```

## Configuración

Todos los ajustes se leen de variables de entorno con prefijo `PARKING_`
(ver [`.env.example`](.env.example)). El intervalo de sondeo por defecto es de
5 minutos, un valor prudente mientras los límites de la API no estén documentados.

## Atribución de datos

Este proyecto usa datos de **[City of Melbourne Open Data](https://data.melbourne.vic.gov.au/)**,
publicados bajo licencia
[Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
Los datos se usan sin modificación de su contenido original; cualquier
transformación aplicada en este repositorio es responsabilidad del autor y no de
City of Melbourne.

## Licencia

Código: MIT (ver [LICENSE](LICENSE)). Los datos conservan su licencia original (CC BY 4.0).
