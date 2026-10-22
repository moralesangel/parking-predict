# parking-predict

Ocupación de aparcamiento en calle en tiempo real en Melbourne y predicción de la
probabilidad de encontrar plaza en una zona **en el momento de llegada**.

La pregunta que el proyecto intenta responder es por qué predecir si ya existen
datos en tiempo real: cuando conduces 12 minutos hasta el destino, lo que importa
no es el estado actual de la zona sino su estado al llegar. La referencia a batir
es, precisamente, "usar el estado actual" (baseline de persistencia).

> Estado: **Fase 6 (prediccion en el producto) completada**. Ver [PLAN.md](PLAN.md) para el roadmap.

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

# Recalcular el informe de calidad de datos
uv run python scripts/quality_report.py

# Construir el dataset de entrenamiento desde el historico de 2019
uv run python scripts/build_dataset.py --months 2019-01:2019-12

# Entrenar baselines + LightGBM y generar el informe de resultados
uv run python scripts/train_evaluate.py
uv run python scripts/plot_results.py

# Geometria de referencia, que /predict necesita para situar los tramos
uv run python scripts/fetch_reference.py
```

Mapa en tiempo real (API + frontend en http://localhost:8000):

```bash
uv run python -m uvicorn --factory parking.api.app:create_app --port 8000
```

Con Docker, poller + API + mapa:

```bash
docker compose up --build       # -> http://localhost:8000
docker compose logs -f poller
```

## API

| Endpoint | Para que sirve |
|---|---|
| `GET /bays/live` | Estado de cada plaza con su bandera de fiabilidad |
| `GET /predict?lat=&lon=&eta_minutes=` | Probabilidad de encontrar sitio al llegar, por zona |
| `GET /zones/coverage` | Cobertura de sensores por zona |
| `GET /health` | `ok`, `stale` o `no_data`, con la antiguedad del ultimo lote |
| `GET /docs` | OpenAPI interactivo |

Dos decisiones que vienen de la Fase 2:

- `/bays/live` devuelve **todas** las plazas por defecto, tambien las no
  fiables, con `state: "unknown"`. Omitirlas haria que el mapa pintara como
  libre una plaza que quiza esta ocupada. `?reliable_only=true` las filtra.
- La API **publica la antiguedad de sus propios datos**. Si el poller muere, la
  API seguiria sirviendo el ultimo snapshot; `is_stale` y `/health` convierten
  ese fallo silencioso en uno visible, y el mapa muestra un aviso.
- `/predict` devuelve **varias zonas y no la mejor**, ordenadas por distancia.
  Senalar un unico hueco mandaria a todo el que pregunte al mismo sitio.
- La respuesta incluye `model_caveat`: el modelo esta entrenado con 2019 y
  aplicado a datos de hoy, asi que esto es una **demostracion del flujo**, no un
  servicio fiable. Ver [`docs/results.md`](docs/results.md) §6.

## Documentacion

| Documento | Contenido |
|---|---|
| [`docs/data_schema.md`](docs/data_schema.md) | Esquema real de la API, verificado campo a campo |
| [`docs/id_crosswalk.md`](docs/id_crosswalk.md) | Por que el historico no cruza con el tiempo real, y que si funciona |
| [`docs/data_quality.md`](docs/data_quality.md) | Los dos modos de fallo de los sensores y cuantas plazas son utilizables |
| [`docs/dataset.md`](docs/dataset.md) | El problema de prediccion, la unidad espacial y como se construye el dataset |
| [`docs/results.md`](docs/results.md) | Comparativa con los baselines, calibracion y donde el modelo si aporta |

Dos hallazgos que condicionan el resto del proyecto:

- **`Present` significa plaza ocupada**, no libre: el campo describe el coche.
- **`lastupdated` es el sello del lote del feed**, no del sensor. Los 5264
  sensores vivos comparten un valor identico y los caidos se quedan congelados
  con el del ultimo lote en que aparecieron.
- **Solo 3557 de las 6324 plazas (56,2%) dan una lectura utilizable.** 1060
  estan fuera del lote del feed y otras 1707 lo reportan pero con el sensor
  congelado (mediana: 150 dias sin cambiar de estado). El segundo fallo es el
  mas numeroso y no se ve si solo se vigila `lastupdated`.

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
