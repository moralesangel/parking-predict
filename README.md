# parking-predict

Ocupación de aparcamiento en calle en tiempo real en Melbourne y predicción de la
probabilidad de encontrar plaza en una zona **en el momento de llegada**.

La pregunta que el proyecto intenta responder es por qué predecir si ya existen
datos en tiempo real: cuando conduces 12 minutos hasta el destino, lo que importa
no es el estado actual de la zona sino su estado al llegar. La referencia a batir
es, precisamente, "usar el estado actual" (baseline de persistencia).

> Estado: **Fase 0 (setup) completada**. Ver [PLAN.md](PLAN.md) para el roadmap.

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
