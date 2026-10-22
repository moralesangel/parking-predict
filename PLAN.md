# Plan: Predicción de aparcamiento en calle (Melbourne)

## Contexto para Claude Code

Proyecto de portfolio de un Data & AI Engineer. Objetivo: un sistema que muestre la ocupación de plazas de aparcamiento en tiempo real y prediga la probabilidad de encontrar sitio en una zona **en el momento de llegada**, demostrando que esa predicción mejora frente a la referencia obvia de "usar el estado actual".

Fuente de datos: City of Melbourne Open Data.
- Tiempo real: dataset `on-street-parking-bay-sensors` (sensores de suelo, presente / no presente, coordenadas, Kerbside id).
- Histórico: datasets `on-street-car-parking-sensor-data-2019` y similares (eventos de llegada/salida, restricción de la plaza).
- Licencia CC BY: incluir atribución a City of Melbourne en el README y en el frontend.

## Reglas de trabajo

1. Trabaja **fase por fase**. Al terminar cada fase, para, resume lo hecho y espera revisión antes de seguir.
2. **No inventes nombres de campos ni endpoints.** Antes de escribir código de ingesta, consulta el esquema real de la API y documéntalo en `docs/data_schema.md`.
3. Respeta los límites de la API. Si no están documentados, usa un intervalo de sondeo prudente (5 min) y cachea.
4. Tests con `pytest` para ingesta, limpieza y features. Nada de lógica sin test en `src/`.
5. Commits pequeños y descriptivos, uno por paso lógico.
6. Si algo del plan resulta inviable con los datos reales, dilo y propón alternativa en lugar de forzarlo.

## Stack

- Python 3.12, `uv` para dependencias, `ruff` para lint y formato.
- Almacenamiento: Parquet particionado por fecha + DuckDB para consultas.
- Modelado: pandas/polars, scikit-learn, LightGBM.
- API: FastAPI. Frontend: página estática con MapLibre GL o Leaflet.
- Docker + docker-compose para el sondeo continuo y la API.

## Estructura del repo

```
parking-predict/
├── src/parking/
│   ├── ingest/        # sondeo tiempo real + descarga histórico
│   ├── quality/       # detección de sensores caídos / datos viejos
│   ├── features/      # agregación espacial y features temporales
│   ├── models/        # baselines y modelos
│   ├── api/           # FastAPI
│   └── config.py
├── web/               # frontend estático
├── data/              # raw/ y processed/ (gitignored)
├── notebooks/         # solo EDA, nada de lógica de producción
├── tests/
├── docs/
├── docker-compose.yml
└── README.md
```

## Fase 0: Setup

- Repo, `uv`, `ruff`, `pytest`, pre-commit, `.gitignore` con `data/`.
- `config.py` con rutas, intervalo de sondeo y URL base de la API leídos de variables de entorno.
- **Entregable:** repo que pasa lint y un test vacío.

## Fase 1: Ingesta

1. Inspeccionar la API de `on-street-parking-bay-sensors` y documentar campos, tipos y semántica del timestamp de actualización en `docs/data_schema.md`.
2. Poller: consulta cada 5 min, guarda snapshot crudo en `data/raw/realtime/date=YYYY-MM-DD/` (Parquet). Idempotente y con reintentos.
3. Descargar el histórico de 2019 a `data/raw/historical/`.
4. **Riesgo a verificar:** los identificadores del histórico pueden no coincidir con los Kerbside id del tiempo real. Comprobar si se pueden cruzar (por id o por coordenadas) y documentar el resultado. Si no se pueden, el histórico sirve para desarrollar el modelo y el tiempo real se usa con su propio histórico acumulado por el poller.
- **Entregable:** poller corriendo en Docker y acumulando datos; informe del cruce de identificadores.

## Fase 2: Calidad de datos

- Detectar sensores con dato viejo (timestamp de actualización muy antiguo), sensores atascados (mismo estado durante un tiempo anómalo) y huecos de red.
- Marcar cada lectura como fiable o no fiable; nunca borrar en silencio.
- Métricas de cobertura por zona y por hora.
- **Entregable:** módulo `quality/` con tests y un informe breve en `docs/data_quality.md`.

## Fase 3: Mapa en tiempo real (MVP sin ML)

- Endpoint `GET /bays/live`: estado actual de cada plaza con flag de fiabilidad.
- Frontend: mapa con plazas coloreadas (libre, ocupada, sin dato fiable).
- **Entregable:** `docker compose up` levanta poller + API + web y se ve el mapa.

## Fase 4: Definición del problema de predicción

- Unidad espacial: celdas H3 (resolución a elegir con datos, probablemente 9 o 10) o tramos de calle. Justificar la elección.
- Objetivo: probabilidad de que haya **al menos una plaza libre** en la zona en `t + h`, con horizontes `h` = 5, 10, 15 y 30 min.
- Alternativa a evaluar: predecir el número de plazas libres y derivar la probabilidad.
- **Entregable:** dataset de entrenamiento reproducible a partir de los datos crudos, con split temporal (nunca aleatorio).

## Fase 5: Baselines y modelo

Baselines obligatorios, en este orden:
1. **Persistencia:** el estado en `t+h` es el estado actual. Es la referencia a batir.
2. **Media histórica** por zona, día de la semana y franja horaria.
3. **LightGBM** con features: ocupación actual de la zona y zonas vecinas, lags y tendencias recientes, hora, día, festivos, tipo de restricción de las plazas.

Evaluación:
- Brier score, log loss y curvas de calibración, por horizonte.
- Resultados desglosados por hora del día y por nivel de ocupación (el caso interesante es cuando la zona está casi llena).
- Si el modelo no bate a la persistencia en horizontes cortos, documentarlo tal cual. Es un resultado válido.
- **Entregable:** `docs/results.md` con tabla comparativa y gráficos.

## Fase 6: Predicción en el producto

- Endpoint `GET /predict?lat=&lon=&eta_minutes=`: probabilidad de encontrar sitio en la zona de destino y zonas vecinas.
- En el mapa: el usuario elige destino y tiempo de llegada; se muestran las zonas con su probabilidad en lugar de plazas sueltas (evita mandar a todos al mismo hueco).
- **Entregable:** demo funcional de punta a punta.

## Fase 7: Operación

- Reentrenamiento programado con los datos acumulados.
- Monitorización: degradación de métricas frente a la persistencia, drift en la distribución de ocupación, cobertura de sensores.
- **Entregable:** job de reentrenamiento y un panel o informe simple de monitorización.

## Fase 8: Documentación

- README: problema, por qué predecir si ya hay datos en tiempo real, arquitectura, resultados honestos (incluido dónde no mejora), limitaciones y cómo se aplicaría a otra ciudad sin sensores.
- Diagrama de arquitectura.
- Atribución de datos de City of Melbourne.
