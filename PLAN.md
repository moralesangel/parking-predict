# Plan: on-street parking prediction (Melbourne)

## Context

Portfolio project for a Data & AI Engineer. Goal: a system that shows real-time
parking bay occupancy and predicts the probability of finding a space in a zone
**at the moment of arrival**, demonstrating that this prediction improves on the
obvious benchmark of "use the current state".

Data source: City of Melbourne Open Data.
- Real time: `on-street-parking-bay-sensors` dataset (in-ground sensors,
  present / not present, coordinates, Kerbside id).
- Historical: `on-street-car-parking-sensor-data-2019` and similar datasets
  (arrival/departure events, bay restriction).
- CC BY licence: include attribution to City of Melbourne in the README and in
  the frontend.

## Working rules

1. Work **phase by phase**. At the end of each phase, stop, summarise what was
   done and wait for review.
2. **Do not invent field names or endpoints.** Before writing any ingestion
   code, consult the real API schema and document it in `docs/data_schema.md`.
3. Respect the API's limits. If they are not documented, use a prudent polling
   interval (5 min) and cache.
4. Tests with `pytest` for ingestion, cleaning and features. No logic in `src/`
   without a test.
5. Small, descriptive commits, one per logical step.
6. If something in the plan turns out to be unworkable with the real data, say
   so and propose an alternative rather than forcing it.

## Stack

- Python 3.12, `uv` for dependencies, `ruff` for linting and formatting.
- Storage: Parquet partitioned by date + DuckDB for queries.
- Modelling: pandas/polars, scikit-learn, LightGBM.
- API: FastAPI. Frontend: static page with MapLibre GL or Leaflet.
- Docker + docker-compose for continuous polling and the API.

## Repository layout

```
parking-predict/
├── src/parking/
│   ├── ingest/        # real-time polling + historical download
│   ├── quality/       # detection of dead / stale sensors
│   ├── features/      # spatial aggregation and temporal features
│   ├── models/        # baselines and models
│   ├── api/           # FastAPI
│   └── config.py
├── web/               # static frontend
├── data/              # raw/ and processed/ (gitignored)
├── notebooks/         # EDA only, no production logic
├── tests/
├── docs/
├── docker-compose.yml
└── README.md
```

## Phase 0: Setup

- Repo, `uv`, `ruff`, `pytest`, pre-commit, `.gitignore` with `data/`.
- `config.py` with paths, polling interval and API base URL read from
  environment variables.
- **Deliverable:** a repo that passes linting and an empty test.

## Phase 1: Ingestion

1. Inspect the `on-street-parking-bay-sensors` API and document the fields,
   types and the semantics of the update timestamp in `docs/data_schema.md`.
2. Poller: query every 5 min, store the raw snapshot in
   `data/raw/realtime/date=YYYY-MM-DD/` (Parquet). Idempotent and with retries.
3. Download the 2019 archive into `data/raw/historical/`.
4. **Risk to verify:** the archive's identifiers may not match the real-time
   Kerbside ids. Check whether they can be joined (by id or by coordinates) and
   document the result. If they cannot, the archive serves to develop the model
   and the real-time feed is used with its own history accumulated by the
   poller.
- **Deliverable:** the poller running in Docker and accumulating data; a report
  on the identifier crosswalk.

## Phase 2: Data quality

- Detect sensors with stale data (very old update timestamp), stuck sensors
  (same state for an anomalous time) and network gaps.
- Flag each reading as reliable or unreliable; never delete silently.
- Coverage metrics per zone and per hour.
- **Deliverable:** a `quality/` module with tests and a short report in
  `docs/data_quality.md`.

## Phase 3: Real-time map (MVP without ML)

- Endpoint `GET /bays/live`: current state of every bay with a reliability flag.
- Frontend: a map with colour-coded bays (free, occupied, no reliable data).
- **Deliverable:** `docker compose up` brings up poller + API + web and the map
  is visible.

## Phase 4: Defining the prediction problem

- Spatial unit: H3 cells (resolution to be chosen from the data, probably 9 or
  10) or street segments. Justify the choice.
- Target: probability that **at least one bay is free** in the zone at `t + h`,
  for horizons `h` = 5, 10, 15 and 30 min.
- Alternative to evaluate: predict the number of free bays and derive the
  probability from it.
- **Deliverable:** a reproducible training dataset built from the raw data, with
  a temporal split (never random).

## Phase 5: Baselines and model

Mandatory baselines, in this order:
1. **Persistence:** the state at `t+h` is the current state. This is the
   benchmark to beat.
2. **Historical mean** per zone, weekday and hour bucket.
3. **LightGBM** with features: current occupancy of the zone and neighbouring
   zones, recent lags and trends, hour, day, holidays, bay restriction type.

Evaluation:
- Brier score, log loss and calibration curves, per horizon.
- Results broken down by hour of day and by occupancy level (the interesting
  case is when the zone is nearly full).
- If the model does not beat persistence at short horizons, document that as-is.
  It is a valid result.
- **Deliverable:** `docs/results.md` with a comparison table and plots.

## Phase 6: Prediction in the product

- Endpoint `GET /predict?lat=&lon=&eta_minutes=`: probability of finding a space
  in the destination zone and neighbouring zones.
- On the map: the user picks a destination and an arrival time; zones are shown
  with their probability instead of individual bays (this avoids sending
  everyone to the same spot).
- **Deliverable:** a working end-to-end demo.

## Phase 7: Operations

- Scheduled retraining with the accumulated data.
- Monitoring: metric degradation against persistence, drift in the occupancy
  distribution, sensor coverage.
- **Deliverable:** a retraining job and a simple monitoring dashboard or report.

## Phase 8: Documentation

- README: the problem, why predict when real-time data already exists,
  architecture, honest results (including where it does not improve),
  limitations, and how this would apply to another city without sensors.
- Architecture diagram.
- City of Melbourne data attribution.
