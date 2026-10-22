# Esquema de datos

Verificado contra la API real de
[City of Melbourne Open Data](https://data.melbourne.vic.gov.au/) el **2026-10-22**.
Todos los campos y endpoints de este documento se han comprobado ejecutando las
consultas; ninguno está supuesto (regla 2 del plan). Datos bajo licencia CC BY 4.0.

La API es una instancia de **Opendatasoft Explore v2.1**:

```
https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/{dataset_id}
```

## 1. Tiempo real: `on-street-parking-bay-sensors`

Sensores de suelo en bahías del CBD. **6324 registros**, uno por plaza. El
dataset se **sobreescribe** en cada actualización: es una foto del estado
actual, no una serie temporal. Conservar la historia es precisamente el trabajo
del poller.

### Campos

| Campo | Tipo API | Semántica verificada |
|---|---|---|
| `kerbsideid` | `int` | Identificador de la plaza. **Clave única**: 6324 valores distintos en 6324 filas, 0 duplicados, 0 nulos. Rango observado [5691, 90812]. |
| `status_description` | `text` | Solo dos valores: `Present` y `Unoccupied`. 0 nulos. Ver §1.1. |
| `status_timestamp` | `datetime` (UTC) | Instante del **último cambio de estado** de esa plaza. 0 nulos. |
| `lastupdated` | `datetime` (UTC) | Instante del **último lote del feed** en que apareció la plaza. 0 nulos. Ver §1.2. |
| `zone_number` | `int` | Zona de restricción (`parkingzone` en otros datasets). 536 zonas distintas, rango [7001, 7996], **520 nulos (8,2 %)**. |
| `location` | `geo_point_2d` | Coordenadas de la plaza. 0 nulos. El formato depende del endpoint, ver §1.3. |

### 1.1 Trampa de nomenclatura: `Present` significa ocupada

`status_description` describe el **coche**, no la plaza:

- `Present` → hay un coche → la plaza está **OCUPADA** (3581 filas, 56,7 %)
- `Unoccupied` → no hay coche → la plaza está **LIBRE** (2743 filas, 43,3 %)

Ese 56,7 % es el reparto crudo de las 6324 filas en un instante concreto, no la
ocupación del parque: incluye sensores caídos y atascados. La cifra utilizable
se calcula solo sobre lecturas fiables y se mueve con la hora del día; ver
[`data_quality.md`](data_quality.md).

Es un error fácil de cometer y silencioso: invierte el target del modelo sin
provocar ningún fallo. La ingesta normaliza a un booleano explícito
`is_occupied` y la cadena cruda no llega nunca a las features.

### 1.2 Los dos timestamps no son lo mismo

Es el hallazgo principal de la Fase 1 y condiciona toda la Fase 2.

**`lastupdated` es un sello del lote, no del sensor.** Sobre los 6324 registros
hay solo **49 valores distintos**, y uno cubre 5264 plazas
(`2026-10-22T09:26:50Z` en la medición). La publicación va por relés de red:
cada ciclo reescribe el dataset y todas las plazas que reportaron reciben el
**mismo** `lastupdated`. Una plaza que deja de reportar se queda **congelada**
con el sello del último lote en que apareció.

Eso convierte la detección de sensores caídos en una regla exacta en lugar de
una heurística de umbral:

```
sensor_vivo  ⟺  lastupdated == max(lastupdated) del snapshot
```

Los 49 valores distintos son "cohortes de caída": grupos de sensores que
desaparecieron en el mismo ciclo (83 plazas el 2025-10-30, 79 el 2026-01-15, 70
el 2026-07-22…). Que caigan en bloque apunta a relés enteros, no a fallos de
sensor individuales — una distinción útil para el informe de calidad.

Antigüedad de `lastupdated` medida el 2026-10-22:

| Antigüedad | Plazas | % |
|---|---|---|
| ≤ 15 min | 5264 | 83,2 % |
| ≤ 24 h | 5264 | 83,2 % |
| ≤ 7 d | 5284 | 83,6 % |
| ≤ 30 d | 5345 | 84,5 % |
| **> 30 d** | **979** | **15,5 %** |

**El parque útil es de ~5264 plazas, no 6324.** Las 979 con más de 30 días son
sensores retirados o permanentemente caídos que el dataset no purga. Cualquier
métrica de ocupación calculada sobre las 6324 está sesgada.

**`status_timestamp` es por plaza** y marca el último cambio de estado. Mediana
de 4,0 h; solo el 8,8 % cambió en los últimos 15 min. Su antigüedad **no**
indica avería: una plaza legítimamente ocupada toda la noche tiene
`status_timestamp` viejo y `lastupdated` fresco. Confundir ambos campos es el
error que la Fase 2 tiene que evitar — y es la razón de documentarlos por
separado aquí.

| Antigüedad de `status_timestamp` | Plazas | % |
|---|---|---|
| ≤ 15 min | 558 | 8,8 % |
| ≤ 1 h | 1531 | 24,2 % |
| ≤ 6 h | 3328 | 52,6 % |
| ≤ 24 h | 3556 | 56,2 % |
| > 30 d | 2427 | 38,4 % |

La propia fuente avala vigilar `lastupdated`; su descripción dice:

> This dataset is updated through network relays and can be disrupted causing
> delays in data updates. Users of this data are encouraged to monitor the
> dataset's last updated timestamp (Lastupdated column) to check for delays in
> the data being updated.

### 1.3 Endpoint elegido para el poller

Tres formas de leer el dataset, con diferencias que importan:

| Endpoint | Peticiones/snapshot | `location` | Veredicto |
|---|---|---|---|
| `/records?limit=100` | 64 (paginado, tope 100/req) | `{"lon":…,"lat":…}` | Descartado: 64 peticiones por snapshot |
| `/exports/parquet?limit=-1` | 1 (204 KB) | **`Binary` (WKB)** | Descartado: obliga a decodificar geometría |
| **`/exports/json?limit=-1`** | **1 (1,5 MB)** | `{"lon":…,"lat":…}` | **Elegido** |

`/exports/json?limit=-1` devuelve los 6324 registros en **una sola petición** con
coordenadas ya en lon/lat. A 5 min por snapshot son 288 peticiones al día,
holgado para cualquier límite razonable.

Que el export Parquet entregue `location` como `Binary` (WKB) es la razón de no
usarlo, aunque el almacenamiento local sí sea Parquet.

Bounding box de los sensores (medido): lon [144.93644, 144.98861], lat
[-37.84824, -37.79491] — unos 4,6 × 5,9 km del centro de Melbourne.

## 2. Datasets auxiliares (consultables por API)

### `on-street-parking-bays` — geometría de bahías

29 053 filas. Campos: `roadsegmentid` (int), `kerbsideid` (**text**),
`roadsegmentdescription` (text), `latitude`, `longitude`, `lastupdated`,
`location`.

Tres avisos verificados:

- `kerbsideid` es **texto aquí** y `int` en el dataset de sensores. Hay que
  castear para cruzar.
- **23 981 de 29 053 filas (82,5 %) tienen `kerbsideid` vacío.** Solo 5070 lo
  traen, cifra coherente con las ~5264 plazas con sensor vivo: la mayoría del
  inventario de bahías no está sensorizada.
- Dos filas (`7568N`, `7570N`) traen un sufijo de letra que rompe el formato
  numérico. Son anomalías puntuales, no un patrón.

`roadsegmentdescription` sigue el formato `"X Street between A Street and B
Street"`, que es la clave del cruce con el histórico (ver
[`id_crosswalk.md`](id_crosswalk.md)). 1035 de 29 053 descripciones no encajan
en ese patrón.

Aporta `roadsegmentid`, la unidad natural de "tramo de calle" para la Fase 4.

### `on-street-car-park-bay-restrictions` — restricciones por plaza

4263 filas. Claves **`bayid`** y **`deviceid`** (ambos int), seguidas de seis
bloques repetidos de restricción: `description1..6`, `typedesc1..6`,
`duration1..6`, `starttime1..6`, `endtime1..6`, `fromday1..6`, `today1..6`,
`effectiveonph1..6`, `disabilityext1..6`, `exemption1..6`.

Ejemplo real:
`{bayid: 9003, deviceid: 29960, description1: "4P MTR M-F 10:00-18:30",
typedesc1: "4P Meter", duration1: 240}`.

**No contiene `kerbsideid`**, y eso es el centro del problema de cruce.

### Otros

- `sign-plates-located-in-each-parking-zone` (1821 filas): `parkingzone` +
  `restriction_days`, `time_restrictions_start/finish`, `restriction_display`.
  Cruza con `zone_number` del tiempo real.
- `parking-zones-linked-to-street-segments`: `parkingzone` → tramos de calle.

## 3. Histórico anual de sensores

### No es consultable por la API

Los diez datasets históricos (`on-street-car-parking-sensor-data-2011` … `-2019`
y `-2020-jan-may`) tienen todos **`has_records: false` y `records_count: 0`**:
la API no sirve sus filas. Solo existen como ZIP de archivo enlazado desde la
descripción HTML del dataset. Para 2019:

```
https://opendatasoft-s3.s3.amazonaws.com/downloads/archive/7pgd-bdf2.zip
```

717 085 379 bytes (717 MB), 42,7 millones de filas, un único CSV dentro:
`On-street_Car_Parking_Sensor_Data_-_2019.csv`.

El S3 responde `Accept-Ranges: bytes`, así que el esquema se verificó con una
petición `Range` inflando solo el prefijo del stream deflate, sin descargar el
fichero entero. El mismo mecanismo da reanudación gratis al descargador.

Nota: la serie termina en mayo de 2020. No hay histórico publicado de 2021 en
adelante, aunque el feed en tiempo real sigue vivo.

### Campos (20, verificados sobre el CSV real)

| Campo | Ejemplo | Notas |
|---|---|---|
| `DeviceId` | `23915` | Identificador del sensor |
| `ArrivalTime` | `11/03/2019 03:00:56 AM` | **Formato US `MM/DD/YYYY hh:mm:ss AM/PM`**, hora local de Melbourne |
| `DepartureTime` | `11/03/2019 03:07:26 AM` | Igual formato |
| `DurationMinutes` | `7` | Redondeado: 6 min 30 s se publica como `7` |
| `StreetMarker` | `2374N` | Marca física en calzada: dígitos + letra de orientación |
| `SignPlateID` | `287` | A menudo vacío |
| `Sign` | `1P MTR M-SAT 7:30-18:30` | Restricción en texto; a menudo vacío |
| `AreaName` | `Windsor`, `Banks` | Nombre de área |
| `StreetId` / `StreetName` | `123` / `BOURKE STREET` | |
| `BetweenStreet1ID` / `BetweenStreet1` | `1221` / `RUSSELL STREET` | Tramo entre dos calles |
| `BetweenStreet2ID` / `BetweenStreet2` | `647` / `EXHIBITION STREET` | |
| `SideOfStreet` / `SideOfStreetCode` / `SideName` | `3` / `N` / `North` | |
| `BayId` | `1753` | Identificador de bahía |
| `InViolation` | `false` | Excedió la restricción |
| `VehiclePresent` | `false` | |

Puntos críticos al parsear:

- **Fechas en formato US.** `11/03/2019` es el 3 de noviembre, no el 11 de
  marzo. Parsear con `%m/%d/%Y %I:%M:%S %p` explícito, nunca con inferencia
  automática: una inferencia que acierte en los primeros registros fallará en
  silencio en los días ≤ 12.
- **Sin zona horaria.** Es hora local de Melbourne (`Australia/Melbourne`) con
  horario de verano: en las transiciones hay instantes ambiguos y otros
  inexistentes. Convertir a UTC requiere decidir explícitamente qué hacer con
  ellos.
- **Es un esquema de eventos, no de snapshots.** Cada fila es una estancia
  (llegada → salida). Reconstruir la ocupación en un instante dado exige pasar
  de eventos a serie temporal: trabajo de la Fase 4.
- **Sin coordenadas.** Ninguna columna de latitud/longitud. Es el núcleo del
  problema de cruce.
- El fichero viene **ordenado por `DeviceId`**, no por tiempo. Un prefijo del
  fichero no es una muestra temporal representativa, sino un subconjunto de
  sensores: relevante al muestrear para EDA.

## 4. El cruce histórico ↔ tiempo real

Resultado en el informe dedicado: [`docs/id_crosswalk.md`](id_crosswalk.md).

## Atribución

Datos de [City of Melbourne Open Data](https://data.melbourne.vic.gov.au/),
licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
