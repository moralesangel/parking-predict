# El problema de predicción y el dataset

Informe de la Fase 4. Reproducible con:

```bash
uv run python scripts/build_dataset.py --months 2019-01:2019-12
```

## El problema

> Dado el estado de un tramo de calle en `t`, ¿cuál es la probabilidad de que
> haya **al menos una plaza libre** en `t + h`, con `h` ∈ {5, 10, 15, 30} min?

La referencia a batir es la **persistencia**: responder con el estado actual.
Todo el proyecto existe para comprobar si se puede mejorar eso, porque cuando
conduces 12 minutos hasta el destino lo que importa no es cómo está la zona
ahora sino cómo estará al llegar.

El dataset guarda **dos objetivos por horizonte**:

- `target_h5`, `target_h10`, `target_h15`, `target_h30` — booleano, hay al menos
  una plaza libre. Es la clasificación que pide el plan.
- `free_bays_h5` … `free_bays_h30` — el recuento de plazas libres. Permite la
  alternativa que el plan manda evaluar: predecir el número y derivar de ahí la
  probabilidad, en lugar de clasificar directamente.

## La unidad espacial es el tramo de calle

Decidido: `segment_id`, no celdas H3. Tres razones, por orden de peso:

1. **Es el único nivel en el que histórico y tiempo real se reconcilian.** El
   histórico no trae coordenadas, así que asignar celdas H3 exigiría unas
   latitudes que no existen y dejaría los 42,7 M de eventos fuera de cualquier
   feature espacial. Medido en [`id_crosswalk.md`](id_crosswalk.md): el cruce
   por tramo cubre el 88 % de los eventos.
2. **Coincide con la decisión del conductor.** "¿Hay sitio en esta manzana?" es
   la pregunta real; un hexágono de 100 m no corresponde a nada que se vea
   desde el coche.
3. **Los fallos de sensor son por zona, no por plaza** (ver
   [`data_quality.md`](data_quality.md) §4), así que una unidad agregada por
   calle encaja con cómo se degradan los datos.

Un tramo es `CALLE|CRUCE_A|CRUCE_B` con las dos calles delimitadoras ordenadas
alfabéticamente, porque los datos usan ambos órdenes para el mismo tramo.

**Las dos aceras van juntas.** Un conductor que busca sitio en una manzana mira
las dos. Además es la granularidad del `roadsegmentid` de la geometría, con lo
que el cruce se mantiene.

### Vecindad sin geometría

Las features de "zonas vecinas" necesitan un grafo, y el histórico no tiene
coordenadas. Se deriva de los propios nombres: un tramo ocupa dos
**intersecciones** (`{CALLE, CRUCE_A}` y `{CALLE, CRUCE_B}`), y dos tramos son
vecinos si comparten una.

Esto captura la manzana contigua y la esquina perpendicular, y —lo importante—
**no** une dos tramos de Collins Street en extremos opuestos del CBD, que es el
error en el que caería una regla de "misma calle".

Resultado: mediana de 3 vecinos por tramo, máximo 7, y 24 tramos aislados que
reciben nulos en esas features.

## El error que casi arruina el dataset: `VehiclePresent`

Vale la pena documentarlo porque no produce ningún fallo, solo números
plausibles y equivocados.

El histórico **no es una lista de estancias**. Es una partición del tiempo de
cada plaza en intervalos, y `VehiclePresent` dice qué es cada uno:

- `true` → la plaza estuvo **ocupada** en ese intervalo
- `false` → la plaza estuvo **libre** en ese intervalo

La descripción de la fuente lo dice con todas las letras: un evento indica si
había vehículo o no, *"which can determine stay time **and vacancy time**"*.

Medido sobre la semana del 1 al 8 de marzo de 2019:

| Intervalos | Filas | Cubren del tiempo-plaza |
|---|---|---|
| `VehiclePresent = true` | 378 792 | **43,7 %** |
| `VehiclePresent = false` | 340 829 | 47,9 % |
| Total | 719 621 | 91,6 % |

Tratar todas las filas como estancias da una ocupación del **91,6 %**, que no es
una ocupación sino la suma de ocupado más vacío. El síntoma que lo delató fue el
perfil horario: 91,8 % a las 3 de la madrugada y 91,5 % a mediodía, es decir,
sin ciclo diario. Un aparcamiento de CBD sin ciclo diario no existe.

Con el filtro correcto el perfil es el esperado, con doble pico:

```
  04h  21.7%  ##########
  07h  34.4%  #################
  09h  48.3%  ########################
  11h  64.0%  ###############################   <- pico de mediodia
  17h  43.7%  #####################
  19h  59.4%  #############################     <- pico de cena
  23h  34.7%  #################
```

Y el 43,7 % concuerda con lo que mide hoy el feed en tiempo real (44-55 % según
la hora), que es una comprobación independiente.

El 8,4 % restante del tiempo-plaza no está cubierto por ningún intervalo: son
huecos de observación del propio sensor.

## Reconstrucción de la ocupación

De intervalos a serie temporal, sin el producto cartesiano que sobre 42,7 M de
filas no termina:

1. Cada intervalo ocupado se parte en dos deltas: `+1` al llegar, `-1` al salir.
2. Se ordenan por tiempo dentro de cada tramo.
3. La **suma acumulada** da las plazas ocupadas tras cada evento.
4. Un `join_asof` lleva ese valor a la rejilla de 5 minutos.

Coste: una ordenación en vez de un cruce. El año entero se procesa en ~8 min.

Dos detalles que cambian el resultado:

- **Se incluyen los intervalos que solapan la ventana**, no solo los que
  empiezan en ella. Un coche aparcado desde antes ocupa su plaza igual, y
  filtrar por `arrival >= start` haría parecer el tramo más vacío.
- **`total_bays` se cuenta sobre los dos tipos de intervalo.** Una plaza libre
  toda la ventana no genera ningún intervalo ocupado pero existe; derivar el
  inventario solo de los ocupados la perdería justo en los tramos más vacíos.

Rejilla de **5 minutos** porque es el horizonte más corto del plan: una rejilla
más gruesa no podría expresar el objetivo a 5 min.

## El dataset

| | |
|---|---|
| Periodo | 2019-01-01 a 2019-12-31 |
| Filas | **29 123 136** |
| Tramos | 301 |
| Features | 32 |
| Objetivos | 4 booleanos + 4 recuentos |
| Formato | Parquet particionado en `data/processed/dataset/month=YYYY-MM/` |

Se procesa **mes a mes** para acotar la memoria, y cada mes se calcula con un
margen de 60 min por la izquierda (el retardo más largo) y 30 min por la derecha
(el horizonte más largo), recortando después. Sin ese margen las primeras filas
de cada mes tendrían los retardos nulos y las últimas el objetivo nulo, y
aparecería un hueco artificial cada día 1.

Se descartan los tramos con menos de 3 plazas observadas: con 1 o 2, "hay al
menos una libre" es prácticamente el estado de una sola plaza y no un problema
de zona.

### Features

Todas calculables en `t`. Ninguna mira al futuro.

- **Estado actual**: `free_bays`, `occupied_bays`, `total_bays`, `occupancy_rate`
- **Historia reciente**: `free_bays_lag{5,10,15,30,60}` y sus equivalentes en
  tasa; `free_bays_trend{...}` (cambio respecto al retardo);
  `free_bays_mean{15,30,60}` (medias móviles)
- **Vecinos**: `neighbour_free_bays`, `neighbour_occupancy_rate`,
  `neighbour_count`
- **Calendario**: `hour`, `minute_of_day`, `minute_sin`, `minute_cos`,
  `day_of_week`, `is_weekend`, `is_holiday`

`minute_of_day` va además como seno y coseno: la medianoche está a un minuto de
las 23:59 y un entero plano afirma que están a 1439 de distancia.

Las tendencias existen porque dos tramos con la misma ocupación pueden tener
futuros opuestos según se estén llenando o vaciando.

`feature_columns()` deriva la lista de los datos en vez de escribirla a mano,
para que al añadir un horizonte nuevo no se cuele como feature.

### Split temporal, nunca aleatorio

`temporal_split()` corta por `timestamp`. Un split aleatorio pondría `t+5` en
entrenamiento y `t` en test, que están correlacionadísimos: las métricas serían
excelentes y el modelo incapaz de predecir el futuro real.

**Fuga conocida en la frontera.** Con horizonte `h`, la etiqueta de la última
fila de entrenamiento procede de `train_end + h`, que ya cae en validación. Con
`h ≤ 30 min` sobre meses de datos es despreciable; si se acortan los bloques hay
que descartar las últimas `h` filas de cada uno.

## La tasa base: el dato más importante para la Fase 5

**El objetivo es positivo el 94-96 % de las veces.**

| Mes | `target_h5` positivo |
|---|---|
| 2019-01 | 95,96 % |
| 2019-06 | 93,90 % |
| 2019-12 | 94,37 % |

Esto condiciona toda la evaluación de la Fase 5:

- **"Siempre hay sitio" acierta el 95 %.** La exactitud es inútil como métrica.
  Por eso el plan pide Brier score, log loss y curvas de calibración.
- **El caso interesante es el 5 % restante**, que es exactamente lo que el plan
  anticipaba al pedir desglosar por nivel de ocupación. Un modelo que mejore la
  media global y no distinga el caso "zona casi llena" no sirve de nada: nadie
  necesita que le digan que hay sitio en una calle vacía.
- La persistencia va a ser **muy fuerte** a 5 minutos. Si no se la bate, hay que
  documentarlo tal cual.

## Limitaciones

1. **Una plaza sin ningún intervalo es invisible.** No se puede distinguir entre
   "no existe", "estuvo ocupada todo el periodo" y "el sensor estaba muerto".
   `total_bays` es el inventario *observado*, no el real.
2. **58 filas de 606 816 (0,01 %) tienen `occupied_bays > total_bays`**, por
   intervalos solapados en la misma plaza. `free_bays` se recorta a 0; no se
   corrigen los intervalos de origen.
3. **Hora local sin zona horaria.** Las fechas del histórico son de Melbourne
   con horario de verano. Se procesan como instantes ingenuos, de modo que los
   dos cambios de hora de 2019 tienen una hora duplicada y otra inexistente. A
   escala anual es ruido, pero invalida cualquier análisis fino de esos dos días.
4. **Los festivos de Victoria 2019 están tomados del registro público y
   conviene verificarlos** contra la fuente oficial, sobre todo los de fecha
   móvil (viernes previo a la final de la AFL y Melbourne Cup). Un festivo mal
   puesto no rompe nada, solo mete ruido en una feature.
5. **Este dataset es de 2019 y no sirve para producción.** Las razones están en
   [`id_crosswalk.md`](id_crosswalk.md) §5: deriva temporal y renovación del
   parque de sensores. Sirve para construir y validar el pipeline; el modelo de
   producción se entrenará con lo que acumule el poller.
6. **El tiempo real todavía no pasa por este pipeline.** Las funciones están
   escritas contra el esquema del histórico. Adaptarlas a los snapshots del
   poller es trabajo de la Fase 7, cuando haya semanas acumuladas.

## Atribución

Datos de [City of Melbourne Open Data](https://data.melbourne.vic.gov.au/),
licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
