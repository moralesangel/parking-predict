# Calidad de los datos de sensores

Informe de la Fase 2. Cifras medidas sobre el snapshot de
**2026-10-22 11:46:49 UTC** salvo donde se indique otra cosa, y recalculables con:

```bash
uv run python scripts/quality_report.py
```

## Resumen

**De las 6324 plazas que publica el feed, solo 3557 (56,2 %) dan una lectura
utilizable.** Las otras 2767 fallan de dos maneras distintas, que hay que
separar porque tienen causas y consecuencias diferentes.

| Estado | Plazas | % | Ocupación aparente |
|---|---|---|---|
| `ok` — reporta y cambia de estado | 3557 | 56,2 % | 44,5 % |
| `stuck` — reporta pero congelado | 1707 | 27,0 % | 39,3 % |
| `offline` — ausente del lote del feed | 1060 | 16,8 % | 53,9 % |

## 1. Los dos modos de fallo

### `offline`: el relé no reporta

Detección **exacta**, sin umbral: `lastupdated < max(lastupdated)` del snapshot.
Como `lastupdated` es el sello del lote y no del sensor
(`docs/data_schema.md` §1.2), una plaza que no aparece en el lote actual arrastra
el sello del último en que estuvo.

### `stuck`: el sensor está congelado

Es el fallo **más numeroso y el que no se ve** si solo se vigila `lastupdated`,
que es lo que recomienda la documentación de la fuente. La plaza sí está en el
lote actual —el relé la reporta— pero su `status_timestamp` no se mueve desde
hace una eternidad: **mediana de 150 días**, con casos de hasta **1432 días**
(casi cuatro años).

1707 plazas, el **32,4 % de las que están en línea**. Ignorarlo significaría
tratar como buenas 1707 lecturas que llevan meses siendo la misma.

## 2. El umbral de 24 h sale del dato, no de una corazonada

La antigüedad de `status_timestamp` entre sensores en línea es claramente
**bimodal**, y el umbral se pone en el valle:

```
  0-1h          1584  ###############################################################
  1-2h           718  ############################
  2-4h           633  #########################
  4-8h           421  ################
  8-12h          125  #####
  12-18h          64  ##
  18-24h          12            <-- valle: 12 plazas de 5264
  24-48h          59  ##
  48-72h          22
  72-168h         43  #
  168-336h       140  #####
  336-720h        60  ##
  720-2160h      274  ##########
  2160-8760h     602  ########################
  >8760h         507  ####################
```

El modo sano se agota hacia las 12 h, la franja 18-24 h está casi vacía, y a
partir de 720 h (30 días) aparece el segundo modo. Cualquier corte entre 18 y 24
horas separa las dos poblaciones; se elige 24 h por ser el ciclo natural de
rotación de un aparcamiento.

La elección además **apenas mueve el resultado**, que es la señal de que el
umbral cae en un valle y no en una pendiente:

| Umbral | Plazas fiables | Cobertura | Ocupación |
|---|---|---|---|
| 6 h | 3227 | 51,0 % | 44,4 % |
| 12 h | 3482 | 55,1 % | 44,6 % |
| 18 h | 3545 | 56,1 % | 44,5 % |
| **24 h** | **3557** | **56,2 %** | **44,5 %** |
| 48 h | 3616 | 57,2 % | 44,6 % |
| 7 d | 3681 | 58,2 % | 44,7 % |
| 30 d | 3881 | 61,4 % | 44,2 % |

Configurable en `PARKING_STUCK_AFTER_HOURS`.

**Falso positivo conocido:** una plaza sin restricción horaria legítimamente
ocupada más de 24 h se marcaría como atascada. Dos cosas acotan el riesgo: la
mediana de los marcados es de 150 días, muy lejos de una estancia larga
plausible; y el 60,7 % de los atascados aparecen **libres**, y una plaza libre
durante meses en el CBD de Melbourne no es una estancia larga, es un sensor
muerto. Con los datos del histórico y las restricciones por plaza
(`on-street-car-park-bay-restrictions`, campo `duration1..6`) se podría afinar
el umbral por tipo de restricción; queda pendiente y no bloquea la Fase 3.

## 3. El sesgo de no filtrar no es constante (y es peor en hora punta)

Esto corrige una estimación preliminar que comparaba snapshots de horas
distintas y daba una cifra fija de ~12 puntos. Medido **dentro de cada
snapshot**, el sesgo varía:

| Snapshot (UTC) | Todas las plazas | Solo fiables | Sesgo |
|---|---|---|---|
| 10:04:49 | 54,7 % | 62,3 % | **+7,6 pts** |
| 10:22:49 | 53,6 % | 60,5 % | +6,9 pts |
| 11:42:49 | 45,0 % | 45,0 % | 0,0 pts |
| 11:44:49 | 44,7 % | 44,5 % | −0,1 pts |
| 11:46:49 | 44,7 % | 44,5 % | −0,1 pts |

La razón es mecánica y hace el problema predecible: **las lecturas atascadas
tienen una ocupación congelada en torno al 39,3 % que no sigue el ciclo
diario.** Al promediarlas con las buenas, tiran de la media hacia ese 39,3 %.
El sesgo es entonces proporcional a la distancia entre la ocupación real y
39,3 %: casi nulo cuando el parque ronda ese valor (como a las 11:46) y máximo
cuando el CBD está lleno.

Es decir, **el sesgo es peor justo en hora punta, que es cuando la predicción
importa.** Motivo suficiente para que `is_reliable` sea obligatorio en las
features y no una opción.

## 4. Los fallos son de relé, no de sensor

Si los sensores fallaran de forma independiente, una zona tendría una mezcla de
estados. Lo que se observa es lo contrario:

| | Zonas |
|---|---|
| Zonas con ≥ 5 plazas | 368 |
| **Enteramente fuera de línea** | **37** |
| **Enteramente atascadas** | **44** |

**81 zonas (22 %) fallan en bloque.** Coincide con lo visto en
`docs/data_schema.md` §1.2: los 49 valores distintos de `lastupdated` son
cohortes de caída, grupos de plazas que desaparecieron en el mismo ciclo.

Consecuencia para el producto: la unidad de fallo es la zona, así que la
degradación no es gradual sino a saltos. Una zona no pierde fiabilidad poco a
poco; se apaga entera. El mapa de la Fase 3 debe poder decir **"sin datos en
esta zona"** como un estado de primera clase, no pintar una zona vacía como si
estuviera libre.

## 5. Cobertura por zona

Sobre las 191 zonas con al menos 10 plazas:

- Cobertura **mediana: 72,2 %**
- **56 zonas (29 %) están al 0 %**: ninguna lectura fiable

Las zonas al 0 % no se pueden usar para predecir nada. En la Fase 4 hay que
excluirlas del conjunto de entrenamiento —no como filas borradas, sino marcadas—
y en la Fase 6 el endpoint debe responder "sin datos" en lugar de una
probabilidad inventada sobre cero observaciones.

## 6. Sensores intermitentes

9 plazas alternan entre fiable y no fiable en la ventana observada. Son más
dañinas de lo que su número sugiere: un sensor que entra y sale del lote **mueve
el denominador de su zona** sin que se haya movido ningún coche, así que la
ocupación agregada cambia por un artefacto de medición. Candidatos a excluir de
las features aunque individualmente parezcan sanos.

`parking.quality.coverage.flapping_sensors` los detecta sobre la serie
acumulada. Con más días de datos esta lista crecerá y conviene revisarla antes
de la Fase 4.

## 7. Continuidad de la serie

`parking.quality.gaps` compara los sellos de lote consecutivos contra la cadencia
del feed (120 s) con una tolerancia de 3 ciclos, para no marcar como hueco cada
retraso normal.

Funcionando: en la ventana de prueba detectó correctamente los dos huecos que
dejaron las ejecuciones manuales de `--once` antes de arrancar el poller
continuo (18 min y 1 h 20 min). Con el poller en marcha la captura es de un lote
cada 2 minutos sin pérdidas.

El informe no atribuye causa al hueco, y es deliberado: "el poller no capturó" y
"la fuente no publicó" se ven idénticos en los ficheros guardados, y
distinguirlos requiere los logs del poller. Mezclarlos daría una falsa sensación
de diagnóstico.

## 8. Limitaciones de este informe

Hay que ser explícito en lo que **todavía no se puede afirmar**:

- **La cobertura por hora no está caracterizada.** La ventana observada son
  ~2 horas (6 snapshots), así que `coverage_by_hour` solo devuelve dos filas,
  ambas al 56,2 %. Para saber si la cobertura se degrada de noche hace falta al
  menos una semana de poller. La función está escrita y probada; faltan los
  datos.
- **Los sensores intermitentes están infra-contados** por la misma razón.
- **El reparto entre `stuck` y `offline` es una foto.** Las cohortes de caída
  indican que cambia por saltos cuando un relé se cae o se recupera, así que
  conviene repetir el informe periódicamente (Fase 7).

## Qué se lleva la Fase 3

1. **El parque utilizable son ~3557 plazas, no 6324.** Cualquier métrica sobre
   las 6324 está mal.
2. **`is_reliable` es obligatorio** en features y métricas, porque el sesgo de
   omitirlo es máximo en hora punta.
3. **"Sin datos" tiene que ser un estado visible** en el mapa y en la API: 56
   zonas no tienen ni una lectura fiable, y pintarlas como libres sería mentir.

## Atribución

Datos de [City of Melbourne Open Data](https://data.melbourne.vic.gov.au/),
licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
