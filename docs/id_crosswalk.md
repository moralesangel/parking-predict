# Cruce de identificadores: histórico ↔ tiempo real

Informe del **riesgo a verificar** de la Fase 1 del plan: *"los identificadores
del histórico pueden no coincidir con los Kerbside id del tiempo real"*.

Medido el **2026-10-22** contra los datos reales.

## Resumen

**El riesgo se confirma: no existe cruce a nivel de plaza.** Los dos espacios de
identificadores son disjuntos y el histórico no trae coordenadas, así que
tampoco hay cruce geométrico.

**Pero sí existe cruce a nivel de tramo de calle**, por el nombre de la calle y
las dos calles que delimitan el tramo: cubre el **88,0 % de los eventos** del
histórico. Esa es la vía utilizable, y tiene una consecuencia de diseño para la
Fase 4 (ver §5).

## 1. Qué identificadores hay

| Dataset | Identificadores | Coordenadas |
|---|---|---|
| Histórico 2019 (CSV) | `DeviceId`, `BayId`, `StreetMarker` | **no** |
| `on-street-parking-bay-sensors` (tiempo real) | `kerbsideid`, `zone_number` | sí (lat/lon) |
| `on-street-parking-bays` (geometría) | `kerbsideid`, `roadsegmentid` | sí (lat/lon) |
| `on-street-car-park-bay-restrictions` | `bayid`, `deviceid` | no |

Se recorrieron los **238 datasets** del catálogo buscando campos con nombre de
identificador de plaza o zona. Resultado:

- `kerbsideid` aparece **solo** en los dos datasets de tiempo real y geometría.
- `bayid` / `deviceid` aparecen **solo** en el de restricciones.
- **Ningún dataset del catálogo contiene a la vez `kerbsideid` y
  `bayid`/`deviceid`/`StreetMarker`.** No hay tabla de equivalencias publicada.

Lectura: `DeviceId`/`BayId` son la numeración de la era de los sensores
(2011-2020) y `kerbsideid` es la numeración posterior. La migración no se
documentó públicamente.

## 2. Método

Muestra del histórico: prefijo de 150 MB del ZIP, inflado en streaming →
**9 413 822 eventos (22 % de los 42,7 M)** con **2409 `DeviceId`** y **2346
`BayId`/`StreetMarker` distintos**.

Para cada par de datasets se mide el solape de conjuntos de identificadores como
porcentaje del conjunto menor. Como **control** se incluye un cruce que debe
funcionar (`kerbsideid` entre tiempo real y geometría): marca cómo se ve un join
real, y por tanto cuánto solape es ruido numérico.

## 3. Resultados

```
=== A) histórico -> tiempo real, directo ===
  ~   hist.DeviceId          n realtime.kerbsideid        179 comunes =   7.4 %
  NO  hist.BayId             n realtime.kerbsideid         21 comunes =   0.9 %
  NO  hist.DeviceId          n bays.kerbsideid             66 comunes =   2.7 %
  NO  hist.BayId             n bays.kerbsideid              1 comunes =   0.0 %

=== B) histórico -> restricciones (puente candidato) ===
  OK  hist.DeviceId          n restr.deviceid            1459 comunes =  60.6 %
  OK  hist.BayId             n restr.bayid               1779 comunes =  75.8 %

=== C) restricciones -> tiempo real (cerraría el puente) ===
  ~   restr.deviceid         n realtime.kerbsideid        314 comunes =   7.4 %
  NO  restr.bayid            n realtime.kerbsideid         76 comunes =   1.8 %

=== D) control: geometría -> tiempo real (debe cruzar) ===
  OK  realtime.kerbsideid    n bays.kerbsideid           4825 comunes =  95.5 %
```

Interpretación:

- **(D) es la referencia: un join real da 95,5 %.**
- **(A) y (C) no son joins.** El 7,4 % de `DeviceId` ∩ `kerbsideid` es
  coincidencia aritmética: `DeviceId` vive en [23 913, 27 596] y `kerbsideid` en
  [5691, 90 812], así que los rangos se superponen y algunos números caen por
  azar en ambos. Unir por ahí produciría emparejamientos silenciosamente falsos
  — peor que no unir.
- **(B) sí es un join.** 60,6 % y 75,8 % confirman que histórico y
  restricciones comparten numeración. El dataset de restricciones es, de hecho,
  un residuo de la era de los sensores.
- El puente **histórico → restricciones → tiempo real se rompe en el segundo
  salto** (C). No sirve.

### Vía geométrica: descartada en origen

Cruzar por coordenadas es imposible: el CSV del histórico **no tiene ninguna
columna de latitud/longitud** (ver `docs/data_schema.md` §3). No es que el
cruce salga mal, es que no hay con qué intentarlo a nivel de plaza.

Se exploró y descartó una hipótesis intermedia: `on-street-parking-bays.kerbsideid`
es de tipo texto y contiene dos valores con sufijo de letra (`7568N`, `7570N`)
con el mismo aspecto que los `StreetMarker` del histórico (`2374N`, `1581S`).
Resultó ser un callejón sin salida: solo **2 de 29 053 filas** tienen ese
formato, son anomalías de carga y no un identificador alternativo.

## 4. Lo que sí funciona: cruce por tramo de calle

El histórico describe la ubicación con texto: `StreetName`,
`BetweenStreet1`, `BetweenStreet2`, `SideOfStreet`. Y
`on-street-parking-bays.roadsegmentdescription` usa el formato
`"X Street between A Street and B Street"`.

Normalizando ambos a la misma clave —mayúsculas, espacios colapsados, y el par
de calles delimitadoras **sin orden** (porque "A and B" y "B and A" son el mismo
tramo)— el cruce funciona:

| Métrica | Resultado |
|---|---|
| Tramos del histórico presentes en geometría | 225 / 297 = **75,8 %** |
| **Eventos del histórico cubiertos** | 8 288 708 / 9 413 823 = **88,0 %** |
| Plazas (`StreetMarker`) cubiertas | 1903 / 2343 = **81,2 %** |
| Tramos ambiguos (→ más de un `roadsegmentid`) | **1** |

La cobertura en eventos (88,0 %) es mayor que en tramos (75,8 %): los tramos que
no cruzan son sobre todo periféricos y de bajo tráfico. Los seis con más eventos
perdidos son Cobden St, William St (entre Victoria y Walsh), Spring St (entre
Lonsdale y Victoria Pde), Dudley St, O'Connell St y Rosslyn St — zonas de borde
del CBD, probablemente retiradas del inventario actual o renombradas.

Con ese cruce, cada plaza del histórico hereda un `roadsegmentid` y, por tanto,
coordenadas: el histórico queda situado en el espacio y es comparable con el
tiempo real **al nivel de tramo, nunca de plaza individual**.

## 5. Consecuencias para el plan

**La unidad espacial de la Fase 4 debe ser el tramo de calle (`roadsegmentid`),
no la celda H3.** El plan dejaba la elección abierta ("celdas H3 … o tramos de
calle. Justificar la elección"). La justificación es ésta: el tramo es el único
nivel en el que histórico y tiempo real se reconcilian. Con H3 habría que
asignar celdas por coordenadas, y el histórico no las tiene — se perderían los
42,7 M de eventos para cualquier feature espacial. El tramo además coincide con
la decisión real del conductor ("¿hay sitio en esta manzana?") mejor que una
celda hexagonal arbitraria.

**El histórico de 2019 no sirve para entrenar el modelo de producción**, por dos
razones independientes del cruce:

1. **Deriva temporal.** 2019 es pre-COVID; la serie publicada se corta en mayo
   de 2020 y no hay datos de 2021 en adelante. Los patrones de ocupación del
   CBD de Melbourne de hace siete años no describen los de hoy.
2. **El parque de sensores cambió.** El cambio de numeración
   (`DeviceId` → `kerbsideid`) acompaña una renovación del parque: no hay
   garantía de que una plaza de 2019 exista hoy, ni al revés.

Se adopta por tanto la alternativa que el propio plan anticipaba: **el histórico
sirve para desarrollar y validar el pipeline** (hay 42,7 M de eventos reales con
los que construir features, probar la conversión de eventos a serie temporal y
comparar baselines), **y el modelo de producción se entrena con el histórico que
acumule el poller.**

Esto tiene un coste honesto que conviene decir ya: hasta que el poller acumule
semanas de datos, los resultados de la Fase 5 serán resultados *sobre 2019*. La
`docs/results.md` tendrá que distinguir explícitamente entre ambos conjuntos.

## 6. Reproducibilidad

El cruce no se congela como fichero de datos: se recalcula desde los datasets
crudos. Los números de este informe salen de los scripts de
`scripts/crosswalk_report.py`, que vuelve a descargar las entradas por API y
reimprime las tablas de §3 y §4.

## Atribución

Datos de [City of Melbourne Open Data](https://data.melbourne.vic.gov.au/),
licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
