# Operación

Informe de la Fase 7: reentrenamiento programado y monitorización.

```bash
uv run python scripts/monitor.py    # informe de monitorizacion
uv run python scripts/retrain.py    # job de reentrenamiento
```

Panel en `http://localhost:8000/monitoring.html`, que consume `GET /monitoring`.

## 1. El reentrenamiento tiene una puerta, y es la parte importante

El reentrenamiento automático corre un riesgo que el despliegue manual no tiene:
**nadie mira el resultado antes de que llegue a producción**. Si un mes sale
malo —una caída larga del feed, una obra que vacía media ciudad— el modelo
reentrenado puede ser peor que el anterior, y peor que no predecir nada.

Por eso un modelo **nunca se promociona por el hecho de haberse entrenado**.
Tiene que pasar dos pruebas sobre un holdout temporal:

1. **Batir a la persistencia calibrada.** Es la conclusión directa de la
   Fase 5. Un modelo que no la supera no aporta nada que no estuviera ya en el
   estado actual, y desplegarlo añadiría un pipeline de ML entero a cambio de
   nada.
2. **No empeorar respecto al modelo en producción** más de un 2 %. Reentrenar no
   debe ser un paso atrás. El margen del 2 % absorbe el ruido de un holdout
   finito; sin él, el modelo cambiaría cada semana por diferencias que no
   significan nada.

**Un reentrenamiento que no promociona es un resultado correcto, no un fallo.**
El job sale con código 0 y registra el motivo.

### La puerta ya ha hecho su trabajo

No es un salvaguarda teórico. En la primera ejecución real sobre los datos
acumulados:

```
train 10976 filas | holdout 3808 filas
candidato 0.00776 | persistencia 0.00672 | produccion ninguno
NO promociona: no bate a la persistencia (candidato 0.00776 >= persistencia
0.00672); desplegarlo anadiria un pipeline de ML a cambio de nada
```

Con 6 horas de datos, LightGBM sale **peor** que la persistencia. La puerta lo
rechazó y producción se quedó como estaba.

### Antes de eso, la puerta de "hay datos suficientes"

El job comprueba primero que haya con qué entrenar y **se niega** si no:

- **14 días mínimo.** Cubren dos ciclos semanales completos; con menos, el
  modelo no ha visto un fin de semana suficientes veces para distinguirlo del
  ruido.
- **50 000 filas etiquetadas mínimo**, para no entrenar sobre un puñado de
  tramos durante una caída del feed.

Son juicios, no umbrales derivados de los datos, y por eso van como parámetros
(`--min-days`, `--min-rows`).

Ahora mismo el job se niega, y eso es lo correcto:

```
datos acumulados: 0.24 dias | 224 tramos | 15456 filas
NO se reentrena. datos insuficientes: solo 0.24 dias de datos (minimo 14);
solo 14784 filas etiquetadas (minimo 50000)
```

### El registro guarda también los rechazos

`data/processed/registry/history.jsonl` lleva una línea por decisión,
promocionada o no. Saber que el modelo lleva cinco semanas sin promocionar
porque no bate a la persistencia es información operativa de primer orden;
guardar solo las victorias dejaría el registro mintiendo por omisión.

Las versiones promocionadas se archivan en `versions/<timestamp>/`, así que
volver atrás es copiar un directorio.

## 2. Las tres señales de monitorización

Cada una responde a una pregunta distinta, y un sistema puede ir bien en dos y
mal en la tercera.

### Degradación: ¿el modelo sigue batiendo a la persistencia *hoy*?

Es la señal que de verdad importa, y funciona porque **la etiqueta llega sola**:
la predicción hecha en `t` para `t+h` se comprueba en cuanto el poller captura
`t+h`. No hay que anotar nada a mano, solo esperar `h` minutos.

Medido sobre los datos acumulados, con el modelo entrenado en 2019:

| Horizonte | n | Brier modelo | Brier persistencia | Skill |
|---|---|---|---|---|
| 5 min | 10 936 | 0,01146 | 0,01161 | **+1,3 %** |
| 10 min | 10 936 | 0,01484 | 0,01655 | +10,3 % |
| 15 min | 10 713 | 0,01585 | 0,01895 | +16,3 % |
| 30 min | 10 044 | 0,02144 | 0,02688 | **+20,3 %** |

**El modelo de 2019 sigue batiendo a la persistencia sobre datos de 2026.** Es
mejor de lo que advertí en la Fase 6, donde dije que podría no transferir.

Pero hay una degradación real y conviene no taparla: **a 5 minutos el skill cae
del 12,8 % que se midió sobre 2019 al 1,3 % de hoy**. A horizontes largos aguanta
(15,0 % → 20,3 %). La lectura es que la parte del modelo que capturaba dinámica
de corto plazo es la que peor ha envejecido.

El backtest **excluye las filas interpoladas**. Evaluarlas mediría la calidad
del relleno de huecos, no la del modelo: una fila arrastrada del instante
anterior hace que la persistencia parezca perfecta.

### Drift: ¿la ocupación de hoy se parece a la de entrenamiento?

PSI (Population Stability Index) sobre la tasa de ocupación, con cortes fijos.
Se compara la **ocupación** y no el objetivo porque es la señal que domina el
modelo: el 90 % de la ganancia sale del estado actual y su media reciente
(`docs/results.md` §5).

Umbrales: < 0,10 estable · 0,10-0,25 moderado · > 0,25 fuerte. Son **convención
de la industria, no un contraste de hipótesis**: sirven para disparar una
revisión, no para concluir nada por sí solos.

Medición actual: **PSI 0,0451 (estable)** frente a junio de 2019.

**Esta cifra hay que leerla con mucha cautela** y es la limitación más seria del
informe: la ventana actual son ~6 horas **nocturnas** y la referencia es un mes
completo. El PSI está comparando distribuciones con cobertura horaria distinta,
así que mezcla "la ciudad ha cambiado" con "ahora es de noche". Con una semana
de poller la comparación pasará a ser hora contra hora y la cifra será
interpretable; hasta entonces, el número es un placeholder que demuestra que la
tubería funciona, no una medida de drift.

### Cobertura: ¿sobre cuánta ciudad seguimos teniendo datos?

La más fácil de pasar por alto: **un modelo puede ir perfecto mientras la mitad
de los sensores se apagan**, porque las métricas se calculan sobre los que
quedan. La cobertura cayendo significa que el producto cubre menos ciudad,
aunque acierte igual.

Estado actual: **56,2 %** (3554 de 6324 plazas), 56 zonas sin ningún dato, 1710
sensores congelados y 1060 fuera de línea. Coherente con la Fase 2.

## 3. Lo que el despliegue programado hace

El servicio `operations` del compose ejecuta en bucle, cada 24 h por defecto:

1. `scripts/monitor.py` → regenera `report.json`
2. `scripts/retrain.py --horizon 15` → intenta reentrenar y promocionar

Es un bucle de `sleep` y no cron: el contenedor es de un solo proceso, y añadir
un demonio cron solo para esto complicaría los logs y la señal de parada. Si el
job crece, se saca a un planificador de verdad.

`GET /monitoring` sirve el último informe **sin recalcularlo**: hacerlo
recorrería toda la serie acumulada dentro de una petición HTTP. La respuesta
incluye `generated_at` y `is_stale`, porque **un informe de monitorización viejo
es en sí mismo un síntoma**: significa que el job programado no está corriendo.

## 4. Limitaciones

1. **El drift no es interpretable todavía**, por lo dicho arriba: se comparan
   ventanas con cobertura horaria distinta. Es la pieza más débil del informe.
2. **El reentrenamiento nunca ha promocionado nada**, porque no hay datos
   suficientes. El camino completo está ejercitado (entrena, evalúa, decide) pero
   la rama de promoción solo está probada con tests, no con datos reales.
3. **Solo se reentrena el horizonte de 15 min.** Los otros tres siguen con el
   modelo de 2019. Ampliarlo es un bucle, pero cuadruplica el tiempo del job.
4. **No hay alertas, solo un informe.** Nadie recibe un aviso si `is_degraded`
   se pone a `true`; hay que mirar el panel. Conectarlo a un canal real queda
   fuera de esta fase.
5. **El umbral de 24 h para "informe obsoleto" es arbitrario**, elegido para
   encajar con un job diario.

## Atribución

Datos de [City of Melbourne Open Data](https://data.melbourne.vic.gov.au/),
licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
