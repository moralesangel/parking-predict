"""Métricas y desgloses para comparar modelos.

La exactitud no sirve aquí y conviene decir por qué: el objetivo es positivo el
94-96 % de las veces (ver `docs/dataset.md`), así que contestar siempre "sí"
acierta el 95 %. Una métrica que premia eso no distingue un modelo útil de uno
inútil.

Se usan en su lugar:

- **Brier score**: error cuadrático medio de la probabilidad. Más bajo, mejor.
- **Log loss**: penaliza con dureza la confianza equivocada.
- **Curva de calibración**: de las veces que el modelo dijo 0,8, ¿pasó el 80 %?
  Un modelo puede ordenar bien y estar mal calibrado, y para este producto la
  calibración *es* el entregable: la pantalla muestra una probabilidad.

Y dos desgloses que el plan pide explícitamente, porque la media global esconde
justo el caso que importa:

- **Por hora del día**.
- **Por nivel de ocupación actual**. Nadie necesita que le digan que hay sitio
  en una calle vacía; el valor está en las zonas casi llenas.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from parking.models.baselines import clip_probabilities


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Error cuadrático medio entre probabilidad y resultado."""
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Log loss con recorte previo, para que un 0/1 equivocado no dé infinito."""
    clipped = clip_probabilities(y_prob)
    return float(-np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped)))


def base_rate(y_true: np.ndarray) -> float:
    """Proporción de positivos. Es el contexto sin el que las métricas no se leen."""
    return float(np.mean(y_true))


def skill_score(metric: float, reference: float) -> float:
    """Mejora relativa frente a una referencia. Positivo = mejor que la referencia.

    `1 - metric/reference`, la forma habitual del Brier skill score. Es lo que
    responde a la pregunta del proyecto: ¿cuánto aporta predecir frente a usar
    el estado actual?
    """
    if reference == 0:
        return float("nan")
    return float(1.0 - metric / reference)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Métricas principales de un conjunto de predicciones."""
    return {
        "n": float(len(y_true)),
        "base_rate": base_rate(y_true),
        "brier": brier_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "mean_prediction": float(np.mean(y_prob)),
    }


def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, *, bins: int = 10) -> pl.DataFrame:
    """Curva de calibración por deciles de probabilidad predicha.

    Devuelve, por tramo de probabilidad, cuántas observaciones cayeron en él,
    la probabilidad media que predijo el modelo y la frecuencia real. Si el
    modelo está calibrado, las dos últimas columnas coinciden.

    Los tramos vacíos no aparecen: un baseline que solo emite dos valores
    distintos ocupa dos tramos, y mostrar ocho filas vacías sugeriría que falta
    información en vez de que el modelo es así.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    # `right=False` y el clip final evitan que la probabilidad 1,0 caiga en un
    # tramo inexistente por encima del último.
    indices = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, bins - 1)

    frame = pl.DataFrame({"bin": indices.astype(np.int32), "y_true": y_true, "y_prob": y_prob})
    return (
        frame.group_by("bin")
        .agg(
            n=pl.len(),
            predicted=pl.col("y_prob").mean(),
            observed=pl.col("y_true").mean(),
        )
        .with_columns(
            bin_low=(pl.col("bin") / bins).round(2),
            bin_high=((pl.col("bin") + 1) / bins).round(2),
            gap=(pl.col("observed") - pl.col("predicted")).round(4),
        )
        .sort("bin")
        .select("bin_low", "bin_high", "n", "predicted", "observed", "gap")
    )


def breakdown(
    frame: pl.DataFrame,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    *,
    by: str,
) -> pl.DataFrame:
    """Brier por modelo y por grupo (`by`), más el skill frente a la persistencia.

    `predictions` es `{nombre_modelo: probabilidades}`. La columna `by` debe
    existir en `frame`.
    """
    data = {"group": frame[by].to_numpy(), "y_true": y_true}
    for name, values in predictions.items():
        data[f"se__{name}"] = (values - y_true) ** 2

    grouped = (
        pl.DataFrame(data)
        .group_by("group")
        .agg(
            [pl.len().alias("n"), pl.col("y_true").mean().alias("base_rate")]
            + [pl.col(f"se__{name}").mean().alias(name) for name in predictions]
        )
        .sort("group")
    )
    return grouped.rename({"group": by})


def occupancy_bucket() -> pl.Expr:
    """Expresión que clasifica cada fila por lo llena que está la zona **ahora**.

    Se usa como `frame.with_columns(occupancy_bucket())`.

    Los cortes no son uniformes a propósito: lo interesante ocurre cerca del
    lleno, así que los tramos superiores son más finos. Separar "lleno" (cero
    plazas) de "casi lleno" importa porque son situaciones distintas para el
    conductor y para el modelo.
    """
    return (
        pl.when(pl.col("free_bays") == 0)
        .then(pl.lit("0 libres (lleno)"))
        .when(pl.col("free_bays") == 1)
        .then(pl.lit("1 libre"))
        .when(pl.col("free_bays") == 2)
        .then(pl.lit("2 libres"))
        .when(pl.col("free_bays") <= 5)
        .then(pl.lit("3-5 libres"))
        .otherwise(pl.lit("6+ libres"))
        .alias("occupancy_bucket")
    )


def comparison_table(results: dict[str, dict[str, float]]) -> pl.DataFrame:
    """Tabla comparativa de modelos, ordenada por Brier.

    Incluye el skill frente a `persistence` si está presente, que es la columna
    que responde a la pregunta del proyecto.
    """
    reference = results.get("persistence", {}).get("brier")
    rows = []
    for name, metrics in results.items():
        row = {"model": name, **metrics}
        if reference is not None:
            row["brier_skill_vs_persistence"] = skill_score(metrics["brier"], reference)
        rows.append(row)
    return pl.DataFrame(rows).sort("brier")
