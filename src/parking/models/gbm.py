"""Modelo LightGBM para la probabilidad de encontrar plaza.

Se entrena un modelo **por horizonte**. Un único modelo con el horizonte como
feature compartiría los árboles entre problemas que son distintos: a 5 minutos
casi todo lo explica el estado actual y a 30 pesa mucho más el patrón horario.
Separarlos cuesta cuatro entrenamientos y permite además comparar cada uno con
su baseline.

El objetivo está desbalanceado (94-96 % positivos) pero **no se reequilibra**.
Reponderar o remuestrear desplaza las probabilidades predichas y rompe la
calibración, que es justamente el entregable de este proyecto: la pantalla
muestra una probabilidad, no una etiqueta. Con `objective="binary"` y sin
reponderar, LightGBM optimiza log loss y sale calibrado de fábrica.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

#: Parámetros conservadores. No se buscan hiperparámetros: con 29 M de filas y
#: un baseline tan fuerte, lo que decide el resultado es si las features
#: aportan algo, no el tercer decimal de `learning_rate`. Un ajuste fino sin
#: haber batido antes a la persistencia sería optimizar ruido.
DEFAULT_PARAMS: dict[str, object] = {
    "objective": "binary",
    "metric": ["binary_logloss"],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "num_threads": 0,
}


@dataclass
class GradientBoostingModel:
    """LightGBM envuelto con la misma interfaz que los baselines."""

    features: list[str]
    params: dict[str, object] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    num_boost_round: int = 400
    early_stopping_rounds: int = 30
    booster: lgb.Booster | None = None
    best_iteration: int | None = None

    def _matrix(self, frame: pl.DataFrame) -> np.ndarray:
        # float32 a propósito: con millones de filas, float64 duplica la memoria
        # sin cambiar el resultado de un árbol.
        return frame.select(self.features).to_numpy().astype(np.float32)

    def fit(
        self,
        frame: pl.DataFrame,
        target: str,
        *,
        validation: pl.DataFrame | None = None,
    ) -> GradientBoostingModel:
        """Entrena. Si hay validación, para pronto para no sobreajustar.

        La validación debe ser **posterior** al entrenamiento: parar pronto
        mirando datos del pasado elegiría el número de árboles con información
        que en producción no existiría.
        """
        usable = frame.filter(pl.col(target).is_not_null())
        train_set = lgb.Dataset(
            self._matrix(usable),
            label=usable[target].to_numpy().astype(np.float32),
            feature_name=self.features,
        )

        valid_sets, callbacks = [], [lgb.log_evaluation(period=0)]
        if validation is not None:
            usable_validation = validation.filter(pl.col(target).is_not_null())
            valid_sets.append(
                lgb.Dataset(
                    self._matrix(usable_validation),
                    label=usable_validation[target].to_numpy().astype(np.float32),
                    reference=train_set,
                )
            )
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        self.booster = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets or None,
            callbacks=callbacks,
        )
        self.best_iteration = self.booster.best_iteration or self.num_boost_round
        logger.info("entrenado %s con %d arboles", target, self.best_iteration)
        return self

    def predict_proba(self, frame: pl.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("llama antes a fit")
        return np.asarray(
            self.booster.predict(self._matrix(frame), num_iteration=self.best_iteration),
            dtype=np.float64,
        )

    def importances(self) -> pl.DataFrame:
        """Importancia por ganancia, que es la que indica cuánto aporta cada feature.

        `split` contaría cuántas veces se usa cada una, y una feature ruidosa
        con muchos valores distintos se usa mucho sin aportar nada.
        """
        if self.booster is None:
            raise RuntimeError("llama antes a fit")
        gains = self.booster.feature_importance(importance_type="gain")
        total = float(gains.sum()) or 1.0
        return (
            pl.DataFrame({"feature": self.features, "gain": gains})
            .with_columns(share_pct=(pl.col("gain") / total * 100).round(2))
            .sort("gain", descending=True)
        )
