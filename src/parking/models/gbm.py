"""LightGBM model for the probability of finding a parking space.

One model is trained **per horizon**. A single model with the horizon as a
feature would share trees between problems that are genuinely different: at 5
minutes almost everything is explained by the current state, while at 30 the
hourly pattern weighs far more. Separating them costs four fits and lets each
one be compared against its own baseline.

The target is imbalanced (94-96% positive) but it is **not rebalanced**.
Reweighting or resampling shifts the predicted probabilities and breaks
calibration, which is exactly this project's deliverable: the screen shows a
probability, not a label. With `objective="binary"` and no reweighting,
LightGBM optimises log loss and comes out calibrated by construction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

#: Conservative parameters. No hyperparameter search: with 29M rows and such a
#: strong baseline, what decides the outcome is whether the features add
#: anything, not the third decimal of `learning_rate`. Fine-tuning before
#: beating persistence would be optimising noise.
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
    """LightGBM wrapped behind the same interface as the baselines."""

    features: list[str]
    params: dict[str, object] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    num_boost_round: int = 400
    early_stopping_rounds: int = 30
    booster: lgb.Booster | None = None
    best_iteration: int | None = None

    def _matrix(self, frame: pl.DataFrame) -> np.ndarray:
        # float32 on purpose: over millions of rows, float64 doubles memory
        # without changing a tree's result.
        return frame.select(self.features).to_numpy().astype(np.float32)

    def fit(
        self,
        frame: pl.DataFrame,
        target: str,
        *,
        validation: pl.DataFrame | None = None,
    ) -> GradientBoostingModel:
        """Fit. With a validation set, stop early to avoid overfitting.

        The validation set must be **later** than training: early-stopping
        against past data would pick the number of trees using information that
        would not exist in production.
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
        logger.info("trained %s with %d trees", target, self.best_iteration)
        return self

    def predict_proba(self, frame: pl.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("call fit first")
        return np.asarray(
            self.booster.predict(self._matrix(frame), num_iteration=self.best_iteration),
            dtype=np.float64,
        )

    def save(self, directory: Path, name: str) -> Path:
        """Save the booster and its feature list.

        The list is stored separately and is essential: the booster keeps the
        column count but not where they came from, and scoring with the
        features in a different order would produce plausible, wrong numbers
        without raising anything.
        """
        if self.booster is None:
            raise RuntimeError("call fit first")
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / f"{name}.txt"
        self.booster.save_model(str(model_path), num_iteration=self.best_iteration)
        (directory / f"{name}.json").write_text(
            json.dumps({"features": self.features, "best_iteration": self.best_iteration}),
            encoding="utf-8",
        )
        return model_path

    @classmethod
    def load(cls, directory: Path, name: str) -> GradientBoostingModel:
        """Load a model saved with `save`."""
        metadata = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        model = cls(features=metadata["features"])
        model.booster = lgb.Booster(model_file=str(directory / f"{name}.txt"))
        model.best_iteration = metadata["best_iteration"]
        return model

    def importances(self) -> pl.DataFrame:
        """Gain-based importance, which is what says how much each feature adds.

        `split` would count how often each is used, and a noisy feature with
        many distinct values gets used a lot while adding nothing.
        """
        if self.booster is None:
            raise RuntimeError("call fit first")
        gains = self.booster.feature_importance(importance_type="gain")
        total = float(gains.sum()) or 1.0
        return (
            pl.DataFrame({"feature": self.features, "gain": gains})
            .with_columns(share_pct=(pl.col("gain") / total * 100).round(2))
            .sort("gain", descending=True)
        )
