"""Metrics and breakdowns for comparing models.

Accuracy is useless here, and it is worth saying why: the target is positive
94-96% of the time (see `docs/dataset.md`), so always answering "yes" scores
95%. A metric that rewards that cannot tell a useful model from a useless one.

Used instead:

- **Brier score**: mean squared error of the probability. Lower is better.
- **Log loss**: punishes confident mistakes hard.
- **Calibration curve**: of the times the model said 0.8, did it happen 80% of
  the time? A model can rank well and be badly calibrated, and for this product
  calibration *is* the deliverable: the screen shows a probability.

Plus two breakdowns the plan explicitly asks for, because the global mean hides
exactly the case that matters:

- **By hour of day**.
- **By current occupancy level.** Nobody needs to be told there is space on an
  empty street; the value is in the nearly full zones.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from parking.models.baselines import clip_probabilities


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between probability and outcome."""
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Log loss with prior clipping, so a wrong 0/1 does not return infinity."""
    clipped = clip_probabilities(y_prob)
    return float(-np.mean(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped)))


def base_rate(y_true: np.ndarray) -> float:
    """Share of positives. The context without which the metrics cannot be read."""
    return float(np.mean(y_true))


def skill_score(metric: float, reference: float) -> float:
    """Relative improvement over a reference. Positive means better.

    `1 - metric/reference`, the usual form of the Brier skill score. It answers
    the project's question: how much does predicting add over using the current
    state?
    """
    if reference == 0:
        return float("nan")
    return float(1.0 - metric / reference)


def evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Headline metrics for one set of predictions."""
    return {
        "n": float(len(y_true)),
        "base_rate": base_rate(y_true),
        "brier": brier_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "mean_prediction": float(np.mean(y_prob)),
    }


def calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, *, bins: int = 10) -> pl.DataFrame:
    """Calibration curve over deciles of predicted probability.

    Returns, per probability band, how many observations fell in it, the mean
    probability the model predicted and the observed frequency. For a
    calibrated model the last two columns match.

    Empty bands are omitted: a baseline emitting only two distinct values
    occupies two bands, and showing eight empty rows would suggest missing
    information rather than a model that simply behaves that way.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    # `right=False` plus the final clip keep probability 1.0 from landing in a
    # non-existent band above the last one.
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
    """Brier per model and per group (`by`), for comparing against persistence.

    `predictions` is `{model_name: probabilities}`. The `by` column must exist
    in `frame`.
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
    """Expression classifying each row by how full the zone is **right now**.

    Used as `frame.with_columns(occupancy_bucket())`.

    The cuts are deliberately uneven: what matters happens near full, so the
    upper bands are finer. Separating "full" (zero bays) from "nearly full"
    matters because they are different situations for the driver and for the
    model.
    """
    return (
        pl.when(pl.col("free_bays") == 0)
        .then(pl.lit("0 free (full)"))
        .when(pl.col("free_bays") == 1)
        .then(pl.lit("1 free"))
        .when(pl.col("free_bays") == 2)
        .then(pl.lit("2 free"))
        .when(pl.col("free_bays") <= 5)
        .then(pl.lit("3-5 free"))
        .otherwise(pl.lit("6+ free"))
        .alias("occupancy_bucket")
    )


def comparison_table(results: dict[str, dict[str, float]]) -> pl.DataFrame:
    """Model comparison table, sorted by Brier.

    Includes the skill against `persistence` when present, which is the column
    that answers the project's question.
    """
    reference = results.get("persistence", {}).get("brier")
    rows = []
    for name, metrics in results.items():
        row = {"model": name, **metrics}
        if reference is not None:
            row["brier_skill_vs_persistence"] = skill_score(metrics["brier"], reference)
        rows.append(row)
    return pl.DataFrame(rows).sort("brier")
