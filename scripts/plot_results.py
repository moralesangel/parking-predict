"""Generate the plots for `docs/results.md` from the result CSVs.

    uv run python scripts/plot_results.py

Reads `data/processed/results/` and writes PNGs into `docs/img/`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this runs in a console and in CI
import matplotlib.pyplot as plt
import polars as pl

from parking.config import get_settings

logger = logging.getLogger("plot")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2de"

#: Colour per entity, in fixed order. Validated with the data-visualisation
#: guide's script: CVD and normal-vision separation above the thresholds.
#: Colour follows the model, not its rank: if a model changes position between
#: charts, its colour does not change.
COLORS = {
    "lightgbm": "#2a78d6",
    "persistence": "#eb6834",
    "calibrated_persistence": "#1baf7a",
    "historical_mean": "#4a3aa7",
}
LABELS = {
    "lightgbm": "LightGBM",
    "persistence": "Persistence",
    "calibrated_persistence": "Calibrated persistence",
    "historical_mean": "Historical mean",
}
ORDER = ["lightgbm", "calibrated_persistence", "persistence", "historical_mean"]


def style(ax: plt.Axes) -> None:
    """Recessive axes and grid: the data leads, the scaffolding follows."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)


def new_figure(width: float, height: float) -> tuple[plt.Figure, plt.Axes]:
    figure, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    style(ax)
    return figure, ax


def save(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    logger.info("wrote %s", path)


def plot_brier_by_horizon(results: Path, images: Path) -> None:
    """Brier per horizon. Lines rather than bars: the x axis is continuous and
    what matters is the *trend* as the horizon grows, not comparing cells."""
    summary = pl.read_csv(results / "summary.csv")
    figure, ax = new_figure(7.2, 4.4)

    for model in ORDER:
        rows = summary.filter(pl.col("model") == model).sort("horizon_minutes")
        if rows.is_empty():
            continue
        x = rows["horizon_minutes"].to_list()
        y = rows["brier"].to_list()
        ax.plot(
            x,
            y,
            color=COLORS[model],
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=LABELS[model],
            zorder=3,
        )
        # Direct label at the end: identify the series without jumping to a legend.
        ax.annotate(
            LABELS[model],
            (x[-1], y[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=INK_SOFT,
            fontsize=9,
            va="center",
        )

    ax.set_xticks([5, 10, 15, 30])
    ax.set_xlabel("Prediction horizon (minutes)", color=INK_SOFT, fontsize=10)
    ax.set_ylabel("Brier score (lower is better)", color=INK_SOFT, fontsize=10)
    ax.set_title(
        "The margin over persistence grows with the horizon",
        color=INK,
        fontsize=12,
        fontweight="600",
        loc="left",
        pad=12,
    )
    ax.set_xlim(3, 42)
    ax.set_ylim(0, None)
    save(figure, images / "brier_by_horizon.png")


def plot_calibration(results: Path, images: Path, *, horizon: int = 15) -> None:
    """Calibration: of the times the model said p, did it happen p% of the time?"""
    figure, ax = new_figure(5.6, 5.4)
    ax.plot([0, 1], [0, 1], color="#b9b8b3", linewidth=1.5, linestyle=(0, (5, 4)), zorder=1)
    ax.annotate(
        "perfect calibration",
        (0.70, 0.665),
        color=INK_SOFT,
        fontsize=8.5,
        rotation=41,
        ha="center",
        rotation_mode="anchor",
    )

    for model in ("lightgbm", "calibrated_persistence", "persistence"):
        path = results / f"calibration_h{horizon}_{model}.csv"
        if not path.exists():
            continue
        curve = pl.read_csv(path)
        # The baselines only emit two distinct probabilities. Joining those two
        # points with a line would suggest they also produce the intermediate
        # values, which is exactly what they do NOT do: they are drawn loose.
        emits_a_range = curve.height > 3
        ax.plot(
            curve["predicted"],
            curve["observed"],
            color=COLORS[model],
            linewidth=2 if emits_a_range else 0,
            linestyle="-" if emits_a_range else "none",
            marker="o",
            markersize=9,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=LABELS[model] + ("" if emits_a_range else " (only 2 values)"),
            zorder=3,
        )

    ax.set_xlabel("Predicted probability", color=INK_SOFT, fontsize=10)
    ax.set_ylabel("Observed frequency", color=INK_SOFT, fontsize=10)
    ax.set_title(
        f"Calibration at {horizon} minutes",
        color=INK,
        fontsize=12,
        fontweight="600",
        loc="left",
        pad=12,
    )
    ax.grid(True, color=GRID, linewidth=0.8, axis="both")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    legend = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK_SOFT)
    save(figure, images / "calibration.png")


def plot_by_occupancy(results: Path, images: Path, *, horizon: int = 15) -> None:
    """Brier per current occupancy level. This is the breakdown the plan asks
    for, because the global mean hides the only case that really matters."""
    order = ["0 free (full)", "1 free", "2 free", "3-5 free", "6+ free"]
    frame = pl.read_csv(results / f"by_occupancy_h{horizon}.csv")
    frame = frame.with_columns(
        rank=pl.col("occupancy_bucket").replace_strict({k: i for i, k in enumerate(order)})
    ).sort("rank")

    models = ["lightgbm", "calibrated_persistence", "persistence", "historical_mean"]
    figure, ax = new_figure(8.0, 4.6)
    positions = range(frame.height)
    width = 0.2

    for index, model in enumerate(models):
        offset = (index - (len(models) - 1) / 2) * width
        values = frame[model].to_list()
        # A 2px gap between adjacent bars: separates without needing a border.
        ax.bar(
            [p + offset for p in positions],
            values,
            width=width * 0.92,
            color=COLORS[model],
            label=LABELS[model],
            zorder=3,
        )

    ax.set_xticks(list(positions))
    ax.set_xticklabels(frame["occupancy_bucket"].to_list(), fontsize=9)
    ax.set_ylabel("Brier score (lower is better)", color=INK_SOFT, fontsize=10)
    ax.set_xlabel("Free bays in the segment right now", color=INK_SOFT, fontsize=10)
    ax.set_title(
        f"At {horizon} min: the hard problem is on the left",
        color=INK,
        fontsize=12,
        fontweight="600",
        loc="left",
        pad=12,
    )
    legend = ax.legend(frameon=False, fontsize=9, ncol=2)
    for text in legend.get_texts():
        text.set_color(INK_SOFT)
    save(figure, images / "by_occupancy.png")


def plot_by_hour(results: Path, images: Path, *, horizon: int = 15) -> None:
    """Brier per hour of day, the other breakdown the plan asks for."""
    frame = pl.read_csv(results / f"by_hour_h{horizon}.csv").sort("hour")
    figure, ax = new_figure(8.0, 4.2)

    for model in ("lightgbm", "calibrated_persistence", "persistence"):
        ax.plot(
            frame["hour"],
            frame[model],
            color=COLORS[model],
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1,
            label=LABELS[model],
            zorder=3,
        )

    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day (UTC)", color=INK_SOFT, fontsize=10)
    ax.set_ylabel("Brier score (lower is better)", color=INK_SOFT, fontsize=10)
    ax.set_title(
        f"At {horizon} min: the error follows the occupancy cycle",
        color=INK,
        fontsize=12,
        fontweight="600",
        loc="left",
        pad=12,
    )
    legend = ax.legend(frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SOFT)
    save(figure, images / "by_hour.png")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    settings = get_settings()
    results = settings.processed_dir / "results"
    if not results.exists():
        print(f"no results in {results}; run train_evaluate.py first", file=sys.stderr)
        return 1
    images = Path(__file__).resolve().parents[1] / "docs" / "img"

    plot_brier_by_horizon(results, images)
    plot_calibration(results, images)
    plot_by_occupancy(results, images)
    plot_by_hour(results, images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
