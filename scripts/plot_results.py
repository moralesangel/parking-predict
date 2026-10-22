"""Genera los gráficos de `docs/results.md` a partir de los CSV de resultados.

    uv run python scripts/plot_results.py

Lee `data/processed/results/` y escribe PNG en `docs/img/`.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: esto corre en consola y en CI
import matplotlib.pyplot as plt
import polars as pl

from parking.config import get_settings

logger = logging.getLogger("plot")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2de"

#: Color por entidad, en orden fijo. Validado con el script de la guía de
#: visualización: separación CVD y de visión normal por encima de los umbrales.
#: El color sigue al modelo, no a su posición en el ranking: si un modelo cambia
#: de puesto entre gráficos, su color no cambia.
COLORS = {
    "lightgbm": "#2a78d6",
    "persistence": "#eb6834",
    "calibrated_persistence": "#1baf7a",
    "historical_mean": "#4a3aa7",
}
LABELS = {
    "lightgbm": "LightGBM",
    "persistence": "Persistencia",
    "calibrated_persistence": "Persistencia calibrada",
    "historical_mean": "Media histórica",
}
ORDER = ["lightgbm", "calibrated_persistence", "persistence", "historical_mean"]


def style(ax: plt.Axes) -> None:
    """Ejes y rejilla recesivos: los datos mandan, el andamiaje acompaña."""
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
    logger.info("escrito %s", path)


def plot_brier_by_horizon(results: Path, images: Path) -> None:
    """Brier por horizonte. Líneas y no barras: el eje x es continuo y lo que
    importa es la *tendencia* al alejarse el horizonte, no comparar celdas."""
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
        # Etiqueta directa al final: identificar la serie sin saltar a la leyenda.
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
    ax.set_xlabel("Horizonte de predicción (minutos)", color=INK_SOFT, fontsize=10)
    ax.set_ylabel("Brier score (menor es mejor)", color=INK_SOFT, fontsize=10)
    ax.set_title(
        "El margen sobre la persistencia crece con el horizonte",
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
    """Calibración: de las veces que el modelo dijo p, ¿ocurrió el p %?"""
    figure, ax = new_figure(5.6, 5.4)
    ax.plot([0, 1], [0, 1], color="#b9b8b3", linewidth=1.5, linestyle=(0, (5, 4)), zorder=1)
    ax.annotate(
        "calibración perfecta",
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
        # Los baselines solo emiten dos probabilidades distintas. Unir esos dos
        # puntos con una linea sugeriria que tambien producen los valores
        # intermedios, que es justo lo que NO hacen: se dibujan sueltos.
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
            label=LABELS[model] + ("" if emits_a_range else " (solo 2 valores)"),
            zorder=3,
        )

    ax.set_xlabel("Probabilidad predicha", color=INK_SOFT, fontsize=10)
    ax.set_ylabel("Frecuencia observada", color=INK_SOFT, fontsize=10)
    ax.set_title(
        f"Calibración a {horizon} minutos",
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
    """Brier por nivel de ocupación actual. Es el desglose que el plan pide,
    porque la media global esconde el único caso que de verdad importa."""
    order = ["0 libres (lleno)", "1 libre", "2 libres", "3-5 libres", "6+ libres"]
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
        # Hueco de 2 px entre barras adyacentes: separa sin necesitar borde.
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
    ax.set_ylabel("Brier score (menor es mejor)", color=INK_SOFT, fontsize=10)
    ax.set_xlabel("Plazas libres en el tramo ahora", color=INK_SOFT, fontsize=10)
    ax.set_title(
        f"A {horizon} min: el problema difícil está a la izquierda",
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
    """Brier por hora del día, el otro desglose que pide el plan."""
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
    ax.set_xlabel("Hora del día (UTC)", color=INK_SOFT, fontsize=10)
    ax.set_ylabel("Brier score (menor es mejor)", color=INK_SOFT, fontsize=10)
    ax.set_title(
        f"A {horizon} min: el error sigue el ciclo de ocupación",
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
        print(f"faltan resultados en {results}; ejecuta antes train_evaluate.py", file=sys.stderr)
        return 1
    images = Path(__file__).resolve().parents[1] / "docs" / "img"

    plot_brier_by_horizon(results, images)
    plot_calibration(results, images)
    plot_by_occupancy(results, images)
    plot_by_hour(results, images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
