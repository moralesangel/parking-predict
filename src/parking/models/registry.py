"""Registro de modelos: versiones, promoción y rechazo.

El reentrenamiento automático tiene un riesgo que el despliegue manual no tiene:
nadie mira el resultado antes de que llegue a producción. Si un mes de datos
sale malo —una caída larga del feed, una obra que vacía media ciudad— el modelo
reentrenado puede ser peor que el que ya había, y peor que no predecir nada.

Por eso un modelo **nunca se promociona por el hecho de haberse entrenado**.
Tiene que pasar dos pruebas sobre un holdout temporal:

1. **Batir a la persistencia.** Es la conclusión directa de la Fase 5: la
   persistencia calibrada es durísima, y un modelo que no la supera no aporta
   nada que no estuviera ya en el estado actual. Desplegarlo añadiría un
   pipeline de ML entero a cambio de nada.
2. **No empeorar respecto al modelo en producción**, más allá de un margen de
   tolerancia. Reentrenar no debe ser un paso atrás.

Si falla cualquiera de las dos, el candidato se archiva con el motivo y el
modelo en producción se queda donde está. Un reentrenamiento que no promociona
es un resultado correcto, no un fallo del job.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Margen por el que un candidato puede ser peor que producción sin que se le
#: rechace. Un 2 % absorbe el ruido de evaluar sobre un holdout finito; sin
#: margen, el modelo cambiaría cada semana por diferencias que no significan
#: nada.
REGRESSION_TOLERANCE = 0.02


@dataclass(frozen=True)
class PromotionDecision:
    """Resultado de evaluar un candidato."""

    promoted: bool
    reason: str
    candidate_brier: float
    persistence_brier: float
    production_brier: float | None

    @property
    def beats_persistence(self) -> bool:
        return self.candidate_brier < self.persistence_brier

    def as_dict(self) -> dict:
        return asdict(self) | {"beats_persistence": self.beats_persistence}


def decide(
    candidate_brier: float,
    persistence_brier: float,
    production_brier: float | None,
    *,
    tolerance: float = REGRESSION_TOLERANCE,
) -> PromotionDecision:
    """Decide si un candidato merece sustituir al modelo en producción.

    `production_brier` es `None` la primera vez, cuando todavía no hay nada
    desplegado: entonces basta con batir a la persistencia.
    """
    if not candidate_brier < persistence_brier:
        return PromotionDecision(
            promoted=False,
            reason=(
                f"no bate a la persistencia (candidato {candidate_brier:.5f} "
                f">= persistencia {persistence_brier:.5f}); desplegarlo anadiria "
                "un pipeline de ML a cambio de nada"
            ),
            candidate_brier=candidate_brier,
            persistence_brier=persistence_brier,
            production_brier=production_brier,
        )

    if production_brier is not None:
        limit = production_brier * (1 + tolerance)
        if candidate_brier > limit:
            return PromotionDecision(
                promoted=False,
                reason=(
                    f"empeora respecto a produccion mas alla del margen "
                    f"(candidato {candidate_brier:.5f} > limite {limit:.5f})"
                ),
                candidate_brier=candidate_brier,
                persistence_brier=persistence_brier,
                production_brier=production_brier,
            )

    improvement = (
        "primer modelo desplegado"
        if production_brier is None
        else f"mejora un {(1 - candidate_brier / production_brier) * 100:.1f}% sobre produccion"
    )
    return PromotionDecision(
        promoted=True,
        reason=f"bate a la persistencia y {improvement}",
        candidate_brier=candidate_brier,
        persistence_brier=persistence_brier,
        production_brier=production_brier,
    )


class ModelRegistry:
    """Modelos versionados en disco, con un puntero a producción.

    Estructura::

        models/
          production/          <- lo que sirve la API
          versions/2026-10-22T18-30-00/
          history.jsonl        <- una linea por decision, promocionada o no

    El histórico guarda **también los rechazos**. Saber que el modelo lleva
    cinco semanas sin promocionar porque no bate a la persistencia es
    información operativa de primer orden, y borrarla dejaría el registro
    contando solo las victorias.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.production_dir = root / "production"
        self.versions_dir = root / "versions"
        self.history_path = root / "history.jsonl"

    def production_metrics(self) -> dict | None:
        """Métricas con las que se promocionó el modelo actual, si lo hay."""
        path = self.production_dir / "metrics.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def record(
        self, decision: PromotionDecision, *, version: str, extra: dict | None = None
    ) -> None:
        """Añade una línea al histórico, promocionara o no."""
        self.root.mkdir(parents=True, exist_ok=True)
        entry = {
            "recorded_at": dt.datetime.now(dt.UTC).isoformat(),
            "version": version,
            **decision.as_dict(),
            **(extra or {}),
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def history(self) -> list[dict]:
        """Todas las decisiones, de la más antigua a la más reciente."""
        if not self.history_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def promote(self, candidate_dir: Path, *, version: str, metrics: dict) -> Path:
        """Archiva el candidato y lo apunta como producción.

        El candidato se copia a `versions/<version>/` **antes** de tocar
        producción, para que un fallo a mitad no deje producción vacía.
        """
        archived = self.versions_dir / version
        if archived.exists():
            shutil.rmtree(archived)
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(candidate_dir, archived)
        (archived / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        staging = self.root / ".production.new"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(archived, staging)
        if self.production_dir.exists():
            shutil.rmtree(self.production_dir)
        staging.replace(self.production_dir)

        logger.info("promocionado %s a produccion", version)
        return self.production_dir
