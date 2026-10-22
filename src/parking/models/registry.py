"""Model registry: versions, promotion and rejection.

Automated retraining carries a risk that manual deployment does not: nobody
looks at the result before it reaches production. If a month of data turns out
bad - a long feed outage, roadworks emptying half the city - the retrained
model can be worse than the one already deployed, and worse than not predicting
at all.

That is why a model is **never promoted merely for having been trained**. It
has to pass two tests on a temporal holdout:

1. **Beat persistence.** This follows directly from phase 5: calibrated
   persistence is brutally strong, and a model that cannot beat it adds nothing
   that was not already in the current state. Deploying it would add an entire
   ML pipeline in exchange for nothing.
2. **Not regress against the model in production**, beyond a tolerance margin.
   Retraining must not be a step backwards.

If either test fails, the candidate is archived with the reason and the
production model stays put. A retraining run that does not promote is a correct
outcome, not a job failure.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: How much worse than production a candidate may be without being rejected. 2%
#: absorbs the noise of evaluating on a finite holdout; without a margin the
#: model would change every week over differences that mean nothing.
REGRESSION_TOLERANCE = 0.02


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of evaluating a candidate."""

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
    """Decide whether a candidate deserves to replace the production model.

    `production_brier` is `None` the first time, when nothing is deployed yet:
    then beating persistence is enough.
    """
    if not candidate_brier < persistence_brier:
        return PromotionDecision(
            promoted=False,
            reason=(
                f"does not beat persistence (candidate {candidate_brier:.5f} "
                f">= persistence {persistence_brier:.5f}); deploying it would add "
                "an ML pipeline in exchange for nothing"
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
                    f"regresses against production beyond the margin "
                    f"(candidate {candidate_brier:.5f} > limit {limit:.5f})"
                ),
                candidate_brier=candidate_brier,
                persistence_brier=persistence_brier,
                production_brier=production_brier,
            )

    improvement = (
        "first model deployed"
        if production_brier is None
        else f"improves on production by {(1 - candidate_brier / production_brier) * 100:.1f}%"
    )
    return PromotionDecision(
        promoted=True,
        reason=f"beats persistence and {improvement}",
        candidate_brier=candidate_brier,
        persistence_brier=persistence_brier,
        production_brier=production_brier,
    )


class ModelRegistry:
    """Versioned models on disk, with a pointer to production.

    Layout::

        models/
          production/          <- what the API serves
          versions/2026-10-22T18-30-00/
          history.jsonl        <- one line per decision, promoted or not

    The history keeps **rejections too**. Knowing the model has gone five weeks
    without promoting because it cannot beat persistence is first-order
    operational information, and discarding it would leave the registry
    recording only the wins.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.production_dir = root / "production"
        self.versions_dir = root / "versions"
        self.history_path = root / "history.jsonl"

    def production_metrics(self) -> dict | None:
        """Metrics the current production model was promoted with, if any."""
        path = self.production_dir / "metrics.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def record(
        self, decision: PromotionDecision, *, version: str, extra: dict | None = None
    ) -> None:
        """Append one line to the history, whether it promoted or not."""
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
        """Every decision, oldest first."""
        if not self.history_path.exists():
            return []
        return [
            json.loads(line)
            for line in self.history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def promote(self, candidate_dir: Path, *, version: str, metrics: dict) -> Path:
        """Archive the candidate and point production at it.

        The candidate is copied into `versions/<version>/` **before** production
        is touched, so a failure halfway through does not leave production
        empty.
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

        logger.info("promoted %s to production", version)
        return self.production_dir
