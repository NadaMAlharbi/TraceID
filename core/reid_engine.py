"""
ReID Engine - Long-term identity matching using appearance embeddings.

Why ReID is necessary:
  - YOLO alone: detects objects each frame independently, no identity continuity.
  - Tracking alone (e.g. BoT-SORT): gives short-term IDs that reset when a
    person leaves and re-enters the frame (e.g. exits elevator, comes back).
  - ReID: extracts a compact appearance descriptor that is stable across
    re-appearances. We compare new detections against stored embeddings
    using cosine similarity, enabling long-term identity persistence.

Duplicate alert prevention:
  - Each confirmed identity has an `alerted` flag.
  - The first time an identity is created, `alerted` is set to True and an
    alert is fired.
  - On subsequent matches the flag is already True so no alert is emitted.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Identity:
    global_id: str
    label: str                          # YOLO class name
    embedding: np.ndarray               # unit-norm appearance vector
    first_seen: float = field(default_factory=time.time)
    last_seen: float  = field(default_factory=time.time)
    times_seen: int   = 1
    alerted: bool     = True            # alert fired on creation
    snapshot: Optional[np.ndarray] = None   # BGR crop thumbnail

    def update(self, embedding: np.ndarray, snapshot=None):
        # Exponential moving average keeps embedding fresh
        alpha = 0.3
        merged = (1 - alpha) * self.embedding + alpha * embedding
        norm = np.linalg.norm(merged)
        self.embedding = merged / (norm + 1e-8)
        self.last_seen = time.time()
        self.times_seen += 1
        if snapshot is not None:
            self.snapshot = snapshot


@dataclass
class MatchResult:
    matched: bool
    identity: Optional[Identity]
    similarity: float
    is_new: bool = False        # True → new alert should fire
    status: str  = "Unknown"    # "New" | "Seen before"


# ---------------------------------------------------------------------------
# ReID Engine
# ---------------------------------------------------------------------------

class ReIDEngine:
    """
    Manages a database of known identities and matches new detections.

    Parameters
    ----------
    similarity_threshold : float
        Minimum cosine similarity to consider a match (0–1).
        Lower = stricter (fewer false matches), higher = looser.
    max_identities : int
        Cap on stored identities to keep memory bounded.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.65,
        max_identities: int = 200,
    ):
        self.threshold = similarity_threshold
        self.max_identities = max_identities
        self.identities: dict[str, Identity] = {}   # global_id → Identity
        self.alerts: list[dict] = []                # alert history

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        embedding: np.ndarray,
        label: str,
        snapshot=None,
    ) -> MatchResult:
        """
        Match *embedding* against the known identity database.

        Returns a MatchResult describing whether this is a new or known
        identity, and fires an alert (appended to self.alerts) only on
        first encounter.
        """
        embedding = self._normalize(embedding)

        best_id, best_sim = self._find_best_match(embedding, label)

        if best_id is not None:
            # --- Known identity ---
            ident = self.identities[best_id]
            ident.update(embedding, snapshot)
            return MatchResult(
                matched=True,
                identity=ident,
                similarity=best_sim,
                is_new=False,
                status="Seen before",
            )
        else:
            # --- New identity ---
            ident = self._create_identity(label, embedding, snapshot)
            self._fire_alert(ident)
            return MatchResult(
                matched=False,
                identity=ident,
                similarity=0.0,
                is_new=True,
                status="New",
            )

    def reset(self):
        self.identities.clear()
        self.alerts.clear()

    def get_identities(self) -> list[Identity]:
        return sorted(
            self.identities.values(),
            key=lambda i: i.last_seen,
            reverse=True,
        )

    def get_alerts(self) -> list[dict]:
        return list(reversed(self.alerts))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-8)

    def _find_best_match(
        self, embedding: np.ndarray, label: str
    ) -> tuple[Optional[str], float]:
        best_id, best_sim = None, -1.0
        for gid, ident in self.identities.items():
            if ident.label != label:
                continue
            sim = float(np.dot(embedding, ident.embedding))
            if sim > best_sim:
                best_sim = sim
                best_id = gid
        if best_sim >= self.threshold:
            return best_id, best_sim
        return None, best_sim

    def _create_identity(
        self, label: str, embedding: np.ndarray, snapshot
    ) -> Identity:
        if len(self.identities) >= self.max_identities:
            # Evict the oldest identity
            oldest = min(self.identities, key=lambda k: self.identities[k].last_seen)
            del self.identities[oldest]

        gid = f"ID-{str(uuid.uuid4())[:8].upper()}"
        ident = Identity(
            global_id=gid,
            label=label,
            embedding=embedding,
            snapshot=snapshot,
        )
        self.identities[gid] = ident
        return ident

    def _fire_alert(self, ident: Identity):
        self.alerts.append({
            "time": time.time(),
            "global_id": ident.global_id,
            "label": ident.label,
            "message": f"New {ident.label} detected → {ident.global_id}",
        })
