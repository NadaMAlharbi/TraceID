"""
Tests for the ReID engine logic.
Run with:  python -m pytest tests/ -v
or:        python tests/test_reid_engine.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from core.reid_engine import ReIDEngine, Identity


def rand_emb(seed=None):
    rng = np.random.default_rng(seed)
    v = rng.random(512).astype(np.float32)
    return v / np.linalg.norm(v)


class TestReIDEngine:

    def test_new_identity_created_on_first_encounter(self):
        engine = ReIDEngine(similarity_threshold=0.70)
        emb = rand_emb(0)
        result = engine.process(emb, "person")
        assert result.is_new
        assert result.status == "New"
        assert len(engine.identities) == 1
        assert len(engine.alerts) == 1

    def test_same_embedding_recognized_as_known(self):
        engine = ReIDEngine(similarity_threshold=0.70)
        emb = rand_emb(42)
        engine.process(emb, "person")      # first → New
        result = engine.process(emb, "person")   # same → Seen before
        assert not result.is_new
        assert result.status == "Seen before"
        assert len(engine.identities) == 1    # no new identity
        assert len(engine.alerts) == 1        # no new alert

    def test_different_embeddings_create_different_identities(self):
        engine = ReIDEngine(similarity_threshold=0.70)
        # Construct two embeddings that are clearly dissimilar:
        # emb_a points mostly in the +first-half direction,
        # emb_b points mostly in the +second-half direction → low dot product.
        emb_a = np.zeros(512, dtype=np.float32)
        emb_a[:256] = 1.0
        emb_a /= np.linalg.norm(emb_a)

        emb_b = np.zeros(512, dtype=np.float32)
        emb_b[256:] = 1.0
        emb_b /= np.linalg.norm(emb_b)

        # Sanity: they should be orthogonal (cosine = 0)
        assert np.dot(emb_a, emb_b) < 0.01

        engine.process(emb_a, "person")
        engine.process(emb_b, "person")
        assert len(engine.identities) == 2
        assert len(engine.alerts) == 2

    def test_no_duplicate_alerts(self):
        engine = ReIDEngine(similarity_threshold=0.65)
        emb = rand_emb(7)
        # First time
        engine.process(emb, "person")
        # Repeated encounters
        for _ in range(5):
            engine.process(emb, "person")
        assert len(engine.alerts) == 1   # still only 1 alert

    def test_reset_clears_all(self):
        engine = ReIDEngine()
        engine.process(rand_emb(1), "person")
        engine.process(rand_emb(2), "person")
        engine.reset()
        assert len(engine.identities) == 0
        assert len(engine.alerts) == 0

    def test_class_label_isolation(self):
        """A person embedding should NOT match a car embedding."""
        engine = ReIDEngine(similarity_threshold=0.70)
        emb = rand_emb(10)
        engine.process(emb, "person")
        result = engine.process(emb, "car")   # same vector, different class
        assert result.is_new                  # treated as new identity
        assert len(engine.identities) == 2

    def test_times_seen_increments(self):
        engine = ReIDEngine(similarity_threshold=0.65)
        emb = rand_emb(3)
        engine.process(emb, "person")
        for _ in range(4):
            engine.process(emb, "person")
        ident = list(engine.identities.values())[0]
        assert ident.times_seen == 5

    def test_max_identities_eviction(self):
        engine = ReIDEngine(similarity_threshold=0.99, max_identities=3)
        for i in range(5):
            engine.process(rand_emb(i * 100), "person")
        assert len(engine.identities) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
