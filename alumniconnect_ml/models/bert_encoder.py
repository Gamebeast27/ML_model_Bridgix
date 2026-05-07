"""
AlumniConnect ML — BERT Encoder
Generates dense semantic embeddings using sentence-transformers.
Falls back to TF-IDF if torch/sentence-transformers are not installed.
"""
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    ARTIFACTS_DIR,
    BERT_MODEL_NAME,
    BERT_BATCH_SIZE,
    BERT_MAX_SEQ_LENGTH,
)

logger = logging.getLogger(__name__)

BERT_EMBEDDINGS_ARTIFACT = ARTIFACTS_DIR / "bert_embeddings.pkl"
BERT_IDS_ARTIFACT = ARTIFACTS_DIR / "bert_user_ids.pkl"

# ── Check if sentence-transformers is available ───────────────────────────
_SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning(
        "sentence-transformers not installed. BERT encoder will be unavailable. "
        "Install with: pip install sentence-transformers"
    )


class BERTEncoder:
    """
    Encodes profile texts into dense 384-dim vectors using MiniLM.
    If sentence-transformers is not installed, all methods return None / empty
    and the hybrid recommender falls back to TF-IDF similarity.
    """

    def __init__(self):
        self.model = None
        self.embeddings: Optional[np.ndarray] = None  # (n_profiles, 384)
        self.user_ids: List[str] = []
        self._id_to_idx: dict = {}
        self._available = _SENTENCE_TRANSFORMERS_AVAILABLE

    @property
    def is_available(self) -> bool:
        return self._available

    # ── Load the underlying transformer model ──────────────────────────────
    def load_model(self) -> "BERTEncoder":
        """Load the sentence-transformer model into memory."""
        if not self._available:
            logger.warning("BERT unavailable — sentence-transformers not installed")
            return self
        if self.model is None:
            logger.info("Loading BERT model: %s ...", BERT_MODEL_NAME)
            self.model = SentenceTransformer(BERT_MODEL_NAME)
            self.model.max_seq_length = BERT_MAX_SEQ_LENGTH
            logger.info("BERT model loaded")
        return self

    # ── Encode ─────────────────────────────────────────────────────────────
    def fit(self, user_ids: List[str], texts: List[str]) -> "BERTEncoder":
        """
        Encode all profile texts into dense embeddings.

        Args:
            user_ids: List of user IDs.
            texts:    Preprocessed profile texts (same order).
        """
        if not self._available:
            logger.warning("BERT fit skipped — not available")
            return self

        self.load_model()
        logger.info("Encoding %d profiles with BERT ...", len(texts))
        self.embeddings = self.model.encode(
            texts,
            batch_size=BERT_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,  # unit vectors → dot product = cosine sim
        )
        self.user_ids = list(user_ids)
        self._id_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        logger.info(
            "BERT encoding done: shape=%s", self.embeddings.shape
        )
        return self

    def encode_single(self, text: str) -> Optional[np.ndarray]:
        """Encode a single text into a dense vector."""
        if not self._available or self.model is None:
            return None
        vec = self.model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0]

    # ── Similarity ─────────────────────────────────────────────────────────
    def similarity(self, user_a: str, user_b: str) -> Optional[float]:
        """Cosine similarity between two user embeddings. Returns None if unavailable."""
        if self.embeddings is None:
            return None
        idx_a = self._id_to_idx.get(user_a)
        idx_b = self._id_to_idx.get(user_b)
        if idx_a is None or idx_b is None:
            return None
        # Embeddings are L2-normalized, so dot product = cosine sim
        sim = float(np.dot(self.embeddings[idx_a], self.embeddings[idx_b]))
        return sim

    def query_similarity(
        self, query_text: str, top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Given a free-text query, find top_k most semantically similar profiles.
        """
        if self.embeddings is None or not self._available:
            return []
        q_vec = self.encode_single(query_text)
        if q_vec is None:
            return []
        # dot product with all embeddings (already normalized)
        scores = self.embeddings @ q_vec
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            (self.user_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0.0
        ]

    # ── Persistence ────────────────────────────────────────────────────────
    def save(self):
        if self.embeddings is None:
            logger.warning("No BERT embeddings to save")
            return
        with open(BERT_EMBEDDINGS_ARTIFACT, "wb") as f:
            pickle.dump(self.embeddings, f)
        with open(BERT_IDS_ARTIFACT, "wb") as f:
            pickle.dump(self.user_ids, f)
        logger.info("BERT embeddings saved")

    def load(self) -> "BERTEncoder":
        """Load pre-computed embeddings (model not needed for similarity lookups)."""
        try:
            with open(BERT_EMBEDDINGS_ARTIFACT, "rb") as f:
                self.embeddings = pickle.load(f)
            with open(BERT_IDS_ARTIFACT, "rb") as f:
                self.user_ids = pickle.load(f)
            self._id_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
            logger.info("BERT embeddings loaded: %d profiles", len(self.user_ids))
        except FileNotFoundError:
            logger.warning("No BERT embeddings found on disk")
        return self

    @property
    def is_fitted(self) -> bool:
        return self.embeddings is not None and len(self.user_ids) > 0
