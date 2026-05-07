"""
AlumniConnect ML — TF-IDF Encoder
Vectorizes user profiles and computes cosine similarity for content-based filtering.
"""
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    ARTIFACTS_DIR,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_SUBLINEAR_TF,
)

logger = logging.getLogger(__name__)

TFIDF_ARTIFACT = ARTIFACTS_DIR / "tfidf_vectorizer.pkl"
TFIDF_MATRIX_ARTIFACT = ARTIFACTS_DIR / "tfidf_matrix.pkl"
TFIDF_IDS_ARTIFACT = ARTIFACTS_DIR / "tfidf_user_ids.pkl"


class TFIDFEncoder:
    """
    Fits a TF-IDF vectorizer on profile text corpus.
    Provides similarity lookup between any two profiles.
    """

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None          # sparse CSR matrix of shape (n_profiles, n_features)
        self.user_ids: List[str] = []
        self._id_to_idx: dict = {}

    # ── Fit ────────────────────────────────────────────────────────────────
    def fit(self, user_ids: List[str], texts: List[str]) -> "TFIDFEncoder":
        """
        Fit TF-IDF on a corpus of profile texts.

        Args:
            user_ids: List of user IDs (same order as texts).
            texts:    List of preprocessed profile texts.

        Returns:
            self (for chaining).
        """
        if len(user_ids) != len(texts):
            raise ValueError("user_ids and texts must have same length")

        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            sublinear_tf=TFIDF_SUBLINEAR_TF,
            stop_words="english",
        )
        self.matrix = self.vectorizer.fit_transform(texts)
        self.user_ids = list(user_ids)
        self._id_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        logger.info(
            "TF-IDF fitted: %d profiles, %d features",
            len(self.user_ids),
            self.matrix.shape[1],
        )
        return self

    # ── Transform a single text ────────────────────────────────────────────
    def transform(self, text: str) -> np.ndarray:
        """Transform a single text into a TF-IDF vector."""
        if self.vectorizer is None:
            raise RuntimeError("TFIDFEncoder not fitted. Call fit() first.")
        return self.vectorizer.transform([text])

    # ── Similarity ─────────────────────────────────────────────────────────
    def similarity(self, user_a: str, user_b: str) -> float:
        """Cosine similarity between two known user profiles."""
        idx_a = self._id_to_idx.get(user_a)
        idx_b = self._id_to_idx.get(user_b)
        if idx_a is None or idx_b is None:
            return 0.0
        sim = cosine_similarity(
            self.matrix[idx_a], self.matrix[idx_b]
        )[0, 0]
        return float(sim)

    def query_similarity(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Given a free-text query, find the top_k most similar profiles.

        Returns:
            List of (user_id, similarity_score) sorted descending.
        """
        if self.vectorizer is None or self.matrix is None:
            return []

        q_vec = self.transform(query_text)
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            (self.user_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0.0
        ]

    # ── Persistence ────────────────────────────────────────────────────────
    def save(self):
        """Persist fitted vectorizer, matrix, and ID mapping."""
        with open(TFIDF_ARTIFACT, "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(TFIDF_MATRIX_ARTIFACT, "wb") as f:
            pickle.dump(self.matrix, f)
        with open(TFIDF_IDS_ARTIFACT, "wb") as f:
            pickle.dump(self.user_ids, f)
        logger.info("TF-IDF artifacts saved to %s", ARTIFACTS_DIR)

    def load(self) -> "TFIDFEncoder":
        """Load previously fitted artifacts."""
        with open(TFIDF_ARTIFACT, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(TFIDF_MATRIX_ARTIFACT, "rb") as f:
            self.matrix = pickle.load(f)
        with open(TFIDF_IDS_ARTIFACT, "rb") as f:
            self.user_ids = pickle.load(f)
        self._id_to_idx = {uid: i for i, uid in enumerate(self.user_ids)}
        logger.info("TF-IDF artifacts loaded: %d profiles", len(self.user_ids))
        return self

    @property
    def is_fitted(self) -> bool:
        return self.vectorizer is not None and self.matrix is not None
