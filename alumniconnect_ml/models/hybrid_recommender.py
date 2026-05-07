"""
AlumniConnect ML — Hybrid Recommender
Combines collaborative filtering, semantic similarity (BERT/TF-IDF),
graph signals, and fairness-aware re-ranking.

Scoring formula:
    S'(u,v) = α·CF(u,v) + β·Sim(u,v) + γ·PR(v) − λ·Bias(v)
"""
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    ALPHA_CF,
    BETA_SEMANTIC,
    GAMMA_GRAPH,
    LAMBDA_FAIRNESS,
    CF_MIN_INTERACTIONS,
    CF_SIMILARITY_NEIGHBORS,
    DEFAULT_TOP_K,
)
from models.tfidf_encoder import TFIDFEncoder
from models.bert_encoder import BERTEncoder

logger = logging.getLogger(__name__)


class CollaborativeFilter:
    """
    User-based collaborative filtering on implicit feedback.
    Interaction matrix: rows = students, cols = alumni.
    Values = weighted interaction signals.
    """

    # Interaction weights
    SIGNAL_WEIGHTS = {
        "profile_view": 1.0,
        "mentorship_request": 3.0,
        "mentorship_accept": 5.0,
        "message_sent": 2.0,
        "event_coattend": 1.5,
        "bookmark": 2.0,
    }

    def __init__(self):
        self.student_ids: List[str] = []
        self.alumni_ids: List[str] = []
        self.matrix: Optional[np.ndarray] = None   # (n_students, n_alumni)
        self._student_idx: Dict[str, int] = {}
        self._alumni_idx: Dict[str, int] = {}

    def build_matrix(
        self,
        student_ids: List[str],
        alumni_ids: List[str],
        interactions: List[Dict],
    ) -> "CollaborativeFilter":
        """
        Build the interaction matrix from raw interaction logs.

        Each interaction dict: {"student_id", "alumni_id", "type"}
        """
        self.student_ids = list(student_ids)
        self.alumni_ids = list(alumni_ids)
        self._student_idx = {sid: i for i, sid in enumerate(self.student_ids)}
        self._alumni_idx = {aid: i for i, aid in enumerate(self.alumni_ids)}

        n_s, n_a = len(self.student_ids), len(self.alumni_ids)
        self.matrix = np.zeros((n_s, n_a), dtype=np.float32)

        for ix in interactions:
            s_idx = self._student_idx.get(ix.get("student_id"))
            a_idx = self._alumni_idx.get(ix.get("alumni_id"))
            if s_idx is not None and a_idx is not None:
                weight = self.SIGNAL_WEIGHTS.get(ix.get("type", ""), 1.0)
                self.matrix[s_idx, a_idx] += weight

        logger.info(
            "CF matrix built: %d students × %d alumni, %d non-zero",
            n_s, n_a, int(np.count_nonzero(self.matrix)),
        )
        return self

    def score(self, student_id: str, alumni_id: str) -> float:
        """
        CF score for a (student, alumni) pair.
        Uses user-based kNN: find k students most similar to this student,
        then aggregate their interaction scores for this alumni.
        """
        if self.matrix is None:
            return 0.0

        s_idx = self._student_idx.get(student_id)
        a_idx = self._alumni_idx.get(alumni_id)
        if s_idx is None or a_idx is None:
            return 0.0

        # Check minimum interactions
        if self.matrix[s_idx].sum() < CF_MIN_INTERACTIONS:
            return 0.0

        # Cosine similarity between this student and all others
        student_vec = self.matrix[s_idx].reshape(1, -1)
        all_sims = cosine_similarity(student_vec, self.matrix).flatten()
        all_sims[s_idx] = -1.0  # exclude self

        # Top-k neighbors
        k = min(CF_SIMILARITY_NEIGHBORS, len(self.student_ids) - 1)
        neighbor_indices = all_sims.argsort()[::-1][:k]

        # Weighted average of neighbors' scores for this alumni
        numerator = 0.0
        denominator = 0.0
        for n_idx in neighbor_indices:
            sim = all_sims[n_idx]
            if sim <= 0:
                continue
            numerator += sim * self.matrix[n_idx, a_idx]
            denominator += abs(sim)

        if denominator == 0:
            return 0.0
        return float(numerator / denominator)


class HybridRecommender:
    """
    Orchestrates all scoring signals and produces final ranked recommendations.

    S'(u,v) = α·CF(u,v) + β·Sim(u,v) + γ·PR(v) − λ·Bias(v)
    """

    def __init__(
        self,
        tfidf_encoder: TFIDFEncoder,
        bert_encoder: BERTEncoder,
    ):
        self.tfidf = tfidf_encoder
        self.bert = bert_encoder
        self.cf = CollaborativeFilter()

        # Graph signals: PageRank / centrality scores per user
        self._pagerank: Dict[str, float] = {}

        # Fairness: recommendation frequency counter (prevents majority-cohort bias)
        self._rec_counts: Dict[str, int] = defaultdict(int)

    # ── Setup Methods ──────────────────────────────────────────────────────
    def set_cf_data(
        self,
        student_ids: List[str],
        alumni_ids: List[str],
        interactions: List[Dict],
    ):
        """Load interaction data for collaborative filtering."""
        self.cf.build_matrix(student_ids, alumni_ids, interactions)

    def set_pagerank(self, scores: Dict[str, float]):
        """
        Load pre-computed PageRank or centrality scores.
        In production, these come from Neo4j.
        scores: {user_id: pagerank_value}
        """
        self._pagerank = dict(scores)
        logger.info("PageRank scores loaded for %d users", len(scores))

    # ── Core Scoring ───────────────────────────────────────────────────────
    def _semantic_score(self, student_id: str, alumni_id: str) -> float:
        """
        Semantic similarity: prefer BERT if available, fall back to TF-IDF.
        """
        # Try BERT first
        if self.bert.is_fitted:
            bert_sim = self.bert.similarity(student_id, alumni_id)
            if bert_sim is not None:
                return bert_sim

        # Fallback to TF-IDF
        if self.tfidf.is_fitted:
            return self.tfidf.similarity(student_id, alumni_id)

        return 0.0

    def _graph_score(self, alumni_id: str) -> float:
        """PageRank centrality score for the alumni (normalized 0-1)."""
        return self._pagerank.get(alumni_id, 0.0)

    def _bias_penalty(self, alumni_id: str) -> float:
        """
        Fairness penalty: alumni who have been recommended many times
        get a small penalty to give less-visible mentors a chance.
        Uses log-dampened frequency.
        """
        count = self._rec_counts.get(alumni_id, 0)
        if count == 0:
            return 0.0
        return float(np.log1p(count) / 10.0)  # gentle penalty

    def score_pair(self, student_id: str, alumni_id: str) -> Dict:
        """
        Compute the full hybrid score for a (student, alumni) pair.
        Returns breakdown of all signal components.
        """
        cf_score = self.cf.score(student_id, alumni_id)
        sem_score = self._semantic_score(student_id, alumni_id)
        graph_score = self._graph_score(alumni_id)
        bias = self._bias_penalty(alumni_id)

        final = (
            ALPHA_CF * cf_score
            + BETA_SEMANTIC * sem_score
            + GAMMA_GRAPH * graph_score
            - LAMBDA_FAIRNESS * bias
        )

        return {
            "final_score": round(float(final), 4),
            "cf_score": round(cf_score, 4),
            "semantic_score": round(sem_score, 4),
            "graph_score": round(graph_score, 4),
            "bias_penalty": round(bias, 4),
            "weights": {
                "alpha_cf": ALPHA_CF,
                "beta_semantic": BETA_SEMANTIC,
                "gamma_graph": GAMMA_GRAPH,
                "lambda_fairness": LAMBDA_FAIRNESS,
            },
        }

    # ── Recommendation ─────────────────────────────────────────────────────
    def recommend(
        self,
        student_id: str,
        candidate_alumni_ids: List[str],
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict]:
        """
        Rank candidate alumni for a student and return top_k recommendations.

        Args:
            student_id:          The querying student.
            candidate_alumni_ids: Pool of alumni to rank.
            top_k:               How many to return.

        Returns:
            Sorted list of dicts with alumni_id, score breakdown.
        """
        results = []
        for aid in candidate_alumni_ids:
            breakdown = self.score_pair(student_id, aid)
            breakdown["alumni_id"] = aid
            results.append(breakdown)

        # Sort descending by final score
        results.sort(key=lambda x: x["final_score"], reverse=True)
        top = results[:top_k]

        # Update recommendation counts for fairness tracking
        for r in top:
            self._rec_counts[r["alumni_id"]] += 1

        return top

    def recommend_by_text(
        self,
        query_text: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> List[Dict]:
        """
        Cold-start recommendation: when we only have a free-text description
        (new user, search query), find similar profiles using semantic search.
        Tries BERT first, falls back to TF-IDF.
        """
        # BERT path
        if self.bert.is_fitted:
            bert_results = self.bert.query_similarity(query_text, top_k)
            if bert_results:
                return [
                    {"alumni_id": uid, "semantic_score": score, "method": "bert"}
                    for uid, score in bert_results
                ]

        # TF-IDF fallback
        if self.tfidf.is_fitted:
            tfidf_results = self.tfidf.query_similarity(query_text, top_k)
            return [
                {"alumni_id": uid, "semantic_score": score, "method": "tfidf"}
                for uid, score in tfidf_results
            ]

        return []
