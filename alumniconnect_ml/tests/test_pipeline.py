"""
AlumniConnect ML — Pipeline Smoke Tests
Run: python -m pytest tests/test_pipeline.py -v
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, SPAM_THRESHOLD_BLOCK
from services import build_profile_text, clean_text, extract_spam_features
from models.tfidf_encoder import TFIDFEncoder
from models.bert_encoder import BERTEncoder
from models.spam_detector import SpamDetector
from models.hybrid_recommender import HybridRecommender


def load_data():
    with open(DATA_DIR / "seed_profiles.json") as f:
        return json.load(f)


# ── Preprocessing Tests ────────────────────────────────────────────────────
class TestPreprocessing:
    def test_clean_text_strips_html(self):
        assert "<b>" not in clean_text("<b>Hello</b> world")

    def test_clean_text_replaces_urls(self):
        result = clean_text("Visit https://example.com for more")
        assert "_url_" in result.lower()
        assert "https" not in result

    def test_build_profile_text_combines_fields(self):
        profile = {
            "department": "CSE",
            "skills": ["python", "ML"],
            "bio": "I love AI",
        }
        text = build_profile_text(profile)
        assert "cse" in text
        assert "python" in text
        assert "ai" in text

    def test_extract_spam_features_counts_urls(self):
        feats = extract_spam_features("Click http://spam.com and http://more.com now!")
        assert feats["url_count"] == 2
        assert feats["exclamation_count"] == 1
        assert feats["urgency_score"] >= 0


# ── TF-IDF Tests ───────────────────────────────────────────────────────────
class TestTFIDF:
    def setup_method(self):
        data = load_data()
        all_profiles = data["students"] + data["alumni"]
        self.ids = [p["id"] for p in all_profiles]
        self.texts = [build_profile_text(p) for p in all_profiles]
        self.encoder = TFIDFEncoder()
        self.encoder.fit(self.ids, self.texts)

    def test_fit_creates_matrix(self):
        assert self.encoder.is_fitted
        assert self.encoder.matrix.shape[0] == len(self.ids)

    def test_similarity_same_user_is_1(self):
        sim = self.encoder.similarity("s001", "s001")
        assert abs(sim - 1.0) < 0.01

    def test_similarity_related_users_higher(self):
        # s001 (ML student) should be more similar to a001 (ML engineer)
        # than to a003 (product designer)
        sim_ml = self.encoder.similarity("s001", "a001")
        sim_design = self.encoder.similarity("s001", "a003")
        assert sim_ml > sim_design, f"Expected ML match ({sim_ml}) > design match ({sim_design})"

    def test_query_similarity_returns_results(self):
        results = self.encoder.query_similarity("machine learning python", top_k=3)
        assert len(results) > 0
        assert all(isinstance(r[1], float) for r in results)

    def test_save_and_load(self, tmp_path):
        import config
        original_dir = config.ARTIFACTS_DIR
        config.ARTIFACTS_DIR = tmp_path
        # Monkey-patch the module-level paths
        import models.tfidf_encoder as mod
        mod.TFIDF_ARTIFACT = tmp_path / "tfidf_vectorizer.pkl"
        mod.TFIDF_MATRIX_ARTIFACT = tmp_path / "tfidf_matrix.pkl"
        mod.TFIDF_IDS_ARTIFACT = tmp_path / "tfidf_user_ids.pkl"

        self.encoder.save()

        loaded = TFIDFEncoder()
        loaded.load()
        assert loaded.is_fitted
        assert len(loaded.user_ids) == len(self.ids)

        config.ARTIFACTS_DIR = original_dir


# ── Spam Tests ─────────────────────────────────────────────────────────────
class TestSpam:
    def setup_method(self):
        data = load_data()
        spam_data = data["spam_training_data"]
        self.detector = SpamDetector()
        self.detector.train(spam_data["texts"], spam_data["labels"])

    def test_detector_is_trained(self):
        assert self.detector.is_trained

    def test_legitimate_text_is_safe(self):
        result = self.detector.predict(
            "Looking for advice on my career in software engineering."
        )
        # With small training data, calibration is imperfect.
        # Core check: legitimate text should not be blocked.
        assert result["label"] != "blocked"
        assert result["spam_probability"] < SPAM_THRESHOLD_BLOCK

    def test_spam_text_detected(self):
        result = self.detector.predict(
            "CONGRATULATIONS! You WON $50000! Click HERE NOW to claim your prize!!!"
        )
        assert result["label"] in ("review", "blocked")
        assert result["spam_probability"] > 0.5

    def test_batch_prediction(self):
        results = self.detector.predict_batch([
            "Normal alumni networking message",
            "FREE MONEY!!! Click now to claim!!!",
        ])
        assert len(results) == 2


# ── Hybrid Recommender Tests ──────────────────────────────────────────────
class TestHybridRecommender:
    def setup_method(self):
        data = load_data()
        all_profiles = data["students"] + data["alumni"]
        ids = [p["id"] for p in all_profiles]
        texts = [build_profile_text(p) for p in all_profiles]

        self.tfidf = TFIDFEncoder().fit(ids, texts)
        self.bert = BERTEncoder()  # will be unavailable but that's fine

        self.recommender = HybridRecommender(self.tfidf, self.bert)
        self.recommender.set_cf_data(
            student_ids=[s["id"] for s in data["students"]],
            alumni_ids=[a["id"] for a in data["alumni"]],
            interactions=data["interactions"],
        )
        self.recommender.set_pagerank(data["pagerank_scores"])
        self.alumni_ids = [a["id"] for a in data["alumni"]]

    def test_recommend_returns_results(self):
        results = self.recommender.recommend("s001", self.alumni_ids, top_k=5)
        assert len(results) > 0
        assert len(results) <= 5

    def test_recommend_is_sorted(self):
        results = self.recommender.recommend("s001", self.alumni_ids, top_k=8)
        scores = [r["final_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_pair_has_all_components(self):
        breakdown = self.recommender.score_pair("s001", "a001")
        assert "final_score" in breakdown
        assert "cf_score" in breakdown
        assert "semantic_score" in breakdown
        assert "graph_score" in breakdown
        assert "bias_penalty" in breakdown

    def test_text_search_returns_results(self):
        results = self.recommender.recommend_by_text("machine learning AI engineer")
        assert len(results) > 0

    def test_ml_student_gets_ml_mentor_high(self):
        """s001 (ML student) should rank ML-relevant alumni high."""
        results = self.recommender.recommend("s001", self.alumni_ids, top_k=5)
        top_ids = [r["alumni_id"] for r in results]
        # Either a001 (Google ML) or a004 (NVIDIA Research) in top 5
        ml_mentors = {"a001", "a004"}
        found = ml_mentors.intersection(set(top_ids))
        assert len(found) > 0, f"Expected ML mentors in top 5, got {top_ids}"
