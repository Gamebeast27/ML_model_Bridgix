"""
AlumniConnect ML — Spam Detector
NLP pipeline for detecting spam in messages, job posts, and event descriptions.
Uses TF-IDF + hand-crafted features + Logistic Regression (configurable).
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

from config import (
    ARTIFACTS_DIR,
    SPAM_TFIDF_MAX_FEATURES,
    SPAM_TFIDF_NGRAM_RANGE,
    SPAM_THRESHOLD_BLOCK,
    SPAM_THRESHOLD_REVIEW,
    SPAM_MODEL_TYPE,
)
from services import extract_spam_features, clean_text

logger = logging.getLogger(__name__)

SPAM_VECTORIZER_PATH = ARTIFACTS_DIR / "spam_tfidf.pkl"
SPAM_MODEL_PATH = ARTIFACTS_DIR / "spam_model.pkl"

# ── Feature names for the hand-crafted augmentation ───────────────────────
AUGMENTED_FEATURE_NAMES = [
    "url_count", "email_count", "exclamation_count",
    "caps_ratio", "digit_ratio", "word_count",
    "avg_word_len", "urgency_score",
]


class SpamDetector:
    """
    Two-stage spam classifier:
    1. TF-IDF on cleaned text → sparse vector
    2. Append hand-crafted features (URL count, caps ratio, urgency, etc.)
    3. Feed combined vector into a trained classifier
    4. Output: probability + label (safe / review / blocked)
    """

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.model = None

    # ── Build feature matrix ───────────────────────────────────────────────
    def _build_features(self, texts: List[str]) -> csr_matrix:
        """Combine TF-IDF vectors with hand-crafted features."""
        tfidf_matrix = self.vectorizer.transform([clean_text(t) for t in texts])

        # Build augmented feature matrix
        aug_rows = []
        for text in texts:
            feats = extract_spam_features(text)
            aug_rows.append([feats[k] for k in AUGMENTED_FEATURE_NAMES])
        aug_matrix = csr_matrix(np.array(aug_rows, dtype=np.float64))

        return hstack([tfidf_matrix, aug_matrix])

    # ── Train ──────────────────────────────────────────────────────────────
    def train(
        self,
        texts: List[str],
        labels: List[int],
        cv_folds: int = 5,
    ) -> Dict:
        """
        Train the spam classifier.

        Args:
            texts:  Raw text content (messages, posts, descriptions).
            labels: 0 = legitimate, 1 = spam.
            cv_folds: Number of cross-validation folds.

        Returns:
            Dict with training metrics.
        """
        logger.info("Training spam detector on %d samples ...", len(texts))

        # Fit TF-IDF
        cleaned = [clean_text(t) for t in texts]
        self.vectorizer = TfidfVectorizer(
            max_features=SPAM_TFIDF_MAX_FEATURES,
            ngram_range=SPAM_TFIDF_NGRAM_RANGE,
            sublinear_tf=True,
            stop_words="english",
        )
        self.vectorizer.fit(cleaned)

        # Build combined features
        X = self._build_features(texts)
        y = np.array(labels)

        # Select model
        if SPAM_MODEL_TYPE == "naive_bayes":
            # MultinomialNB needs non-negative features
            # TF-IDF is non-negative but augmented features could be; clip them
            base_model = MultinomialNB(alpha=1.0)
            # Wrap for probability calibration
            self.model = CalibratedClassifierCV(base_model, cv=3)
        elif SPAM_MODEL_TYPE == "svm":
            base_model = LinearSVC(C=1.0, max_iter=5000)
            self.model = CalibratedClassifierCV(base_model, cv=3)
        else:
            # Default: Logistic Regression
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",  # handle class imbalance
            )

        # Cross-validation
        skf = StratifiedKFold(n_splits=min(cv_folds, len(y)), shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X, y, cv=skf, scoring="f1")
        logger.info("CV F1 scores: %s | Mean: %.4f", cv_scores, cv_scores.mean())

        # Final fit on all data
        self.model.fit(X, y)

        # Full-data evaluation
        y_pred = self.model.predict(X)
        report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)

        metrics = {
            "cv_f1_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_std": round(float(cv_scores.std()), 4),
            "train_f1": round(report["1"]["f1-score"], 4) if "1" in report else None,
            "train_precision": round(report["1"]["precision"], 4) if "1" in report else None,
            "train_recall": round(report["1"]["recall"], 4) if "1" in report else None,
            "confusion_matrix": cm.tolist(),
            "model_type": SPAM_MODEL_TYPE,
            "n_samples": len(texts),
        }
        logger.info("Spam detector trained: %s", metrics)
        return metrics

    # ── Predict ────────────────────────────────────────────────────────────
    def predict(self, text: str) -> Dict:
        """
        Classify a single text.

        Returns:
            {
                "label": "safe" | "review" | "blocked",
                "spam_probability": float,
                "features": dict  (hand-crafted feature values)
            }
        """
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("SpamDetector not trained. Call train() first.")

        X = self._build_features([text])
        prob = self.model.predict_proba(X)[0, 1]  # probability of class 1 (spam)
        feats = extract_spam_features(text)

        if prob >= SPAM_THRESHOLD_BLOCK:
            label = "blocked"
        elif prob >= SPAM_THRESHOLD_REVIEW:
            label = "review"
        else:
            label = "safe"

        return {
            "label": label,
            "spam_probability": round(float(prob), 4),
            "features": feats,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Classify a batch of texts."""
        return [self.predict(t) for t in texts]

    # ── Persistence ────────────────────────────────────────────────────────
    def save(self):
        with open(SPAM_VECTORIZER_PATH, "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(SPAM_MODEL_PATH, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Spam model saved")

    def load(self) -> "SpamDetector":
        with open(SPAM_VECTORIZER_PATH, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(SPAM_MODEL_PATH, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Spam model loaded")
        return self

    @property
    def is_trained(self) -> bool:
        return self.vectorizer is not None and self.model is not None
