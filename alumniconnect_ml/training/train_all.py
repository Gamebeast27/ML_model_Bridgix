"""
AlumniConnect ML — Training Pipeline
Fits TF-IDF, BERT, spam detector, and collaborative filtering from data.
Can run against seed data (for testing) or a MongoDB JSON export (production).
"""
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, ARTIFACTS_DIR
from services import build_profile_text
from models.tfidf_encoder import TFIDFEncoder
from models.bert_encoder import BERTEncoder
from models.spam_detector import SpamDetector
from models.hybrid_recommender import CollaborativeFilter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-5s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_seed_data() -> dict:
    """Load the built-in seed dataset."""
    path = DATA_DIR / "seed_profiles.json"
    with open(path) as f:
        return json.load(f)


def train_tfidf(data: dict) -> TFIDFEncoder:
    """Fit TF-IDF on all profiles (students + alumni)."""
    logger.info("=" * 60)
    logger.info("TRAINING: TF-IDF Encoder")
    logger.info("=" * 60)

    all_profiles = data["students"] + data["alumni"]
    user_ids = [p["id"] for p in all_profiles]
    texts = [build_profile_text(p) for p in all_profiles]

    encoder = TFIDFEncoder()
    encoder.fit(user_ids, texts)
    encoder.save()

    # Quick sanity check
    logger.info("Sanity check — similarity(s001, a001): %.4f",
                encoder.similarity("s001", "a001"))
    logger.info("Sanity check — similarity(s003, a003): %.4f",
                encoder.similarity("s003", "a003"))

    return encoder


def train_bert(data: dict) -> BERTEncoder:
    """Fit BERT embeddings on all profiles."""
    logger.info("=" * 60)
    logger.info("TRAINING: BERT Encoder")
    logger.info("=" * 60)

    encoder = BERTEncoder()
    if not encoder.is_available:
        logger.warning("BERT skipped — sentence-transformers not installed")
        return encoder

    all_profiles = data["students"] + data["alumni"]
    user_ids = [p["id"] for p in all_profiles]
    texts = [build_profile_text(p) for p in all_profiles]

    encoder.fit(user_ids, texts)
    encoder.save()

    # Sanity check
    sim = encoder.similarity("s001", "a001")
    logger.info("Sanity check — BERT similarity(s001, a001): %s", sim)

    return encoder


def train_spam(data: dict) -> SpamDetector:
    """Train spam detection model."""
    logger.info("=" * 60)
    logger.info("TRAINING: Spam Detector")
    logger.info("=" * 60)

    spam_data = data["spam_training_data"]
    texts = spam_data["texts"]
    labels = spam_data["labels"]

    detector = SpamDetector()
    metrics = detector.train(texts, labels)
    detector.save()

    logger.info("Spam metrics: %s", json.dumps(metrics, indent=2))

    # Quick inference test
    test_cases = [
        "Hi, I'd love to connect and discuss career paths in ML.",
        "CONGRATULATIONS! You won $10000! Click NOW to claim!!!",
        "Our department is hosting a workshop on cloud computing next week.",
    ]
    for text in test_cases:
        result = detector.predict(text)
        logger.info("  '%s...' → %s (p=%.3f)",
                     text[:50], result["label"], result["spam_probability"])

    return detector


def train_all():
    """Run the full training pipeline."""
    logger.info("=" * 60)
    logger.info("AlumniConnect ML — Full Training Pipeline")
    logger.info("=" * 60)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    data = load_seed_data()

    logger.info("Loaded seed data: %d students, %d alumni, %d interactions, %d spam samples",
                len(data["students"]), len(data["alumni"]),
                len(data["interactions"]),
                len(data["spam_training_data"]["texts"]))

    # 1. TF-IDF
    tfidf = train_tfidf(data)

    # 2. BERT
    bert = train_bert(data)

    # 3. Spam
    spam = train_spam(data)

    # 4. Print summary
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("  TF-IDF:  fitted=%s", tfidf.is_fitted)
    logger.info("  BERT:    fitted=%s (available=%s)", bert.is_fitted, bert.is_available)
    logger.info("  Spam:    trained=%s", spam.is_trained)
    logger.info("  Artifacts saved to: %s", ARTIFACTS_DIR)

    return tfidf, bert, spam


if __name__ == "__main__":
    train_all()
