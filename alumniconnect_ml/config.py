"""
AlumniConnect ML — Configuration
Single source of truth for all model hyperparameters, weights, and paths.
"""
import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ─── TF-IDF Settings ─────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 1
TFIDF_SUBLINEAR_TF = True

# ─── BERT Settings ────────────────────────────────────────────────────────────
BERT_MODEL_NAME = "all-MiniLM-L6-v2"   # 80MB, fast, good quality
BERT_BATCH_SIZE = 32
BERT_MAX_SEQ_LENGTH = 256

# ─── Hybrid Scoring Weights ──────────────────────────────────────────────────
# S'(u,v) = α·CF(u,v) + β·Sim_BERT(u,v) + γ·PR(v) − λ·Bias(v)
ALPHA_CF = 0.25          # Collaborative filtering weight
BETA_SEMANTIC = 0.40     # Semantic similarity weight (BERT or TF-IDF fallback)
GAMMA_GRAPH = 0.25       # Graph signal weight (PageRank / centrality)
LAMBDA_FAIRNESS = 0.10   # Fairness penalty weight

# ─── Spam Detection ──────────────────────────────────────────────────────────
SPAM_TFIDF_MAX_FEATURES = 10000
SPAM_TFIDF_NGRAM_RANGE = (1, 3)
SPAM_THRESHOLD_BLOCK = 0.85      # Auto-block above this
SPAM_THRESHOLD_REVIEW = 0.50     # Send to admin review above this
SPAM_MODEL_TYPE = "logistic_regression"  # "naive_bayes" | "logistic_regression" | "svm"

# ─── Collaborative Filtering ─────────────────────────────────────────────────
CF_MIN_INTERACTIONS = 2          # Minimum interactions before CF kicks in
CF_SIMILARITY_NEIGHBORS = 20    # k-nearest neighbors for user-based CF

# ─── API Settings ─────────────────────────────────────────────────────────────
API_HOST = os.getenv("ML_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("ML_API_PORT", "8000"))

# ─── Recommendation Output ───────────────────────────────────────────────────
DEFAULT_TOP_K = 10
MAX_TOP_K = 50
