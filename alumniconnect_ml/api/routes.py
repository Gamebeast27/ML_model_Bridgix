"""
AlumniConnect ML — API Routes
REST endpoints that your Express.js backend calls.
"""
import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import DATA_DIR
from services import build_profile_text
from models.tfidf_encoder import TFIDFEncoder
from models.bert_encoder import BERTEncoder
from models.spam_detector import SpamDetector
from models.hybrid_recommender import HybridRecommender
from api.schemas import (
    RecommendRequest,
    RecommendResponse,
    TextSearchRequest,
    SpamCheckRequest,
    SpamCheckResponse,
    SpamBatchRequest,
    HealthResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AlumniConnect ML Service",
    description="Hybrid recommendation + spam detection for the AlumniConnect platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global model instances (loaded at startup) ────────────────────────────
tfidf_encoder = TFIDFEncoder()
bert_encoder = BERTEncoder()
spam_detector = SpamDetector()
recommender: HybridRecommender = None


@app.on_event("startup")
async def startup():
    """Load all trained artifacts into memory."""
    global recommender

    # Load TF-IDF
    try:
        tfidf_encoder.load()
    except FileNotFoundError:
        logger.warning("TF-IDF artifacts not found. Run training first.")

    # Load BERT embeddings (model not needed if we only do lookups)
    bert_encoder.load()

    # Load spam detector
    try:
        spam_detector.load()
    except FileNotFoundError:
        logger.warning("Spam model not found. Run training first.")

    # Build hybrid recommender
    recommender = HybridRecommender(tfidf_encoder, bert_encoder)

    # Load CF data and PageRank from seed
    try:
        with open(DATA_DIR / "seed_profiles.json") as f:
            data = json.load(f)

        student_ids = [s["id"] for s in data["students"]]
        alumni_ids = [a["id"] for a in data["alumni"]]
        recommender.set_cf_data(student_ids, alumni_ids, data["interactions"])
        recommender.set_pagerank(data["pagerank_scores"])
    except Exception as e:
        logger.warning("Could not load CF/PageRank data: %s", e)

    logger.info("ML service startup complete")


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        tfidf_fitted=tfidf_encoder.is_fitted,
        bert_fitted=bert_encoder.is_fitted,
        bert_available=bert_encoder.is_available,
        spam_trained=spam_detector.is_trained,
    )


# ── Recommendation ─────────────────────────────────────────────────────────
@app.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    """
    Rank candidate alumni for a student using the hybrid scoring formula.
    Called by your Express.js backend when a student opens the mentor discovery page.
    """
    if recommender is None:
        raise HTTPException(500, "Recommender not initialized")

    results = recommender.recommend(
        student_id=req.student_id,
        candidate_alumni_ids=req.candidate_alumni_ids,
        top_k=req.top_k,
    )
    return RecommendResponse(
        student_id=req.student_id,
        recommendations=results,
        method="hybrid",
    )


@app.post("/recommend/search")
async def recommend_by_text(req: TextSearchRequest):
    """
    Cold-start / text-based search: find mentors matching a free-text query.
    Used when a new student hasn't filled their profile yet, or for search bar.
    """
    if recommender is None:
        raise HTTPException(500, "Recommender not initialized")

    results = recommender.recommend_by_text(
        query_text=req.query,
        top_k=req.top_k,
    )
    return {"query": req.query, "results": results}


@app.post("/recommend/score")
async def score_pair(student_id: str, alumni_id: str):
    """
    Get the full score breakdown for a specific (student, alumni) pair.
    Useful for debugging and transparency.
    """
    if recommender is None:
        raise HTTPException(500, "Recommender not initialized")

    breakdown = recommender.score_pair(student_id, alumni_id)
    return breakdown


# ── Spam Detection ─────────────────────────────────────────────────────────
@app.post("/spam/check", response_model=SpamCheckResponse)
async def check_spam(req: SpamCheckRequest):
    """
    Check if a message/post/description is spam.
    Called by your Express.js backend on every message send / post create.
    """
    if not spam_detector.is_trained:
        raise HTTPException(500, "Spam detector not trained")

    result = spam_detector.predict(req.text)
    return SpamCheckResponse(**result)


@app.post("/spam/batch")
async def check_spam_batch(req: SpamBatchRequest):
    """Batch spam check for multiple texts."""
    if not spam_detector.is_trained:
        raise HTTPException(500, "Spam detector not trained")

    results = spam_detector.predict_batch(req.texts)
    return {"results": results}
