"""
AlumniConnect ML — API Schemas
Pydantic models for request validation and response serialization.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ── Recommendation ─────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    student_id: str = Field(..., description="Student user ID")
    candidate_alumni_ids: List[str] = Field(
        ..., description="Pool of alumni IDs to rank"
    )
    top_k: int = Field(default=10, ge=1, le=50)


class TextSearchRequest(BaseModel):
    query: str = Field(..., description="Free-text search query or profile description")
    top_k: int = Field(default=10, ge=1, le=50)


class ScoreBreakdown(BaseModel):
    alumni_id: str
    final_score: float
    cf_score: float
    semantic_score: float
    graph_score: float
    bias_penalty: float
    weights: Dict[str, float]


class RecommendResponse(BaseModel):
    student_id: str
    recommendations: List[Dict]
    method: str = "hybrid"


# ── Spam Detection ─────────────────────────────────────────────────────────
class SpamCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text content to check")


class SpamCheckResponse(BaseModel):
    label: str  # "safe" | "review" | "blocked"
    spam_probability: float
    features: Dict


class SpamBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)


# ── Health ─────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    tfidf_fitted: bool
    bert_fitted: bool
    bert_available: bool
    spam_trained: bool
