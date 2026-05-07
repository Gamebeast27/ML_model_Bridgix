"""
AlumniConnect ML — Text Preprocessing
Shared cleaning and normalization used by both recommendation and spam pipelines.
"""
import re
import string
from typing import List, Optional


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, strip HTML, normalize whitespace."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML
    text = re.sub(r"http\S+|www\.\S+", " _URL_ ", text)  # replace URLs with token
    text = re.sub(r"[^\w\s@._-]", " ", text)       # keep alphanumeric + common chars
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_profile_text(profile: dict) -> str:
    """
    Combine structured profile fields into a single text block for vectorization.
    Works for both students and alumni.
    """
    parts = []

    # Core identity fields
    for field in ["department", "branch", "specialization", "designation",
                  "organization", "industry", "sector"]:
        val = profile.get(field, "")
        if val:
            parts.append(str(val))

    # List fields — skills, interests, expertise
    for field in ["skills", "interests", "expertise", "mentoring_tags",
                  "technical_interests", "preferred_domains"]:
        val = profile.get(field, [])
        if isinstance(val, list):
            parts.extend([str(v) for v in val if v])
        elif isinstance(val, str) and val:
            parts.append(val)

    # Free-text fields
    for field in ["bio", "about", "mentorship_purpose", "career_goal"]:
        val = profile.get(field, "")
        if val:
            parts.append(str(val))

    combined = " ".join(parts)
    return clean_text(combined)


def extract_spam_features(text: str) -> dict:
    """
    Extract hand-crafted features for spam detection beyond TF-IDF.
    These augment the bag-of-words representation.
    """
    raw = text if text else ""
    cleaned = clean_text(raw)

    url_count = len(re.findall(r"http\S+|www\.\S+", raw))
    email_count = len(re.findall(r"\S+@\S+\.\S+", raw))
    exclamation_count = raw.count("!")
    caps_ratio = sum(1 for c in raw if c.isupper()) / max(len(raw), 1)
    digit_ratio = sum(1 for c in raw if c.isdigit()) / max(len(raw), 1)
    word_count = len(cleaned.split())
    avg_word_len = (sum(len(w) for w in cleaned.split()) / max(word_count, 1))

    # Spam trigger patterns
    urgency_keywords = ["urgent", "limited", "act now", "immediately",
                        "guaranteed", "free", "click here", "apply now",
                        "congratulations", "winner", "earn money"]
    urgency_score = sum(1 for kw in urgency_keywords if kw in cleaned)

    return {
        "url_count": url_count,
        "email_count": email_count,
        "exclamation_count": exclamation_count,
        "caps_ratio": round(caps_ratio, 4),
        "digit_ratio": round(digit_ratio, 4),
        "word_count": word_count,
        "avg_word_len": round(avg_word_len, 2),
        "urgency_score": urgency_score,
    }


def normalize_department(dept: str) -> str:
    """Map common department name variants to canonical form."""
    if not dept:
        return ""
    dept = dept.lower().strip()
    mapping = {
        "cse": "computer science",
        "cs": "computer science",
        "computer science and engineering": "computer science",
        "computer science & engineering": "computer science",
        "it": "information technology",
        "ece": "electronics and communication",
        "ee": "electrical engineering",
        "eee": "electrical engineering",
        "me": "mechanical engineering",
        "mech": "mechanical engineering",
        "ce": "civil engineering",
    }
    return mapping.get(dept, dept)


def normalize_skills(skills: List[str]) -> List[str]:
    """Deduplicate and lowercase skill tags."""
    if not skills:
        return []
    seen = set()
    out = []
    for s in skills:
        key = s.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out
