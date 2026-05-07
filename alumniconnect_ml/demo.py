"""
AlumniConnect ML — Interactive Demo
Run: python demo.py
Shows the recommender and spam detector in action.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR
from services import build_profile_text
from models.tfidf_encoder import TFIDFEncoder
from models.bert_encoder import BERTEncoder
from models.spam_detector import SpamDetector
from models.hybrid_recommender import HybridRecommender


def divider(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def main():
    # ── Load data ──────────────────────────────────────────────────────
    with open(DATA_DIR / "seed_profiles.json") as f:
        data = json.load(f)

    all_profiles = data["students"] + data["alumni"]
    profile_map = {p["id"]: p for p in all_profiles}

    # ── Load trained models ────────────────────────────────────────────
    tfidf = TFIDFEncoder().load()
    bert = BERTEncoder().load()
    spam = SpamDetector().load()

    recommender = HybridRecommender(tfidf, bert)
    recommender.set_cf_data(
        [s["id"] for s in data["students"]],
        [a["id"] for a in data["alumni"]],
        data["interactions"],
    )
    recommender.set_pagerank(data["pagerank_scores"])

    alumni_ids = [a["id"] for a in data["alumni"]]

    # ═══════════════════════════════════════════════════════════════════
    divider("MENTOR RECOMMENDATIONS")
    # ═══════════════════════════════════════════════════════════════════

    for student in data["students"][:3]:
        sid = student["id"]
        print(f"Student: {student['name']} ({sid})")
        print(f"  Skills: {', '.join(student.get('skills', []))}")
        print(f"  Goal:   {student.get('career_goal', 'N/A')}")
        print()

        results = recommender.recommend(sid, alumni_ids, top_k=5)
        for rank, r in enumerate(results, 1):
            mentor = profile_map[r["alumni_id"]]
            print(f"  #{rank}  {mentor['name']} — {mentor.get('designation', '')} @ {mentor.get('organization', '')}")
            print(f"       Score: {r['final_score']:.3f}  "
                  f"(CF={r['cf_score']:.3f}, Sem={r['semantic_score']:.3f}, "
                  f"Graph={r['graph_score']:.3f}, Bias={r['bias_penalty']:.3f})")
        print()

    # ═══════════════════════════════════════════════════════════════════
    divider("TEXT-BASED SEARCH (Cold Start)")
    # ═══════════════════════════════════════════════════════════════════

    queries = [
        "machine learning AI deep learning career",
        "frontend react design portfolio",
        "embedded systems IoT hardware",
    ]
    for q in queries:
        print(f"  Query: \"{q}\"")
        results = recommender.recommend_by_text(q, top_k=3)
        for r in results:
            mentor = profile_map.get(r["alumni_id"], {})
            name = mentor.get("name", r["alumni_id"])
            print(f"    → {name} (score={r['semantic_score']:.3f}, method={r['method']})")
        print()

    # ═══════════════════════════════════════════════════════════════════
    divider("SPAM DETECTION")
    # ═══════════════════════════════════════════════════════════════════

    test_messages = [
        "Hi, I graduated from the same college in 2019. Happy to help with your ML career questions.",
        "CONGRATULATIONS!! You have been SELECTED for our EXCLUSIVE program! Click NOW: http://scam.com/claim",
        "We are hiring for SDE-1 role at our company. Required: 2+ years experience in Java, Spring Boot.",
        "EARN $10000 PER WEEK!!! No experience needed. Join our crypto group NOW for guaranteed profits!!!",
        "Can someone share resources for system design interview preparation?",
        "URGENT: Your account will be DELETED unless you verify NOW. Send OTP to this number immediately.",
    ]

    for msg in test_messages:
        result = spam.predict(msg)
        label = result["label"].upper()
        prob = result["spam_probability"]

        if label == "BLOCKED":
            icon = "🚫"
        elif label == "REVIEW":
            icon = "⚠️"
        else:
            icon = "✅"

        print(f"  {icon} [{label:7s}] (p={prob:.3f})  \"{msg[:80]}{'...' if len(msg) > 80 else ''}\"")

    # ═══════════════════════════════════════════════════════════════════
    divider("SCORE BREAKDOWN (Detailed)")
    # ═══════════════════════════════════════════════════════════════════

    print("  Pair: s001 (Mohit, ML student) → a001 (Rahul, Google ML Engineer)")
    breakdown = recommender.score_pair("s001", "a001")
    for key, val in breakdown.items():
        if key == "weights":
            print(f"  {key}:")
            for wk, wv in val.items():
                print(f"    {wk}: {wv}")
        else:
            print(f"  {key}: {val}")

    print(f"\n{'=' * 60}")
    print("  Demo complete. Run 'python run.py serve' to start the API.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
