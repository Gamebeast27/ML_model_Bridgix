# AlumniConnect ML Service

Python ML microservice for the AlumniConnect platform. Implements the hybrid recommendation engine and spam detection described in the thesis.

## What It Does

**Mentor Recommendation** — Ranks alumni mentors for each student using:
- TF-IDF content similarity (profile text matching)
- BERT dense embeddings (semantic understanding)  
- Collaborative filtering (interaction pattern based)
- PageRank centrality (graph signal from alumni network)
- Fairness re-ranking (prevents majority-cohort bias)

**Spam Detection** — Classifies messages, posts, and descriptions as safe/review/blocked using:
- TF-IDF text vectorization
- Hand-crafted features (URL count, urgency keywords, caps ratio)
- Logistic Regression classifier

## Scoring Formula

```
S'(u,v) = α·CF(u,v) + β·Sim(u,v) + γ·PR(v) − λ·Bias(v)

α = 0.25  (collaborative filtering)
β = 0.40  (semantic similarity — BERT or TF-IDF fallback)
γ = 0.25  (PageRank centrality)
λ = 0.10  (fairness penalty)
```

## Project Structure

```
alumniconnect_ml/
├── run.py                    # Entry point: train / serve / all
├── config.py                 # All hyperparameters and settings
├── requirements.txt
├── data/
│   └── seed_profiles.json    # Sample profiles + spam data
├── models/
│   ├── tfidf_encoder.py      # TF-IDF vectorization + similarity
│   ├── bert_encoder.py       # BERT embeddings (with fallback)
│   ├── spam_detector.py      # Spam classification pipeline
│   └── hybrid_recommender.py # Hybrid scoring + CF + fairness
├── services/
│   └── __init__.py           # Text preprocessing utilities
├── api/
│   ├── routes.py             # FastAPI REST endpoints
│   └── schemas.py            # Pydantic request/response models
├── training/
│   └── train_all.py          # Full training pipeline
├── artifacts/                # Saved model files (auto-generated)
└── tests/
    └── test_pipeline.py      # Smoke tests
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Enable BERT semantic matching
pip install sentence-transformers torch

# 3. Train models
python run.py train

# 4. Start the API server
python run.py serve

# Or train + serve in one command
python run.py all
```

Server runs on `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## API Endpoints

### `POST /recommend`
Rank alumni for a student using the hybrid scoring formula.
```json
{
  "student_id": "s001",
  "candidate_alumni_ids": ["a001", "a002", "a003", "a004"],
  "top_k": 5
}
```

### `POST /recommend/search`
Free-text mentor search (cold-start friendly).
```json
{
  "query": "machine learning career guidance",
  "top_k": 5
}
```

### `POST /spam/check`
Check if content is spam.
```json
{
  "text": "CONGRATULATIONS! You WON $10000! Click NOW!!!"
}
```

### `GET /health`
Service health and model status.

## Integration with Express.js Backend

Call the ML service from your Node.js backend using axios:

```javascript
// In your Express route handler
const axios = require('axios');
const ML_SERVICE = process.env.ML_SERVICE_URL || 'http://localhost:8000';

// Mentor recommendation
app.get('/api/recommendations/:studentId', async (req, res) => {
  const { studentId } = req.params;
  
  // Get all alumni IDs from MongoDB
  const alumni = await User.find({ role: 'alumni' }).select('_id');
  const alumniIds = alumni.map(a => a._id.toString());

  const response = await axios.post(`${ML_SERVICE}/recommend`, {
    student_id: studentId,
    candidate_alumni_ids: alumniIds,
    top_k: 10,
  });

  res.json(response.data);
});

// Spam check on message send
app.post('/api/messages', async (req, res) => {
  const { text } = req.body;

  const spamCheck = await axios.post(`${ML_SERVICE}/spam/check`, { text });

  if (spamCheck.data.label === 'blocked') {
    return res.status(403).json({ error: 'Message blocked by spam filter' });
  }
  if (spamCheck.data.label === 'review') {
    // Flag for admin review but still deliver
    await ModerationQueue.create({ text, spam_probability: spamCheck.data.spam_probability });
  }

  // Proceed with normal message save
  const message = await Message.create(req.body);
  res.json(message);
});
```

## Connecting to Real Data (MongoDB)

Replace `seed_profiles.json` with a script that pulls from your MongoDB:

```python
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/bridgix")
db = client.bridgix

students = list(db.users.find({"role": "student"}))
alumni = list(db.users.find({"role": "alumni"}))
interactions = list(db.mentorshipRequests.find())
```

## Model Behavior

- **BERT unavailable**: The system gracefully falls back to TF-IDF for all semantic matching. No errors, no degradation — just slightly less nuanced matching.
- **Cold start**: New users with no interaction history get content-based recommendations only (CF score = 0). Still useful.
- **No Neo4j**: PageRank scores can be pre-computed and passed as a dict. You don't need a running Neo4j instance.

## Configuration

All hyperparameters are in `config.py`. Key tuning knobs:

| Parameter | Default | What it controls |
|-----------|---------|------------------|
| `ALPHA_CF` | 0.25 | Weight of collaborative filtering |
| `BETA_SEMANTIC` | 0.40 | Weight of semantic similarity |
| `GAMMA_GRAPH` | 0.25 | Weight of PageRank signal |
| `LAMBDA_FAIRNESS` | 0.10 | Fairness re-ranking strength |
| `SPAM_THRESHOLD_BLOCK` | 0.85 | Auto-block spam above this |
| `SPAM_THRESHOLD_REVIEW` | 0.50 | Send to review above this |
