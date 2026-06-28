# Customer Support Intelligence Platform

An end-to-end MLOps project that analyzes 2.8M real customer support conversations from Twitter, turning raw interaction data into operational intelligence for support teams.

## What It Does

Given a customer tweet, the platform predicts:
- **Priority** — urgent or standard (sentiment classification)
- **Estimated response time** — in minutes, based on company and time of day
- **Complaint topic** — one of 10 discovered categories (delivery, billing, iOS issues, travel delays, etc.)

A live Streamlit UI and a REST API make it usable in real time or at scale via batch evaluation.

## Dataset

**Twitter Customer Support Dataset** — 2.8M tweets from 108 companies (Oct–Dec 2017), sourced from Kaggle. Customer messages were paired with company replies and engineered into conversation-level features including response time, time of day, and company identity.

## Key Findings

- 82% of customer tweets received a company reply — median response time 21 minutes
- Financial companies (Visa, SC Support) are 10–20x slower than tech companies (Postmates, TMobile)
- Tweet activity peaks 3–7pm on weekdays — useful for staffing decisions
- Tweet length has near-zero correlation with response time — ruled out as a useful feature
- 7.3% of tweets are non-English (Spanish, French, Japanese, Portuguese, Italian) — discovered through LDA topic modeling

## ML Tasks   

### Classification — Sentiment Triage
Predict whether a customer tweet is negative (urgent) or non-negative (standard).

| Model | Accuracy | F1 (negative) |
|---|---|---|
| Logistic Regression | 87.9% | 0.771 |
| SGD Classifier | 86.1% | 0.734 |

Winner: Logistic Regression — higher recall on the negative class (0.87), catching 87% of unhappy customers.

### Regression — Response Time Prediction
Predict how long a company will take to respond, in minutes.

| Model | R² | MAE |
|---|---|---|
| Ridge — text only | 0.176 | 89.6 mins |
| Ridge — text + label encoded company | 0.182 | 89.3 mins |
| Ridge — text + one-hot company + hour | **0.445** | **77.7 mins** |

Key insight: company identity is the strongest predictor of response time — more than the tweet content itself.

### Clustering — Complaint Topic Discovery
Discover recurring complaint categories without manual labeling.

Compared KMeans and LDA on TF-IDF representations. KMeans produced weak clusters (silhouette 0.011–0.017) with one cluster capturing 63% of data. LDA discovered 10 meaningful topics:

- iOS/iPhone Issues
- Billing & Account
- Delivery & Orders
- Travel Delays (flights, trains)
- Internet & Network Outages
- App & Streaming
- Gaming & Online Shopping
- Rideshare
- Service Quality Complaints
- Support Response Quality

A multilingual proof of concept was also implemented — detecting and translating non-English tweets (Spanish, French, Japanese, Portuguese, German) before clustering, removing language noise from topic discovery.

## Stack

| Component | Tool |
|---|---|
| Environment | uv |
| Experiment tracking | MLflow |
| API | FastAPI |
| UI | Streamlit |
| Containerization | Docker + Docker Compose |
| Version control | Git + GitHub |
| Config management | config.yaml |

## Project Structure

```
notebooks/          # EDA, preprocessing, modeling, clustering
src/
  data/             # preprocessing pipeline
  models/           # training logic
  api/              # FastAPI endpoint
data/
  raw/              # raw data (gitignored)
  processed/        # train/val/test splits, artifacts
models/             # saved joblib models
config.yaml         # hyperparameters and paths
Dockerfile          # API container
Dockerfile.ui       # Streamlit container
docker-compose.yml  # full stack deployment
```

## Setup

```bash
git clone <repo-url>
cd support-ticket-triage
uv sync
```

## Run Locally

```bash
# API
uv run uvicorn src.api.main:app --reload

# UI (new terminal)
uv run streamlit run app.py

# MLflow UI
uv run mlflow ui --backend-store-uri sqlite:///notebooks/mlruns.db
```

## Run with Docker

```bash
# Full stack — API + UI
docker-compose up --build

# API only
docker build -t support-triage .
docker run -p 8000:8000 support-triage
```

API available at `http://localhost:8000` — interactive docs at `http://localhost:8000/docs`
UI available at `http://localhost:8501`

## API

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "@AmazonHelp my package never arrived", "timestamp": "2017-10-31 14:00:00"}'

# Batch prediction (up to 1000 tweets)
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"tweets": [{"text": "@AppleSupport my phone is broken"}]}'
```

## Milestones

- [x] Environment setup — repo, uv, dependencies pinned
- [x] EDA — data quality, distributions, response time analysis, text exploration
- [x] Feature engineering pipeline — time features, text cleaning, sentiment labels, response time transform
- [x] Baseline model + problem framing
- [x] Model comparison with MLflow tracking — 5 models across 3 tasks
- [x] Notebook → .py modules — preprocessing.py, train.py
- [x] Config and reproducibility — config.yaml
- [x] FastAPI /predict endpoint — single + batch
- [x] Docker deployment — Dockerfile + Docker Compose
- [x] Model registry — MLflow model registry
- [ ] Model card
- [ ] Final demo and retrospective