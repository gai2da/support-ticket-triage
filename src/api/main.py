import re
import joblib
import numpy as np
import yaml
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.sparse import hstack, csr_matrix
from src.data.preprocessing import clean_text, extract_company, extract_hour, extract_is_weekend

with open("config.yaml") as f:
    config = yaml.safe_load(f)

app = FastAPI(
    title="Customer Support Intelligence API",
    description="Predicts sentiment and estimated response time for customer support tweets.",
    version="1.0.0"
)

sentiment_model = joblib.load("models/sentiment_model.joblib")
regression_artifacts = joblib.load("models/regression_model.joblib")
reg_model = regression_artifacts["model"]
reg_tfidf = regression_artifacts["tfidf"]
reg_ohe = regression_artifacts["ohe"]

OVERALL_MEDIAN = config["api"]["overall_median_response"]
MAX_TWEET_LENGTH = config["api"]["max_tweet_length"]


class TweetRequest(BaseModel):
    text: str
    company: str = ""
    timestamp: str = ""


class PredictionResponse(BaseModel):
    sentiment: str
    estimated_response_mins: float
    estimated_response_label: str
    company: str
    hour: int
    is_weekend: int


class BatchRequest(BaseModel):
    tweets: List[TweetRequest]


def response_label(mins: float) -> str:
    if mins <= 15:
        return "Very fast (under 15 mins)"
    elif mins <= 60:
        return "Fast (under 1 hour)"
    elif mins <= 180:
        return "Moderate (1-3 hours)"
    else:
        return "Slow (over 3 hours)"


def predict_single(text: str, company: str = "", timestamp: str = "") -> dict:
    company = company or extract_company(text)
    hour = extract_hour(timestamp)
    is_weekend = extract_is_weekend(timestamp)
    cleaned = clean_text(text)

    sentiment = sentiment_model.predict([cleaned])[0]

    text_features = reg_tfidf.transform([cleaned])
    company_features = reg_ohe.transform([[company]])
    extra_features = csr_matrix([[hour, is_weekend]])
    combined = hstack([text_features, company_features, extra_features])

    response_log = reg_model.predict(combined)[0]
    response_mins = float(np.expm1(response_log))
    response_mins = max(0, min(response_mins, 500))

    return {
        "sentiment": sentiment,
        "estimated_response_mins": round(response_mins, 1),
        "estimated_response_label": response_label(response_mins),
        "company": company,
        "hour": hour,
        "is_weekend": is_weekend
    }


@app.get("/")
def root():
    return {
        "message": "Customer Support Intelligence API",
        "endpoints": {
            "/predict": "POST — predict single tweet",
            "/predict/batch": "POST — predict batch of tweets",
            "/health": "GET — health check",
            "/docs": "GET — interactive API docs"
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TweetRequest):
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Tweet text cannot be empty")
    if len(request.text) > MAX_TWEET_LENGTH:
        raise HTTPException(status_code=400, detail=f"Tweet text too long (max {MAX_TWEET_LENGTH} characters)")
    return predict_single(request.text, request.company, request.timestamp)


@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    if len(request.tweets) == 0:
        raise HTTPException(status_code=400, detail="No tweets provided")
    if len(request.tweets) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 tweets per batch")
    return [predict_single(t.text, t.company, t.timestamp) for t in request.tweets]