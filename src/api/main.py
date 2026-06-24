import json
import re
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Customer Support Intelligence API",
    description="Predicts sentiment and estimated response time for customer support tweets.",
    version="1.0.0"
)

with open("data/processed/company_medians.json", "r") as f:
    COMPANY_MEDIANS = json.load(f)

with open("models/sentiment_model.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

OVERALL_MEDIAN = 21.1


class TweetRequest(BaseModel):
    text: str
    company: str = "unknown"


class PredictionResponse(BaseModel):
    sentiment: str
    estimated_response_mins: float
    estimated_response_label: str
    company: str


def clean_text(text: str) -> str:
    text = re.sub(r'@\w+', '', str(text))
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\s!?]', '', text)
    return text.lower().strip()


def estimate_response_time(company: str) -> float:
    return COMPANY_MEDIANS.get(company, OVERALL_MEDIAN)


def response_label(mins: float) -> str:
    if mins <= 15:
        return "Very fast (under 15 mins)"
    elif mins <= 60:
        return "Fast (under 1 hour)"
    elif mins <= 180:
        return "Moderate (1-3 hours)"
    else:
        return "Slow (over 3 hours)"


@app.get("/")
def root():
    return {
        "message": "Customer Support Intelligence API",
        "endpoints": {
            "/predict": "POST — predict sentiment and response time",
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

    if len(request.text) > 560:
        raise HTTPException(status_code=400, detail="Tweet text too long (max 560 characters)")

    cleaned = clean_text(request.text)
    sentiment = sentiment_model.predict([cleaned])[0]
    response_mins = estimate_response_time(request.company)
    label = response_label(response_mins)

    return PredictionResponse(
        sentiment=sentiment,
        estimated_response_mins=response_mins,
        estimated_response_label=label,
        company=request.company
    )