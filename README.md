# Customer Support Tweet Intelligence

An end-to-end ML pipeline for analyzing customer support interactions on Twitter.

## What it does
- Classifies tweets as customer complaints or company responses
- Predicts response time for customer tweets
- Segments customer complaints into topic clusters

## Dataset
Twitter Customer Support Dataset (Kaggle) — 2.8M real tweets between customers and 108 companies (Oct-Dec 2017).


## Setup
```bash
git clone <repo-url>
cd support-ticket-triage
uv sync
```

## Project Structure
```
notebooks/    # EDA and experimentation
src/
  data/       # data loading and preprocessing
  models/     # model training and evaluation
  api/        # FastAPI endpoint
data/raw/     # raw data (gitignored)
tests/        # unit tests
config.yaml   # hyperparameters and paths
```

## Run
```bash
uv run uvicorn src.api.main:app --reload
```