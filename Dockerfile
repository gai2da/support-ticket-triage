FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .

RUN pip install uv && uv sync --frozen && apt-get update && apt-get install -y curl --no-install-recommends && rm -rf /var/lib/apt/lists/*

COPY src/ src/
COPY config.yaml .
COPY data/processed/company_medians.json data/processed/company_medians.json
COPY models/sentiment_model.joblib models/sentiment_model.joblib
COPY models/regression_model.joblib models/regression_model.joblib
COPY models/lda_model.joblib models/lda_model.joblib
COPY models/lda_tfidf.joblib models/lda_tfidf.joblib

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]