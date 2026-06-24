FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .

RUN pip install uv
RUN uv sync --frozen

COPY src/ src/
COPY config.yaml .
COPY data/processed/company_medians.json data/processed/company_medians.json
COPY models/sentiment_model.joblib models/sentiment_model.joblib
COPY models/regression_model.pkl models/regression_model.pkl

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]