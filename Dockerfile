FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml .
COPY uv.lock .

RUN pip install uv
RUN uv sync --frozen

COPY src/ src/
COPY data/processed/company_medians.json data/processed/company_medians.json

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]