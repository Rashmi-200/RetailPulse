FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config/ ./config/
COPY sql/ ./sql/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

RUN mkdir -p /app/data /app/logs /app/models

EXPOSE 8000

CMD ["sh", "-c", "python scripts/bootstrap_ci_env.py && exec uvicorn src.api.control_plane:app --host 0.0.0.0 --port 8000 --workers 1"]
