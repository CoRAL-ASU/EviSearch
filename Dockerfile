FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-cloudrun.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements-cloudrun.txt

COPY src ./src
COPY web ./web
COPY experiment-scripts ./experiment-scripts
COPY dataset ./dataset
COPY new_pipeline_outputs ./new_pipeline_outputs

RUN mkdir -p /app/web/uploads

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "web.main_app:app"]
