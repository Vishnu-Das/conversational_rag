FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY . .

RUN mkdir -p chroma_db
RUN mkdir -p src/data
RUN mkdir -p storage

EXPOSE 8501

CMD sh -c "\
if [ ! -d chroma_db ] || [ -z \"$(ls -A chroma_db)\" ]; then \
    echo 'No vector database found. Running ingestion...'; \
    .venv/bin/python -m src.ingest; \
fi && \
.venv/bin/streamlit run app.py --server.address=0.0.0.0"