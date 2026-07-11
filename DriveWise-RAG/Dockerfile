FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed by PyMuPDF and SQLite
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmupdf-dev \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements or install directly to avoid missing files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir fastapi uvicorn PyMuPDF qdrant-client sentence-transformers fastembed requests pydantic python-dotenv

# Copy codebase
COPY . .

# Expose FastAPI + Frontend static port
EXPOSE 8000

# Environment variables defaults
ENV PORT=8000
ENV HOST=0.0.0.0
ENV MOCK_LLM=true

CMD ["python", "src/app.py"]
