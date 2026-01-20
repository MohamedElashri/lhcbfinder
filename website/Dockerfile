FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV WORKERS=3
ENV TIMEOUT=60

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY app.py .
COPY helpers.py .
COPY paper.py .
COPY query_processor.py .
COPY static static/
COPY templates templates/
COPY requirements.txt .

# Create logs directory
RUN mkdir -p logs

# Install uv for faster package installation
RUN pip install --no-cache-dir uv

# Install Python dependencies using uv
RUN uv pip install --system --no-cache -r requirements.txt \
    && uv pip install --system --no-cache gunicorn redis

# Create ChromaDB mount point directory
# Volume will be mounted at /app/chroma_db (see docker-compose.yml)
RUN mkdir -p /app/chroma_db

# Create and switch to non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run gunicorn with environment variables
ENTRYPOINT ["gunicorn"]
CMD ["--bind", "0.0.0.0:8000", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "60", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--capture-output", \
     "--log-level", "info", \
     "app:app"]