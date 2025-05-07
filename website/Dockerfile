FROM python:3.10.6-slim

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

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn redis

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