# Dockerfile for OpenCV 5 Agentic Vision Backend
# Compatible with AWS App Runner, AWS ECS (Fargate), and local Docker

FROM python:3.12-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest and install
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code
COPY backend /app

# Expose FastAPI port
EXPOSE 8000

# Health check for AWS App Runner / ECS
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start Uvicorn production server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
