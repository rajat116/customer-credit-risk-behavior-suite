# Dockerfile for FastAPI + ML models (root level)

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src /app/src
COPY models /app/models
COPY service /app/service

# Environment
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]