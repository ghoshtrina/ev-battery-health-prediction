# ---- Base image ----
FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Install system deps (if you later need e.g. gcc for some libs, add here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy only requirements first
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the project
COPY . /app

# Default port
EXPOSE 8000

# Start FastAPI app
# (If you want fewer logs, add: --log-level warning)
CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
