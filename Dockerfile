FROM python:3.13-slim

WORKDIR /app

# 1. Install system dependencies
#    Added 'git' to fix the error if any other git deps exist
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Install the Shared Core Library
#    We copy it from the build context (project root) into a temp folder
COPY pvp-core-lib/ /tmp/pvp-core-lib/
RUN pip install /tmp/pvp-core-lib/

# 3. Install Backend Requirements
COPY pvp-mockup-be/requirements.txt /app/requirements.txt

# 4. Remove the remote git reference robustly
RUN cat /app/requirements.txt > /app/requirements.clean.txt && pip install --no-cache-dir -r /app/requirements.clean.txt

# 5. Copy Backend Code
COPY pvp-mockup-be/ /app/

# 6. Runtime Configuration
ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]