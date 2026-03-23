#!/bin/bash
set -e

podman-compose down
podman-compose up -d

# čekáme, až api-wrapper bude poslouchat port 8000
echo "Waiting for api-wrapper to be ready..."

sleep 20

echo "API is ready, starting kali..."
podman exec -it kali bash -c "
    python3 -m venv /app/venv && \
    source /app/venv/bin/activate && \
    python3 -m pip install --upgrade pip requests && \
    python3 /app/app.py
"
