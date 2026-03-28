FROM python:3.12-slim

WORKDIR /app

RUN apt update && apt install -y net-tools git && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        requests \
        pydantic \
        openai \
        mcp \
        python-dotenv \
        typing-extensions \
        aiohttp \
        dotenv \
        numpy




CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

