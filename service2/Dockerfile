FROM python:3.12-slim

WORKDIR /app

# Aktualizace, instalace net-tools a git, a Python balíčků
RUN apt update && apt install -y net-tools git && \
    pip install --no-cache-dir --default-timeout=1000 \
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
	torch \
	transformers \
	accelerate \
	vllm

# Přidání tvých souborů
COPY main.py /app/main.py
COPY mcp_server.py /app/mcp_server.py
COPY security.py /app/security.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
