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
        numpy \
        torch

RUN git clone https://github.com/YihanWang617/llm-jailbreaking-defense.git /app/

RUN pip install --no-cache-dir -e /app/

COPY main.py /app/main.py
COPY mcp_server.py /app/mcp_server.py
COPY security.py /app/security.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

