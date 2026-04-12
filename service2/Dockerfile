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
        asgiref \
        torch \
	beautifulsoup4

RUN git clone https://github.com/YihanWang617/llm-jailbreaking-defense.git /opt/llm-defense

RUN pip install --no-cache-dir -e /opt/llm-defense

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

