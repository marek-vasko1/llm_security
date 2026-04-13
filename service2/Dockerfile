FROM python:3.12-slim

WORKDIR /app

RUN apt update && apt install -y net-tools git libgomp1 && \
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
        transformers \
        tqdm \
        lightgbm \
        pandas \
        joblib \
        accelerate \
	scikit-learn \
	beautifulsoup4






CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

