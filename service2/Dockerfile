FROM python:3.12-slim

WORKDIR /app

# Aktualizace, instalace net-tools a git, a Python balíčků
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
	matplotlib \
	nltk \
    	numpy \
    	openai \
    	Pillow \
    	spacy \
    	spacy-legacy \
    	spacy-loggers \
    	textaugment \
    	textblob \
    	torch \
    	torchaudio \
    	torchdata \
    	torchtext \
    	torchvision \
	&& python -m spacy download en_core_web_md

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


