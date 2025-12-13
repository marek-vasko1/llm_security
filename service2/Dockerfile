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
        numpy \
        torch

# Klonování repozitáře
RUN git clone https://github.com/YihanWang617/llm-jailbreaking-defense.git /app/

# Instalace v editable módu
RUN pip install --no-cache-dir -e /app/

# Přidání tvých souborů
COPY main.py /app/main.py
COPY mcp_server.py /app/mcp_server.py
COPY security.py /app/security.py

# Pokud chceš použít konkrétní soubory z klonovaného repozitáře
# COPY /app/smooth-llm/lib/defenses.py /app/lib/defenses.py
# COPY /app/smooth-llm/lib/perturbations.py /app/lib/perturbations.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

