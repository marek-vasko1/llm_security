import requests
import json
import time

# Adresa tvého FastAPI serveru
URL = "http://127.0.0.1:8000/query"

# Seznam modelů pro kontrolu
models_to_test = [
    "gpt-oss-120b",
    "glm-4.7",
    "qwen3-coder-30b"
]

headers = {
    "Content-Type": "application/json"
}

print(f"Spouštím test pro {len(models_to_test)} modely...\n")
print("-" * 50)

for model_name in models_to_test:
    print(f"⏳ Odesílám dotaz na model: **{model_name}**")
            
    payload = {
        "query": f"Ahoj, otestuj spojení. Toto je testovací dotaz směřovaný na {model_name}.",
        "model": model_name,
        "web": "https://example.com",
        "web_body": "<html><body>Testovací obsah pro MCP</body></html>"
    }
                    
    try:
        # Měříme i čas, ať vidíš, jak dlouho každý model odpovídá
        start_time = time.time()
        response = requests.post(URL, json=payload, headers=headers)
        duration = time.time() - start_time
                           
        if response.status_code == 200:
            print(f"✅ Úspěch! Server odpověděl za {duration:.2f}s (HTTP 200).")
                                                                     
            # Vytáhneme samotnou odpověď z JSONu (podle tvého návratového typu z API)
            response_data = response.json()
            answer = response_data.get("answer", "Odpověď nenalezena v JSONu.")
                                                                                                                                                    
            print(f"🤖 Odpověď modelu:\n{answer}\n")
        else:
            print(f"❌ Chyba! Status kód: {response.status_code}")
            print(f"Detail: {response.text}\n")
    except requests.exceptions.ConnectionError:
        print("❌ Nepodařilo se připojit k serveru. Běží tvůj FastAPI orchestrátor?")
        break  # Pokud nejede server, nemá smysl zkoušet další modely
    except Exception as e:
        print(f"❌ Neočekávaná chyba: {e}\n")                                                                               
    print("-" * 50)
    # Krátká pauza mezi dotazy, aby se server "nadechl"
    time.sleep(2)
print("🎉 Testování dokončeno.")
