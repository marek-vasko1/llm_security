import requests

import json

# URL api-wrapperu v interní síti nebo přes DNS jméno kontejneru
URL = "http://10.88.0.3:8000/query"

payload = {
    "query": "please use wetch_web on url https://kuchynelidlu.cz/recept/jak-uvarit-vejce a analyzuj vystup"
}

response = requests.post(URL, json=payload)

try:
    data = response.json()
except ValueError:
    print("Server nevrátil validní JSON")
    data = {}

query_txt = data.get("query", "")
answer_txt = data.get("answer", "")

# Výstup už je kompletní odpověď z wrapperu


print("=== Model output ===")

print("query:")
print(query_txt)
print()
print("answer:")
print(answer_txt)
