import requests

import json

# URL api-wrapperu v interní síti nebo přes DNS jméno kontejneru
URL = "http://10.89.0.3:8000/query"

payload = {
    "query": "Potrebuju poradit s vyberem graficke karty. Pouzij funkci na fetch the web na https://www.alza.cz/graficke-karty/18842862.htm?kampan=adwkom_komponenty_bee_gen_search_komponenty-graficke-karty-c18842862&ppcbee-adtext-variant=rsa_gen_seg1&gad_source=1&gad_campaignid=17543560951&gclid=CjwKCAjwspPOBhB9EiwATFbi5DZSC0Vvs7oHCTRiSt8Fbxl4bdTFjg9jEImY7I4poKoMlUloEvZmIxoC8PIQAvD_BwE" 
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
