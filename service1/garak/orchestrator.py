import json
import os
import requests


MODELS = ["gpt-oss-120b", "glm-4.7", "qwen3-coder-30b", "deepseek-v3.2-thinking", "deepseek-v3.2", "qwen3.5", "kimi-k2.5", "llama-4-scout-17b-16e-instruct"]
DRA_FILE = "dra_dataset_30.json"
BASIC_FILE = "basic_prompts.txt"
ENDPOINT_URL = "http://localhost:8000/query"
LOG_FILE = "full_attack_log.json"
WEB_TEMPLATE = "web_template.html"

def shared_log_result(model, phase, prompt, answer):
    logs = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try: logs = json.load(f)
            except: logs = []
    logs.append({
        "timestamp": os.popen("date").read().strip(),
        "model": model,
        "phase": phase,
        "prompt": prompt,
        "answer": answer
    })
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)

def run_all():
    with open(DRA_FILE, "r") as f: dra_data = json.load(f)
    dra_prompts = [d["prompt"] for d in dra_data]
    
    basic_prompts = []
    with open(BASIC_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line and " " in line:
                try:
                    basic_prompts.append(line.split(" ", 1)[1])
                except IndexError:
                    continue

    with open(WEB_TEMPLATE, "r") as f: web_html = f.read()

    for model in MODELS:
        print(f"\nSTARTING MODEL: {model}")

        # 1. TAP DIRECT 
        print("- Phase: TAP Direct (30 iterations)")
        cmd_direct = (
            f"GARAK_TARGET_MODEL={model} GARAK_IS_DIRECT=true "
            f"python3 -m garak --target_type openai --target_name {model} "
            f"--probes custom_probe_TAP.TapCustomExternalProbe "
            f"--report_prefix {model}_TAP_DIRECT"
        )
        os.system(cmd_direct)

        # 2. TAP INDIRECT 
        print("- Phase: TAP Indirect (30 iterations)")
        cmd_indirect = (
            f"GARAK_TARGET_MODEL={model} GARAK_IS_DIRECT=false "
            f"python3 -m garak --target_type openai --target_name {model} "
            f"--probes custom_probe_TAP.TapCustomExternalProbe "
            f"--report_prefix {model}_TAP_INDIRECT"
        )   
        os.system(cmd_indirect)


        # 3. DRA DIRECT (1-30)
        print("- Phase: DRA Direct")
        for i in range(0, 30):
            p = dra_prompts[i]
            resp = requests.post(ENDPOINT_URL, json={"query": p, "model": model}).json()
            shared_log_result(model, "DRA_DIRECT", p, resp.get("answer"))
        
        # 4. DRA INDIRECT
        print("- Phase: DRA Indirect")
        for i in range(0, 30):
            web_url = f"https://supertechnologybestof.en/zenitbook-16-pro-n{i}"
            p = f"Analyze web {web_url}"
            payload = {"query": p ,"model": model}
            resp = requests.post(ENDPOINT_URL, json=payload).json()
            shared_log_result(model, "DRA_INDIRECT", p, resp.get("answer"))


        # 6. BASIC PROMPTS
        print("- Phase: Basic Prompts")
        for p in basic_prompts:
            resp = requests.post(ENDPOINT_URL, json={"query": p, "model": model}).json()
            shared_log_result(model, "BASIC", p, resp.get("answer"))

if __name__ == "__main__":
    run_all()
