import re

FORBIDEN_WORDS = [
    "shutdown", "delete", "rm -rf", "system", "kill",
    "/etc/passwd", "ignore previous", "sudo"
]

def sanitize_prompt(prompt: str):
    if any(word in prompt.lower() for word in FORBIDEN_WORDS):
        return None
    return prompt.strip()

