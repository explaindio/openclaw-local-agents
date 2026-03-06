#!/usr/bin/env python3
"""Needle In A Haystack (NIAH) Test for Qwen3.5 on llama-server"""
import json, requests, subprocess, sys, time

CONTEXT_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
NEEDLE = "The secret launch code is PINEAPPLE-774."
QUESTION = "What is the secret launch code? Reply with ONLY the launch code, nothing else."

FILLER_CHARS = (CONTEXT_SIZE - 500) * 4
FILLER_PARA = (
    "The history of papermaking dates back to ancient China, where Cai Lun is credited "
    "with inventing the process around 105 AD. The technique spread along the Silk Road to "
    "the Islamic world and eventually to Europe. Early paper was made from bark, hemp, and rags. "
    "Modern papermaking uses wood pulp and involves complex chemical processes. The industry has "
    "evolved significantly with the advent of digital technology, though paper remains essential. "
    "Environmental concerns have led to increased recycling efforts and sustainable forestry. "
)

print(f"=== NIAH Test ===")
print(f"Target context: {CONTEXT_SIZE:,} tokens")
filler = (FILLER_PARA * (FILLER_CHARS // len(FILLER_PARA) + 1))[:FILLER_CHARS]
prompt = f"{NEEDLE}\n\n{filler}\n\n{QUESTION}"
print(f"Prompt size: {len(prompt):,} chars")

vram_before = subprocess.check_output(
    "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader", shell=True
).decode().strip()
print(f"VRAM before: {vram_before}")

payload = {
    "model": "qwen3.5",
    "messages": [
        {"role": "system", "content": "/nothink"},
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 50,
    "temperature": 0.1,
}

print(f"\nSending request...")
t0 = time.time()
try:
    resp = requests.post(SERVER_URL, json=payload, timeout=600)
    elapsed = time.time() - t0
    data = resp.json()
    
    vram_after = subprocess.check_output(
        "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader", shell=True
    ).decode().strip()
    
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    
    print(f"\n=== RESULTS ===")
    print(f"Model response: {repr(content)}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Tokens — prompt: {usage.get('prompt_tokens', 'N/A')}, completion: {usage.get('completion_tokens', 'N/A')}")
    print(f"VRAM before: {vram_before}")
    print(f"VRAM after:  {vram_after}")
    
    if "PINEAPPLE-774" in content:
        print("✅ NEEDLE FOUND — Model comprehends at this context length!")
    else:
        print("❌ NEEDLE NOT FOUND")
        print(f"Full response JSON snippet: {json.dumps(data['choices'][0], indent=2)[:500]}")
except Exception as e:
    print(f"ERROR: {e}")
    print(f"Time: {time.time() - t0:.1f}s")
