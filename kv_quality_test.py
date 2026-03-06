#!/usr/bin/env python3
"""
KV Cache Quality Comparison: f16 vs q8_0
Tests the same prompts on both KV cache modes and compares outputs.
"""
import json, requests, subprocess, sys, time, os, re

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

# Test suite — diverse tasks to detect quality degradation
TESTS = [
    {
        "name": "Math Reasoning",
        "prompt": "Solve step by step: If a train travels 120 km in 1.5 hours, then stops for 30 minutes, then travels 80 km in 1 hour, what is the average speed for the entire journey including the stop?",
        "check": lambda r: "66" in r or "67" in r,  # ~66.67 km/h
    },
    {
        "name": "Factual Recall",
        "prompt": "List the first 10 prime numbers and their sum.",
        "check": lambda r: "129" in r,  # 2+3+5+7+11+13+17+19+23+29=129
    },
    {
        "name": "Logic Puzzle",
        "prompt": "There are 3 boxes. Box A says 'The gold is in Box B'. Box B says 'The gold is not in this box'. Box C says 'The gold is in this box'. Only one box tells the truth. Which box has the gold?",
        "check": lambda r: "B" in r,
    },
    {
        "name": "Code Generation",
        "prompt": "Write a Python function that checks if a string is a palindrome, ignoring spaces and case. Show just the function.",
        "check": lambda r: "def" in r and "palindrome" in r.lower(),
    },
    {
        "name": "NIAH (5K context)",
        "prompt": None,  # Generated below
        "check": lambda r: "PINEAPPLE-774" in r,
    },
]

def make_niah_prompt():
    needle = "The secret launch code is PINEAPPLE-774."
    filler_para = (
        "The history of papermaking dates back to ancient China. "
        "Modern papermaking uses wood pulp and involves complex chemical processes. "
        "Environmental concerns have led to increased recycling efforts. "
    ) * 50
    filler = (filler_para * 3)[:20000]
    return f"{needle}\n\n{filler}\n\nWhat is the secret launch code? Reply with ONLY the launch code."

TESTS[4]["prompt"] = make_niah_prompt()

def query(prompt, max_tokens=200, timeout=120):
    payload = {
        "model": "qwen3.5",
        "messages": [
            {"role": "system", "content": "/nothink"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000,
        "temperature": 0.0,  # Deterministic for comparison
        "seed": 42,
    }
    t0 = time.time()
    try:
        resp = requests.post(SERVER_URL, json=payload, timeout=timeout)
        elapsed = time.time() - t0
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content", "")
        reasoning = msg.get("reasoning_content", "")
        full = content + " " + reasoning
        usage = data.get("usage", {})
        return {
            "content": content,
            "reasoning": reasoning[:300],
            "full": full,
            "time": elapsed,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
    except Exception as e:
        return {"content": f"ERROR: {e}", "reasoning": "", "full": str(e), "time": time.time() - t0, "prompt_tokens": 0, "completion_tokens": 0}

def run_suite(label):
    print(f"\n{'='*60}")
    print(f"  Running test suite: {label}")
    print(f"{'='*60}")
    results = []
    for i, test in enumerate(TESTS):
        print(f"\n[{i+1}/{len(TESTS)}] {test['name']}...")
        result = query(test["prompt"])
        passed = test["check"](result["full"])
        results.append({
            "name": test["name"],
            "passed": passed,
            "time": result["time"],
            "content": result["content"][:200],
            "reasoning": result["reasoning"][:200],
            "prompt_tokens": result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
        })
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} | {result['time']:.1f}s | tokens: {result['prompt_tokens']}+{result['completion_tokens']}")
        if result["content"]:
            print(f"  Content: {result['content'][:150]}")
        elif result["reasoning"]:
            print(f"  Reasoning: {result['reasoning'][:150]}")
    return results

def wait_for_server(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get("http://127.0.0.1:8080/health", timeout=3)
            if r.json().get("status") == "ok":
                return True
        except:
            pass
        time.sleep(2)
    return False

if __name__ == "__main__":
    print("KV Cache Quality Comparison Test")
    print("Waiting for server...")
    if not wait_for_server():
        print("ERROR: Server not available")
        sys.exit(1)
    
    vram = subprocess.check_output(
        "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader", shell=True
    ).decode().strip()
    print(f"Server ready. VRAM: {vram}")
    
    results = run_suite(sys.argv[1] if len(sys.argv) > 1 else "unknown")
    
    # Save results
    output_file = sys.argv[2] if len(sys.argv) > 2 else "/tmp/kv_quality_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    passed = sum(1 for r in results if r["passed"])
    print(f"\n{'='*60}")
    print(f"  TOTAL: {passed}/{len(results)} passed")
    print(f"{'='*60}")
