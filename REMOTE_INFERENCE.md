# Qwen3.5-27B Remote Inference Guide

## Server Info

| | |
|--|--|
| **Model** | Qwen3.5-27B-Q4_K_M (4-bit quantized, 16.7GB) |
| **Parameters** | 26.9 billion |
| **KV Cache** | q8_0 (8-bit) |
| **Context Window** | 180,000 tokens |
| **Trained Context** | 262,144 tokens |
| **VRAM Used** | ~22,950 MiB / 24,576 MiB |
| **Base URL** | `http://192.168.1.7:8080/v1` |
| **API Key** | Not required (use any string like `none`) |
| **API** | OpenAI-compatible |
| **GPU** | RTX 3090 24GB |
| **Thinking** | OFF (`--reasoning-budget 0`) |

---

## Quick Test

```bash
curl http://192.168.1.7:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-27b",
    "messages": [{"role": "user", "content": "Hello, what model are you?"}],
    "max_tokens": 200
  }'
```

---

## Thinking Mode (Important!)

Qwen3.5 is a **thinking model**. **Thinking is controlled at the SERVER level only — it cannot be toggled per-request.**

The server is currently running with `--reasoning-budget 0` (thinking OFF). To change this, the server must be restarted with different flags.

| Mode | Server Flag | Behavior |
|------|------------|----------|
| Thinking ON | *(default, no flag)* | Internal reasoning chain → final answer. Set `max_tokens` to **2000+**. |
| **Thinking OFF** | `--reasoning-budget 0` | **Direct answers only, no reasoning chain. Current setting.** |

> **⚠️ Warning**: The `/nothink` system message does NOT work with Qwen3.5 in llama.cpp — it causes Jinja template errors. The ONLY way to control thinking is the `--reasoning-budget` server flag at startup.

### Server commands
```bash
# Thinking OFF (current):
llama-server -m Qwen3.5-27B-Q4_K_M.gguf -c 180000 -ngl 99 \
  -ctk q8_0 -ctv q8_0 --flash-attn on --reasoning-budget 0 \
  --host 0.0.0.0 --port 8080

# Thinking ON (restart with this to enable):
llama-server -m Qwen3.5-27B-Q4_K_M.gguf -c 180000 -ngl 99 \
  -ctk q8_0 -ctv q8_0 --flash-attn on \
  --host 0.0.0.0 --port 8080
```

---

## Python (openai library)

```bash
pip install openai
```

### Basic usage (thinking OFF — current server config)
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://192.168.1.7:8080/v1",
    api_key="none"
)

response = client.chat.completions.create(
    model="qwen3.5-27b",
    messages=[
        {"role": "user", "content": "Explain quicksort in simple terms"}
    ],
    max_tokens=500,
    temperature=0.7
)

print(response.choices[0].message.content)
```

> **Note**: If the server is restarted with thinking ON (no `--reasoning-budget 0`), set `max_tokens` to 2000+ and check `response.choices[0].message.reasoning_content` for the thinking chain.

---

## Available Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (main) |
| `POST` | `/v1/completions` | Text completion |
| `GET`  | `/v1/models` | List loaded models |
| `GET`  | `/health` | Server health check |

---

## Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `model` | any string | Server ignores this, always uses loaded model |
| `max_tokens` | — | **500 for thinking OFF, 2000+ for thinking ON** |
| `temperature` | 1.0 | 0.0 = deterministic, 0.7 = good default |
| `top_p` | 1.0 | Nucleus sampling |
| `stream` | false | Set true for streaming responses |
| `seed` | — | Set for reproducible outputs |

---

## Performance

| Metric | Value |
|--------|-------|
| Prompt processing | ~410 tokens/sec |
| Token generation | ~35 tokens/sec |
| Typical response (thinking OFF) | 1-5 seconds |
| Typical response (thinking ON) | 30-60 seconds |
