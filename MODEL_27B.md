# Qwen3.5-27B — Full Model Card & Benchmark

## Model Identity

| | |
|--|--|
| **Model** | Qwen3.5-27B |
| **Architecture** | Dense transformer (all parameters active per token) |
| **Total Parameters** | 26.9 billion |
| **Active Parameters** | 26.9 billion (100%) |
| **Quantization** | Q4_K_M (4-bit k-quant medium) |
| **File** | `Qwen3.5-27B-Q4_K_M.gguf` |
| **File Size** | 16 GB |
| **Native Max Context** | 262,144 tokens |
| **Thinking Model** | Yes — reasons internally before answering |
| **Vocab Size** | 248,320 tokens |
| **Embedding Dim** | 5,120 |

---

## Download

**Source**: [Hugging Face — unsloth/Qwen3.5-27B-GGUF](https://huggingface.co/unsloth/Qwen3.5-27B-GGUF)

```bash
# Direct download (16 GB)
wget -c \
  "https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q4_K_M.gguf" \
  -O Qwen3.5-27B-Q4_K_M.gguf
```

**Stored at**: `/home/code10/qwen3_5/models/Qwen3.5-27B-Q4_K_M.gguf`

---

## Inference Stack

### llama.cpp

| | |
|--|--|
| **Version** | v8148 (commit `244641955`) |
| **Repo** | https://github.com/ggerganov/llama.cpp |
| **Binary** | `/home/code10/qwen3_5/llama.cpp/build/bin/llama-server` |
| **API** | OpenAI-compatible (`/v1/chat/completions`) |

### Build from Source

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with CUDA (adjust CUDA_ARCHITECTURES for your GPU)
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)

# Verify
./build/bin/llama-server --version
```

**CUDA_ARCHITECTURES by GPU**:
| GPU | Architecture |
|-----|-------------|
| RTX 3090 / 3080 / 3070 | `86` |
| RTX 4090 / 4080 / 4070 | `89` |
| A100 | `80` |
| H100 | `90` |

### Build Environment (Tested)

| | |
|--|--|
| **OS** | Ubuntu 24.04.4 LTS |
| **GCC** | 13.3.0 |
| **CMake** | 3.28.3 |
| **CUDA Toolkit** | 12.0 (V12.0.140) |
| **NVIDIA Driver** | 590.48.01 |
| **GPU** | NVIDIA GeForce RTX 3090 (24,576 MiB) |
| **Compute Capability** | 8.6 |

### Flash Attention

**Status**: Enabled (`--flash-attn on`)

Flash Attention is compiled into llama.cpp when built with `GGML_CUDA=ON`. It runs as a fused CUDA kernel (`launch_fattn<256, 8, 8>`) that processes Q/K/V attention in a single pass without materializing the full attention matrix. This is critical for large contexts — without it, attention would require O(n²) VRAM for the attention scores.

**How it's coupled**: Flash Attention is part of the GGML CUDA backend (`libggml-cuda.so`). When `--flash-attn on` is set, the server selects the fused FA kernel instead of the standard multi-step attention kernel. No external FlashAttention library is needed — it's built directly into llama.cpp's CUDA code.

**Important**: Flash attention scratch buffers need ~1-1.5GB of temporary VRAM during compute. Pre-allocating the full KV cache with no VRAM headroom will cause OOM during inference even if the server starts successfully.

---

## Max Context Benchmark (RTX 3090, 24GB)

### f16 KV Cache (Default)

| Context | VRAM Used | VRAM Free | Status |
|---------|-----------|-----------|--------|
| 65,536 | 20,970 MiB | 3,606 MiB | ✅ |
| 80,000 | 21,882 MiB | 2,694 MiB | ✅ |
| 90,000 | 22,506 MiB | 2,070 MiB | ✅ |
| 100,000 | 23,130 MiB | 1,446 MiB | ✅ |
| 110,000 | 23,754 MiB | 822 MiB | ✅ |
| **115,000** | **24,074 MiB** | **502 MiB** | **✅ MAX STABLE** |
| 117,500 | — | — | ❌ OOM |

**Max stable (f16 KV)**: **115,000 tokens**
**KV cost**: ~62 MiB per 1,000 tokens

### q8_0 KV Cache (8-bit)

| Context | VRAM Used | VRAM Free | Status |
|---------|-----------|-----------|--------|
| 131,072 | 21,228 MiB | 3,348 MiB | ✅ |
| 200,000 | 23,672 MiB | 904 MiB | ✅ |
| **210,000** | **24,032 MiB** | **544 MiB** | **✅ MAX STABLE (startup)** |
| 215,000 | — | — | ❌ OOM |

**Max stable (q8_0 KV)**: **210,000 tokens** (startup), **~180,000 tokens** (safe for actual inference with FA scratch buffers)
**KV cost**: ~35 MiB per 1,000 tokens
**Context gain vs f16**: **1.83×**

### VRAM Breakdown

| Component | Size |
|-----------|------|
| Model weights (Q4_K_M) | ~16.5 GB |
| System / Xorg | ~164 MiB |
| Available for KV cache | ~7.5 GB |
| Flash Attention scratch | ~1-1.5 GB (needed during compute) |

---

## Quality: f16 vs q8_0 KV

Tested with `temperature=0`, `seed=42`, `max_tokens=2000`:

| Test | f16 | q8_0 |
|------|-----|------|
| Math Reasoning | ✅ 36.3s | ✅ 36.2s |
| Factual Recall | ✅ 28.4s | ✅ 28.6s |
| Logic Puzzle | ✅ 59.1s | ✅ 59.1s |
| Code Generation | ✅ 59.3s | ✅ 59.0s |
| NIAH (5K context) | ✅ 12.0s | ✅ 12.1s |
| **Score** | **5/5** | **5/5** |

**Zero quality degradation from f16 → q8_0.**

---

## NIAH (Needle In A Haystack)

- **Needle**: `"The secret launch code is PINEAPPLE-774."`
- **Context**: 10,000 tokens (7,024 prompt tokens)
- **Result**: ✅ Found correctly
- **Time**: 12.0s

---

## Performance

| Metric | Value |
|--------|-------|
| Prompt processing | ~410 tokens/sec |
| Token generation | ~35 tokens/sec |
| Model load time | ~25 seconds |
| Typical response (thinking ON) | 30-60 seconds |
| Typical response (thinking OFF) | 1-5 seconds |

---

## Server Commands

### Production (recommended — 180K with inference headroom)

```bash
export LD_LIBRARY_PATH="/path/to/llama.cpp/build/bin:$LD_LIBRARY_PATH"

# WITH THINKING (default) — model reasons step-by-step before answering
llama-server \
  -m /path/to/models/Qwen3.5-27B-Q4_K_M.gguf \
  -c 180000 \
  -ngl 99 \
  -ctk q8_0 \
  -ctv q8_0 \
  --flash-attn on \
  --host 0.0.0.0 \
  --port 8080

# WITHOUT THINKING — direct answers, much faster
llama-server \
  -m /path/to/models/Qwen3.5-27B-Q4_K_M.gguf \
  -c 180000 \
  -ngl 99 \
  -ctk q8_0 \
  -ctv q8_0 \
  --flash-attn on \
  --reasoning-budget 0 \
  --host 0.0.0.0 \
  --port 8080
```

**Currently running on 192.168.1.7:8080** with `--reasoning-budget 0` (thinking OFF), confirmed stable under real inference load.

### Maximum context (startup stable, may OOM under heavy load)

```bash
llama-server \
  -m Qwen3.5-27B-Q4_K_M.gguf \
  -c 210000 \
  -ngl 99 \
  -ctk q8_0 -ctv q8_0 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080
```

### f16 KV (default, lower context but no quantization on cache)

```bash
llama-server \
  -m Qwen3.5-27B-Q4_K_M.gguf \
  -c 115000 \
  -ngl 99 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080
```

### Flag Reference

| Flag | Meaning |
|------|---------|
| `-m` | Model file path |
| `-c` | Context size in tokens |
| `-ngl 99` | Offload all layers to GPU |
| `-ctk q8_0` | KV cache key type: 8-bit quantized |
| `-ctv q8_0` | KV cache value type: 8-bit quantized |
| `--flash-attn on` | Enable Flash Attention |
| `--reasoning-budget 0` | Disable thinking (default: -1 = unlimited thinking) |
| `--host 0.0.0.0` | Listen on all interfaces (for remote access) |
| `--port 8080` | Server port |

---

## Thinking Mode

Qwen3.5 is a thinking model — it can reason step-by-step internally before answering.

**Thinking is controlled at the SERVER level, NOT per-request.** You choose at startup whether the server runs with thinking on or off. There is no way to toggle it per-request via the API.

| Mode | Server Flag | Behavior | Speed |
|------|------------|----------|-------|
| Thinking ON | *(default, no flag needed)* | Internal reasoning chain → final answer | ~35 tok/s, 30-60s responses |
| Thinking OFF | `--reasoning-budget 0` | Direct answer only, no reasoning | ~35 tok/s, 1-5s responses |

### With thinking ON (default)
```bash
llama-server -m Qwen3.5-27B-Q4_K_M.gguf -c 180000 -ngl 99 \
  -ctk q8_0 -ctv q8_0 --flash-attn on --host 0.0.0.0 --port 8080
```
Response includes `reasoning_content` (thinking chain) and `content` (final answer). Set `max_tokens` to **2000+** to allow room for the reasoning chain.

### With thinking OFF
```bash
llama-server -m Qwen3.5-27B-Q4_K_M.gguf -c 180000 -ngl 99 \
  -ctk q8_0 -ctv q8_0 --flash-attn on --reasoning-budget 0 --host 0.0.0.0 --port 8080
```
Response contains only `content` — direct answer with no reasoning chain. `max_tokens` of 500 is usually enough.

> **⚠️ Note**: The `/nothink` system message does NOT work with Qwen3.5 in llama.cpp — it causes Jinja template errors. The ONLY way to disable thinking is the `--reasoning-budget 0` server flag.

---

## Reproduce on Another Machine

```bash
# 1. Install dependencies
sudo apt install cmake nvidia-cuda-toolkit build-essential

# 2. Clone and build llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="YOUR_ARCH" -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# 3. Download model (no auth needed from unsloth)
mkdir -p models
wget -c \
  "https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q4_K_M.gguf" \
  -O models/Qwen3.5-27B-Q4_K_M.gguf

# 4. Run server (adjust -c for your GPU's VRAM)
export LD_LIBRARY_PATH="$(pwd)/build/bin:$LD_LIBRARY_PATH"

# Thinking ON:
./build/bin/llama-server \
  -m models/Qwen3.5-27B-Q4_K_M.gguf \
  -c 180000 -ngl 99 \
  -ctk q8_0 -ctv q8_0 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080

# Thinking OFF (add --reasoning-budget 0):
./build/bin/llama-server \
  -m models/Qwen3.5-27B-Q4_K_M.gguf \
  -c 180000 -ngl 99 \
  -ctk q8_0 -ctv q8_0 \
  --flash-attn on \
  --reasoning-budget 0 \
  --host 0.0.0.0 --port 8080

# 5. Test
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```
