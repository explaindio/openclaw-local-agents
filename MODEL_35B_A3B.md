# Qwen3.5-35B-A3B — Full Model Card & Benchmark

## Model Identity

| | |
|--|--|
| **Model** | Qwen3.5-35B-A3B |
| **Architecture** | Mixture of Experts (MoE) |
| **Total Parameters** | 35 billion |
| **Active Parameters** | ~3 billion per token (~8.6%) |
| **Experts** | Multiple expert FFN blocks, top-k routing per token |
| **Quantization** | Q4_K_M (4-bit k-quant medium) |
| **File** | `Qwen3.5-35B-A3B-Q4_K_M.gguf` |
| **File Size** | 20 GB |
| **Native Max Context** | 262,144 tokens |
| **Thinking Model** | Yes — reasons internally before answering |

---

## Download

**Source**: [Hugging Face — unsloth/Qwen3.5-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF)

```bash
# Direct download (20 GB)
wget -c \
  "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-Q4_K_M.gguf" \
  -O Qwen3.5-35B-A3B-Q4_K_M.gguf
```

**Stored at**: `/home/code10/qwen3_5/models/Qwen3.5-35B-A3B-Q4_K_M.gguf`

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

## Why MoE Gets More Context

The 35B-A3B MoE model has **larger weights** (20GB vs 16GB) but achieves **longer context** than the 27B dense model. This is because:

1. **KV cache only covers attention layers** — the expert FFN layers don't contribute to KV cache
2. MoE has fewer attention parameters relative to total parameters
3. KV cost is **~19 MiB per 1K tokens** (f16) vs 62 MiB for the 27B dense = **3.3× more efficient**

| | 27B Dense | 35B-A3B MoE |
|--|-----------|-------------|
| Weights VRAM | ~16.5 GB | ~20.2 GB |
| Free for KV | ~7.5 GB | ~3.5 GB |
| KV per 1K tokens (f16) | 62 MiB | 19 MiB |
| Max context (f16) | 115K | 145K |
| Max context (q8_0) | 210K | 250K |

Despite having half the free VRAM, MoE still wins because each token's KV footprint is 3.3× smaller.

---

## Max Context Benchmark (RTX 3090, 24GB)

### f16 KV Cache (Default)

| Context | VRAM Used | VRAM Free | Status |
|---------|-----------|-----------|--------|
| 32,768 | 21,822 MiB | 2,754 MiB | ✅ |
| 45,000 | 22,062 MiB | 2,514 MiB | ✅ |
| 55,000 | 22,258 MiB | 2,318 MiB | ✅ |
| 70,000 | 22,552 MiB | 2,024 MiB | ✅ |
| 90,000 | 22,942 MiB | 1,634 MiB | ✅ |
| 100,000 | 23,138 MiB | 1,438 MiB | ✅ |
| 120,000 | 23,528 MiB | 1,048 MiB | ✅ |
| 140,000 | 23,918 MiB | 658 MiB | ✅ |
| **145,000** | **24,018 MiB** | **558 MiB** | **✅ MAX STABLE** |
| 150,000 | — | — | ❌ OOM |

**Max stable (f16 KV)**: **145,000 tokens**
**KV cost**: ~19 MiB per 1,000 tokens

### q8_0 KV Cache (8-bit)

| Context | VRAM Used | VRAM Free | Status |
|---------|-----------|-----------|--------|
| 200,000 | 23,390 MiB | 1,186 MiB | ✅ |
| 230,000 | 23,788 MiB | 788 MiB | ✅ |
| **250,000** | **24,054 MiB** | **522 MiB** | **✅ MAX STABLE (startup)** |
| 255,000 | — | — | ❌ OOM |

**Max stable (q8_0 KV)**: **250,000 tokens** (startup), **~220,000 tokens** recommended for safe inference with FA scratch
**KV cost**: ~13 MiB per 1,000 tokens
**Context gain vs f16**: **1.72×**

---

## Quality: f16 vs q8_0 KV

Tested with `temperature=0`, `seed=42`, `max_tokens=2000`:

| Test | f16 | q8_0 |
|------|-----|------|
| Math Reasoning | ✅ 13.7s | ✅ 18.9s |
| Factual Recall | ✅ 12.2s | ✅ 8.0s |
| Logic Puzzle | ✅ 18.7s | ✅ 19.1s |
| Code Generation | ✅ 4.0s | ✅ 4.1s |
| NIAH (5K context) | ✅ 4.1s | ✅ 4.1s |
| **Score** | **5/5** | **5/5** |
| **Total Time** | **52.7s** | **54.2s** |

**Zero quality degradation from f16 → q8_0.**

---

## NIAH (Needle In A Haystack)

- **Needle**: `"The secret launch code is PINEAPPLE-774."`
- **Context**: 10,000 tokens (7,024 prompt tokens)
- **Result**: ✅ Found correctly
- **Time**: 3.2s (2.4× faster than 27B dense)

---

## Performance

| Metric | Value |
|--------|-------|
| Prompt processing | ~830 tokens/sec |
| Token generation | ~100 tokens/sec |
| Model load time | ~35 seconds |
| Typical response (thinking ON) | 4-20 seconds |
| Typical response (thinking OFF) | 1-3 seconds |

The MoE model is **~3× faster** than the 27B dense because only ~3B parameters are active per token.

---

## Server Commands

### Production (recommended — leaves inference headroom)

```bash
export LD_LIBRARY_PATH="/path/to/llama.cpp/build/bin:$LD_LIBRARY_PATH"

# WITH THINKING (default) — model reasons step-by-step before answering
llama-server \
  -m /path/to/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -c 220000 \
  -ngl 99 \
  -ctk q8_0 \
  -ctv q8_0 \
  --flash-attn on \
  --host 0.0.0.0 \
  --port 8080

# WITHOUT THINKING — direct answers, much faster
llama-server \
  -m /path/to/models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -c 220000 \
  -ngl 99 \
  -ctk q8_0 \
  -ctv q8_0 \
  --flash-attn on \
  --reasoning-budget 0 \
  --host 0.0.0.0 \
  --port 8080
```

### Maximum context (startup stable, may OOM under heavy FA compute)

```bash
llama-server \
  -m Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -c 250000 \
  -ngl 99 \
  -ctk q8_0 -ctv q8_0 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080
```

### f16 KV (default, lower but no KV quantization)

```bash
llama-server \
  -m Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -c 130000 \
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
| Thinking ON | *(default, no flag needed)* | Internal reasoning chain → final answer | ~100 tok/s, 4-20s responses |
| Thinking OFF | `--reasoning-budget 0` | Direct answer only, no reasoning | ~100 tok/s, 1-3s responses |

### With thinking ON (default)
```bash
llama-server -m Qwen3.5-35B-A3B-Q4_K_M.gguf -c 220000 -ngl 99 \
  -ctk q8_0 -ctv q8_0 --flash-attn on --host 0.0.0.0 --port 8080
```
Response includes `reasoning_content` (thinking chain) and `content` (final answer). Set `max_tokens` to **2000+** to allow room for the reasoning chain.

### With thinking OFF
```bash
llama-server -m Qwen3.5-35B-A3B-Q4_K_M.gguf -c 220000 -ngl 99 \
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
  "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-Q4_K_M.gguf" \
  -O models/Qwen3.5-35B-A3B-Q4_K_M.gguf

# 4. Run server (adjust -c for your GPU's VRAM)
export LD_LIBRARY_PATH="$(pwd)/build/bin:$LD_LIBRARY_PATH"

# Thinking ON:
./build/bin/llama-server \
  -m models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -c 220000 -ngl 99 \
  -ctk q8_0 -ctv q8_0 \
  --flash-attn on \
  --host 0.0.0.0 --port 8080

# Thinking OFF (add --reasoning-budget 0):
./build/bin/llama-server \
  -m models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -c 220000 -ngl 99 \
  -ctk q8_0 -ctv q8_0 \
  --flash-attn on \
  --reasoning-budget 0 \
  --host 0.0.0.0 --port 8080

# 5. Test
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```

---

## vs 27B Dense (Quick Comparison)

| | 27B Dense | 35B-A3B MoE |
|--|-----------|-------------|
| File size | 16 GB | 20 GB |
| Speed | ~35 tok/s | ~100 tok/s |
| Max context (q8_0) | 210K (180K safe) | 250K (220K safe) |
| KV per 1K tokens | 35 MiB | 13 MiB |
| Quality score | 5/5 | 5/5 |
| Best for | Max reasoning quality | Speed + long context |
