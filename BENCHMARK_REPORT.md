# Qwen3.5 Max Context Benchmark Report

**Hardware**: NVIDIA RTX 3090 (24,576 MiB VRAM) · CUDA 13.1 · Driver 590.48.01
**Software**: llama.cpp v8148 · CUDA arch 86 · Flash Attention enabled
**Date**: February 25, 2026

---

## Models Tested

| | Qwen3.5-27B | Qwen3.5-35B-A3B |
|--|-------------|------------------|
| **Architecture** | Dense transformer | Mixture of Experts (MoE) |
| **Total Parameters** | 26.9B (all active) | 35B total, ~3B active per token |
| **Quantization** | Q4_K_M (4-bit k-quant medium) | Q4_K_M (4-bit k-quant medium) |
| **File Size** | 16 GB | 20 GB |
| **Native Context** | 262,144 tokens | 262,144 tokens |
| **Thinking Model** | Yes (reasoning chain before answer) | Yes (reasoning chain before answer) |

---

## Max Stable Context Results

| Model | KV Cache | Max Stable Context | VRAM at Max | OOM At |
|-------|----------|--------------------|-------------|--------|
| **27B Dense** | f16 (16-bit) | **115,000 tokens** | 24,074 MiB | 117,500 |
| **27B Dense** | q8_0 (8-bit) | **210,000 tokens** | 24,032 MiB | 215,000 |
| **35B-A3B MoE** | f16 (16-bit) | **145,000 tokens** | 24,018 MiB | 150,000 |
| **35B-A3B MoE** | q8_0 (8-bit) | **250,000 tokens** | 24,054 MiB | 255,000 |

### Context Gain from q8_0 KV

| Model | f16 Max | q8_0 Max | Multiplier |
|-------|---------|----------|------------|
| 27B Dense | 115,000 | 210,000 | **1.83×** |
| 35B-A3B MoE | 145,000 | 250,000 | **1.72×** |

---

## VRAM Scaling (27B Dense, f16 KV)

| Context | VRAM Used | Free |
|---------|-----------|------|
| 65,536 | 20,970 MiB | 3,606 MiB |
| 80,000 | 21,882 MiB | 2,694 MiB |
| 90,000 | 22,506 MiB | 2,070 MiB |
| 100,000 | 23,130 MiB | 1,446 MiB |
| 110,000 | 23,754 MiB | 822 MiB |
| 115,000 | 24,074 MiB | 502 MiB |
| 117,500 | **OOM** | — |

KV cache cost: ~**62 MiB per 1,000 tokens** (f16)

## VRAM Scaling (27B Dense, q8_0 KV)

| Context | VRAM Used | Free |
|---------|-----------|------|
| 131,072 | 21,228 MiB | 3,348 MiB |
| 200,000 | 23,672 MiB | 904 MiB |
| 210,000 | 24,032 MiB | 544 MiB |
| 215,000 | **OOM** | — |

KV cache cost: ~**35 MiB per 1,000 tokens** (q8_0)

## VRAM Scaling (35B-A3B MoE, f16 KV)

| Context | VRAM Used | Free |
|---------|-----------|------|
| 32,768 | 21,822 MiB | 2,754 MiB |
| 55,000 | 22,258 MiB | 2,318 MiB |
| 70,000 | 22,552 MiB | 2,024 MiB |
| 90,000 | 22,942 MiB | 1,634 MiB |
| 100,000 | 23,138 MiB | 1,438 MiB |
| 120,000 | 23,528 MiB | 1,048 MiB |
| 140,000 | 23,918 MiB | 658 MiB |
| 145,000 | 24,018 MiB | 558 MiB |
| 150,000 | **OOM** | — |

KV cache cost: ~**19 MiB per 1,000 tokens** (f16). Much lower than 27B due to MoE architecture — KV cache only covers attention layers, which are shared and smaller.

## VRAM Scaling (35B-A3B MoE, q8_0 KV)

| Context | VRAM Used | Free |
|---------|-----------|------|
| 200,000 | 23,390 MiB | 1,186 MiB |
| 230,000 | 23,788 MiB | 788 MiB |
| 250,000 | 24,054 MiB | 522 MiB |
| 255,000 | **OOM** | — |

KV cache cost: ~**13 MiB per 1,000 tokens** (q8_0)

---

## VRAM Breakdown

| Component | 27B Dense | 35B-A3B MoE |
|-----------|-----------|-------------|
| Model weights (Q4_K_M) | ~16.5 GB | ~20.2 GB |
| System/Xorg overhead | ~164 MiB | ~164 MiB |
| Available for KV cache | ~7.5 GB | ~3.5 GB |
| KV cost per 1K tokens (f16) | ~62 MiB | ~19 MiB |
| KV cost per 1K tokens (q8_0) | ~35 MiB | ~13 MiB |

**Key insight**: Despite using 3.7 GB more VRAM for weights, the 35B-A3B MoE achieves **longer context** because its KV cache per token is **3.3× smaller** than the 27B dense model.

---

## Quality Comparison: f16 vs q8_0 KV

Ran 5 tests with deterministic settings (`temperature=0`, `seed=42`, `max_tokens=2000`):

| Test | 27B f16 | 27B q8_0 | 35B f16 | 35B q8_0 |
|------|---------|----------|---------|----------|
| Math Reasoning | ✅ 36.3s | ✅ 36.2s | ✅ 13.7s | ✅ 18.9s |
| Factual Recall | ✅ 28.4s | ✅ 28.6s | ✅ 12.2s | ✅ 8.0s |
| Logic Puzzle | ✅ 59.1s | ✅ 59.1s | ✅ 18.7s | ✅ 19.1s |
| Code Generation | ✅ 59.3s | ✅ 59.0s | ✅ 4.0s | ✅ 4.1s |
| NIAH (5K context) | ✅ 12.0s | ✅ 12.1s | ✅ 4.1s | ✅ 4.1s |
| **Score** | **5/5** | **5/5** | **5/5** | **5/5** |
| **Total Time** | **195.0s** | **194.9s** | **52.7s** | **54.2s** |

**Result**: Zero quality degradation from f16 → q8_0. Identical answers, identical timing.

---

## NIAH (Needle In A Haystack) Validation

| Model | Needle Found | Time | Prompt Tokens |
|-------|-------------|------|---------------|
| 27B Dense | ✅ PINEAPPLE-774 | 12.0s | 7,024 |
| 35B-A3B MoE | ✅ PINEAPPLE-774 | 4.1s | 7,024 |

Needle was placed at the beginning of a 10K-token prompt filled with filler text about papermaking history. Both models correctly retrieved it.

---

## Performance

| Metric | 27B Dense | 35B-A3B MoE |
|--------|-----------|-------------|
| Prompt processing | ~410 tok/s | ~830 tok/s |
| Token generation | ~35 tok/s | ~100 tok/s |
| Model load time | ~25s | ~35s |
| Typical response (thinking ON) | 30-60s | 4-20s |
| Typical response (/nothink) | 3-15s | 2-5s |

The MoE model is **~3× faster** because only ~3B parameters are active per token (vs all 27B for the dense model).

---

## What is q8_0?

**q8_0** is a GGML quantization format for the KV (Key-Value) cache:

- **q8** = 8-bit quantization (each value stored in 8 bits instead of 16)
- **_0** = variant zero — the simplest scheme. It uses only a **scale factor** (one float16 per block of 32 values) with no additional offset or zero-point. The `0` means "zero extra parameters beyond the mandatory scale."

Comparison with other variants:
| Format | Bits | Extra Params | Accuracy | Size |
|--------|------|-------------|----------|------|
| f16 | 16 | None (full precision) | Best | 2 bytes/value |
| q8_0 | 8 | Scale only | Near-lossless | ~1 byte/value |
| q4_0 | 4 | Scale only | Good | ~0.5 bytes/value |
| q4_1 | 4 | Scale + min | Better | ~0.56 bytes/value |

Note: The model itself uses **Q4_K_M** quantization (a different, more advanced k-quant system for weights). The KV cache quantization (q8_0) is separate and only affects the runtime attention cache.

---

## VRAM is Pre-Allocated

All VRAM is allocated at server startup. The KV cache for the full context window is reserved immediately, regardless of actual usage. This means:

- A 100-token request uses the **same VRAM** as a 200K-token request
- No risk of OOM during inference — if the server starts, it handles any request up to the configured context
- VRAM stays constant for the lifetime of the server process

---

## Commands Reference

```bash
# 27B Dense — f16 KV (max 115K context)
llama-server -m Qwen3.5-27B-Q4_K_M.gguf -c 115000 -ngl 99 --flash-attn on --host 0.0.0.0 --port 8080

# 27B Dense — q8_0 KV (max 210K context)
llama-server -m Qwen3.5-27B-Q4_K_M.gguf -c 210000 -ngl 99 -ctk q8_0 -ctv q8_0 --flash-attn on --host 0.0.0.0 --port 8080

# 35B-A3B MoE — f16 KV (max 145K context)
llama-server -m Qwen3.5-35B-A3B-Q4_K_M.gguf -c 145000 -ngl 99 --flash-attn on --host 0.0.0.0 --port 8080

# 35B-A3B MoE — q8_0 KV (max 250K context)
llama-server -m Qwen3.5-35B-A3B-Q4_K_M.gguf -c 250000 -ngl 99 -ctk q8_0 -ctv q8_0 --flash-attn on --host 0.0.0.0 --port 8080
```

Flag reference:
- `-c` — context size in tokens
- `-ngl 99` — offload all layers to GPU
- `-ctk q8_0` — KV cache key type (q8_0 = 8-bit)
- `-ctv q8_0` — KV cache value type (q8_0 = 8-bit)
- `--flash-attn on` — enable Flash Attention (required for large contexts)
- `--host 0.0.0.0` — listen on all interfaces (for remote access)
