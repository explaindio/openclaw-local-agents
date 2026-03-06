# 14 OpenClaw AI Agents Running 100% Locally on a Single RTX 3090

### Zero cloud APIs. Zero subscriptions. Full reasoning on one GPU from 2020.

---

A home computer from 2019 with an RTX 3090 GPU from 2020 — running **Qwen3.5** models released this year. No cloud API. No subscription. Everything runs on one machine sitting under a desk.

Without the optimizations in this setup, every request with a 200K-token context would take **5.8 minutes** just to process the prompt on the 9B model. With KV cache offloading and agent routing, the same request starts generating in **52 milliseconds**.

> **That's 6,667× faster. Same GPU. Same model. Just smarter caching.**

---

## The Problem

We have **14 AI agents** with very different needs:
- Some need **deep reasoning** (strategy, compliance, code architecture)
- Some need **raw speed** (trend scouting, short copy, voice workflows)

Running all 14 on a single large model wastes GPU on simple tasks. Running them all on a small model degrades quality on complex ones.

## The Solution: Two Models, One GPU

We run **two LLM servers simultaneously** on a single RTX 3090 (24 GB), each optimized for its role:

| | 🧠 Quality Server | ⚡ Speed Server |
|--|---|---|
| **Model** | Qwen3.5-9B | Qwen3.5-2B |
| **Port** | 8090 | 8091 |
| **Thinking** | ✅ ON (budgeted) | ❌ OFF |
| **Slots** | 1 (serial) | 4 (parallel) |
| **Context** | 262K tokens | 262K tokens |
| **Speed** | ~100 tokens/sec | ~210 tokens/sec |
| **VRAM** | ~10.4 GB | ~4.0 GB |
| **Total VRAM** | **14.3 GB** | **10.3 GB free** |

---

## What is "Thinking" and Why Budget It?

Qwen3.5 models can **reason internally** before answering. The model generates hidden "thinking tokens" inside `<think>...</think>` tags, then produces the actual answer.

```
User: "Review this ad for compliance issues"

Model thinking (hidden):  ← these tokens cost time but improve quality
  1. Check claims... "50% off" needs substantiation
  2. Check disclaimers... missing terms link
  3. FTC guidelines require...

Model answer (visible):
  "Two issues found: 1) The '50% off' claim needs a reference price..."
```

**The tradeoff**: More thinking = better answers, but slower and uses more of `max_tokens`.

**Thinking budget** controls this per-agent:

| Budget | Behavior | Use case |
|--------|----------|----------|
| `0` | No thinking at all | Fast tasks: routing, short copy |
| `512` | Brief reasoning | Content writing, SEO |
| `1024` | Moderate reasoning | Compliance, analytics |
| `2048` | Deep reasoning | Orchestration, architecture |

> ⚠️ **Important**: `max_tokens` includes BOTH thinking AND answer tokens.
> Formula: **max_tokens = thinking_budget + expected_output**

---

## Agent Assignment

### 🧠 Quality Server (9B, port 8090) — 7 agents

These agents handle tasks where **quality matters more than speed**. The 9B model thinks before answering, with each agent getting a thinking budget matched to its complexity.

| # | Agent | Thinking Budget | Output | max_tokens | What it does |
|---|-------|----------------|--------|------------|--------------|
| 1 | **main** (orchestrator) | 2048 | 2048 | 4096 | Routes work, coordinates all agents |
| 4 | **campaign-strategist** | 1024 | 3072 | 4096 | Campaign strategy, plans, positioning |
| 7 | **seo-article-writer** | 512 | 6144 | 8192 | Long-form SEO articles |
| 10 | **compliance-qa** | 1024 | 2048 | 3072 | Policy and compliance review |
| 11 | **analytics-optimizer** | 512 | 2048 | 3072 | Performance analysis, optimizations |
| 12 | **ui_engineer** | 1024 | 8192 | 9216 | Frontend/UI code generation |
| 13 | **backend_architect** | 2048 | 8192 | 10240 | Backend design, APIs, system code |

> 💡 **Why code agents get larger budgets**: Code generation needs more thinking to plan the approach, consider edge cases, and reason about architecture. Output budgets are also larger — a single component can be 200-400 lines (~3000-6000 tokens). A 4096-token limit would cut a component mid-function.

These 7 agents **queue on 1 slot** — one request at a time. The orchestrator naturally serializes work.

### ⚡ Speed Server (2B, port 8091) — 7 agents

These agents handle tasks where **speed matters more than depth**. No thinking overhead — instant responses.

| # | Agent | Thinking | max_tokens | What it does |
|---|-------|----------|------------|--------------|
| 2 | **trend-scout** | OFF | 2048 | Trends, market signals |
| 3 | **offer-hunter** | OFF | 2048 | Competitive deals, offers |
| 5 | **shortscript-writer** | OFF | 2048 | Short-form scripts, hooks |
| 6 | **post-writer** | OFF | 2048 | Social posts, short copy |
| 8 | **voice-agent** | OFF | 1024 | Voice workflows |
| 9 | **ugc-video-agent** | OFF | 2048 | UGC and video concepts |
| 14 | **agile_dev** | OFF | 4096 | Bug fixes, small code patches |

> 💡 **Why `agile_dev` stays on the 2B server**: It handles small, scoped patches — "fix this null check", "add this field". For full features or architecture, `backend_architect` on the 9B takes over. 2B is fast (210 t/s) which keeps the dev loop tight for small fixes.

These 7 agents run on **4 parallel slots** — up to 4 can execute simultaneously.

---

## How Agent Offloading Works

```
┌──────────────────────────────────────────────────┐
│                   OpenClaw Gateway                │
│                                                   │
│  Incoming request → Which agent? → Route to port  │
└──────────┬───────────────────────┬────────────────┘
           │                       │
     Complex task              Simple task
           │                       │
           ▼                       ▼
┌─────────────────────┐  ┌─────────────────────┐
│  🧠 9B Server        │  │  ⚡ 2B Server        │
│  Port 8090           │  │  Port 8091           │
│  1 slot (serial)     │  │  4 slots (parallel)  │
│  Thinking: ON        │  │  Thinking: OFF       │
│  ~100 tok/s          │  │  ~210 tok/s          │
│                      │  │                      │
│  orchestrator        │  │  trend-scout         │
│  campaign-strategist │  │  offer-hunter        │
│  seo-article-writer  │  │  shortscript-writer  │
│  compliance-qa       │  │  post-writer         │
│  analytics-optimizer │  │  voice-agent         │
│  ui_engineer         │  │  ugc-video-agent     │
│  backend_architect   │  │  agile_dev           │
└─────────────────────┘  └─────────────────────┘
           │                       │
           └───────┬───────────────┘
                   ▼
          ┌────────────────┐
          │  RTX 3090 GPU  │
          │  24 GB VRAM    │
          │  14.3 GB used  │
          │  10.3 GB free  │
          └────────────────┘
```

**Why this works**:
- The 2B model is **2× faster** than the 9B and handles 4 requests in parallel
- Simple tasks (short copy, trend scanning) don't benefit from a bigger model
- The 9B model focuses its capacity on tasks that actually need reasoning
- Both models share the same GPU — total VRAM is only 14.3 GB out of 24 GB available

---

## Per-Request Thinking Control

Each agent sets its own thinking budget via the API:

```bash
# Compliance agent (9B) — needs deep thinking
POST http://server:8090/v1/chat/completions
{
  "reasoning_budget": 1024,    ← up to 1024 tokens of thinking
  "max_tokens": 3072           ← 1024 thinking + 2048 answer
}

# Post writer (2B) — no thinking needed
POST http://server:8091/v1/chat/completions
{
  "reasoning_budget": 0,       ← no thinking at all
  "max_tokens": 512            ← pure answer output
}
```

When the thinking budget is reached, the model is **forced to stop thinking** and immediately start producing the answer. The thinking it already did still helps — it stays in the model's context, so even partial thinking improves answer quality.

---

## Resource Summary

| Resource | Value |
|----------|-------|
| **GPU** | RTX 3090, 24 GB VRAM |
| **VRAM used** | 14.3 GB (60%) |
| **VRAM free** | 10.3 GB (headroom for KV cache growth) |
| **System RAM** | 64 GB (only ~14 GB used by servers + OS) |
| **Context window** | 262K tokens per server |
| **KV cache** | q8_0 (8-bit, saves ~50% VRAM vs default) |
| **Total agents** | 14 |
| **Models loaded** | 2 (simultaneously) |
