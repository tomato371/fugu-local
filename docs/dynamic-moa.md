# Dynamic Mixture-of-Agents in Fugu-Local

This note explains the core idea behind Fugu-Local: a **Conductor LLM that decides,
per query, how much machinery to spend**. It is written to be read alongside
`fugu_local.py`.

## Static vs. dynamic MoA

Classic (static) Mixture-of-Agents runs a **fixed** set of proposer models on *every*
query and merges them with a fixed aggregator. It is robust but wasteful: a trivial
question pays the same cost as a hard one.

Fugu-Local is **dynamic**. A lightweight **Conductor** model first reads the question
and emits a JSON execution plan — *is one model enough? which proposers, how many, how
many rounds? is web search or RAG needed?* Cheap questions stay cheap; hard questions
escalate.

## The pipeline

```
question
   │
   ▼
Conductor ──JSON plan──►  single model?  ──►  answer ──► Critic: good enough?
   │                                                        │ yes → done
   │                                                        │ no  → escalate
   └────────── council? ─────────────────────────────────► │
                                                            ▼
                            Proposers (A/B/C/D personas) → Aggregator merges
                                                            │
                                                    Enough? ─ yes → done
                                                            └ no  → bounded extra round
```

1. **Plan.** The Conductor produces a JSON plan (single vs. council, selected proposers,
   round budget, and routing flags for web search / RAG / office documents).
2. **Fast path.** Simple questions are answered by a single model — no MoA overhead.
3. **Escalate.** A **Critic** inspects a weak single answer and escalates to a council.
4. **Recurse.** If the council's answer is still insufficient, bounded additional rounds
   run (capped by `MAX_ROUNDS`).

## Personas → local models

The Conductor selects abstract proposer **personas** (`Proposer A`…`D`), which are then
resolved to concrete local models (e.g. `gpt-oss:20b`, `qwen3-coder:30b`, `gemma4:26b`,
plus a math/physics specialist). Routing (Conductor + Critic) uses a small, fast
`qwen3:4b` to keep VRAM residency low. The aggregator merges proposer answers.

## Engineering under an 8 GB VRAM budget

The design is shaped by running on a laptop RTX 4060 (8 GB):

- **Native `/api/chat`, not `/v1`.** The OpenAI-compatible `/v1` endpoint ignores
  `num_ctx` and tries to allocate each model's maximum context (e.g. qwen3's 262 144),
  which overflows the KV cache and crashes `llama-server`. Talking to Ollama's native
  `/api/chat` over `urllib` lets `options.num_ctx` be pinned per request to a safe value.
- **Sequential by default.** On 8 GB, proposers run sequentially so only one large model
  is resident at a time (large models offload to the 48 GB RAM). A high-VRAM profile
  enables parallel proposers.
- **Zero third-party dependencies** for the core path — only the standard library.

## Why it matters

Dynamic routing turns MoA from "always expensive" into "expensive only when it helps":
the Critic/escalation loop spends a council on hard questions while a single fast model
handles easy ones. This is the same cost/quality lever that production LLM systems pull —
here, made explicit and measurable, entirely on local hardware.
