# Inferix

> A decentralized AI compute cooperative where contributors earn inference credits by donating idle GPU/CPU time — measured in FLOPs, not crypto.

Inferix is an open-source framework that allows anyone to contribute idle compute resources in exchange for AI inference credits — denominated in floating-point operations (FLOPs), not cryptocurrency.

**The core idea:** You donate idle GPU/CPU time → you earn GigaFLOP-Tokens (GFT) → you spend GFT to access AI inference. No speculation. No financial value. Just compute for compute.

---

## Motivation

AI inference is increasingly powerful but increasingly centralized. Access is gated by capital, not contribution. Meanwhile, millions of GPUs sit idle every night across the world, in different time zones, on different grids.

Inferix turns that idle capacity into a global cooperative — one where the communities that bear the cost of AI infrastructure are also empowered to benefit from it.

A secondary benefit: because contributor nodes are globally distributed across time zones, computation naturally migrates toward regions with low electricity demand and high renewable availability at any given hour — a passive energy-efficiency property that centralized data centers cannot replicate.

---

## How It Works

```
Contribute idle compute → earn GFT tokens → spend tokens on AI inference
```

Token formula:
```
T_earned = FLOPs_delivered × Hardware_Multiplier × Reliability_Factor
```

Token cost per inference:
```
T_spent = FLOPs_per_model × Output_tokens × Precision_factor
```

All tokens are:
- Denominated in GigaFLOPs (objective, hardware-agnostic)
- Non-transferable and non-tradeable by design
- Tied to the wallet that earned them

---

## Architecture

The framework has four components:

| Component | Role |
|---|---|
| **Contributor Client** | Benchmarks device, serves transformer model layers, manages token wallet |
| **Coordination Layer** | Dispatches tasks, assigns hardware multipliers, issues tokens, routes inference |
| **Token Ledger** | Append-only hash-chained log of all GFT issuance and redemption |
| **Inference Cluster** | Custom distributed pipeline (volunteer nodes) primary, centralized fallback |

### Distributed Inference Architecture

Inferix implements a custom layer-sharding system for distributed inference:

- **Primary:** Volunteer contributor nodes each serve a contiguous slice of transformer layers. Inference requests flow through the pipeline — activations pass from node to node until the final output is generated. Contributors earn tokens proportional to FLOPs delivered. Any HuggingFace-compatible transformer model can be served — LLaMA 3, Mistral, Mixtral, and others.
- **Fallback:** A centralized inference service — used when the volunteer pipeline is unavailable, ensuring reliable access at all times. Fallback integration with a specific inference provider is a community contribution opportunity (see Contributing).

This makes the token economy architecturally honest — earned tokens are backed by compute that directly contributes to real AI inference.

---

## Hardware Multiplier Tiers

| Hardware Class | Example Devices | Multiplier |
|---|---|---|
| Mobile / Edge | Smartphone SoCs, Raspberry Pi | 0.05× |
| CPU Only | Desktop / server CPUs | 0.10× |
| Entry Consumer GPU *(integrated)* | Intel UHD, AMD Radeon integrated | 0.50× |
| Mid Consumer GPU *(baseline)* | RTX 3060, RX 6700 XT | 1.00× |
| High Consumer GPU | RTX 4080, RX 7900 XTX | 2.00× |
| Prosumer GPU | RTX 4090, RTX 6000 Ada | 3.00× |
| Professional Accelerator | A40, L40S | 5.00× |
| Data Center Accelerator | A100, H100, H200 | 8.00× |

Multipliers are derived from a composite AI Performance Index (API) combining matrix multiplication throughput, memory bandwidth, and batch inference latency. Hardware changes trigger a quarantine period to prevent swap attacks.

---

## Grid-Aware Scheduling

The coordination layer implements energy-aware task scheduling using real-time grid data:

```
P_grid = w1×(1−L) + w2×(1−C) + w3×R
```

Where L = grid load, C = carbon intensity, R = renewable fraction. Tasks are preferentially routed to nodes in regions with low grid demand and high renewable generation.

---

## Paper

**Inferix: A FLOP-Based Token Framework for Decentralized AI Access**
Aslan Kose — Independent Researcher

- [arXiv preprint](https://arxiv.org/abs/XXXX.XXXXX) *(coming soon)*
- [LaTeX source](docs/paper/inferix.tex)

---

## Status

- [x] Framework design and specification complete
- [x] Academic paper drafted
- [x] Coordination Layer — FastAPI + PostgreSQL
- [x] Token Ledger — hash-chained, tamper-evident
- [x] Hardware multiplier system — 8 tiers with interpolation
- [x] Grid-aware scheduler — P_grid formula
- [x] Shard registry + pipeline scheduler
- [x] Distributed inference pipeline — custom layer sharding proven
- [x] Auto-reconnect on coordinator restart
- [x] Inference challenge verification + reliability factor
- [x] Test suite — 52 passing tests
- [x] Contributor client — cross-platform, simulation + production mode
- [x] Public beta release
- [ ] Real model weights — load HuggingFace models on volunteer nodes *(community)*
- [ ] Fallback inference integration — wire a centralized inference provider *(community)*

---

## Contributing

We welcome community contributions, particularly:

- **Real model serving** — integrate `torch` + `transformers` for production inference on volunteer nodes. The activation server already supports real model loading via HuggingFace — GPU nodes with sufficient VRAM can load and serve model layer slices today.
- **Fallback inference** — wire a centralized inference provider (e.g. Groq, Together AI, or self-hosted Ollama) into the `_fallback_inference()` function in `coordination/api/inference.py`.
- **Grid API integration** — replace mock grid data with real ENTSO-E or EIA grid operator APIs.
- **Security hardening** — formal analysis of the token issuance and challenge verification protocols.

Please open an Issue to discuss before submitting a pull request.

---

## License

MIT License

---

## Author

**Aslan Kose** — IT professional, independent researcher
GitHub: [@aslankose](https://github.com/aslankose)

*Inferix emerged from a conviction that the communities bearing the cost of AI infrastructure should also be empowered to benefit from it.*
