# nanoMegatron

A minimal distributed training framework for learning. Hand-written DDP → ZeRO-1/2/3 → TP → SP → PP → EP, all in ~2k lines, trained on a real 3.8B MoE model.

📖 **[中文版本](README_zh.md)** | 🐛 **[Pitfalls & Bug Fixes](docs/PITFALLS.md)** | 📊 **[Benchmark Log](benchmark_logs/BENCHMARK_LOG.md)**

---

## Why this exists

Production frameworks (Megatron-LM, DeepSpeed, PyTorch FSDP) are battle-tested but each has 10k–50k lines of source with heavy abstractions, async streams, and bucketing. nanoMegatron strips these away to show the **core idea** of each parallelism strategy in 100–350 lines per file.

You get a working SFT pipeline on [Phi-tiny-MoE](https://huggingface.co/microsoft/Phi-tiny-MoE-instruct) (3.8B params, 16 experts, top-2 routing) with [GSM8k](https://huggingface.co/datasets/openai/gsm8k), and real benchmark numbers on consumer GPUs.

## Implemented strategies

| File | Lines | Strategy |
|------|------:|----------|
| `ddp.py` | 31 | DDP — full model/GPU, AllReduce gradients |
| `zero.py` | 201 | ZeRO-1 / ZeRO-2 (with `post_accumulate_grad_hook`) |
| `fsdp.py` | 347 | ZeRO-3 / FSDP via `autograd.Function` (no hook deadlock), shards experts |
| `tensor_parallel.py` | 352 | TP with NCCL coalescing (1 AllReduce per layer instead of 17) |
| `sequence_parallel.py` | 177 | SP — sequence-dim sharding on top of TP |
| `pipeline_parallel.py` | 188 | PP — GPipe-style |
| `expert_parallel.py` | 231 | EP — AllToAll dispatch (production-grade) |

Plus `model.py` (Phi-tiny-MoE, 312 lines) and `trainer.py` (161 lines). **Total ~2k lines.**

## Benchmark results

### 4× NVIDIA RTX A5000 (24 GB), full 3.8B model, 32 layers

| Strategy | Memory/GPU | Throughput | Status |
|----------|-----------|------------|--------|
| DDP (fp16) | – | – | OOM (Adam fp32 copies need ~30 GB) |
| ZeRO-1 | ~22 GB peak | – | OOM at the boundary |
| **ZeRO-2** (v2 hooks) | ~12 GB | (slow*) | ✓ fits |
| **ZeRO-3** (v2, sharded experts) | ~9 GB | (slow*) | ✓ fits |
| **TP-4** | 15.5 GB | **154 tok/s** | ✓ |
| **EP-4** (AllToAll) | 21.1 GB | **284 tok/s** | ✓ |

*ZeRO-2/3 throughput is bound by per-param sync NCCL on this PCIe machine — see [PITFALLS.md](docs/PITFALLS.md#trade-offs-still-on-the-table) for the bucketing trade-off.

### TP throughput improvement after the v2 fix

| Version | NCCL calls/step | Throughput (TP-2 fwd+bwd, A5000) |
|---------|----------------:|----------------------------------|
| v1 (per-Linear SplitFunc/AllReduce) | 2158 | 79 tok/s |
| **v2 (coalesced)** | **288** | **95 tok/s** |

On 4× A5000 the same fix takes TP-4 from "slower than ZeRO" to **154 tok/s** — a 3.4× wall-clock speedup over the old V100×4 baseline of 45 tok/s, despite the A5000 having ~50% the fp16 throughput of a V100. Roughly **6.8× equivalent speedup**, all from dropping NCCL launches.

## Comparison with production frameworks

What we have, what we don't.

| Aspect | nanoMegatron | Megatron-LM | DeepSpeed | PyTorch FSDP |
|--------|--------------|-------------|-----------|--------------|
| **Lines of code** | ~2k total | ~30k+ | ~50k+ | ~10k+ |
| **TP coalesced AllReduce** | ✓ (1 per layer) | ✓ (fused QKV + async) | – | – |
| **Async TP overlap** | ✗ | ✓ (`CUDA_DEVICE_MAX_CONNECTIONS=1`) | – | – |
| **Sequence Parallel** | stub | ✓ (LayerNorm/Dropout sliced) | – | – |
| **ZeRO bucketed gradient reduction** | ✗ (per-param) | – | ✓ (flat buffer) | ✓ |
| **ZeRO async overlap with backward** | ✗ | – | ✓ (multi-stream) | ✓ |
| **CPU/NVMe offload** | ✗ | – | ✓ (ZeRO-Infinity) | ✓ (CPUOffload) |
| **FlatParameter / contiguous shards** | ✗ | – | ✓ | ✓ |
| **ZeRO-3 backward prefetch** | ✗ | – | ✓ | ✓ |
| **MoE EP via AllToAll** | ✓ | ✓ (Megatron-MoE) | ✓ (DeepSpeed-MoE) | – |
| **MoE load balancing (aux loss + capacity factor)** | ✗ | ✓ | ✓ | – |
| **GPipe pipeline** | ✓ | ✓ | – | – |
| **1F1B pipeline** | ✗ | ✓ | – | – |
| **Auto-mixed precision policy** | manual fp32 copies | – | ✓ | ✓ (`MixedPrecision` policy) |
| **FlashAttention** | uses PyTorch SDPA (auto) | dispatches explicitly | – | – |

**The big trade-off**: with ~2k lines we get the **memory savings** of these frameworks but not the **throughput tricks** (bucketing, async, multi-stream prefetch). On a normal NVLink box those tricks give a 5–10× speedup on top of the algorithm itself; that's ~80% of what makes production frameworks production-ready.

**When to use what**:

| Goal | Use |
|------|-----|
| Learn how each parallelism algorithm works under the hood | nanoMegatron |
| Train production models 1B–100B | DeepSpeed or PyTorch FSDP |
| Train 100B+ with TP+PP+SP | Megatron-LM, NeMo |
| Sparse MoE at scale | DeepSpeed-MoE, Megatron-MoE |
| Inference for serving | vLLM, TensorRT-LLM, SGLang |

## Quick start

```bash
pip install -r requirements.txt

# Run unit tests
python -m pytest tests/test_all.py -v

# Single GPU sanity check
python scripts/train.py --config configs/default.yaml

# 4-GPU TP (fastest in our benchmark for fitting + throughput trade-off)
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/default.yaml --strategy tp --tp_size 4

# 4-GPU EP (highest throughput overall)
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/default.yaml --strategy ep --ep_size 4
```

> **If NCCL hangs on the first AllReduce**: your machine may have a PCIe P2P bug. Set `NCCL_P2P_DISABLE=1`. We hit this on RTX A5000s — see [PITFALLS.md](docs/PITFALLS.md#nccl-p2p-deadlock-on-pcie-machines).

## Architecture

| Component | Value |
|-----------|-------|
| Total params | 3.8B |
| Active params (per token) | 1.1B |
| Layers | 32 |
| Hidden size | 4096 |
| Heads | 16 (Q) / 4 (KV, GQA) |
| Head dim | 128 |
| Experts | 16 (top-2) |
| Expert intermediate | 448 (SlimMoE compression) |
| Vocab | 32064 |

## Project structure

```
nanoMegatron/
├── nano_megatron/
│   ├── model.py                  # Phi-tiny-MoE (loads HF weights as-is)
│   ├── data.py                   # GSM8k loader + chat formatting
│   ├── trainer.py                # Training loop
│   └── parallel/
│       ├── ddp.py                # DDP wrapper
│       ├── zero.py               # ZeRO-1/2 (v2: backward hook)
│       ├── fsdp.py               # ZeRO-3 (v2: shards experts)
│       ├── tensor_parallel.py    # TP (v2: coalesced NCCL)
│       ├── sequence_parallel.py  # SP
│       ├── pipeline_parallel.py  # PP (GPipe)
│       └── expert_parallel.py    # EP (AllToAll)
├── scripts/
│   ├── train.py                  # Training entry
│   ├── eval.py                   # GSM8k accuracy eval
│   ├── profile_tp.py             # NCCL call counter for TP
│   ├── benchmark_deepspeed.py    # DeepSpeed comparison
│   ├── benchmark_fsdp.py         # PyTorch FSDP comparison
│   └── run_4gpu_benchmarks.sh    # 4-GPU benchmark suite
├── configs/
│   ├── default.yaml              # Full training (1000 steps)
│   ├── benchmark.yaml            # 50-step benchmark
│   ├── benchmark_4gpu.yaml       # 10-step 4-GPU benchmark
│   └── benchmark_small.yaml      # 8-layer config for ZeRO verification
├── docs/
│   ├── PITFALLS.md               # 🐛 every NaN, OOM, and deadlock we hit
│   └── PITFALLS_zh.md            # 🐛 中文版
├── benchmark_logs/
│   └── BENCHMARK_LOG.md          # Raw experiment log
├── tests/
│   └── test_all.py               # 20+ unit tests
├── README.md                     # This file
└── README_zh.md                  # 中文版
```

## Documentation

- 🐛 **[Pitfalls & Bug Fixes](docs/PITFALLS.md)** — every NaN, OOM, deadlock, and silently-wrong-gradient we hit, with the fix
- 📊 **[Benchmark Log](benchmark_logs/BENCHMARK_LOG.md)** — raw experiment output and methodology
- 🇨🇳 **[中文版本](README_zh.md)**

## Design philosophy

- **Minimal** — no wandb, no fancy CLI, no custom dataloader abstractions
- **Transparent** — every parallelism strategy in one file, heavily commented
- **Educational** — small enough to read in an afternoon, runnable on a desktop
- **Honest** — documents what we get right *and* what we trade off vs production frameworks

## References

- [DeepSpeed ZeRO paper](https://arxiv.org/abs/1910.02054)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (TP foundations)
- [PyTorch FSDP paper](https://arxiv.org/abs/2304.11277)
- [GPipe paper](https://arxiv.org/abs/1811.06965) (PP)
- [GShard / Switch Transformer](https://arxiv.org/abs/2006.16668) (MoE EP)
- [Tiny-FSDP](https://github.com/liangyuwang/Tiny-FSDP) (FSDP minimal reference)
