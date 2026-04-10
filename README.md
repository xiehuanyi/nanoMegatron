# nanoMegatron

A minimal distributed training framework for learning. Hand-written DDP, ZeRO-1/2/3, TP, SP, PP, EP — same algorithms as Megatron-LM / DeepSpeed / PyTorch FSDP, in ~2k lines.

📖 **[中文版本](README_zh.md)** | 🐛 **[Pitfalls & history](docs/PITFALLS.md)** | 📊 **[Benchmark log](benchmark_logs/BENCHMARK_LOG.md)**

---

## What's in the box

A working SFT pipeline on [Phi-tiny-MoE](https://huggingface.co/microsoft/Phi-tiny-MoE-instruct) (3.8B params, 16 experts, top-2 routing) with [GSM8k](https://huggingface.co/datasets/openai/gsm8k), and one parallelism strategy per file:

| File | Strategy |
|------|----------|
| `parallel/ddp.py` | DDP — full model on every GPU, AllReduce gradients |
| `parallel/zero.py` | ZeRO-1 / ZeRO-2 (backward `post_accumulate_grad_hook` to free non-owner grads) |
| `parallel/fsdp.py` | ZeRO-3 / FSDP via `autograd.Function`, shards every Linear including MoE experts |
| `parallel/tensor_parallel.py` | TP — Q/K/V/O coalesced into one SplitFunc + one AllReduce per layer; same for MoE |
| `parallel/sequence_parallel.py` | SP — sequence-dim sharding on top of TP |
| `parallel/pipeline_parallel.py` | PP — GPipe schedule |
| `parallel/expert_parallel.py` | EP — AllToAll dispatch (production-grade) |

## Comparison with production frameworks

| Aspect | nanoMegatron | Megatron-LM | DeepSpeed | PyTorch FSDP |
|--------|--------------|-------------|-----------|--------------|
| **Source size** | ~2k lines | ~30k+ | ~50k+ | ~10k+ |
| **TP coalesced AllReduce** | ✓ (1/layer) | ✓ (fused QKV + async) | – | – |
| **Async TP overlap with GEMM** | ✗ | ✓ (`CUDA_DEVICE_MAX_CONNECTIONS=1`) | – | – |
| **Sequence Parallel** | stub | ✓ (LayerNorm/Dropout sliced) | – | – |
| **ZeRO bucketed reduction** | ✗ (per-param) | – | ✓ (flat buffer) | ✓ |
| **ZeRO async overlap** | ✗ | – | ✓ (multi-stream) | ✓ |
| **CPU/NVMe offload** | ✗ | – | ✓ (ZeRO-Infinity) | ✓ (CPUOffload) |
| **FlatParameter / contiguous shards** | ✗ | – | ✓ | ✓ |
| **ZeRO-3 backward prefetch** | ✗ | – | ✓ | ✓ |
| **MoE EP via AllToAll** | ✓ | ✓ (Megatron-MoE) | ✓ (DeepSpeed-MoE) | – |
| **MoE load balancing (aux loss + capacity)** | ✗ | ✓ | ✓ | – |
| **GPipe / 1F1B pipeline** | GPipe only | both | – | – |
| **Mixed precision policy** | manual fp32 master copies | – | ✓ | ✓ |
| **FlashAttention** | via PyTorch SDPA | explicit dispatch | – | – |

**The trade-off**: with ~2k lines we get the **memory savings** of these frameworks, but not the **throughput tricks** (bucketing, async overlap, multi-stream prefetch). On a normal NVLink box those tricks give a 5–10× speedup on top of the algorithm; that's most of what makes production frameworks production-ready.

**When to use what**:

| Goal | Use |
|------|-----|
| Learn how each parallelism algorithm works under the hood | nanoMegatron |
| Train production models 1B–100B | DeepSpeed or PyTorch FSDP |
| Train 100B+ with TP+PP+SP | Megatron-LM, NeMo |
| Sparse MoE at scale | DeepSpeed-MoE, Megatron-MoE |
| Inference for serving | vLLM, TensorRT-LLM, SGLang |

## Benchmarks

**4× NVIDIA RTX A5000 (24 GB)**, full 3.8B Phi-tiny-MoE, `seq_len=96`, `batch_size=1`, `grad_accum=1`, gradient checkpointing on, `NCCL_P2P_DISABLE=1`. nanoMegatron numbers are head-to-head against the official frameworks.

### Memory (peak GB / GPU)

| Strategy | nanoMegatron | DeepSpeed | PyTorch FSDP |
|----------|:------------:|:---------:|:------------:|
| **DDP** (fp16) | OOM | – | – |
| **ZeRO-1** | OOM (~22 GB peak) | **20.6 GB** ✓ | – |
| **ZeRO-2** | **12.0 GB** ✓ | (similar to ZeRO-1) | – |
| **ZeRO-3 / FSDP** | **9.6 GB** ✓ | **10.8 GB** ✓ | **18.8 GB** ✓ |
| **TP-4** | **15.5 GB** ✓ | n/a (DS has no TP) | n/a |
| **EP-4** (AllToAll) | **21.1 GB** ✓ | n/a | n/a |

### Throughput (tok/s)

| Strategy | nanoMegatron | DeepSpeed | PyTorch FSDP |
|----------|:------------:|:---------:|:------------:|
| **ZeRO-2** | (slow*) | (hung**) | – |
| **ZeRO-3 / FSDP** | (slow*) | (hung**) | **24 tok/s** ✓ |
| **TP-4** | **154 tok/s** ✓ | n/a | n/a |
| **EP-4** | **284 tok/s** ✓ | n/a | n/a |

\* nanoMegatron ZeRO-2/3 use per-param sync hooks (no bucketing) — on this PCIe-SHM machine that means ~2k NCCL calls per backward, each with ~100μs latency. Throughput is bound by NCCL launch cost, not algorithm. On a NVLink machine with bucketing this would be 5–10× faster.

\*\* DeepSpeed's first step never completes on this machine in 4+ minutes (we tried both fp16 and bf16 with bucket sizes set to 500 MB). PyTorch FSDP completes happily because it wraps at the layer granularity and its multi-stream prefetch is more PCIe-SHM friendly. Memory numbers above were captured from `nvidia-smi` after DeepSpeed engine init.

**Three things this table tells you:**

1. **Memory: we match the official frameworks.** nanoMegatron ZeRO-3 (9.6 GB) is actually a bit more compact than PyTorch FSDP (18.8 GB) on the same model — FSDP keeps fp32 master copies for *all* params, we only do that for the sharded ones, and our backward hooks free non-owner grads more aggressively. DeepSpeed ZeRO-3 (10.8 GB) is the closest match. Our ZeRO-1 OOMs where DeepSpeed's fits, because their flat-buffer design avoids the per-param peak we hit.

2. **Throughput: TP and EP are competitive; ZeRO-2/3 are bottlenecked by lack of bucketing.** The sweet spot for nanoMegatron on this hardware is TP-4 / EP-4. The ZeRO numbers are limited by *engineering* (no bucketing, no async overlap), not algorithm. See [PITFALLS.md](docs/PITFALLS.md#trade-offs-still-on-the-table) for the gap to production-framework optimizations.

3. **DeepSpeed is more sensitive to PCIe-SHM than PyTorch FSDP.** Even DeepSpeed couldn't complete a step on this NCCL-P2P-disabled machine, while PyTorch FSDP did. This is hardware-specific — on a normal NVLink box DeepSpeed is fast.

## Quick start

```bash
pip install -r requirements.txt

# Run unit tests
python -m pytest tests/test_all.py -v

# Single GPU sanity check
python scripts/train.py --config configs/default.yaml

# 4-GPU TP
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/default.yaml --strategy tp --tp_size 4

# 4-GPU EP (highest throughput in our benchmark)
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/default.yaml --strategy ep --ep_size 4
```

> **If NCCL hangs on the first AllReduce**: your machine may have a PCIe P2P bug (we hit it on RTX A5000s). Set `NCCL_P2P_DISABLE=1`. See [PITFALLS.md](docs/PITFALLS.md#nccl-p2p-deadlock-on-pcie-machines).

## Model

[Phi-tiny-MoE](https://huggingface.co/microsoft/Phi-tiny-MoE-instruct): 3.8B total params, 1.1B active per token, 32 layers, 16 experts (top-2), GQA (16 Q heads / 4 KV heads), hidden 4096. Implementation in `nano_megatron/model.py` loads HF weights as-is.

## Project structure

```
nanoMegatron/
├── nano_megatron/
│   ├── model.py              # Phi-tiny-MoE
│   ├── data.py               # GSM8k loader
│   ├── trainer.py            # Training loop
│   └── parallel/             # one strategy per file
├── scripts/
│   ├── train.py              # Training entry
│   ├── eval.py               # GSM8k eval
│   ├── profile_tp.py         # NCCL call counter
│   └── benchmark_*.py        # DeepSpeed / FSDP comparison scripts
├── configs/                  # YAML configs
├── docs/
│   ├── PITFALLS.md           # 🐛 every NaN/OOM/deadlock + history
│   └── PITFALLS_zh.md
├── benchmark_logs/
│   └── BENCHMARK_LOG.md      # raw experiment output
├── tests/
└── README.md
```

## Documentation

- 🐛 **[docs/PITFALLS.md](docs/PITFALLS.md)** — every NaN, OOM, deadlock, and silently-wrong-gradient we hit, with the fix and the diagnostic story
- 📊 **[benchmark_logs/BENCHMARK_LOG.md](benchmark_logs/BENCHMARK_LOG.md)** — raw benchmark output and methodology
- 🇨🇳 **[README_zh.md](README_zh.md)** — Chinese version

## Design philosophy

- **Minimal** — no wandb, no fancy CLI, no custom dataloader abstractions
- **Transparent** — every parallelism strategy in one file, heavily commented
- **Educational** — small enough to read in an afternoon, runnable on a desktop
- **Honest** — documents what we trade off vs production frameworks

## References

- [DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [PyTorch FSDP paper](https://arxiv.org/abs/2304.11277)
- [GPipe](https://arxiv.org/abs/1811.06965)
- [GShard / Switch Transformer](https://arxiv.org/abs/2006.16668)
- [Tiny-FSDP](https://github.com/liangyuwang/Tiny-FSDP)
