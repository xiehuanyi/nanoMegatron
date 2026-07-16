# nanoMegatron

A minimal distributed training framework for learning. Hand-written DDP, ZeRO-1/2/3, TP, SP, PP, EP — same algorithms as Megatron-LM / DeepSpeed / PyTorch FSDP, in ~2k lines.

📖 **[中文版本](README_zh.md)** | 🐛 **[Pitfalls & history](docs/PITFALLS.md)** | 📊 **[Benchmark log](benchmark_logs/BENCHMARK_LOG.md)**

---

## What's in the box

A working SFT pipeline on [Phi-tiny-MoE](https://huggingface.co/microsoft/Phi-tiny-MoE-instruct) (3.8B params, 16 experts, top-2 routing) with [GSM8k](https://huggingface.co/datasets/openai/gsm8k), and one parallelism strategy per file:

| File | Strategy |
|------|----------|
| `parallel/ddp.py` | DDP — full model on every GPU, AllReduce gradients |
| `parallel/zero.py` | ZeRO-1 / ZeRO-2 with flat buffers, sharded Adam, ReduceScatter, and AllGather overlap |
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
| **ZeRO bucketed reduction** | ✓ (flat buffer) | ✓ | ✓ | ✓ |
| **ZeRO async overlap** | partial (backward RS + next-forward AG) | ✓ | ✓ | ✓ |
| **CPU/NVMe offload** | ✗ | – | ✓ (ZeRO-Infinity) | ✓ (CPUOffload) |
| **FlatParameter / contiguous shards** | ✓ (ZeRO-2) | ✓ | ✓ | ✓ |
| **ZeRO-3 backward prefetch** | ✗ | – | ✓ | ✓ |
| **MoE EP via AllToAll** | ✓ | ✓ (Megatron-MoE) | ✓ (DeepSpeed-MoE) | – |
| **MoE load balancing (aux loss + capacity)** | ✗ | ✓ | ✓ | – |
| **GPipe / 1F1B pipeline** | GPipe only | both | – | – |
| **Mixed precision policy** | manual fp32 master copies | – | ✓ | ✓ |
| **FlashAttention** | via PyTorch SDPA | explicit dispatch | – | – |

**The trade-off**: the core algorithms and the first layer of production optimizations are here, but fused kernels, selective recomputation, communication scheduling, and allocator integration are still much simpler than Megatron Core. The benchmark below tracks that remaining gap rather than calling an implementation "done" merely because it runs.

**When to use what**:

| Goal | Use |
|------|-----|
| Learn how each parallelism algorithm works under the hood | nanoMegatron |
| Train production models 1B–100B | DeepSpeed or PyTorch FSDP |
| Train 100B+ with TP+PP+SP | Megatron-LM, NeMo |
| Sparse MoE at scale | DeepSpeed-MoE, Megatron-MoE |
| Inference for serving | vLLM, TensorRT-LLM, SGLang |

## Benchmarks

### Qwen3-0.6B vs Megatron Core: optimization log

This is the current parity target: 2× V100 PCIe, FP16, sequence length 1024,
micro/global batch 1/2, no recomputation, portable/unfused attention, and a real
Adam update every step. Megatron Core is pinned to `v0.15.3`. Formal acceptance
requires the median of three independent jobs to reach at least 95% throughput
and at most 105% peak HBM under the same topology.

| Iteration | nano tok/s | Megatron tok/s | Speed ratio | nano / Megatron peak HBM | What changed |
|---|---:|---:|---:|---:|---|
| D0 | 3647.8 | 5818.2 | 0.627 | 1.147 | Whole-bucket AllReduce, whole-parameter owners, serial Broadcast |
| D1 | 4770.0 | 5812.2 | 0.821 | 1.328 | dtype-flat buffers, balanced tensor shards, backward ReduceScatter, bucket AllGather |
| D2* | 4974.5 | 5813.2 | 0.856 | 1.272 | fused QKV and SwiGLU input GEMMs, fused AdamW |
| D3* | 5020.1 | 5815.9 | 0.863 | 1.272 | launch AllGather in forward order and wait from module pre-hooks |
| D4 (running) | - | - | - | - | Megatron-style math attention, RMSNorm/RoPE/CE alignment, exact bucket packing |

`D0` and `D1` are three-job medians. `D2` and `D3` are one-job exploratory
runs and are not promoted to formal results yet. Raw summaries and the exact
protocol live in `benchmark_logs/qwen3_0.6b/`; the maintained feature and
acceptance matrix is in `docs/MEGATRON_PARITY.md`.

What the attempts taught us:

1. Replacing owner-based AllReduce/Broadcast with balanced ReduceScatter/AllGather
   was the largest speed win: +30.8%. It also exposed a memory regression because
   the educational implementation keeps full model-dtype gradient buffers plus
   local FP32 master gradients alive.
2. Fusing Q/K/V and gate/up projections removed duplicate saved inputs and GEMM
   launches. It added another 4.3% throughput and reduced sampled HBM by 642 MiB,
   but peak allocated memory is still 13.12 GiB.
3. Parameter-gather overlap is correct and measurable, but only added 0.9% on
   this two-GPU PCIe topology. The remaining stable gap is about 56 ms/step and
   3.1 GiB of sampled HBM.
4. D4 removes two benchmark mismatches that favored nanoMegatron: it predicts
   all 1024 labels from a 1025-token stream, and computes the distributed gradient
   norm/finite check every step just like the Megatron run.
5. D4 also replaces PyTorch math SDPA with Megatron's unfused
   `baddbmm -> fp16 softmax -> bmm` graph, uses `torch.nn.RMSNorm`, caches RoPE,
   uses a custom in-place cross entropy without returning full logits, and packs
   the distributed optimizer into the same 40M-element, overshoot-after-add
   buckets. This produces 11 buckets instead of 13 for Qwen3-0.6B.
6. After D4, the main known differences are communication scheduling and fusion:
   gradient division and fp16-to-fp32 main-grad copies are separate kernels,
   parameter AllGather waits use Python `Work.wait()`, and there is no Megatron
   multi-tensor optimizer/overflow infrastructure. Activation checkpointing is
   not counted as a fix while the Megatron side has recomputation disabled.

Jobs: D1 `48899170`, `48901884`, `48901885`; D2 `48908022`; D3 `48908122`;
D4 `48977678` (`48975002` was cancelled before start to remove the debug partition pin).

Two head-to-head runs on Ibex, full 3.8B Phi-tiny-MoE (32 layers, 16 experts top-2), `seq_len=96`, `batch_size=1`, `grad_accum=1`, gradient checkpointing on, fp16, 10 steps. nanoMegatron, DeepSpeed 0.18.9, PyTorch FSDP all run on the same checkpoint with the same script (`scripts/run_v100_benchmark.sh` / `scripts/run_4gpu_benchmarks.sh`).

### 4× Tesla V100 SXM2-32GB (NVLink)

PyTorch 2.8.0 + CUDA 12.8 + DeepSpeed 0.18.9, NVLink enabled. Memory from `torch.cuda.max_memory_allocated()`.

#### Memory (peak GB / GPU)

| Strategy        | nanoMegatron        | DeepSpeed           | PyTorch FSDP        |
|-----------------|:-------------------:|:-------------------:|:-------------------:|
| **ZeRO-1**      | OOM¹                | **26.3 GB** ✓       | –                   |
| **ZeRO-2**      | **27.6 GB** ✓       | **26.3 GB** ✓       | –                   |
| **ZeRO-3 / FSDP** | **20.7 GB** ✓     | (slow²)             | **18.8 GB** ✓       |
| **TP-4**        | **15.5 GB** ✓       | n/a                 | n/a                 |
| **EP-4**        | **21.1 GB** ✓       | n/a                 | n/a                 |

#### Throughput (tok/s)

| Strategy          | nanoMegatron        | DeepSpeed           | PyTorch FSDP        |
|-------------------|:-------------------:|:-------------------:|:-------------------:|
| **ZeRO-1**        | OOM                 | **31 tok/s** ✓      | –                   |
| **ZeRO-2**        | **31 tok/s** ✓      | **32 tok/s** ✓      | –                   |
| **ZeRO-3 / FSDP** | **18 tok/s** ✓      | (slow²)             | **32 tok/s** ✓      |
| **TP-4**          | **106 tok/s** ✓     | n/a                 | n/a                 |
| **EP-4**          | **207 tok/s** ✓     | n/a                 | n/a                 |

¹ nanoMegatron ZeRO-1 keeps all fp16 grads alive during backward (no bucketing) → 22+ GB peak → OOM once CUDA context is added. DeepSpeed's flat-buffer design avoids this.
² DeepSpeed ZeRO-3 can't finish one step in 30 min on this hardware. Its per-param AllGather/ReduceScatter overhead dominates when every Linear in a 3.8B MoE model triggers its own collective. Memory sharding works (10–21 GB asymmetric across ranks) but throughput is too low.

### 4× NVIDIA RTX A5000 (24 GB) — `NCCL_P2P_DISABLE=1`

This is the older PCIe-SHM run kept here for comparison; same configuration except nano numbers come from `torch.cuda.max_memory_allocated()` instead of `nvidia-smi`.

#### Memory (peak GB / GPU)

| Strategy | nanoMegatron | DeepSpeed | PyTorch FSDP |
|----------|:------------:|:---------:|:------------:|
| **DDP** (fp16) | OOM | – | – |
| **ZeRO-1** | OOM (~22 GB peak) | **20.6 GB** ✓ | – |
| **ZeRO-2** | **12.0 GB** ✓ | (similar to ZeRO-1) | – |
| **ZeRO-3 / FSDP** | **9.6 GB** ✓ | **10.8 GB** ✓ | **18.8 GB** ✓ |
| **TP-4** | **15.5 GB** ✓ | n/a (DS has no TP) | n/a |
| **EP-4** (AllToAll) | **21.1 GB** ✓ | n/a | n/a |

#### Throughput (tok/s)

| Strategy | nanoMegatron | DeepSpeed | PyTorch FSDP |
|----------|:------------:|:---------:|:------------:|
| **ZeRO-2** | (slow*) | (hung**) | – |
| **ZeRO-3 / FSDP** | (slow*) | (hung**) | **24 tok/s** ✓ |
| **TP-4** | **154 tok/s** ✓ | n/a | n/a |
| **EP-4** | **284 tok/s** ✓ | n/a | n/a |

\* nanoMegatron ZeRO-2/3 use per-param sync hooks (no bucketing) — on this PCIe-SHM machine that means ~2k NCCL calls per backward, each with ~100μs latency. Throughput is bound by NCCL launch cost, not algorithm. On a NVLink machine with bucketing this would be 5–10× faster.

\*\* DeepSpeed's first step never completes on this machine in 4+ minutes (we tried both fp16 and bf16 with bucket sizes set to 500 MB). PyTorch FSDP completes happily because it wraps at the layer granularity and its multi-stream prefetch is more PCIe-SHM friendly. Memory numbers above were captured from `nvidia-smi` after DeepSpeed engine init.

**What both runs tell you:**

1. **nanoMegatron ZeRO-2 matches DeepSpeed ZeRO-2 on throughput** (31 vs 32 tok/s on V100). With gradient bucketing (the `_GradBucket` class in `zero.py`), the ~1957 per-param NCCL calls collapse into ~15 bucket-level AllReduces, putting us on par with DeepSpeed's flat-buffer reducer. Memory is slightly higher (27.6 vs 26.3 GB) because our per-param ownership bucketing has a small overhead vs DeepSpeed's contiguous-grad buffer.

2. **EP-4 is 6× faster than ZeRO on V100** (207 tok/s vs 31 tok/s). TP-4 is 3× faster (106 tok/s). These strategies have minimal NCCL calls (EP: 2 AllToAll/layer, TP: 2 AllReduce/layer) vs ZeRO's per-bucket-AllReduce + broadcast. For MoE models, expert parallelism is the clear winner.

3. **nanoMegatron ZeRO-3 is 1.7× slower than PyTorch FSDP** (18 vs 32 tok/s) at similar memory (20.7 vs 18.8 GB). The gap is wrapping granularity: we do per-Linear AllGather (~1700 calls/forward), FSDP wraps per-DecoderLayer (~32 calls/forward). Closing this gap requires layer-level `FlatParameter` — the same optimization that makes FSDP production-grade.

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
