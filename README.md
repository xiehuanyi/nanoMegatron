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

### nanoMegatron vs Megatron Core feature parity

`DONE` means the current implementation has a tested parity path. `PARTIAL`
means the basic algorithm exists but production semantics, composition, scale,
or performance are not fully aligned. `TODO` means the feature is not
implemented. The detailed acceptance criteria and experiment IDs are maintained
in [`docs/MEGATRON_PARITY.md`](docs/MEGATRON_PARITY.md).

| Area | Megatron Core feature | nanoMegatron status | Current evidence / remaining gap |
|---|---|---|---|
| Model | Dense GPT/Qwen decoder | PARTIAL | Qwen3-0.6B structure, shifted labels, fixed tokens, loss, and gradients are aligned for the DP2 benchmark |
| Precision | FP32/FP16/BF16 mixed precision | PARTIAL | FP16 master weights work; precision policy is not yet uniform across every parallel strategy |
| Data parallel | DDP with bucketed gradient reduction | DONE | Hand-written DDP and flat gradient buffers |
| Distributed optimizer | Sharded optimizer states and main parameters | PARTIAL | DP2 throughput/HBM pass D6; implementation is not yet validated at larger DP sizes |
| Grad overlap | Backward gradient ReduceScatter overlap | PARTIAL | Async post-accumulate hooks are implemented; stream scheduling is simpler than Megatron |
| Param overlap | Next-forward parameter AllGather overlap | PARTIAL | Forward-order launch and module pre-hook waits work; dependencies still use Python `Work.wait()` |
| Tensor parallel | Column/row parallel linear layers | PARTIAL | TP paths exist, but TP+SP composition and communication overlap are incomplete |
| Sequence parallel | Sequence-dimension activation sharding | PARTIAL | Stub/limited paths only |
| Pipeline parallel | GPipe and 1F1B schedules | PARTIAL | GPipe exists; 1F1B, interleaving, and communication overlap are missing |
| Virtual pipeline | Interleaved pipeline stages | TODO | No virtual pipeline schedule |
| Context parallel | Long-sequence context sharding | PARTIAL | Qwen3 has zigzag sequence shards plus synchronous K/V AllGather and backward ReduceScatter; ring overlap and TP/PP/DP composition are missing |
| Expert parallel | MoE AllToAll dispatch | PARTIAL | Dispatch works; production token permutation and grouped GEMM are not aligned |
| Expert tensor parallel | Independent ETP dimension | TODO | No ETP topology |
| Recomputation | Full and selective activation recompute | PARTIAL | Whole-layer checkpointing exists; selective core-attention recompute is missing |
| Attention | Local/unfused, SDPA, Flash/TE backends | PARTIAL | D6 uses aligned local/unfused math; no Megatron TE/Flash backend parity yet |
| Core kernels | Fused QKV and SwiGLU projections | DONE | QKV and gate/up GEMMs are coalesced |
| Norm/RoPE | Fused RMSNorm and RoPE | PARTIAL | RMSNorm semantics align; RoPE is cached but remains a PyTorch pointwise graph |
| Cross entropy | Vocab-parallel fused cross entropy | PARTIAL | DP custom in-place CE is aligned; TP vocab-parallel backward still needs validation |
| Optimizer | Fused Adam and multi-tensor utilities | PARTIAL | nano uses fused torch AdamW; Megatron uses TE FusedAdam in D6; nano main-grad copy/check remains separate |
| Transformer Engine | TE layers and FP8 recipes | TODO | TE 2.9 is installed only for the Megatron reference optimizer; nano has no TE TransformerLayer or FP8 path |
| CUDA Graphs | Captured steady-state execution | TODO | No graph capture |
| MoE training | Router aux/z loss and capacity policies | TODO | Router balancing and token-drop policies are not implemented |
| Grouped GEMM | Batched expert GEMMs | TODO | Experts still execute through Python/module loops |
| Checkpointing | Distributed and async checkpoints | PARTIAL | Basic rank-0 save exists; sharded resharding and async save are missing |
| Distributed RNG | TP/PP-aware RNG trackers | PARTIAL | Basic deterministic seeds exist; no full model-parallel RNG tracker |
| Training loop | Gradient accumulation | DONE | Accumulation and communication suppression are supported |
| Reliability | Global grad norm, overflow, and clipping | PARTIAL | Main paths work, but behavior is not uniform across every strategy |
| Data pipeline | Packed sequences and indexed datasets | TODO | Uses ordinary dataset/DataLoader paths |
| Observability | Timers, MFU, memory, communication traces | PARTIAL | JSON throughput, HBM, software manifest, correctness, and profiler traces exist; MFU/NCCL telemetry is incomplete |
| Resilience | Fault tolerance and straggler detection | TODO | Not implemented |

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

This is the current parity target: 2× V100-SXM2 with NVLink, FP16, sequence length 1024,
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
| D4 | 8435.2 | 7610.8 | 1.108 | 1.017 | Megatron-style math attention, RMSNorm/RoPE/CE alignment, exact bucket packing |
| D5 | 8437.2 | 7729.2 | 1.092 | 1.017 | rank-local input tokens are identical and fixed in both engines |
| D6 | 8436.0 | 8186.8 | 1.031 | 1.017 | install TE 2.9.0; Megatron uses TE FusedAdam and multi-tensor kernels |

`D0`, `D1`, and `D4` through `D6` are three-job medians. `D2` and `D3` are
one-job exploratory runs and are not promoted to formal results. Raw summaries and the exact
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
7. Three D4 jobs pass both parity thresholds on 2× V100 SXM2 nodes. Median
   throughput is 8435.2 vs 7610.8 tok/s, while sampled peak HBM is 11710 vs
   11518 MiB. The remaining speed advantage is real for this exact harness, but
   is likely dominated by nanoMegatron's narrower training loop and direct
   PyTorch modules versus Megatron's generic TP-capable layers, validation,
   timers, and logging. A staged CUDA profile is still needed before assigning
   the 10.8% difference to a particular kernel.
8. Numerical parity is checked separately with identical weights and tokens.
   FP32 logits/loss are exact and all gradient cosines are at least 0.99999988.
   In FP16, max logit difference is 9.77e-4, loss difference is 6.48e-5, max
   gradient difference is 2.44e-4, and minimum gradient cosine is 0.99999905.
9. D4 still reused one random batch in nano while Megatron advanced its mock
   dataloader. D5 removes that mismatch: both engines now generate the same
   rank-local token tensor once and reuse it. Three-run median advantage remains
   9.2%, so changing batches was not the explanation. Rank-0 input sum and the
   first eight token IDs are emitted by both engines and match exactly.
10. PyTorch profiling locates the advantage. Across three profiled steps,
    Megatron executes 763 more CUDA launches per step and 29.9 ms more aggregate
    CUDA kernel time per step. The largest component is optimizer execution:
    Megatron's local fallback multi-tensor Adam takes 28.7 ms for the inner step,
    while nano uses fused `torch.optim.AdamW`. The remaining overhead comes from
    generic TP-capable linear wrappers, FP32 main-grad checks/scaling, extra
    pointwise/cat/copy kernels, and repeated RoPE cos/sin application. Large GEMM
    runtimes are nearly identical.
11. D6 installs Transformer Engine 2.9.0, the version required by the pinned
    Megatron checkout. The first source-build attempt failed on `cudnn.h`; the
    second reached `nccl.h`. Adding the pip CUDA packages' `include` and `lib`
    directories to `CPATH`, `CPLUS_INCLUDE_PATH`, `LIBRARY_PATH`, and
    `LD_LIBRARY_PATH` produced a working V100 build. The benchmark now logs the
    selected optimizer backend and package versions, so a silent torch fallback
    cannot be mistaken for a production Megatron result.
12. TE changes the conclusion materially. Megatron's three-run median rises
    5.9%, from 7729.2 to 8186.8 tok/s, while nano stays flat. Profiling shows the
    inner optimizer step falling from 28.68 to 12.32 ms and total optimizer time
    from 44.46 to 25.38 ms. The remaining median gap is 3.1%, and the profiled
    extra CUDA kernel time shrinks from 29.9 to 7.5 ms/step. A direct 20-step
    optimizer check against fused torch AdamW gives parameter max error
    `2.38e-7` and cosine `0.9999999999999999`, so the gain is not from changing
    the Adam update semantics.

The environment-specific build is captured in
`scripts/install_transformer_engine.sh`; on the V100 benchmark environment it
is reproduced with `TORCH_CUDA_ARCH_LIST=7.0 scripts/install_transformer_engine.sh`.

Jobs: D1 `48899170`, `48901884`, `48901885`; D2 `48908022`; D3 `48908122`;
D4 `48977678`, `48979015`, `48979016`; D5 `48980514`, `48980551`,
`48980552`; D6 `48981508`, `48981561`, `48981562`; profiling D5 `48980515`,
profiling D6 `48981563`.
Correctness: FP32 `48980002`, FP16 `48980005`, TE optimizer `48981695`.

### Context parallel status

The first Qwen3 context-parallel path is intentionally a correctness-first
implementation of Megatron's synchronous `all_gather` CP mode:

1. The global sequence is divided into `2 * CP` chunks. Rank `r` owns chunk
   `r` and its mirrored chunk from the end, which balances causal-attention
   work better than contiguous one-sided shards.
2. Every layer keeps Q local and AllGathers K/V before attention. The custom
   autograd function performs a ReduceScatter in backward so local K/V
   activations receive gradients from every query shard.
3. Causal masks use the original token positions because rank-major gathered
   K/V order is intentionally not monotonic. Loss sums and replicated weight
   gradients are reduced across the CP group.

Run the two-rank path with:

```bash
torchrun --standalone --nproc_per_node=2 scripts/benchmark_qwen3.py \
    --strategy cp --seq-len 2048 \
    --output benchmark_logs/qwen3_0.6b/cp2.json
```

This row is `PARTIAL`, not `DONE`: Qwen3 numerics and a two-rank smoke test are
the first acceptance gate, but nano still lacks Megatron/Transformer Engine's
ring P2P overlap, `a2a` and hierarchical CP modes, packed-sequence handling,
and CP composition with TP, PP, DP, and distributed optimizer groups.

FP32 and FP16 local-logit/global-loss/gradient checks, including activation
recomputation, pass on 2×V100 (jobs `48987179` and `48987328`). The first V100
smoke run (`CP-B0`, job `48986133`) used sequence length 2048.
Against the same one-GPU nano graph (job `48986931`), CP2 improved throughput
from 4447.5 to 5575.8 tok/s and reduced peak allocated memory per GPU from
17.65 to 12.99 GiB. These are single-job implementation checks, not yet a
Megatron parity claim.

The CP profile (`48987975`) shows why `PARTIAL` still matters: synchronous K/V
AllGather plus backward ReduceScatter cost about 8.9 ms/step, while the
post-backward replicated-weight AllReduce costs about 31.4 ms/step. At this
sequence length, overlapping/bucketing weight-gradient reduction is a larger
next win than replacing the K/V path alone.

The first Megatron head-to-head (`CP-B1`, job `48988648`) ran on 2×A100 with
the same Qwen3-0.6B shape, fixed token stream, FP16, sequence length 2048,
CP=2, micro/global batch 1, optimizer hyperparameters, and synchronous
`all_gather` CP communication:

| Engine | tok/s | mean step | peak `nvidia-smi` HBM |
|---|---:|---:|---:|
| nanoMegatron | 11299.8 | 181.24 ms | 14363 MiB |
| Megatron Core | 8587.7 | 238.48 ms | 14333 MiB |

nano/Megatron is `1.316x` for throughput and `1.002x` for sampled HBM, so this
exploratory run passes the 95% speed and 105% memory thresholds. It is not yet
a strict promotion to `DONE`: Transformer Engine disables unfused attention
whenever CP is enabled, so Megatron used TE FusedAttention while nano used its
Megatron-style math backend. FlashAttention was not installed. The V100
attempt (`48988407`) therefore failed on the Megatron side with no supported
CP attention backend. Formal acceptance still needs the same attention backend
and a three-job median.

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
