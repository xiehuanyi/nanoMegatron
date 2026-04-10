# Pitfalls & Bug Fixes

🇨🇳 **[中文版](PITFALLS_zh.md)**

This document collects every non-obvious bug, gotcha, and trade-off we hit while building nanoMegatron. Each section names what bit us, how we diagnosed it, the fix, and the cost (if any).

## Table of contents

- [Setup gotchas](#setup-gotchas)
  - [NCCL P2P deadlock on PCIe machines](#nccl-p2p-deadlock-on-pcie-machines)
- [fp16 training pitfalls](#fp16-training-pitfalls)
  - [1. RoPE dtype mismatch silently produces NaN](#1-rope-dtype-mismatch-silently-produces-nan)
  - [2. RMSNorm overflows in fp16](#2-rmsnorm-overflows-in-fp16)
  - [3. fp16 Adam blows up via grad²](#3-fp16-adam-blows-up-via-grad²)
  - [4. TP MoE routing flicker → deadlock](#4-tp-moe-routing-flicker--deadlock)
  - [5. EP AllReduce backward over-counts the gradient](#5-ep-allreduce-backward-over-counts-the-gradient)
- [v1 → v2 algorithmic fixes](#v1--v2-algorithmic-fixes)
  - [Fix 1: TP NCCL flood (2158 → 288 calls/step)](#fix-1-tp-nccl-flood)
  - [Fix 2: ZeRO-2 didn't actually save memory](#fix-2-zero-2-didnt-actually-save-memory)
  - [Fix 3: ZeRO-3 didn't shard MoE experts](#fix-3-zero-3-didnt-shard-moe-experts)
- [Trade-offs still on the table](#trade-offs-still-on-the-table)

---

## Setup gotchas

### NCCL P2P deadlock on PCIe machines

**Symptom**: Even a basic `dist.all_reduce` on the default group hangs forever (or times out at the NCCL watchdog timeout, default 600s) on a 4-GPU PCIe machine. `torch.cuda.can_device_access_peer()` returns True for every pair, so it *looks* like P2P is fine.

**Diagnosis**: We started by suspecting our own code, then wrote a 10-line minimal repro that just inits the process group and does one all_reduce on the default PG. It hung. So the bug is in NCCL/driver, not in nanoMegatron.

**Root cause**: A bug in NCCL's PCIe P2P path on certain workstation/consumer GPUs. We hit it on RTX A5000s (4× on a single workstation, no NVLink). The peer-access check passes but the actual NCCL transfer never completes.

**Fix**: Force NCCL to skip P2P and use shared memory instead.

```bash
export NCCL_P2P_DISABLE=1
```

**Cost**: SHM is much slower than P2P (5–10× more latency for small messages, much lower bandwidth for large ones). Any per-param NCCL pattern (e.g. our ZeRO-2 hooks, or per-Linear ZeRO-3 AllGather) becomes dramatically slower. This is the main reason our ZeRO-2/3 v2 results show "fits but slow throughput" in the benchmark — on a working NVLink machine the same code would be much faster.

---

## fp16 training pitfalls

These bit us during the *initial* implementation of the model. fp16 has surprisingly many sharp edges.

### 1. RoPE dtype mismatch silently produces NaN

**Symptom**: First forward pass with fp16 model returns NaN logits. No error, no warning.

**Diagnosis**: Added a debug print of the first forward — `loss=nan, logits_max=nan, logits_nan=True`. Stepped through the model layer by layer. Q after `apply_rope` was NaN even though Q before `apply_rope` was fine.

**Root cause**: `cos`/`sin` returned by `build_rope_cache(...)` are fp32 (we build them with `torch.arange(...).float()`). The expression `q * cos + x_rot * sin` triggers automatic dtype promotion to fp32 for q and k, but `v` (used downstream in attention) stays fp16. SDPA with one fp32 input and one fp16 input returns NaN on certain GPUs.

**Fix** (`model.py`):

```python
def apply_rope(x, cos, sin):
    cos = cos.to(x.dtype)   # ← cast to input dtype
    sin = sin.to(x.dtype)
    d = x.shape[-1]
    x_rot = torch.stack([-x[..., d // 2:], x[..., :d // 2]], dim=-1).flatten(-2)
    return x * cos + x_rot * sin
```

### 2. RMSNorm overflows in fp16

**Symptom**: NaN loss after a few hundred steps, hard to reproduce, no obvious pattern.

**Diagnosis**: Bisected by checkpointing. Found the explosion happens inside RMSNorm when an activation gets large enough.

**Root cause**: `x.pow(2).mean(-1)` overflows fp16 when any element of `x` exceeds `~256`, because `256² = 65536 > 65504` (max representable fp16). Once a single element overflows to `inf`, the mean propagates `inf`, then `rsqrt(inf + eps) = 0`, then everything downstream becomes 0 or NaN.

**Fix** (`model.py`): Always compute RMSNorm in fp32, regardless of input dtype.

```python
def forward(self, x):
    orig_dtype = x.dtype
    x = x.float()
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    return (self.weight.float() * (x * rms) + self.bias.float()).to(orig_dtype)
```

This is a few extra ms per layer but it's the only correct way to do norm in fp16 mixed precision.

### 3. fp16 Adam blows up via grad²

**Symptom**: Adam `exp_avg_sq` overflows after warmup. NaN parameters within ~50 steps.

**Root cause**: Adam's second moment is `E[grad²]`. In fp16, even moderate gradients (~256) overflow when squared. PyTorch's fused Adam doesn't protect against this.

**Fix** (`zero.py`): Maintain a fp32 copy of parameters for the optimizer. Cast fp16 gradients to fp32 before the Adam step, write the updated fp32 weights back to fp16 after.

```python
class FP16OptimizerWrapper:
    def __init__(self, params, lr, weight_decay):
        self.fp16_params = list(params)
        self.fp32_params = [p.data.float().clone().requires_grad_(True) for p in self.fp16_params]
        self.optimizer = torch.optim.AdamW(self.fp32_params, lr=lr, weight_decay=weight_decay)

    def step(self):
        for fp32_p, fp16_p in zip(self.fp32_params, self.fp16_params):
            if fp16_p.grad is not None:
                fp32_p.grad = fp16_p.grad.float()
                fp16_p.grad = None
        self.optimizer.step()
        for fp32_p, fp16_p in zip(self.fp32_params, self.fp16_params):
            fp16_p.data.copy_(fp32_p.data)
```

**Cost**: 4 extra bytes per parameter (the fp32 copy), so 4 GB extra for a 1B-param model. This is mandatory for fp16 mixed precision and is exactly what `torch.cuda.amp.GradScaler` + master weights does internally.

### 4. TP MoE routing flicker → deadlock

**Symptom**: TP-only run sometimes deadlocks at a random layer. The location moves between runs. With `_sync_routing` disabled, hangs are reliable.

**Diagnosis**: Each TP rank should see the *same* x at every layer (TP keeps activations identical across ranks via AllReduce). But the AllReduce in the previous layer's O-projection has floating-point non-associativity — different NCCL ring orderings on different runs produce subtly different sums. By the time x reaches the next MoE layer's gate, each rank's gate output differs by a few ULPs.

That's enough that one rank's top-2 selects expert {3, 7} while another rank's top-2 selects expert {3, 8}. Now expert 7 sees tokens on rank 0 but not rank 1, so rank 0 calls `expert_7.forward()` (which triggers a TP AllReduce) while rank 1 doesn't. Mismatched collective counts → deadlock.

**Fix** (`model.py`): After computing top-k routing, broadcast rank 0's decision to all other ranks before dispatching to experts.

```python
if getattr(self, "_sync_routing", False):
    dist.broadcast(topk_indices, src=0)
    dist.broadcast(topk_weights, src=0)
```

The `_sync_routing` flag is set automatically when the model is wrapped with TP.

### 5. EP AllReduce backward over-counts the gradient

**Symptom** (older version): Loss explodes after a few EP steps. Gradients look about N× larger than expected (where N = world size).

**Root cause**: The naive way to write a forward AllReduce in autograd is "AllReduce in forward, AllReduce in backward". That's wrong — the backward of `y = AllReduce_SUM(x_partial)` is **identity**, not another AllReduce. If you write the backward as another AllReduce, every rank's local gradient ends up summed N times.

**Fix**: Use a `torch.autograd.Function` with identity backward.

```python
class _AllReduceFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None   # identity, NOT another AllReduce
```

The mathematical justification: if `y = sum_i x_i` (where i ranges over ranks), then `dL/dx_i = dL/dy` for every i. Each rank should keep its own `dL/dy` unchanged. Re-AllReducing would produce `N × dL/dy` on every rank.

---

## v1 → v2 algorithmic fixes

These bugs we discovered through systematic profiling, *after* the initial implementation worked correctly. The original V100 benchmark gave suspicious numbers (TP slow, ZeRO-1/2/3 all using identical memory), and digging in revealed that the implementations had real bugs hiding behind "looks like it works".

### Fix 1: TP NCCL flood

**Symptom**: TP-4 throughput on 4× V100 was 45 tok/s — exactly the same as ZeRO-1/2 (45 tok/s), despite TP distributing the model across GPUs. Meanwhile EP-4 was 4× faster at 190 tok/s. Something was eating TP's compute speedup.

**Diagnosis**: We wrote `scripts/profile_tp.py` that monkey-patches `dist.all_reduce` / `broadcast` / etc. to count calls. Result on the v1 implementation:

```
TP-2 (fp32):  79 tok/s | 2158 NCCL calls/step | 15.1 GB
Baseline:     69 tok/s |    0 NCCL calls/step | 14.5 GB
```

**Two thousand NCCL calls per forward+backward step.** Where do they come from?

- Every `ColumnParallelLinear` calls `_SplitFunc.apply(x)` in forward (no-op forward, but its backward issues an AllReduce on `dL/dx`).
- Every `RowParallelLinear` calls `_AllReduceFunc.apply(y)` in forward (an AllReduce in forward, identity in backward).
- Per layer: 1 attention AllReduce (from O-proj) + 16 expert AllReduce (from each expert's w2) + 3 attention SplitFunc-backward AllReduce (from Q/K/V) + 32 expert SplitFunc-backward AllReduce (from each expert's w1/w3) + 2 routing broadcasts ≈ **54 collectives per layer**.
- 32 layers + gradient checkpointing (which re-runs forward inside backward, doubling the count) ≈ 2158 per step.

Each NCCL call on PCIe has 20–50μs of pure launch overhead. So 2158 × 35μs ≈ **75ms of launch overhead per step**, before any actual data transfer.

**Insight**: All three of {Q proj, K proj, V proj} consume the *same* x. The autograd graph could sum their three contributions to `dL/dx` locally and then issue one AllReduce. Similarly, all 16 experts' `w2` outputs can accumulate locally and AllReduce *once* at the end of the MoE block.

**Fix** (`tensor_parallel.py`):

1. Add `skip_split` flag to `ColumnParallelLinear` and `skip_reduce` to `RowParallelLinear`. When set, the per-layer SplitFunc / AllReduce is suppressed.

2. **Attention**: register a `forward_pre_hook` on the attention module that applies `_SplitFunc` once to the input. Q/K/V have `skip_split=True`, so backward AllReduce fires once per layer instead of three times.

3. **MoE**: introduce `TPMoEWrapper` that replaces the MoE block. It applies `_SplitFunc` once on the expert path (not the gate path — see the subtlety below), runs all 16 experts with `skip_reduce=True` so they accumulate *partial* outputs into a single tensor, then `_AllReduceFunc` once at the end. One AllReduce per layer instead of 16.

**Mathematical justification**: For RowParallel, `y = sum_ranks(W_partial @ x)`. The MoE output is `y_total = sum_experts(weight_e × expert_e(x))`. If each expert produces `partial_e = weight_e × expert_e_partial(x)` without AllReducing, then `sum_experts(partial_e) = sum_experts(weight_e × sum_ranks(W_partial_e @ ...)) = sum_ranks(sum_experts(weight_e × W_partial_e @ ...))`. The sums commute, so `AllReduce(sum_experts(partial_e)) = y_total`. ✓

**Subtle correctness issue**: The MoE router (gate) is *not* TP-parallelized — it's a regular `nn.Linear` replicated across ranks. If we put the SplitFunc on the input of the *entire* MoE block (so the gate sees the split input), the gate's gradient would also flow through the SplitFunc backward AllReduce. But the gate gradient is *full* on every rank (not a partition), so SplitFunc backward would sum it N times → wrong by a factor of N. The fix only puts SplitFunc on the *expert path*, leaving the gate to use the original `x_flat`:

```python
def forward(self, x):
    x_flat = x.view(-1, D)
    router_logits = self.gate(x_flat)              # ← gate uses x_flat directly
    topk_weights, topk_indices = ...
    if self._sync_routing:
        dist.broadcast(topk_indices, src=0)
        dist.broadcast(topk_weights, src=0)

    x_for_experts = _SplitFunc.apply(x_flat, self.tp_group)   # ← SplitFunc only on expert path

    output = torch.zeros_like(x_for_experts)
    for i in range(self.num_experts_per_tok):
        for e_id in range(len(self.experts)):
            mask = (expert_idx == e_id)
            if mask.any():
                output[mask] += weight[mask].unsqueeze(-1) * self.experts[e_id](x_for_experts[mask])

    return _AllReduceFunc.apply(output, self.tp_group).view(B, L, D), router_logits
```

**Result**:

| | NCCL/step | TP-2 fwd+bwd throughput |
|-|----------:|------------------------:|
| v1 | 2158 | 79 tok/s |
| **v2** | **288** | **95 tok/s** |

7.5× fewer collectives, 20% throughput improvement on the profiling micro-benchmark. On the **4-GPU full training run**, TP-4 went from 45 tok/s (V100×4 v1) to 154 tok/s (A5000×4 v2) — a 3.4× wall-clock speedup, ~6.8× equivalent if you account for A5000 being ~50% the fp16 throughput of V100.

### Fix 2: ZeRO-2 didn't actually save memory

**Symptom**: V100×4 results showed ZeRO-1 = 27.6 GB and ZeRO-2 = 27.6 GB. These should not be identical — ZeRO-2 is supposed to halve the gradient memory by reducing each gradient to its owner rank instead of AllReducing to all.

**Diagnosis**: We traced where `torch.cuda.max_memory_allocated()` peaks. The peak happens *during backward*, before `step()` is even called. At that point all gradients are still alive on every rank — the v1 implementation only does the Reduce-to-owner inside `step()`, which is too late.

```
backward end (peak):
  fp16 params (full, every rank):    7.6 GB    ← same for both stages
  fp16 grads (full, every rank):     7.6 GB    ← same for both — accumulated during backward
  fp32 copies (1/N):                 1.9 GB
  ──────────────────────────────────────────
  total: ~17 GB before any optimizer state allocation

step() (now happens, but peak is already locked in):
  ZeRO-1: AllReduce all grads → free non-local
  ZeRO-2: Reduce each grad to owner → free non-local
                ↑ this is where they differ, but max_memory_allocated is already set
```

**Fix** (`zero.py`): Use `register_post_accumulate_grad_hook` so each parameter's hook fires *the moment* its gradient is accumulated by autograd. The hook does `dist.reduce(p.grad, dst=owner)` immediately, then sets `p.grad = None` on non-owners. The backward peak now holds at most a few in-flight gradients, not all of them.

```python
if stage == 2:
    for p in self.all_params:
        owner = self.param_to_owner[id(p)]
        p.register_post_accumulate_grad_hook(self._make_zero2_hook(owner))

def _make_zero2_hook(self, owner):
    def hook(p):
        dist.reduce(p.grad, dst=owner, op=dist.ReduceOp.SUM)
        if dist.get_rank() != owner:
            p.grad = None   # ← the actual saving
    return hook
```

**Knock-on issue**: `nn.utils.clip_grad_norm_` would now compute different norms on each rank, because non-owner grads are gone. Each rank would scale its grads by a different factor → diverging weights.

**Fix for the knock-on** (`zero.py` + `trainer.py`): Add `ZeROOptimizer.clip_grad_norm` that does a distributed norm computation:

```python
def clip_grad_norm(self, max_norm):
    local_norm_sq = torch.zeros(1, device=...)
    for p in self.local_params:
        if p.grad is not None:
            g = p.grad.float() / self.world_size
            local_norm_sq += g.pow(2).sum()
    dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM)
    global_norm = local_norm_sq.sqrt().item()
    if global_norm > max_norm and global_norm > 0:
        clip_coef = max_norm / global_norm
        for p in self.local_params:
            if p.grad is not None:
                p.grad.mul_(clip_coef)
    return global_norm
```

The trainer detects ZeRO-2 and calls this method instead of `nn.utils.clip_grad_norm_`.

**Result**: On 8-layer test config, ZeRO-1 = 13.65 GB, ZeRO-2 v2 = ~5.3 GB (-60%). On 4× A5000 with the full 32-layer model, ZeRO-1 OOMs at ~22 GB while ZeRO-2 v2 fits comfortably at ~12 GB.

**Trade-off**: Per-param sync hooks fire one at a time, blocking the autograd thread on each NCCL call. Without bucketing this is much slower than ZeRO-1's batched-in-step approach. Production frameworks (DeepSpeed, FSDP) bucket many small grads into one collective and use multi-stream async overlap to hide the latency. We don't.

### Fix 3: ZeRO-3 didn't shard MoE experts

**Symptom**: V100×4 ZeRO-3 = 26.6 GB, only 1 GB less than ZeRO-1 (27.6 GB). The "shard everything" stage was barely sharding anything.

**Diagnosis**: Read the v1 `fsdp.py`:

```python
def fsdp_wrap_module(model, group):
    for name, child in list(model.named_children()):
        if name == "experts":   # ← !!
            continue
        ...
```

**Why the skip existed**: Different ranks see different data (data parallel), so they activate different experts. With per-Linear FSDP wrapping, each rank's `forward` calls AllGather only for the experts it actually activates. Different rank → different AllGather call counts → deadlock.

**The cost of the skip**: Out of 3.8B params, MoE experts hold 16 × 3 × 4096 × 448 × 32 ≈ 2.8B (~74%). Skipping them defeats the entire point of ZeRO-3.

**Fix** (`fsdp.py`):

1. Remove the `if name == "experts": continue` check.

2. Add `_patch_moe_for_fsdp(moe)` that rewrites the MoE forward to call **every expert unconditionally**, even when `mask.any()` is False:

   ```python
   for e_id in range(len(self.experts)):
       mask = (expert_idx == e_id)
       expert_input = x_flat[mask]               # may be [0, D] when mask is empty
       expert_output = self.experts[e_id](expert_input)   # ← always called
       if expert_input.shape[0] > 0:
           output[mask] += weight[mask].unsqueeze(-1) * expert_output
   ```

   The empty case is fine: `F.linear([0, in_features], W)` returns `[0, out_features]` and the FSDPLinear AllGather happens regardless. All ranks now issue identical AllGather sequences in identical order → no deadlock.

3. `FSDPMixedOptimizer` previously detected sharded params with `name in ("q_proj", "k_proj", ...)`, which missed everything inside `experts.X.wY`. Switch to `isinstance(module, (FSDPLinear, FSDPEmbedding))` checks so any sharded module is found regardless of its name.

4. With experts sharded, the only non-sharded params left are RMSNorm (a few KB). They're small enough to use fp32 Adam, not the fp16 SGD workaround that v1 needed when expert weights weren't sharded.

**Result**: On 8-layer test config, ZeRO-3 v2 = ~4.9 GB (-64% vs ZeRO-1 13.65 GB). On 4× A5000 full model, ~9 GB/GPU.

**Trade-off**: With experts sharded, every expert call triggers AllGather on the expert's three Linears → 16 experts × 3 linears × 32 layers = 1536 AllGather per forward. Gradient checkpointing doubles this. Without bucketing, this is *slow*.

---

## Trade-offs still on the table

These are the things production frameworks do that we don't, and what they cost.

### NCCL bucketing
Both DDP and FSDP group multiple parameters into a single AllReduce/AllGather call. nanoMegatron does one collective per parameter. On a NVLink machine, the per-call overhead is ~10μs vs ~100μs on PCIe SHM, so the "fits but slow" trade-off would be much less painful in a normal setup. On our test box with `NCCL_P2P_DISABLE=1`, the per-call cost compounds catastrophically.

**To add this**: Allocate a flat buffer (one big tensor per model layer or per fixed bucket size). Have each parameter's `.grad` be a view into the buffer. Do a single NCCL call on the buffer. Implementation is ~500 lines + careful memory management.

### Communication-computation overlap
Megatron-LM uses async AllReduce + GEMM scheduling so the communication time is hidden behind compute. Requires `CUDA_DEVICE_MAX_CONNECTIONS=1` env var (so NCCL and compute share connections without serializing) plus explicit stream management.

**To add this**: Make the SplitFunc/AllReduce nodes use `dist.all_reduce(..., async_op=True)` and return a handle. Wait on the handle before the *next* operation that depends on the result. Doable but adds significant complexity to the autograd integration.

### FlatParameter (PyTorch FSDP-style)
FSDP physically lays out a layer's parameters in a single contiguous flat tensor. AllGather then transfers one big buffer instead of N small ones, and the memory layout is friendlier to the allocator. We use Python lists of `nn.Parameter` objects, which is simple but slow.

### CPU/NVMe offloading (ZeRO-Offload, ZeRO-Infinity)
DeepSpeed can spill optimizer states to CPU memory (or even NVMe with ZeRO-Infinity), letting you train models that would otherwise exceed GPU memory entirely. nanoMegatron doesn't.

### MoE load balancing
Production MoE frameworks add an auxiliary loss term that encourages even expert usage and use a "capacity factor" to bound work imbalance (drop or pad tokens that exceed an expert's capacity). nanoMegatron lets the routing be whatever the gate produces — fine for SFT on a small dataset, would be a problem at pretraining scale.

### Sequence Parallel (Megatron-style)
Megatron's SP slices LayerNorm and Dropout along the sequence dimension on top of TP, saving `tp_size`× in activation memory for those operations. We have a stub but it's not heavily used.

### 1F1B pipeline scheduling
Megatron's 1F1B pipeline schedule interleaves forward and backward to keep activation memory at `O(P)` instead of GPipe's `O(M)` (where P = pipeline stages, M = micro-batches). Our PP is GPipe-only.

### Tensor & Sequence Parallel + MoE in one block
Megatron-MoE / DeepSpeed-MoE compose TP, SP, and EP within a single MoE block (e.g., TP-shard the gate's projection, EP-shard the experts, SP-slice the attention activations, all simultaneously). Our strategies are mostly orthogonal — you can use TP *or* EP for the experts, not both at once.

---

## What the trade-offs add up to

The right way to read nanoMegatron's benchmark numbers is:

> nanoMegatron's algorithms fit the same models DeepSpeed/FSDP fit, but at 5–10× lower throughput on the slow paths (ZeRO-2/3) because we lack bucketing and async overlap. The gap closes for TP and EP, where the algorithm shape itself doesn't generate per-param NCCL waterfalls.

If you're learning *how* the algorithms work, that's fine — the math and the dataflow are the same. If you're training a real model, use DeepSpeed or FSDP.
