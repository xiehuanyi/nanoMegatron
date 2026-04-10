# 踩坑日记 & Bug 修复

🇬🇧 **[English version](PITFALLS.md)**

这份文档收集了在写 nanoMegatron 的过程中所有不那么显而易见的 bug、坑和 trade-off。每节包括：现象、诊断过程、修复方案、以及代价（如果有）。

## 目录

- [环境坑](#环境坑)
  - [NCCL P2P 在 PCIe 机器上死锁](#nccl-p2p-在-pcie-机器上死锁)
- [fp16 训练坑](#fp16-训练坑)
  - [1. RoPE dtype 不匹配悄悄产生 NaN](#1-rope-dtype-不匹配悄悄产生-nan)
  - [2. RMSNorm 在 fp16 下溢出](#2-rmsnorm-在-fp16-下溢出)
  - [3. fp16 Adam 因 grad² 爆炸](#3-fp16-adam-因-grad²-爆炸)
  - [4. TP MoE 路由抖动 → 死锁](#4-tp-moe-路由抖动--死锁)
  - [5. EP AllReduce 反向把梯度多算 N 倍](#5-ep-allreduce-反向把梯度多算-n-倍)
- [v1 → v2 算法层面修复](#v1--v2-算法层面修复)
  - [Fix 1: TP NCCL 调用爆炸（2158 → 288 次/步）](#fix-1-tp-nccl-调用爆炸)
  - [Fix 2: ZeRO-2 实际并没省显存](#fix-2-zero-2-实际并没省显存)
  - [Fix 3: ZeRO-3 没分片 MoE experts](#fix-3-zero-3-没分片-moe-experts)
- [没补的 trade-off](#没补的-trade-off)

---

## 环境坑

### NCCL P2P 在 PCIe 机器上死锁

**现象**：在一台 4 卡 PCIe 机器上，连最简单的 default group `dist.all_reduce` 都会 hang 住（或者直到 NCCL watchdog 600 秒超时）。`torch.cuda.can_device_access_peer()` 对所有 GPU 对都返回 True，**看上去** P2P 是好的。

**诊断**：先怀疑自己的代码，然后写了一个 10 行的最小复现：只 init process group 然后做一次 default PG 的 all_reduce。也 hang 住了。所以 bug 在 NCCL/驱动层，不在 nanoMegatron。

**根因**：某些 workstation/消费级 GPU 的 NCCL PCIe P2P 路径有 bug。我们在 RTX A5000 上踩到（一台机器 4 卡，没有 NVLink）。peer-access 检查通过，但实际 NCCL 传输永远不完成。

**修复**：强制 NCCL 跳过 P2P，走共享内存。

```bash
export NCCL_P2P_DISABLE=1
```

**代价**：SHM 比 P2P 慢很多（小消息延迟 5–10 倍，大消息带宽也低很多）。任何 per-param 的 NCCL 模式（比如我们的 ZeRO-2 hook、或 per-Linear 的 ZeRO-3 AllGather）都会变得极慢。这就是为什么我们的 ZeRO-2/3 v2 在 benchmark 里表现成"能放下但吞吐慢"——在正常 NVLink 机器上，同样的代码会快得多。

---

## fp16 训练坑

这些是模型**最初**实现时踩到的。fp16 的雷比想象中多。

### 1. RoPE dtype 不匹配悄悄产生 NaN

**现象**：fp16 模型的第一次 forward 返回 NaN logits。无错误、无警告。

**诊断**：在第一次 forward 加 debug print，看到 `loss=nan, logits_max=nan, logits_nan=True`。然后逐层 step 进去，发现 Q 在 `apply_rope` 之后是 NaN，但之前是好的。

**根因**：`build_rope_cache(...)` 返回的 `cos`/`sin` 是 fp32（用 `torch.arange(...).float()` 构造）。表达式 `q * cos + x_rot * sin` 会触发 q 和 k 自动提升为 fp32，但 `v`（attention 中后面用到）还是 fp16。SDPA 在某些 GPU 上对 fp32+fp16 输入会产生 NaN。

**修复**（`model.py`）：

```python
def apply_rope(x, cos, sin):
    cos = cos.to(x.dtype)   # ← 转成输入的 dtype
    sin = sin.to(x.dtype)
    d = x.shape[-1]
    x_rot = torch.stack([-x[..., d // 2:], x[..., :d // 2]], dim=-1).flatten(-2)
    return x * cos + x_rot * sin
```

### 2. RMSNorm 在 fp16 下溢出

**现象**：训了几百步后 NaN loss，难以复现，没有明显规律。

**诊断**：用 checkpoint bisect。发现爆炸发生在 RMSNorm 内部，触发条件是某个激活值变大。

**根因**：`x.pow(2).mean(-1)` 在 `x` 任何元素超过 `~256` 时就会 fp16 溢出，因为 `256² = 65536 > 65504`（fp16 最大可表示数）。一旦某个元素溢出成 `inf`，mean 就会传 `inf`，然后 `rsqrt(inf + eps) = 0`，下游全变 0 或 NaN。

**修复**（`model.py`）：始终用 fp32 算 RMSNorm，不管输入 dtype 是什么。

```python
def forward(self, x):
    orig_dtype = x.dtype
    x = x.float()
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    return (self.weight.float() * (x * rms) + self.bias.float()).to(orig_dtype)
```

每层多花几 ms，但这是 fp16 混精度下做 norm 的唯一正确方法。

### 3. fp16 Adam 因 grad² 爆炸

**现象**：Warmup 之后 Adam 的 `exp_avg_sq` 溢出。~50 步内参数变 NaN。

**根因**：Adam 的二阶矩是 `E[grad²]`。fp16 下，即使中等大小的梯度（~256）平方后也会溢出。PyTorch 的 fused Adam 不会保护这种情况。

**修复**（`zero.py`）：维护一份 fp32 参数副本给 optimizer 用。Adam step 之前把 fp16 梯度转 fp32，step 之后把更新后的 fp32 权重写回 fp16。

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

**代价**：每个参数多 4 字节（fp32 副本），所以 1B 参数要多 4 GB。fp16 混精度下这是必须的，等价于 `torch.cuda.amp.GradScaler` + master weights 内部做的事情。

### 4. TP MoE 路由抖动 → 死锁

**现象**：纯 TP 的 run 偶尔在某一层 deadlock，位置每次不同。如果禁掉 `_sync_routing`，hang 是稳定可复现的。

**诊断**：理论上每个 TP rank 在每一层应该看到**相同**的 x（TP 通过 AllReduce 让激活在所有 rank 上保持一致）。但前一层 O 投影里的 AllReduce 有浮点非结合性 —— 不同 NCCL ring ordering 在不同 run 上会产生 ULP 级的差异。等到 x 进入下一层 MoE 的 gate 时，每个 rank 的 gate 输出已经差了几个 ULP。

这点差异足以让 rank 0 的 top-2 选 expert {3, 7}，rank 1 的 top-2 选 expert {3, 8}。现在 expert 7 在 rank 0 上有 token 但 rank 1 没有，rank 0 调 `expert_7.forward()`（触发一次 TP AllReduce），rank 1 不调 → collective 调用次数错位 → 死锁。

**修复**（`model.py`）：算完 top-k 路由后，把 rank 0 的决策广播给所有其他 rank，再去 dispatch 到 expert。

```python
if getattr(self, "_sync_routing", False):
    dist.broadcast(topk_indices, src=0)
    dist.broadcast(topk_weights, src=0)
```

模型被 TP 包装时自动设置 `_sync_routing` 标志。

### 5. EP AllReduce 反向把梯度多算 N 倍

**现象**（旧版）：EP 跑几步后 loss 爆炸。梯度看上去比预期大约 N 倍（N = world size）。

**根因**：朴素地写一个 forward AllReduce 进 autograd 的方式是"forward AllReduce，backward 也 AllReduce"。这是错的——`y = AllReduce_SUM(x_partial)` 的 backward 是 **identity**，不是再来一次 AllReduce。如果把 backward 写成再 AllReduce，每个 rank 的本地梯度会被 sum N 次。

**修复**：用 `torch.autograd.Function`，backward 是 identity。

```python
class _AllReduceFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None   # identity，不是再来一次 AllReduce
```

**数学解释**：如果 `y = sum_i x_i`（i 是 rank index），那么对每个 i 都有 `dL/dx_i = dL/dy`。每个 rank 应该保留自己那份 `dL/dy` 不变。再 AllReduce 一次会让每个 rank 的梯度变成 `N × dL/dy`。

---

## v1 → v2 算法层面修复

这些 bug 是在最初的实现"看起来工作正常"之后，**通过系统性 profiling** 发现的。最初的 V100 benchmark 给出可疑的数字（TP 慢、ZeRO-1/2/3 显存完全相同），追下去发现实现里藏着真实的 bug，只是被"看上去能跑"掩盖了。

### Fix 1: TP NCCL 调用爆炸

**现象**：4× V100 上 TP-4 吞吐 45 tok/s——和 ZeRO-1/2 一模一样（45 tok/s），尽管 TP 把模型分布到多卡。同时 EP-4 是 4× 那么快（190 tok/s）。一定有什么在吃掉 TP 的计算加速。

**诊断**：写了 `scripts/profile_tp.py`，monkey-patch `dist.all_reduce` / `broadcast` 等来计数。v1 实现的结果：

```
TP-2 (fp32):  79 tok/s | 2158 NCCL calls/step | 15.1 GB
Baseline:     69 tok/s |    0 NCCL calls/step | 14.5 GB
```

**每步 forward+backward 两千多次 NCCL 调用**。从哪儿来的？

- 每个 `ColumnParallelLinear` forward 时调 `_SplitFunc.apply(x)`（forward 是 no-op，但它的 backward 会对 `dL/dx` 做 AllReduce）。
- 每个 `RowParallelLinear` forward 时调 `_AllReduceFunc.apply(y)`（forward 是 AllReduce，backward 是 identity）。
- 每层：1 次 attention AllReduce（O proj） + 16 次 expert AllReduce（每个 expert 的 w2）+ 3 次 attention SplitFunc 反向 AllReduce（Q/K/V）+ 32 次 expert SplitFunc 反向 AllReduce（每个 expert 的 w1/w3）+ 2 次 routing broadcast ≈ **每层 54 次 collective**。
- 32 层 + gradient checkpointing（backward 里重跑 forward，把 forward NCCL 翻倍）≈ 每步 2158 次。

每次 NCCL 调用在 PCIe 上有 20–50μs 的纯 launch overhead。所以 2158 × 35μs ≈ **每步 75ms 的 launch overhead**，这还没算实际数据传输。

**关键 insight**：Q proj、K proj、V proj **共用同一个 x**。autograd 图可以把这三个对 `dL/dx` 的贡献本地相加，然后做**一次**AllReduce。同样，所有 16 个 expert 的 `w2` 输出可以本地累加，在 MoE 块结尾做**一次**AllReduce。

**修复**（`tensor_parallel.py`）：

1. 给 `ColumnParallelLinear` 加 `skip_split` 标志、给 `RowParallelLinear` 加 `skip_reduce` 标志。设置后，per-layer 的 SplitFunc / AllReduce 会被跳过。

2. **Attention**：在 attention 模块上注册一个 `forward_pre_hook`，在入口对 x 做一次 `_SplitFunc`。Q/K/V 设置 `skip_split=True`，所以 backward AllReduce 每层只 fire 一次而不是三次。

3. **MoE**：引入 `TPMoEWrapper` 替换原 MoE 块。它在 expert 路径（不是 gate 路径——见下面那个微妙的问题）上做一次 `_SplitFunc`，所有 16 个 expert 用 `skip_reduce=True` 跑出 *partial* output 累积到一个 tensor 上，最后在结尾做一次 `_AllReduceFunc`。每层一次 AllReduce 而不是 16 次。

**数学正确性**：对 RowParallel，`y = sum_ranks(W_partial @ x)`。MoE 输出是 `y_total = sum_experts(weight_e × expert_e(x))`。如果每个 expert 不 AllReduce，直接产出 `partial_e = weight_e × expert_e_partial(x)`，那么 `sum_experts(partial_e) = sum_experts(weight_e × sum_ranks(W_partial_e @ ...)) = sum_ranks(sum_experts(weight_e × W_partial_e @ ...))`。求和可交换，所以 `AllReduce(sum_experts(partial_e)) = y_total`。✓

**微妙的正确性问题**：MoE 的 router（gate）**没有** TP 并行——它是普通的 `nn.Linear`，每个 rank 都有完整副本。如果在**整个 MoE 块**的输入上做 SplitFunc（gate 也看到 split 后的输入），gate 的梯度也会流过 SplitFunc 反向 AllReduce。但 gate 的梯度在每个 rank 上都是**完整的**（不是分片），SplitFunc 反向会把它 sum N 次 → 错了 N 倍。修复是只把 SplitFunc 放在 **expert 路径**上，让 gate 用原始的 `x_flat`：

```python
def forward(self, x):
    x_flat = x.view(-1, D)
    router_logits = self.gate(x_flat)              # ← gate 直接用 x_flat
    topk_weights, topk_indices = ...
    if self._sync_routing:
        dist.broadcast(topk_indices, src=0)
        dist.broadcast(topk_weights, src=0)

    x_for_experts = _SplitFunc.apply(x_flat, self.tp_group)   # ← 只在 expert 路径上做 SplitFunc

    output = torch.zeros_like(x_for_experts)
    for i in range(self.num_experts_per_tok):
        for e_id in range(len(self.experts)):
            mask = (expert_idx == e_id)
            if mask.any():
                output[mask] += weight[mask].unsqueeze(-1) * self.experts[e_id](x_for_experts[mask])

    return _AllReduceFunc.apply(output, self.tp_group).view(B, L, D), router_logits
```

**结果**：

| | NCCL/步 | TP-2 fwd+bwd 吞吐 |
|-|--------:|---------------:|
| v1 | 2158 | 79 tok/s |
| **v2** | **288** | **95 tok/s** |

7.5× 减少 collective 数，profiling 微 benchmark 上 20% 吞吐提升。在**4 卡完整训练**上，TP-4 从 45 tok/s（V100×4 v1）变成 154 tok/s（A5000×4 v2）—— wall-clock 加速 3.4×，考虑 A5000 的 fp16 算力只有 V100 的 ~50%，**等效加速 ~6.8×**。

### Fix 2: ZeRO-2 实际并没省显存

**现象**：V100×4 上 ZeRO-1 = 27.6 GB，ZeRO-2 = 27.6 GB。这俩不应该一样——ZeRO-2 理论上应该把梯度显存减半（reduce 到 owner 而不是 AllReduce 给所有人）。

**诊断**：追 `torch.cuda.max_memory_allocated()` 的 peak 在哪里。peak 发生在 **backward 期间**，`step()` 还没被调用。这时所有 rank 上的所有梯度还活着 —— v1 实现只在 `step()` 里做 reduce-to-owner，太晚了。

```
backward 结束（peak）:
  fp16 params (完整，每个 rank):    7.6 GB    ← 两个 stage 都一样
  fp16 grads (完整，每个 rank):     7.6 GB    ← 两个 stage 都一样 —— backward 期间累积
  fp32 副本 (1/N):                 1.9 GB
  ──────────────────────────────────────────
  total: ~17 GB（在 optimizer state 分配之前）

step() (现在才发生，但 peak 已经定格)：
  ZeRO-1: AllReduce 所有 grads → 释放非本地
  ZeRO-2: Reduce 每个 grad 到 owner → 释放非本地
                ↑ 这里才有差异，但 max_memory_allocated 已经被锁定了
```

**修复**（`zero.py`）：用 `register_post_accumulate_grad_hook` 让每个参数的 hook 在 autograd 累积完梯度的**那一刻**就 fire。hook 立即 `dist.reduce(p.grad, dst=owner)`，然后在非 owner 上 `p.grad = None`。backward 期间 peak 最多只持有几个 in-flight 梯度，而不是全部。

```python
if stage == 2:
    for p in self.all_params:
        owner = self.param_to_owner[id(p)]
        p.register_post_accumulate_grad_hook(self._make_zero2_hook(owner))

def _make_zero2_hook(self, owner):
    def hook(p):
        dist.reduce(p.grad, dst=owner, op=dist.ReduceOp.SUM)
        if dist.get_rank() != owner:
            p.grad = None   # ← 真正省显存的地方
    return hook
```

**连带问题**：`nn.utils.clip_grad_norm_` 现在会在不同 rank 上算出不一样的 norm，因为非 owner 的 grad 已经被释放了。每个 rank 用不同的 factor 缩放 → 权重发散。

**连带问题的修复**（`zero.py` + `trainer.py`）：加一个 `ZeROOptimizer.clip_grad_norm` 方法做分布式 norm 计算：

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

trainer 检测到 ZeRO-2 时调用这个方法，而不是 `nn.utils.clip_grad_norm_`。

**结果**：8 层测试 config 上，ZeRO-1 = 13.65 GB，ZeRO-2 v2 = ~5.3 GB（-60%）。4× A5000 跑完整 32 层模型时，ZeRO-1 OOM 在 ~22 GB，ZeRO-2 v2 在 ~12 GB 舒服地跑起来。

**Trade-off**：per-param 同步 hook 一次只 fire 一个，每次 NCCL 都阻塞 autograd 线程。没有 bucketing 的话，比 ZeRO-1 在 step() 里批量做要慢得多。生产框架（DeepSpeed、FSDP）把多个小梯度 bucket 成一次 collective，再用 multi-stream async overlap 隐藏延迟。我们没做。

### Fix 3: ZeRO-3 没分片 MoE experts

**现象**：V100×4 ZeRO-3 = 26.6 GB，比 ZeRO-1（27.6 GB）只少 1 GB。"全分片"这个 stage 几乎没分片任何东西。

**诊断**：读 v1 的 `fsdp.py`：

```python
def fsdp_wrap_module(model, group):
    for name, child in list(model.named_children()):
        if name == "experts":   # ← !!
            continue
        ...
```

**为什么有这个跳过**：不同 rank 看不同数据（数据并行），所以激活的 expert 也不同。per-Linear FSDP 包装下，每个 rank 的 forward 只对自己激活的 expert 调 AllGather。不同 rank → 不同 AllGather 调用次数 → deadlock。

**跳过的代价**：3.8B 参数里，MoE experts 占 16 × 3 × 4096 × 448 × 32 ≈ 2.8B（~74%）。跳过 experts 等于把 ZeRO-3 的意义全废了。

**修复**（`fsdp.py`）：

1. 移除 `if name == "experts": continue` 检查。

2. 加 `_patch_moe_for_fsdp(moe)`，重写 MoE forward，让所有 expert **无条件**被调用，即使 `mask.any()` 是 False：

   ```python
   for e_id in range(len(self.experts)):
       mask = (expert_idx == e_id)
       expert_input = x_flat[mask]               # mask 全 False 时是 [0, D]
       expert_output = self.experts[e_id](expert_input)   # ← 始终调用
       if expert_input.shape[0] > 0:
           output[mask] += weight[mask].unsqueeze(-1) * expert_output
   ```

   空 input 没问题：`F.linear([0, in_features], W)` 返回 `[0, out_features]`，FSDPLinear 的 AllGather 照常发生。所有 rank 现在按同样顺序发出同样的 AllGather 序列 → 不死锁。

3. `FSDPMixedOptimizer` 之前用 `name in ("q_proj", "k_proj", ...)` 检测分片参数，漏掉了 `experts.X.wY` 里的所有东西。改成 `isinstance(module, (FSDPLinear, FSDPEmbedding))` 检查，这样不管 module 叫什么名字，分片参数都能被识别。

4. experts 分片之后，剩下的非分片参数只有 RMSNorm（几 KB）。小到完全可以走 fp32 Adam，不再需要 v1 用的 fp16 SGD workaround。

**结果**：8 层测试 config 上，ZeRO-3 v2 = ~4.9 GB（vs ZeRO-1 13.65 GB，-64%）。4× A5000 完整模型上 ~9 GB/卡。

**Trade-off**：experts 分片之后，每个 expert 调用都会触发它 3 个 Linear 的 AllGather → 16 experts × 3 linears × 32 layers = 1536 次 AllGather/forward。gradient checkpointing 翻倍。没有 bucketing 的话，就是慢。

---

## 没补的 trade-off

这些是生产框架做了我们没做的事情，以及代价是什么。

### NCCL bucketing
DDP 和 FSDP 都把多个参数 group 成一次 AllReduce/AllGather 调用。nanoMegatron 是每参数一次 collective。在 NVLink 机器上每次调用的 overhead 是 ~10μs，在 PCIe SHM 上是 ~100μs，所以"能放下但慢"的 trade-off 在正常环境下没那么痛。在我们的测试机加 `NCCL_P2P_DISABLE=1` 之后，per-call cost 灾难性地累积。

**怎么加**：分配一个 flat buffer（每层模型或每个固定 bucket size 一个大 tensor）。让每个参数的 `.grad` 是 buffer 的 view。对 buffer 做一次 NCCL 调用。实现大约 ~500 行 + 仔细的内存管理。

### 通信和计算重叠
Megatron-LM 用 async AllReduce + GEMM 调度，让通信时间藏在计算后面。需要 `CUDA_DEVICE_MAX_CONNECTIONS=1` 环境变量（让 NCCL 和 compute 共用 connection 而不是串行化），加显式 stream 管理。

**怎么加**：让 SplitFunc/AllReduce 节点用 `dist.all_reduce(..., async_op=True)` 返回 handle。在**下一个**依赖结果的操作之前 wait 这个 handle。可行但要给 autograd 集成加不少复杂度。

### FlatParameter（PyTorch FSDP 风格）
FSDP 在物理上把一层的所有参数放在一个连续的 flat tensor 里。AllGather 传一个大 buffer 而不是 N 个小的，内存布局对 allocator 也更友好。我们用 Python list 存 `nn.Parameter`，简单但慢。

### CPU/NVMe offload（ZeRO-Offload, ZeRO-Infinity）
DeepSpeed 可以把 optimizer states 倒到 CPU 内存（甚至 NVMe，用 ZeRO-Infinity），让你训练大到完全超出 GPU 内存的模型。nanoMegatron 没做。

### MoE 负载均衡
生产 MoE 框架加一个辅助 loss 项鼓励 expert 使用均衡，并用 "capacity factor" 限制工作不均衡（超出 expert 容量的 token 会被丢或填充）。nanoMegatron 让 routing 是 gate 输出的样子 —— SFT 小数据集没问题，预训练 scale 上会有问题。

### Sequence Parallel（Megatron 风格）
Megatron 的 SP 在 TP 之上把 LayerNorm 和 Dropout 沿 sequence 维切分，对这些操作的激活内存省 `tp_size` 倍。我们有 stub，但没怎么用。

### 1F1B pipeline 调度
Megatron 的 1F1B pipeline 调度交叉 forward 和 backward，让激活内存保持在 `O(P)` 而不是 GPipe 的 `O(M)`（P = pipeline stage 数，M = micro-batch 数）。我们的 PP 只有 GPipe。

### Tensor + Sequence Parallel + MoE 在一个块里
Megatron-MoE / DeepSpeed-MoE 在一个 MoE 块里同时组合 TP、SP、EP（比如 TP-shard gate 的投影、EP-shard experts、SP-slice attention 的激活，全部同时）。我们的策略大部分是正交的——experts 要么用 TP 要么用 EP，不能同时。

---

## Trade-off 总结

读 nanoMegatron 的 benchmark 数字的正确方式是：

> nanoMegatron 的算法能放下 DeepSpeed/FSDP 能放下的同样模型，但在慢路径（ZeRO-2/3）上吞吐低 5–10×，因为我们没有 bucketing 和异步 overlap。这个差距在 TP 和 EP 上变小，因为这些算法本身的形状不会产生 per-param NCCL 瀑布。

如果你在学习算法**怎么工作**，没问题——数学和数据流是一样的。如果你在训真正的模型，用 DeepSpeed 或 FSDP。
