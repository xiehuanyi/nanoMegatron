# nanoMegatron

极简分布式训练框架。从零实现 DDP → ZeRO-1/2/3 → TP → SP → PP → EP，全部手写，用于学习各种并行策略的核心原理。

基座模型：[Phi-tiny-MoE](https://huggingface.co/microsoft/Phi-tiny-MoE-instruct)（3.8B MoE，16 experts，top-2 路由）
数据集：[GSM8k](https://huggingface.co/datasets/openai/gsm8k)（小学数学，chain-of-thought SFT）

## 实验结果

在 **4× V100 32GB** 上 SFT 1000 步（batch_size=1, seq_len=96~128, grad_accum=4）：

| 策略 | 精度 | Loss (start → end) | 吞吐 | 显存/卡 |
|------|------|---------------------|------|---------|
| **TP-4**   | fp32 | 6.69 → 1.03 | 45 tok/s  | 22.5 GB |
| **ZeRO-1** | fp16 | 3.16 → 0.91 | 45 tok/s  | 27.6 GB |
| **ZeRO-2** | fp16 | 3.10 → 0.94 | 45 tok/s  | 27.6 GB |
| **ZeRO-3** | fp16 | 3.19 → 0.66 | 49 tok/s  | 26.6 GB |
| **EP-4** (AllToAll) | fp16 | 3.50 → 1.00 | 190 tok/s | 21.1 GB |

> 注：DDP 在 V100 32GB 上跑不起来。fp32 DDP 每卡需 ~61 GB：
> 参数 15 GB + 梯度 15 GB + Adam 两个 state（`exp_avg` + `exp_avg_sq`）30 GB。

### 与 DeepSpeed 官方库对比

在 **2× A5000 24GB** 上的对比（同模型、同数据、gradient checkpointing、fp16）：

| 实现 | 策略 | 吞吐 | 显存/卡 | 能否跑起来 |
|------|------|------|---------|-----------|
| **nanoMegatron** | ZeRO-1 (N=2) | - | - | OOM（需 ~27 GB/卡） |
| **nanoMegatron** | ZeRO-2 (N=2) | - | - | OOM（需 ~27 GB/卡） |
| **nanoMegatron** | ZeRO-3 (N=2) | - | - | OOM（experts 不分片，需 ~25 GB） |
| **nanoMegatron** | TP-2 (fp32, 无优化器) | 79 tok/s | 15.1 GB | 仅 forward+backward |
| **DeepSpeed** | ZeRO-3 (N=2) | 极慢* | 16.5 GB | **能跑（全分片）** |

> *注：此机器需设置 `NCCL_P2P_DISABLE=1`（PCIe P2P bug），导致 NCCL 走共享内存，3.8B 模型 ZeRO-3 的 AllGather 极慢（25 分钟 < 10 步）。
> 在正常 NVLink/P2P 机器上 DeepSpeed ZeRO-3 的吞吐会好很多。此处对比重点是**显存占用**。

**为什么 DeepSpeed 能跑但 nanoMegatron 不能？**
- DeepSpeed ZeRO-3 对**所有参数**做全分片（包括 MoE experts），每卡只存 1/N
- nanoMegatron FSDP **跳过了 MoE experts 的分片**（`fsdp.py` 第 192 行：`if name == "experts": continue`），因为 experts 太小（intermediate=448）且路由不一致可能导致 AllGather 死锁
- 结果：nanoMegatron ZeRO-3 只分片了 ~1B/3.8B = 26% 的参数，显存节省有限

### 深度分析：TP 为什么这么慢？

通过 TP profiling（`scripts/profile_tp.py`），在 2× A5000 上测量：

| 指标 | TP-2 (fp32) | Baseline (fp16, 单卡) |
|------|------------|---------------------|
| 吞吐 (forward+backward only) | 79 tok/s | 69 tok/s |
| NCCL 调用次数/步 | **2158** | 0 |
| 峰值显存 | 15.1 GB | 14.5 GB |

**TP-2 的 forward+backward 竟然比单卡 fp16 还快（79 vs 69 tok/s）**，因为每个 GPU 只算一半的参数。但原始 V100 实验 TP-4 只有 45 tok/s，和 ZeRO-1/2 相当——开销都藏在 **完整训练循环** 里。

**TP 慢的三大原因：**

1. **海量 NCCL 调用**：每步 forward+backward **2158 次** collective 操作
   - 32 层 × (1 Attention AllReduce + 16 Expert AllReduce + 2 Routing Broadcast) = ~608/pass
   - Gradient checkpointing 导致 backward 重算 forward → NCCL 调用翻倍 → ~2158/步
   - 对比：ZeRO-1/2 在 forward+backward 期间 **0 次** NCCL（只在 optimizer step 通信）
   - 每次 NCCL 在 PCIe 上有 20-50μs latency → 2158 × 35μs ≈ **75ms overhead/步**

2. **fp32 精度**：TP 不转 fp16（因为参数已分布在多卡，fp32 能放下）
   - 参数存储 2x 大 → 显存带宽 2x 消耗
   - 虽然 autocast 用 fp16 计算，但梯度是 fp32 → optimizer 更慢
   - Adam 状态（fp32）对 TP 更贵：~2B params/GPU × 12 bytes/param = 24 GB → OOM on A5000

3. **同步通信无重叠**：每个 RowParallel 的 AllReduce 是同步的
   - AllReduce 期间 GPU 计算完全暂停
   - 生产框架（Megatron-LM）用 **async AllReduce + GEMM overlap** 隐藏通信延迟

**为什么 EP-4 (190 tok/s) 比 TP-4 (45 tok/s) 快 4x？**
- EP 用 fp16（TP 用 fp32）→ 计算快 ~2x
- EP 每层只有 **2 次 AllToAll**（dispatch + combine），TP 每层有 **~19 次 AllReduce**
- EP 的 AllToAll 可以和 expert 计算 overlap，TP 的 AllReduce 是同步阻塞的

### 深度分析：为什么 ZeRO-1/2/3 显存一样？

V100 4 卡实验：ZeRO-1 = 27.6 GB，ZeRO-2 = 27.6 GB，ZeRO-3 = 26.6 GB。理论上应该递减，为什么几乎相同？

**ZeRO-1 vs ZeRO-2：完全相同（27.6 GB）**

`torch.cuda.max_memory_allocated()` 报告的是 **峰值** 显存。峰值发生在 **backward 结束后、`step()` 之前**：

```
backward 结束时（峰值！）:
  ├── fp16 参数（完整）:     7.6 GB   ← ZeRO-1 和 ZeRO-2 都一样
  ├── fp16 梯度（完整）:     7.6 GB   ← backward 期间累积，两者都是完整的
  └── fp32 参数副本 (1/N):  3.8 GB   ← 已在 __init__ 分配
  总计: ~19 GB + activations + CUDA overhead ≈ 27.6 GB
```

ZeRO-2 的优势（Reduce 到 owner vs AllReduce）**只在 `step()` 内部**才体现：
- ZeRO-1 `step()`: AllReduce 梯度 → 释放非本地 → 优化
- ZeRO-2 `step()`: Reduce 梯度到 owner → 释放非本地 → 优化

但此时 peak 已经被 backward 的完整梯度锁定了！ZeRO-2 省的是 **step 期间** 的瞬时显存，而 `max_memory_allocated` 已经被 backward 的 peak 定格。

**ZeRO-3 为什么也差不多（26.6 GB）？**

nanoMegatron 的 FSDP 实现 **跳过了 MoE experts 的分片**：

```python
# fsdp.py 第 192 行
def fsdp_wrap_module(model, group):
    for name, child in list(model.named_children()):
        if name == "experts":     # ← 跳过！
            continue
        # 只分片 Linear 和 Embedding...
```

模型参数分布：
- **被分片的**（Attention + Embedding）：~1B params = 2 GB fp16
- **未分片的**（MoE experts + gate + norms）：~2.8B params = 5.6 GB fp16

结果：ZeRO-3 只分片了 26% 的参数，省了 ~1 GB → 26.6 GB vs 27.6 GB。

**如何真正降低显存？** → 使用 DeepSpeed ZeRO-3（全分片所有参数），实测 **16.5 GB/卡**（2× A5000），比 nanoMegatron 的 26.6 GB 省了 38%。

## v2 修复：把 bug 都补上

针对上面发现的三个问题，分别做了以下修复（commit 历史可见）：

### Fix 1: TP NCCL 调用合并 → 7.5x 减少

**问题**：每个 ColumnParallel 的 _SplitFunc 和每个 RowParallel 的 _AllReduceFunc 各自触发 1 次 NCCL，32 层 × (3 attn + 16×3 expert) ≈ 2158 次/step。

**修复**（`tensor_parallel.py`）：
- 给 `ColumnParallelLinear` 加 `skip_split` 标志，给 `RowParallelLinear` 加 `skip_reduce` 标志
- Attention：用 `forward_pre_hook` 在入口对 x 做 1 次 SplitFunc，Q/K/V 跳过自己的（合并 3→1）
- MoE：新增 `TPMoEWrapper`，对 expert 路径做 1 次 SplitFunc，所有 expert 的 partial output 累积到 1 个 tensor，最后 1 次 AllReduce（合并 16→1）
  - 注意 gate 不参与 SplitFunc，否则 gate 梯度会被 AllReduce SUM 多算 N 倍

**实测**（2× A5000，profile_tp.py）：

| 指标 | v1 (修复前) | v2 (修复后) | 改善 |
|------|------------|------------|------|
| NCCL calls/step | 2158 | **288** | **7.5x ↓** |
| 吞吐 (forward+backward only) | 79 tok/s | **95 tok/s** | +20% |

### Fix 2: ZeRO-2 backward hook → 显存真正下降

**问题**：朴素 ZeRO-2 在 `step()` 里 reduce-then-free，但 `max_memory_allocated` 的 peak 发生在 backward 结束时（此时所有梯度都还在），所以 ZeRO-1/2 显存相同。

**修复**（`zero.py`）：
- 用 `register_post_accumulate_grad_hook` 在 backward 期间增量 reduce-and-free
- 每个梯度算完立即 `dist.reduce` 到 owner，非 owner 立刻 `p.grad = None`
- backward 期间最多只有 1 个完整梯度活着（而不是全部）
- 副作用：`nn.utils.clip_grad_norm_` 在不同 rank 上算的 norm 不一致 → 新增 `ZeROOptimizer.clip_grad_norm` 做分布式 grad clipping，trainer 自动用它

**实测**（2× A5000，8 层模型 ~1B 参数验证 config）：

| 实现 | Peak Memory | 备注 |
|------|------------|------|
| ZeRO-1 | 13.65 GB | 完整 grads + 1/2 fp32 + 1/2 Adam |
| **ZeRO-2 v2** | **~5.3 GB** | hook 释放非 owner grads → backward peak 下降 |

显存节省 **~60%**，但代价是吞吐变慢（per-param sync hook 通信开销大；生产框架会用 bucketing + async）。

### Fix 3: ZeRO-3 真正分片 MoE experts

**问题**：`fsdp_wrap_module` 跳过了 experts（怕不同 rank 路由不同导致 AllGather 死锁），但 experts 占了 74% 的参数，所以 ZeRO-3 几乎没有节省显存。

**修复**（`fsdp.py`）：
- 移除 `if name == "experts": continue` 跳过逻辑
- 新增 `_patch_moe_for_fsdp`：patch MoE forward 让所有 rank **始终调用所有 expert**（即使 mask 为空），保证 AllGather 调用顺序在所有 rank 上一致 → 不死锁
- `FSDPMixedOptimizer` 改用 module class 检测分片参数（而不是 name pattern），保证 expert 内部的 Linear 也被识别为 sharded
- 非分片参数（只剩 RMSNorm 等小张量）也走 fp32 Adam（之前用 SGD 是因为 experts 没分片，fp32 副本会 OOM）

**实测**（2× A5000，8 层模型 验证 config）：

| 实现 | Peak Memory | 备注 |
|------|------------|------|
| ZeRO-1 | 13.65 GB | 基线 |
| **ZeRO-3 v2** | **~4.9 GB** | 所有参数分片（包括 experts） |

显存节省 **~64%**，代价同样是吞吐：每个 expert 的 3 个 Linear 都各自 AllGather → per-step NCCL 数大涨。

### 总结：v2 修复后的对比

| 维度 | v1 (旧) | v2 (新) | 主要修复 |
|------|---------|---------|---------|
| TP NCCL/step | 2158 | 288 | SplitFunc/AllReduce 合并 |
| TP throughput (fwd+bwd) | 79 tok/s | 95 tok/s | NCCL 减少 |
| ZeRO-2 vs ZeRO-1 显存 | 相同 | -60% | backward hook |
| ZeRO-3 vs ZeRO-1 显存 | -7% (只省 attn) | -64% (含 experts) | expert 分片 + MoE patch |

**未解决的 trade-off**：v2 的 ZeRO-2/3 throughput 比 v1 慢（per-param/per-Linear sync NCCL 成本高）。生产框架（DeepSpeed FSDP）的解决办法：
- **bucketing**：把多个小 grad/param 打包成 1 个大 NCCL 调用
- **async**：通信和计算重叠，用 multi-stream 隐藏延迟
- **prefetch**：提前 AllGather 下一层的参数

这些优化代码量大、容易出 bug，超出了"一看就懂"项目的范围。

## 和生产框架的差距

为了让这个项目"一看就懂"，我刻意做了很多简化。和 Megatron-LM / DeepSpeed 相比：

| 维度 | nanoMegatron (本项目) | 生产框架 (Megatron/DeepSpeed) |
|------|----------------------|------------------------------|
| **ZeRO 通信** | per-param（~2000 次 NCCL/步）| **flat buffer**（1 次/步，省 4-6 ms） |
| **ZeRO backward** | backward 完再通信 | **backward hook overlap**（通信和计算重叠）|
| **TP Attention** | 同步 AllReduce | **async AllReduce + wgrad GEMM overlap** |
| **Sequence Parallel** | 未实现 | **LayerNorm/Dropout 沿 seq 切分**（省 TP× 激活显存）|
| **EP 路由** | ~~AllReduce SUM~~ → **AllToAll** ✅ | AllToAll + permute/unpermute |
| **Pipeline Parallel** | GPipe（存 M 个 activation）| **1F1B**（只存 P 个 activation）|
| **ZeRO-3 参数分片** | 简单的 Linear/Embedding 替换 | **FlatParameter + 预取 + CUDA 多 stream** |
| **Load balancing** | 无 | aux loss + capacity factor |

**为什么有这些差距：**
- 生产框架为了 speedup 做了大量工程（bucketing, overlap, async, prefetch, stream 调度）
- nanoMegatron 为了"可读性"保留了最直观的实现
- 例如 ZeRO flat buffer 需要额外 7.6 GB 内存（完整 flat grad），V100 32GB 放不下 3.8B 模型
- async overlap 需要 `CUDA_DEVICE_MAX_CONNECTIONS=1` + 单独的 CUDA stream，逻辑复杂

**EP 的改动（生产级做法）：**
- **旧版（AllReduce SUM）**：每个 rank 对所有 token 算 router + 跑所有 16 个 expert（非本地的输出为 0）→ AllReduce 求和
- **新版（AllToAll）**：每个 rank 只 router 本地 token → 按目标 expert 排序 → AllToAll 发到对应 rank → 本地 expert 只处理收到的 token → AllToAll 发回
- 吞吐从 170 → 190 tok/s（~12%），通信量从 O(B·S·D) 降到 O(top_k·B·S·D/N)
- 对于更大的 MoE（如 DeepSeek-V3 intermediate=14336），AllToAll 的优势会大得多

## 项目结构

```
nanoMegatron/
├── configs/
│   └── default.yaml              # 统一配置（模型/数据/训练/并行）
├── nano_megatron/
│   ├── model.py                  # Phi-tiny-MoE 模型（手写，可加载 HF 权重）
│   ├── data.py                   # GSM8k 数据加载 + CoT 格式化
│   ├── trainer.py                # 训练循环（train/eval/log/checkpoint）
│   ├── evaluate.py               # GSM8k exact match 评估
│   ├── metrics.py                # throughput / peak memory / GPU util
│   ├── utils.py                  # 配置加载等工具
│   └── parallel/
│       ├── ddp.py                # DDP (AllReduce)
│       ├── zero.py               # ZeRO-1/2（optimizer + gradient 分片）
│       ├── fsdp.py               # ZeRO-3 / FSDP（全分片，autograd.Function）
│       ├── tensor_parallel.py    # TP (Column/Row Parallel)
│       ├── sequence_parallel.py  # SP (seq 维度切分激活)
│       ├── pipeline_parallel.py  # PP (GPipe micro-batch)
│       └── expert_parallel.py    # EP (AllReduce expert 分发)
├── scripts/
│   ├── train.py                  # 训练入口
│   ├── eval.py                   # 评估入口
│   └── run_sft.sh                # SLURM 批处理脚本
└── tests/
    └── test_all.py               # 20 个单元测试
```

## 快速开始

```bash
pip install -r requirements.txt

# 运行全部测试
python -m pytest tests/test_all.py -v
```

## 如何复现实验

### 1. 单机多卡训练（推荐）

```bash
# ZeRO-1
torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml --strategy zero1

# ZeRO-2
torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml --strategy zero2

# ZeRO-3 (FSDP)
torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml --strategy zero3

# Tensor Parallel (4 卡)
torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml --strategy tp --tp_size 4

# Expert Parallel (4 卡, 16 experts / 4 = 4 experts/GPU)
torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml --strategy ep --ep_size 4
```

### 2. SLURM 集群

```bash
# 编辑 scripts/run_sft.sh 指向你的 conda 环境，然后：
sbatch scripts/run_sft.sh
```

### 3. 评估 GSM8k 精度

```bash
python scripts/eval.py --config configs/default.yaml --checkpoint checkpoints/checkpoint_1000.pt
```

### 默认配置（`configs/default.yaml`）

```yaml
data:
  max_seq_len: 128        # V100 32GB 约束
  batch_size: 1
training:
  lr: 2.0e-5
  max_steps: 1000
  grad_accum_steps: 4     # 等效 batch_size=4
  gradient_checkpointing: true
  dtype: "float16"
```

## 并行策略原理

### 数据并行

**DDP (Distributed Data Parallel)** — 每卡完整模型，AllReduce 同步梯度
```
GPU 0: 完整模型 + Batch 0  ──┐
GPU 1: 完整模型 + Batch 1  ──┤── AllReduce 梯度 ──→ 各自更新
GPU 2: 完整模型 + Batch 2  ──┤
GPU 3: 完整模型 + Batch 3  ──┘
```

**ZeRO (Zero Redundancy Optimizer)** — 分片消除冗余

| Stage | 每卡存储 | 通信 | 显存节省 |
|-------|----------|------|---------|
| ZeRO-1 | 完整参数 + 完整梯度 + 1/N Adam states | AllReduce grad → step → Broadcast param | 省 Adam |
| ZeRO-2 | 完整参数 + 1/N 梯度 + 1/N Adam states | Reduce grad → step → Broadcast param | + 省梯度 |
| ZeRO-3 | 1/N 参数 + 1/N 梯度 + 1/N Adam states | 每层 AllGather param → 计算 → ReduceScatter grad | 全省（= FSDP） |

**ZeRO-3 / FSDP 关键实现**：不用 forward/backward hook（会 NCCL 死锁），而是替换 `nn.Linear` 为 `FSDPLinear`，把 AllGather/ReduceScatter 放进 `torch.autograd.Function`。这样 backward 顺序由 autograd 图决定，所有 rank 一致，不会死锁。参考 [Tiny-FSDP](https://github.com/liangyuwang/Tiny-FSDP)。

### 模型并行

**TP (Tensor Parallel)** — 切分单层权重矩阵
```
Attention Q/K/V: ColumnParallel (按 head 切，输出一段)
Attention O:     RowParallel (各卡算一部分 → AllReduce)
Expert w1/w3:    ColumnParallel (按 intermediate 切)
Expert w2:       RowParallel (AllReduce 输出)
```
**TP 坑**：所有 rank 必须看相同数据（不能用 DistributedSampler），否则 MoE routing 不一致导致 AllReduce 大小不匹配 → 死锁。

**SP (Sequence Parallel)** — 在 TP 基础上把 LayerNorm/Dropout 沿 seq 维度切分，激活显存再省 tp 倍。

**PP (Pipeline Parallel)** — GPipe 风格，把不同层分到不同 GPU，micro-batch 流水线。

### 专家并行

**EP (Expert Parallel)** — 每卡只放一部分 expert
```
16 Experts ÷ 4 GPUs = 4 Experts/GPU
所有 rank 相同数据 → Router 路由一致 → 各卡算本地 expert → AllReduce 合并输出
```

## 踩过的坑（fp16 训练）

1. **RoPE dtype 不匹配**：`cos/sin` 是 fp32，`apply_rope` 会把 q/k 升到 fp32，但 v 留 fp16 → attention dtype 不匹配 → NaN。修复：`cos = cos.to(x.dtype)`。
2. **RMSNorm fp16 溢出**：`x.pow(2)` 在 x > 256 时溢出 fp16（65504）。修复：RMSNorm 始终 fp32 计算。
3. **fp16 Adam**：`grad²` 容易溢出。修复：维护 fp32 参数副本做 Adam 更新，同步回 fp16。
4. **TP MoE 路由抖动**：不同 rank 的 AllReduce 浮点非结合性导致 routing 微差，expert 处理 token 数不同 → AllReduce 大小不匹配。修复：broadcast routing 决策。
5. **EP AllReduce 反向**：`AllReduce SUM` 的反向应该是 identity，不是再 AllReduce（否则梯度放大 N 倍 → 爆炸）。

## 模型架构

Phi-tiny-MoE（microsoft/Phi-tiny-MoE-instruct）:

| 参数 | 值 |
|------|-----|
| 总参数量 | 3.8B |
| 激活参数量 | 1.1B (top-2/16) |
| Hidden Size | 4096 |
| Layers | 32 |
| Query Heads | 16 (GQA) |
| KV Heads | 4 |
| Head Dim | 128 |
| Experts | 16 (top-2 路由) |
| FFN Intermediate | 448 (SlimMoE 压缩) |
| Vocab Size | 32064 |

## 设计理念

- **极简**：不引入无端依赖，核心只用 PyTorch + safetensors
- **透明**：每个并行策略独立一个文件，注释解释原理
- **可学习**：小白读完注释就能理解每种并行策略的核心思想
- **可运行**：不是伪代码，每种策略都可以实际跑起来

## 依赖

- PyTorch >= 2.1（SDPA 支持）
- transformers（仅用于 tokenizer）
- datasets（加载 GSM8k）
- safetensors（加载 HF 权重）
- pyyaml（配置文件）

## 参考

- [DeepSpeed ZeRO paper](https://arxiv.org/abs/1910.02054)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)（TP 原理）
- [GPipe](https://arxiv.org/abs/1811.06965)（PP 原理）
- [Tiny-FSDP](https://github.com/liangyuwang/Tiny-FSDP)（FSDP 极简实现参考）
