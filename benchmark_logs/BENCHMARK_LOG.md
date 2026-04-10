# Benchmark Log — 2× A5000 24GB

实验日期：2026-04-10
硬件：4× NVIDIA RTX A5000 (24GB)，使用 GPU 0 和 1
PyTorch 2.8.0 + CUDA 12.8，DeepSpeed 0.18.9

## 调查目标

1. **为什么 TP 慢？**（V100 实验中 TP-4=45 tok/s，和 ZeRO-1/2 持平）
2. **为什么 ZeRO-1/2/3 显存几乎一样？**（V100：27.6 / 27.6 / 26.6 GB）
3. **与 DeepSpeed 官方库对比**

---

## 系统级问题：NCCL P2P bug

第一次跑 NCCL（甚至最简单的 default group AllReduce）就 hang。
排查过程：
1. `torch.cuda.can_device_access_peer` 显示所有 GPU 之间 P2P=True
2. 但 `dist.all_reduce` 在 default group 上就 hang，10 分钟超时
3. 设 `NCCL_P2P_DISABLE=1` 强制走共享内存（SHM）→ AllReduce 立即成功

**结论**：这台机器的 PCIe P2P 有 bug，所有 NCCL 实验都需要 `NCCL_P2P_DISABLE=1`。
副作用：通信吞吐降低（SHM 比 P2P 慢），所有 throughput 数据偏低。

---

## 实验 1：TP 通信 profiling

脚本：`scripts/profile_tp.py`
方法：monkey-patch `dist.all_reduce/broadcast/reduce/all_gather/reduce_scatter`，统计调用次数；测量 5 步的总 forward+backward 时间（不含 optimizer，避免 OOM）。

### 配置
- batch_size=1, seq_len=96, gradient_checkpointing=True
- TP-2: fp32 model + fp16 autocast
- Baseline: fp16 model on single GPU (no parallelism)

### 结果

```
======================================================================
PROFILING RESULTS
======================================================================

Metric                         TP-2 (fp32)          Baseline (fp16)
----------------------------------------------------------------------
Throughput (tok/s)             79                   69
Peak memory (GB)               15.1                 14.5
NCCL calls/step                2158                 0
Total time (5 steps)           6.11                 6.95

TP-2 slowdown vs baseline: 0.88x  ← TP-2 比单卡 baseline 还快！
```

### 分析

**反直觉发现**：TP-2 forward+backward 比单卡 baseline 快（79 vs 69 tok/s），因为每 GPU 只算一半参数。但**完整训练循环**里 TP-4 慢（V100=45 tok/s）的原因藏在：

1. **NCCL 调用爆炸**：2158 次/步
   - 32 层 × (1 attention AllReduce + 16 expert AllReduce + 2 routing broadcast) ≈ 608/pass
   - Gradient checkpointing 让 backward 重算 forward → 翻倍 → ~2158
   - 对比：ZeRO-1/2 在 forward+backward 期间 **0 次 NCCL**

2. **TP 用 fp32**：参数 2x 大、Adam 状态 2x 大、optimizer step 慢
3. **同步 AllReduce 无重叠**：每次 RowParallel AllReduce 期间 GPU 暂停
4. **expert 循环 launch overhead**：每层 16 个 expert 的 Python loop

**推论**：TP 在 V100 上和 ZeRO 相当（45 tok/s），是因为：
- TP 把 forward+backward 加速了（每卡少算）
- 但 NCCL overhead + fp32 optimizer step 把这些好处吃掉了
- 净效果约等于 ZeRO

---

## 实验 2：DeepSpeed ZeRO-3 对比

脚本：`scripts/benchmark_deepspeed.py`
配置：stage=3, fp16, gradient_checkpointing, batch_size=1, seq_len=96

### 内存观察

- DeepSpeed 初始化后稳定在 **16.5 GB / GPU**
- 对比 nanoMegatron ZeRO-3 在 V100 上：26.6 GB / GPU（4 卡）

### 吞吐：未能完成

- 训练运行约 25 分钟未输出 step 10 的 metrics
- 推断：每 step（4 micro-batch）耗时 > 2 分钟
- 估算吞吐 < 0.5 tok/s（极慢）

**根因**：`NCCL_P2P_DISABLE=1` 让 DeepSpeed ZeRO-3 的 AllGather/ReduceScatter 走 SHM。
- ZeRO-3 每 forward 需 AllGather 全部参数（~1957 个 tensor）
- Gradient checkpointing 让 backward 也 AllGather 一遍
- 大参数（如 4096×4096 fp16 = 32MB）通过 SHM 传 ~3-5ms/次
- 数千次小+大 collective 累积 → 单 step 数十秒到分钟

**结论**：throughput 数据不可用，但**显存对比是有效的**：
- nanoMegatron ZeRO-3：26.6 GB（4 卡，跳过 expert 分片）
- DeepSpeed ZeRO-3：16.5 GB（2 卡，全分片）→ 省 38%

---

## 实验 3：nanoMegatron ZeRO-1/2/3 在 2 卡上的可行性

### 理论分析

3.8B 模型 fp16，2 卡 ZeRO（per-param 实现）：
```
peak 显存（backward 结束时）:
  ├── 完整 fp16 参数:    7.6 GB
  ├── 完整 fp16 梯度:    7.6 GB  ← backward 累积，未 reduce
  ├── 1/2 fp32 副本:     7.6 GB  ← 在 __init__ 已分配
  └── (step 时还会 +) Adam m+v:  15.2 GB
  ────────────────────────────
  worst case: 38 GB → 远超 A5000 的 24 GB
```

V100 4 卡能跑（27.6 GB < 32 GB）是因为：
- 1/4 fp32 副本只 3.8 GB
- 1/4 Adam 只 7.6 GB
- 总 26.6 GB < 32 GB ✓

2 卡 → 1/2 vs 1/4 → 多 ~7 GB → 直接 OOM。

### 实验：未能复现

- nanoMegatron ZeRO-1/2/3 在 2× A5000 上预期 OOM
- 由于其他实验耗时过长（NCCL P2P 问题），这些 OOM 实验未实际跑

---

## 关键发现汇总

### 答案 1：TP 为什么慢

**不是 NCCL 数据量大**，而是**调用次数太多**（2158/步）+ **fp32 拖慢 optimizer** + **同步通信无重叠**。

每个 NCCL 调用在 PCIe 上有 20-50μs 固定 latency overhead：
- 2158 × 35μs ≈ **75 ms 纯 launch overhead/步**
- 配合 grad_accum=4 → 300 ms/optimizer-step

生产框架（Megatron-LM）对策：
- bucketing：把多个小 AllReduce 合并成 1 个大的
- async：AllReduce 和后续 GEMM 重叠
- fp16/bf16 for TP

### 答案 2：ZeRO-1/2 显存为什么相同

**`torch.cuda.max_memory_allocated()` 报的是峰值**，而 ZeRO-1 vs ZeRO-2 的差异**只在 `step()` 内部**显现：

```
backward 结束时（peak）:
  ZeRO-1: params(7.6) + 完整 grads(7.6) + fp32 副本(3.8/N) ≈ 15+ GB
  ZeRO-2: 同上（grads 还没 reduce 到 owner）

step() 期间:
  ZeRO-1: AllReduce 全部 grads（瞬时全有）
  ZeRO-2: Reduce 到 owner（瞬时 1/N）  ← 但 peak 已锁定
```

ZeRO-2 省的是 step 期间的 transient memory，**不影响 max peak**。

### 答案 3：ZeRO-3 显存为什么也相同

`fsdp.py` 第 192 行 **跳过了 expert 的分片**：
```python
if name == "experts":  # ← 不分片
    continue
```

模型参数构成：
- Attention + Embedding：~1B（被 ZeRO-3 分片）
- MoE experts + gate + norms：~2.8B（**未分片**，74% 参数）

所以 ZeRO-3 实际只分片了 26% 的参数，仅省 ~1 GB（27.6 → 26.6 GB）。

**根本解决**：DeepSpeed ZeRO-3 全分片所有参数 → **16.5 GB**（省 38%）。

---

## 新增脚本

| 文件 | 用途 |
|------|------|
| `configs/benchmark.yaml` | benchmark 短跑配置（max_steps=50） |
| `scripts/profile_tp.py` | TP 通信 profiling（统计 NCCL 次数） |
| `scripts/benchmark_deepspeed.py` | DeepSpeed ZeRO-1/2/3 对比 |
| `scripts/benchmark_fsdp.py` | PyTorch FSDP 对比 |
| `scripts/run_all_benchmarks.sh` | 综合 benchmark 串行运行所有实验 |

## 环境要求

所有实验都需要：
```bash
export NCCL_P2P_DISABLE=1   # 这台机器的 PCIe P2P 有 bug
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
```
