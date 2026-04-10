# nanoMegatron

极简分布式训练框架，用于学习。从零手写 DDP → ZeRO-1/2/3 → TP → SP → PP → EP，全部 ~2k 行，跑真实的 3.8B MoE 模型。

📖 **[English](README.md)** | 🐛 **[踩坑日记 & Bug 修复](docs/PITFALLS_zh.md)** | 📊 **[Benchmark 日志](benchmark_logs/BENCHMARK_LOG.md)**

---

## 这个项目为什么存在

生产框架（Megatron-LM、DeepSpeed、PyTorch FSDP）经过大量验证，但每个都有 1 万到 5 万行源码，重度抽象、异步流、bucketing 优化层层包裹。nanoMegatron 把这些剥掉，用 100–350 行/文件展示**每种并行策略的核心思想**。

提供一套能跑起来的 SFT pipeline：[Phi-tiny-MoE](https://huggingface.co/microsoft/Phi-tiny-MoE-instruct)（3.8B 参数，16 experts，top-2 路由）+ [GSM8k](https://huggingface.co/datasets/openai/gsm8k)，并附消费级 GPU 上的真实 benchmark 数据。

## 实现的策略

| 文件 | 行数 | 策略 |
|------|------:|------|
| `ddp.py` | 31 | DDP — 每卡完整模型，AllReduce 梯度 |
| `zero.py` | 201 | ZeRO-1 / ZeRO-2（v2 用 `post_accumulate_grad_hook`） |
| `fsdp.py` | 347 | ZeRO-3 / FSDP（用 `autograd.Function` 避免 hook 死锁），分片 experts |
| `tensor_parallel.py` | 352 | TP，NCCL 合并（每层 1 次 AllReduce 而不是 17 次） |
| `sequence_parallel.py` | 177 | SP — 在 TP 之上沿 sequence 维分片 |
| `pipeline_parallel.py` | 188 | PP — GPipe 风格 |
| `expert_parallel.py` | 231 | EP — AllToAll dispatch（生产级实现） |

加上 `model.py`（Phi-tiny-MoE，312 行）和 `trainer.py`（161 行）。**总计 ~2k 行。**

## Benchmark 结果

### 4× NVIDIA RTX A5000 (24 GB)，完整 3.8B 模型，32 层

| 策略 | 显存/卡 | 吞吐 | 状态 |
|------|---------|------|------|
| DDP (fp16) | – | – | OOM（Adam fp32 副本要 ~30 GB） |
| ZeRO-1 | ~22 GB peak | – | OOM，刚好超 24 GB |
| **ZeRO-2** (v2 hooks) | ~12 GB | （慢*） | ✓ 能放下 |
| **ZeRO-3** (v2，experts 分片) | ~9 GB | （慢*） | ✓ 能放下 |
| **TP-4** | 15.5 GB | **154 tok/s** | ✓ |
| **EP-4** (AllToAll) | 21.1 GB | **284 tok/s** | ✓ |

*ZeRO-2/3 的 throughput 受限于 per-param 同步 NCCL（这台机器走 PCIe SHM）—— 详见 [PITFALLS_zh.md](docs/PITFALLS_zh.md#没补的-trade-off) 中的 bucketing trade-off 分析。

### TP v2 修复后的 throughput 提升

| 版本 | NCCL 调用/步 | 吞吐 (TP-2 fwd+bwd, A5000) |
|------|---------:|--------------------------|
| v1（per-Linear SplitFunc/AllReduce） | 2158 | 79 tok/s |
| **v2（合并）** | **288** | **95 tok/s** |

在 4× A5000 上同样的修复让 TP-4 从"比 ZeRO 慢"变成 **154 tok/s** —— 比旧的 V100×4 baseline 的 45 tok/s 快了 3.4×，而 A5000 的 fp16 算力只有 V100 的 ~50%。**等效加速约 6.8×**，全部来自 NCCL launch 数量减少。

## 与生产框架对比

我们有什么、没什么。

| 维度 | nanoMegatron | Megatron-LM | DeepSpeed | PyTorch FSDP |
|------|--------------|-------------|-----------|--------------|
| **代码行数** | ~2k 总计 | ~30k+ | ~50k+ | ~10k+ |
| **TP 合并 AllReduce** | ✓（每层 1 次） | ✓（fused QKV + async） | – | – |
| **TP 异步通信重叠** | ✗ | ✓（`CUDA_DEVICE_MAX_CONNECTIONS=1`） | – | – |
| **Sequence Parallel** | stub | ✓（LayerNorm/Dropout 沿 seq 切） | – | – |
| **ZeRO bucket 化梯度通信** | ✗（per-param） | – | ✓（flat buffer） | ✓ |
| **ZeRO 与 backward 异步重叠** | ✗ | – | ✓（multi-stream） | ✓ |
| **CPU/NVMe offload** | ✗ | – | ✓（ZeRO-Infinity） | ✓（CPUOffload） |
| **FlatParameter / 连续分片** | ✗ | – | ✓ | ✓ |
| **ZeRO-3 backward prefetch** | ✗ | – | ✓ | ✓ |
| **MoE EP 用 AllToAll** | ✓ | ✓（Megatron-MoE） | ✓（DeepSpeed-MoE） | – |
| **MoE 负载均衡（aux loss + capacity factor）** | ✗ | ✓ | ✓ | – |
| **GPipe pipeline** | ✓ | ✓ | – | – |
| **1F1B pipeline** | ✗ | ✓ | – | – |
| **自动混精度策略** | 手动 fp32 副本 | – | ✓ | ✓（`MixedPrecision` policy） |
| **FlashAttention** | 用 PyTorch SDPA（自动） | 显式 dispatch | – | – |

**最大的 trade-off**：~2k 行代码拿到了这些框架的**显存节省**，但拿不到它们的**吞吐技巧**（bucketing、async、multi-stream prefetch）。在正常 NVLink 机器上这些技巧能在算法之上再叠加 5–10× 加速；这部分占了生产框架"production-ready"的 ~80%。

**怎么选**：

| 目的 | 用什么 |
|------|--------|
| 学习每种并行算法的内部原理 | nanoMegatron |
| 训练 1B–100B 生产模型 | DeepSpeed 或 PyTorch FSDP |
| 训练 100B+ 用 TP+PP+SP | Megatron-LM、NeMo |
| 大规模稀疏 MoE | DeepSpeed-MoE、Megatron-MoE |
| 推理服务 | vLLM、TensorRT-LLM、SGLang |

## 快速开始

```bash
pip install -r requirements.txt

# 跑单元测试
python -m pytest tests/test_all.py -v

# 单卡 sanity check
python scripts/train.py --config configs/default.yaml

# 4 卡 TP（benchmark 中的"能放下 + 吞吐折衷"最优解）
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/default.yaml --strategy tp --tp_size 4

# 4 卡 EP（整体吞吐最高）
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/default.yaml --strategy ep --ep_size 4
```

> **如果第一次 NCCL AllReduce 就 hang**：你的机器可能有 PCIe P2P bug。设 `NCCL_P2P_DISABLE=1`。我们在 RTX A5000 上踩过这个坑 —— 详见 [PITFALLS_zh.md](docs/PITFALLS_zh.md#nccl-p2p-在-pcie-机器上死锁)。

## 模型结构

| 字段 | 值 |
|------|---|
| 总参数量 | 3.8B |
| 激活参数量（每个 token） | 1.1B |
| 层数 | 32 |
| Hidden size | 4096 |
| Heads | 16 (Q) / 4 (KV, GQA) |
| Head dim | 128 |
| Experts | 16（top-2） |
| Expert intermediate | 448（SlimMoE 压缩） |
| Vocab | 32064 |

## 项目结构

```
nanoMegatron/
├── nano_megatron/
│   ├── model.py                  # Phi-tiny-MoE（直接加载 HF 权重）
│   ├── data.py                   # GSM8k 加载 + chat 格式化
│   ├── trainer.py                # 训练循环
│   └── parallel/
│       ├── ddp.py                # DDP 包装
│       ├── zero.py               # ZeRO-1/2（v2: backward hook）
│       ├── fsdp.py               # ZeRO-3（v2: 分片 experts）
│       ├── tensor_parallel.py    # TP（v2: 合并 NCCL）
│       ├── sequence_parallel.py  # SP
│       ├── pipeline_parallel.py  # PP（GPipe）
│       └── expert_parallel.py    # EP（AllToAll）
├── scripts/
│   ├── train.py                  # 训练入口
│   ├── eval.py                   # GSM8k 准确率评估
│   ├── profile_tp.py             # TP NCCL 调用计数器
│   ├── benchmark_deepspeed.py    # DeepSpeed 对比
│   ├── benchmark_fsdp.py         # PyTorch FSDP 对比
│   └── run_4gpu_benchmarks.sh    # 4 卡 benchmark 套件
├── configs/
│   ├── default.yaml              # 完整训练（1000 步）
│   ├── benchmark.yaml            # 50 步 benchmark
│   ├── benchmark_4gpu.yaml       # 4 卡 10 步 benchmark
│   └── benchmark_small.yaml      # 8 层小模型（用于 ZeRO 显存验证）
├── docs/
│   ├── PITFALLS.md               # 🐛 所有踩过的 NaN、OOM、deadlock
│   └── PITFALLS_zh.md            # 🐛 中文版
├── benchmark_logs/
│   └── BENCHMARK_LOG.md          # 原始实验日志
├── tests/
│   └── test_all.py               # 20+ 单元测试
├── README.md                     # 英文版
└── README_zh.md                  # 本文件
```

## 文档

- 🐛 **[踩坑日记 & Bug 修复](docs/PITFALLS_zh.md)** —— 所有踩过的 NaN、OOM、deadlock、悄悄算错的梯度，以及对应的修复
- 📊 **[Benchmark 日志](benchmark_logs/BENCHMARK_LOG.md)** —— 原始实验输出和方法论
- 🇬🇧 **[English version](README.md)**

## 设计理念

- **极简** —— 不引入 wandb，没有花哨 CLI，没有自定义 dataloader 抽象
- **透明** —— 每种并行策略一个文件，注释充分
- **可学习** —— 一个下午能读完，普通台式机能跑起来
- **诚实** —— 既写清楚做对的部分，也写清楚和生产框架相比的 trade-off

## 参考

- [DeepSpeed ZeRO paper](https://arxiv.org/abs/1910.02054)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)（TP 基础）
- [PyTorch FSDP paper](https://arxiv.org/abs/2304.11277)
- [GPipe paper](https://arxiv.org/abs/1811.06965)（PP）
- [GShard / Switch Transformer](https://arxiv.org/abs/2006.16668)（MoE EP）
- [Tiny-FSDP](https://github.com/liangyuwang/Tiny-FSDP)（FSDP 极简参考实现）
