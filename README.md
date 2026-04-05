# nanoMegatron

极简分布式训练框架。从零实现 DDP → ZeRO-1/2/3 → TP → SP → PP → EP，全部手写，用于学习各种并行策略的核心原理。

基座模型：[Phi-tiny-MoE](https://huggingface.co/microsoft/Phi-tiny-MoE-instruct)（3.8B MoE，16 experts，top-2 路由）
数据集：[GSM8k](https://huggingface.co/datasets/openai/gsm8k)（小学数学，chain-of-thought SFT）

## 实验结果

在 **4× V100 32GB** 上 SFT 1000 步（batch_size=1, seq_len=128, grad_accum=4）：

| 策略 | 精度 | Loss (start → end) | 吞吐 | 显存/卡 |
|------|------|---------------------|------|---------|
| **TP-4**   | fp32 | 6.69 → 1.03 | 45 tok/s  | 22.5 GB |
| **ZeRO-1** | fp16 | 3.09 → 0.74 | 59 tok/s  | 27.6 GB |
| **ZeRO-2** | fp16 | 3.09 → 0.80 | 57 tok/s  | 27.6 GB |
| **ZeRO-3** | fp16 | 3.19 → 0.66 | 49 tok/s  | 26.6 GB |
| **EP-4**   | fp16 | 3.53 → 1.04 | 170 tok/s | 21.0 GB |

> 注：DDP 在 V100 32GB 上跑不起来（每卡需 ~45 GB：fp32 模型 15GB + Adam 30GB）。

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
