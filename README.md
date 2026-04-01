# nanoMegatron

极简分布式训练框架。从零实现 DDP → ZeRO → TP → SP → PP → EP，全部手写，用于学习各种并行策略的核心原理。

基座模型：[Phi-tiny-MoE](https://huggingface.co/microsoft/Phi-tiny-MoE-instruct)（3.8B MoE，16 experts，top-2 路由）
数据集：[GSM8k](https://huggingface.co/datasets/openai/gsm8k)（小学数学，chain-of-thought SFT）

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
│       ├── ddp.py                # Phase 1: DDP (AllReduce)
│       ├── zero.py               # Phase 1: ZeRO-1/2/3
│       ├── tensor_parallel.py    # Phase 2: TP (Column/Row Parallel)
│       ├── sequence_parallel.py  # Phase 2: SP (seq 维度切分激活)
│       ├── pipeline_parallel.py  # Phase 2: PP (GPipe micro-batch)
│       └── expert_parallel.py    # Phase 3: EP (AlltoAll expert 分发)
├── scripts/
│   ├── train.py                  # 训练入口
│   └── eval.py                   # 评估入口
└── tests/
    └── test_all.py               # 全部测试
```

## 快速开始

```bash
pip install -r requirements.txt

# 运行测试
python -m pytest tests/test_all.py -v

# 单卡训练
python scripts/train.py --config configs/default.yaml

# 多卡 DDP (4 GPU)
torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml --strategy ddp

# ZeRO Stage 2
torchrun --nproc_per_node=4 scripts/train.py --strategy zero2

# Tensor Parallel (2 GPU)
torchrun --nproc_per_node=2 scripts/train.py --strategy tp --tp_size 2

# Pipeline Parallel (4 GPU)
torchrun --nproc_per_node=4 scripts/train.py --strategy pp --pp_size 4

# Expert Parallel (4 GPU, 16 experts / 4 = 4 experts per GPU)
torchrun --nproc_per_node=4 scripts/train.py --strategy ep --ep_size 4

# 评估
python scripts/eval.py --checkpoint checkpoints/checkpoint_1000.pt
```

## 并行策略原理

### Phase 1: 数据并行

**DDP (Distributed Data Parallel)**
```
GPU 0: 完整模型 + Batch 0  ──┐
GPU 1: 完整模型 + Batch 1  ──┤── AllReduce 梯度 ──→ 各自更新
GPU 2: 完整模型 + Batch 2  ──┤
GPU 3: 完整模型 + Batch 3  ──┘
```
每个 GPU 有完整模型，处理不同数据，反向传播后 AllReduce 同步梯度。

**ZeRO (Zero Redundancy Optimizer)**
```
┌──────────┬──────────────────┬──────────────────┐
│ Stage    │ 每个 GPU 分片存储 │ 显存节省          │
├──────────┼──────────────────┼──────────────────┤
│ ZeRO-1   │ Optimizer States │ ~4x (Adam 有 2 份)│
│ ZeRO-2   │ + Gradients      │ ~8x              │
│ ZeRO-3   │ + Parameters     │ ~N 倍 (= FSDP)  │
└──────────┴──────────────────┴──────────────────┘
```

### Phase 2: 模型并行

**TP (Tensor Parallel)** — 切分单个层的权重矩阵
```
Attention Q/K/V: ColumnParallel (按 head 切)
Attention O:     RowParallel (AllReduce 输出)
Expert w1/w3:    ColumnParallel (按 intermediate 切)
Expert w2:       RowParallel (AllReduce 输出)
```

**SP (Sequence Parallel)** — 在 TP 基础上切分 LayerNorm 的 seq 维度
```
[B, L/tp, D] → LayerNorm → AllGather → Attention(TP) → ReduceScatter → [B, L/tp, D]
```

**PP (Pipeline Parallel)** — GPipe 风格，把不同层放到不同 GPU
```
时间 →    t0   t1   t2   t3   t4   t5   t6   t7
Stage 0:  F0   F1   F2   F3   B3   B2   B1   B0
Stage 1:       F0   F1   F2   F3   B3   B2   B1
                                    ↑ pipeline bubble
```

### Phase 3: 专家并行

**EP (Expert Parallel)** — 不同 Expert 放到不同 GPU
```
16 Experts ÷ 4 GPUs = 4 Experts/GPU

Token → Router → AlltoAll(发到 expert 所在卡) → Expert 计算 → AlltoAll(发回) → 输出
```

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

- PyTorch >= 2.1（FlashAttention 需要 SDPA 支持）
- transformers（仅用于 tokenizer）
- datasets（加载 GSM8k）
- safetensors（加载 HF 权重）
- pyyaml（配置文件）
