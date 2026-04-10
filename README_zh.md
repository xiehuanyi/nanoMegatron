# nanoMegatron

极简分布式训练框架，用于学习。从零手写 DDP、ZeRO-1/2/3、TP、SP、PP、EP —— 算法和 Megatron-LM / DeepSpeed / PyTorch FSDP 一致，~2k 行。

📖 **[English](README.md)** | 🐛 **[踩坑日记 & 历史](docs/PITFALLS_zh.md)** | 📊 **[Benchmark 日志](benchmark_logs/BENCHMARK_LOG.md)**

---

## 项目内容

一套能跑起来的 SFT pipeline：[Phi-tiny-MoE](https://huggingface.co/microsoft/Phi-tiny-MoE-instruct)（3.8B 参数，16 experts，top-2 路由）+ [GSM8k](https://huggingface.co/datasets/openai/gsm8k)，每种并行策略一个文件：

| 文件 | 策略 |
|------|------|
| `parallel/ddp.py` | DDP —— 每卡完整模型，AllReduce 梯度 |
| `parallel/zero.py` | ZeRO-1 / ZeRO-2（用 backward `post_accumulate_grad_hook` 释放非 owner 梯度） |
| `parallel/fsdp.py` | ZeRO-3 / FSDP，用 `autograd.Function`，分片所有 Linear（含 MoE experts） |
| `parallel/tensor_parallel.py` | TP —— Q/K/V/O 合并成每层 1 次 SplitFunc + 1 次 AllReduce；MoE 同理 |
| `parallel/sequence_parallel.py` | SP —— 在 TP 之上沿 sequence 维分片 |
| `parallel/pipeline_parallel.py` | PP —— GPipe 调度 |
| `parallel/expert_parallel.py` | EP —— AllToAll dispatch（生产级实现） |

## 与生产框架对比

| 维度 | nanoMegatron | Megatron-LM | DeepSpeed | PyTorch FSDP |
|------|--------------|-------------|-----------|--------------|
| **代码行数** | ~2k | ~30k+ | ~50k+ | ~10k+ |
| **TP 合并 AllReduce** | ✓（每层 1 次） | ✓（fused QKV + async） | – | – |
| **TP 通信和 GEMM 异步重叠** | ✗ | ✓（`CUDA_DEVICE_MAX_CONNECTIONS=1`） | – | – |
| **Sequence Parallel** | stub | ✓（LayerNorm/Dropout 沿 seq 切） | – | – |
| **ZeRO bucket 化梯度通信** | ✗（per-param） | – | ✓（flat buffer） | ✓ |
| **ZeRO 与 backward 异步重叠** | ✗ | – | ✓（multi-stream） | ✓ |
| **CPU/NVMe offload** | ✗ | – | ✓（ZeRO-Infinity） | ✓（CPUOffload） |
| **FlatParameter / 连续分片** | ✗ | – | ✓ | ✓ |
| **ZeRO-3 backward prefetch** | ✗ | – | ✓ | ✓ |
| **MoE EP 用 AllToAll** | ✓ | ✓（Megatron-MoE） | ✓（DeepSpeed-MoE） | – |
| **MoE 负载均衡（aux loss + capacity factor）** | ✗ | ✓ | ✓ | – |
| **GPipe / 1F1B pipeline** | 只有 GPipe | 都有 | – | – |
| **混精度策略** | 手动 fp32 master 副本 | – | ✓ | ✓ |
| **FlashAttention** | 通过 PyTorch SDPA | 显式 dispatch | – | – |

**最大的 trade-off**：~2k 行代码拿到了这些框架的**显存节省**，但拿不到它们的**吞吐技巧**（bucketing、async overlap、multi-stream prefetch）。在正常 NVLink 机器上这些技巧能在算法之上再叠加 5–10× 加速，这部分占了生产框架"production-ready"的大半。

**怎么选**：

| 目的 | 用什么 |
|------|--------|
| 学习每种并行算法的内部原理 | nanoMegatron |
| 训 1B–100B 生产模型 | DeepSpeed 或 PyTorch FSDP |
| 训 100B+ 用 TP+PP+SP | Megatron-LM、NeMo |
| 大规模稀疏 MoE | DeepSpeed-MoE、Megatron-MoE |
| 推理服务 | vLLM、TensorRT-LLM、SGLang |

## Benchmark

**4× NVIDIA RTX A5000 (24 GB)**，完整 3.8B Phi-tiny-MoE，`seq_len=96`，`batch_size=1`，`grad_accum=1`，开启 gradient checkpointing，`NCCL_P2P_DISABLE=1`。nanoMegatron 数据是和官方框架在同样配置下 head-to-head。

### 显存（peak GB / 卡）

| 策略 | nanoMegatron | DeepSpeed | PyTorch FSDP |
|------|:------------:|:---------:|:------------:|
| **DDP** (fp16) | OOM | – | – |
| **ZeRO-1** | OOM (~22 GB peak) | **20.6 GB** ✓ | – |
| **ZeRO-2** | **12.0 GB** ✓ | (类似 ZeRO-1) | – |
| **ZeRO-3 / FSDP** | **9.6 GB** ✓ | **10.8 GB** ✓ | **18.8 GB** ✓ |
| **TP-4** | **15.5 GB** ✓ | n/a (DS 没有 TP) | n/a |
| **EP-4** (AllToAll) | **21.1 GB** ✓ | n/a | n/a |

### 吞吐（tok/s）

| 策略 | nanoMegatron | DeepSpeed | PyTorch FSDP |
|------|:------------:|:---------:|:------------:|
| **ZeRO-2** | （慢*） | （hung**） | – |
| **ZeRO-3 / FSDP** | （慢*） | （hung**） | **24 tok/s** ✓ |
| **TP-4** | **154 tok/s** ✓ | n/a | n/a |
| **EP-4** | **284 tok/s** ✓ | n/a | n/a |

\* nanoMegatron ZeRO-2/3 用 per-param 同步 hook（没有 bucketing）—— 在这台 PCIe-SHM 机器上意味着每次 backward ~2k 次 NCCL 调用，每次 ~100μs 延迟。吞吐被 NCCL launch cost 卡死，不是算法问题。在 NVLink 机器加 bucketing 会快 5–10×。

\*\* DeepSpeed 的第一个 step 在这台机器上 4+ 分钟没完成（fp16/bf16 都试过，bucket size 设了 500 MB）。PyTorch FSDP 顺利跑完，因为它在 layer 粒度包装 + multi-stream prefetch 对 PCIe-SHM 更友好。上面的显存数字是 DeepSpeed engine 初始化完后从 `nvidia-smi` 抓的。

**这张表说明三件事：**

1. **显存：我们和官方框架相当。** nanoMegatron ZeRO-3 (9.6 GB) 实际比 PyTorch FSDP (18.8 GB) 还省一点 —— FSDP 给**所有**参数都保留 fp32 master 副本，我们只对分片参数这么做，加上 backward hook 更激进地释放非 owner 梯度。DeepSpeed ZeRO-3 (10.8 GB) 是最接近的。我们的 ZeRO-1 OOM 而 DeepSpeed 能放下，是因为他们的 flat-buffer 设计避开了我们 per-param 实现的峰值。

2. **吞吐：TP 和 EP 有竞争力，ZeRO-2/3 卡在没有 bucketing。** nanoMegatron 在这台硬件上的甜点是 TP-4 / EP-4。ZeRO 数字慢是被**工程**卡（没 bucketing、没 async overlap），不是被算法卡。和生产框架的工程差距详见 [PITFALLS_zh.md](docs/PITFALLS_zh.md#没补的-trade-off)。

3. **DeepSpeed 比 PyTorch FSDP 更受 PCIe-SHM 影响。** 这台 NCCL-P2P-disabled 机器上 DeepSpeed 跑不动一个 step，PyTorch FSDP 能跑完。这是硬件相关的 —— 在正常 NVLink 机器上 DeepSpeed 很快。

## 快速开始

```bash
pip install -r requirements.txt

# 跑单元测试
python -m pytest tests/test_all.py -v

# 单卡 sanity check
python scripts/train.py --config configs/default.yaml

# 4 卡 TP
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/default.yaml --strategy tp --tp_size 4

# 4 卡 EP（benchmark 中吞吐最高）
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/default.yaml --strategy ep --ep_size 4
```

> **如果第一次 NCCL AllReduce 就 hang**：你的机器可能有 PCIe P2P bug（我们在 RTX A5000 上踩过）。设 `NCCL_P2P_DISABLE=1`。详见 [PITFALLS_zh.md](docs/PITFALLS_zh.md#nccl-p2p-在-pcie-机器上死锁)。

## 模型

[Phi-tiny-MoE](https://huggingface.co/microsoft/Phi-tiny-MoE-instruct)：3.8B 总参数，每个 token 1.1B 激活参数，32 层，16 experts（top-2），GQA（16 Q heads / 4 KV heads），hidden 4096。`nano_megatron/model.py` 的实现直接加载 HF 权重。

## 项目结构

```
nanoMegatron/
├── nano_megatron/
│   ├── model.py              # Phi-tiny-MoE
│   ├── data.py               # GSM8k 加载
│   ├── trainer.py            # 训练循环
│   └── parallel/             # 每种策略一个文件
├── scripts/
│   ├── train.py              # 训练入口
│   ├── eval.py               # GSM8k 评估
│   ├── profile_tp.py         # NCCL 调用计数器
│   └── benchmark_*.py        # DeepSpeed / FSDP 对比脚本
├── configs/                  # YAML 配置
├── docs/
│   ├── PITFALLS.md           # 🐛 所有 NaN/OOM/deadlock + 历史
│   └── PITFALLS_zh.md
├── benchmark_logs/
│   └── BENCHMARK_LOG.md      # 原始实验输出
├── tests/
└── README.md
```

## 文档

- 🐛 **[docs/PITFALLS_zh.md](docs/PITFALLS_zh.md)** —— 所有踩过的 NaN、OOM、deadlock、悄悄算错的梯度，以及修复过程和诊断细节
- 📊 **[benchmark_logs/BENCHMARK_LOG.md](benchmark_logs/BENCHMARK_LOG.md)** —— 原始 benchmark 输出和方法论
- 🇬🇧 **[README.md](README.md)** —— English version

## 设计理念

- **极简** —— 不引入 wandb，没有花哨 CLI，没有自定义 dataloader 抽象
- **透明** —— 每种并行策略一个文件，注释充分
- **可学习** —— 一个下午能读完，普通台式机能跑起来
- **诚实** —— 把和生产框架的 trade-off 写清楚

## 参考

- [DeepSpeed ZeRO](https://arxiv.org/abs/1910.02054)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [PyTorch FSDP paper](https://arxiv.org/abs/2304.11277)
- [GPipe](https://arxiv.org/abs/1811.06965)
- [GShard / Switch Transformer](https://arxiv.org/abs/2006.16668)
- [Tiny-FSDP](https://github.com/liangyuwang/Tiny-FSDP)
