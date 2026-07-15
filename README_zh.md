# nanoMegatron

极简分布式训练框架，用于学习。从零手写 DDP、ZeRO-1/2/3、TP、SP、PP、EP —— 算法和 Megatron-LM / DeepSpeed / PyTorch FSDP 一致，~2k 行。

📖 **[English](README.md)** | 🐛 **[踩坑日记 & 历史](docs/PITFALLS_zh.md)** | 📊 **[Benchmark 日志](benchmark_logs/BENCHMARK_LOG.md)**

---

## 项目内容

一套能跑起来的 SFT pipeline：[Phi-tiny-MoE](https://huggingface.co/microsoft/Phi-tiny-MoE-instruct)（3.8B 参数，16 experts，top-2 路由）+ [GSM8k](https://huggingface.co/datasets/openai/gsm8k)，每种并行策略一个文件：

| 文件 | 策略 |
|------|------|
| `parallel/ddp.py` | DDP —— 每卡完整模型，AllReduce 梯度 |
| `parallel/zero.py` | ZeRO-1 / ZeRO-2，含 flat buffer、分片 Adam、ReduceScatter 和 AllGather overlap |
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
| **ZeRO bucket 化梯度通信** | ✓（flat buffer） | ✓ | ✓ | ✓ |
| **ZeRO 异步重叠** | 部分（backward RS + 下一轮 forward AG） | ✓ | ✓ | ✓ |
| **CPU/NVMe offload** | ✗ | – | ✓（ZeRO-Infinity） | ✓（CPUOffload） |
| **FlatParameter / 连续分片** | ✓（ZeRO-2） | ✓ | ✓ | ✓ |
| **ZeRO-3 backward prefetch** | ✗ | – | ✓ | ✓ |
| **MoE EP 用 AllToAll** | ✓ | ✓（Megatron-MoE） | ✓（DeepSpeed-MoE） | – |
| **MoE 负载均衡（aux loss + capacity factor）** | ✗ | ✓ | ✓ | – |
| **GPipe / 1F1B pipeline** | 只有 GPipe | 都有 | – | – |
| **混精度策略** | 手动 fp32 master 副本 | – | ✓ | ✓ |
| **FlashAttention** | 通过 PyTorch SDPA | 显式 dispatch | – | – |

**最大的 trade-off**：核心算法和第一层生产优化已经具备，但 fused kernel、selective recompute、通信调度和 allocator 集成仍远比 Megatron Core 简单。下面的 benchmark 会持续记录这个差距，而不是把“能运行”直接写成“完成”。

**怎么选**：

| 目的 | 用什么 |
|------|--------|
| 学习每种并行算法的内部原理 | nanoMegatron |
| 训 1B–100B 生产模型 | DeepSpeed 或 PyTorch FSDP |
| 训 100B+ 用 TP+PP+SP | Megatron-LM、NeMo |
| 大规模稀疏 MoE | DeepSpeed-MoE、Megatron-MoE |
| 推理服务 | vLLM、TensorRT-LLM、SGLang |

## Benchmark

### Qwen3-0.6B 对标 Megatron Core：优化记录

当前正式目标采用 2× V100 PCIe、FP16、序列长度 1024、micro/global batch 1/2、
关闭 recompute、两端都走 portable/unfused attention，并且每一步都执行真实 Adam
更新。Megatron Core 固定为 `v0.15.3`。验收要求是三次独立作业中位数达到至少
95% 吞吐，同时峰值 HBM 不超过 Megatron 的 105%。

| 迭代 | nano tok/s | Megatron tok/s | 速度比 | nano / Megatron 峰值显存 | 改动 |
|---|---:|---:|---:|---:|---|
| D0 | 3647.8 | 5818.2 | 0.627 | 1.147 | 整 bucket AllReduce、完整参数 owner、串行 Broadcast |
| D1 | 4770.0 | 5812.2 | 0.821 | 1.328 | dtype flat buffer、均匀 tensor shard、backward ReduceScatter、bucket AllGather |
| D2* | 4974.5 | 5813.2 | 0.856 | 1.272 | 融合 QKV 和 SwiGLU 输入 GEMM，启用 fused AdamW |
| D3* | 5020.1 | 5815.9 | 0.863 | 1.272 | 按 forward 顺序发起 AllGather，由 module pre-hook 等待参数 |

`D0`、`D1` 是三次独立作业的中位数；`D2`、`D3` 目前各跑了一次，只作为探索结果，
暂不升级为正式结论。原始结果和完整协议在 `benchmark_logs/qwen3_0.6b/`，维护中的
特性与验收矩阵在 `docs/MEGATRON_PARITY.md`。

几轮尝试带来的结论：

1. 从 owner-based AllReduce/Broadcast 改成均匀 ReduceScatter/AllGather 是最大收益，
   吞吐提升 30.8%。但完整 model-dtype grad buffer 加本地 FP32 master grad 常驻，
   也暴露出了新的显存回退。
2. 融合 Q/K/V 和 gate/up 投影减少了重复保存的输入与 GEMM launch，吞吐再提升
   4.3%，`nvidia-smi` 峰值下降 642 MiB，但 allocated peak 仍有 13.12 GiB。
3. 参数 AllGather overlap 已正确工作，但在这组双卡 PCIe 拓扑上只带来 0.9%。
   当前剩余的稳定差距约为 56 ms/step 和 3.1 GiB sampled HBM。
4. 下一步继续用数据推进：拆分 forward/backward/RS/Adam/AG 时间，审计 math SDPA
   和 cross entropy 保存的 tensor，复用 main-grad storage，再正式跑三次。Megatron
   未开启 recompute 时，我们不会用 activation checkpointing 冒充显存优化。

作业编号：D1 `48899170`、`48901884`、`48901885`；D2 `48908022`；
D3 `48908122`。

两组 head-to-head 对比都跑在 Ibex 上，统一用完整 3.8B Phi-tiny-MoE（32 层、16 个 expert、top-2 路由），`seq_len=96`，`batch_size=1`，`grad_accum=1`，开启 gradient checkpointing，fp16，10 步。nanoMegatron / DeepSpeed 0.18.9 / PyTorch FSDP 全部基于同一份 HF checkpoint、同一份脚本（`scripts/run_v100_benchmark.sh` 和 `scripts/run_4gpu_benchmarks.sh`）。

### 4× Tesla V100 SXM2-32GB（NVLink）

PyTorch 2.8.0 + CUDA 12.8 + DeepSpeed 0.18.9，NVLink 启用。显存来自 `torch.cuda.max_memory_allocated()`。

#### 显存（peak GB / 卡）

| 策略              | nanoMegatron        | DeepSpeed           | PyTorch FSDP        |
|-------------------|:-------------------:|:-------------------:|:-------------------:|
| **ZeRO-1**        | OOM¹                | **26.3 GB** ✓       | –                   |
| **ZeRO-2**        | **27.6 GB** ✓       | **26.3 GB** ✓       | –                   |
| **ZeRO-3 / FSDP** | **20.7 GB** ✓       | （慢²）              | **18.8 GB** ✓       |
| **TP-4**          | **15.5 GB** ✓       | n/a                 | n/a                 |
| **EP-4**          | **21.1 GB** ✓       | n/a                 | n/a                 |

#### 吞吐（tok/s）

| 策略              | nanoMegatron        | DeepSpeed           | PyTorch FSDP        |
|-------------------|:-------------------:|:-------------------:|:-------------------:|
| **ZeRO-1**        | OOM                 | **31 tok/s** ✓      | –                   |
| **ZeRO-2**        | **31 tok/s** ✓      | **32 tok/s** ✓      | –                   |
| **ZeRO-3 / FSDP** | **18 tok/s** ✓      | （慢²）              | **32 tok/s** ✓      |
| **TP-4**          | **106 tok/s** ✓     | n/a                 | n/a                 |
| **EP-4**          | **207 tok/s** ✓     | n/a                 | n/a                 |

¹ nanoMegatron ZeRO-1 在 backward 期间所有 fp16 梯度都活着（没有 bucketing）→ 22+ GB peak → 加上 CUDA context 直接 OOM。DeepSpeed flat buffer 设计避开了这个问题。
² DeepSpeed ZeRO-3 在这台硬件上 30 分钟跑不完一步。它的 per-param AllGather/ReduceScatter 在 3.8B MoE 模型上每个 Linear 都触发一次 collective，overhead 太高。显存分片是有效的（10–21 GB 不对称），但吞吐太低。

### 4× NVIDIA RTX A5000 (24 GB) — `NCCL_P2P_DISABLE=1`

老的 PCIe-SHM 跑法，留在这里方便对照；除了 nano 显存数字来自 `torch.cuda.max_memory_allocated()` 而非 `nvidia-smi`，配置完全相同。

#### 显存（peak GB / 卡）

| 策略 | nanoMegatron | DeepSpeed | PyTorch FSDP |
|------|:------------:|:---------:|:------------:|
| **DDP** (fp16) | OOM | – | – |
| **ZeRO-1** | OOM (~22 GB peak) | **20.6 GB** ✓ | – |
| **ZeRO-2** | **12.0 GB** ✓ | (类似 ZeRO-1) | – |
| **ZeRO-3 / FSDP** | **9.6 GB** ✓ | **10.8 GB** ✓ | **18.8 GB** ✓ |
| **TP-4** | **15.5 GB** ✓ | n/a (DS 没有 TP) | n/a |
| **EP-4** (AllToAll) | **21.1 GB** ✓ | n/a | n/a |

#### 吞吐（tok/s）

| 策略 | nanoMegatron | DeepSpeed | PyTorch FSDP |
|------|:------------:|:---------:|:------------:|
| **ZeRO-2** | （慢*） | （hung**） | – |
| **ZeRO-3 / FSDP** | （慢*） | （hung**） | **24 tok/s** ✓ |
| **TP-4** | **154 tok/s** ✓ | n/a | n/a |
| **EP-4** | **284 tok/s** ✓ | n/a | n/a |

\* nanoMegatron ZeRO-2/3 用 per-param 同步 hook（没有 bucketing）—— 在这台 PCIe-SHM 机器上意味着每次 backward ~2k 次 NCCL 调用，每次 ~100μs 延迟。吞吐被 NCCL launch cost 卡死，不是算法问题。在 NVLink 机器加 bucketing 会快 5–10×。

\*\* DeepSpeed 的第一个 step 在这台机器上 4+ 分钟没完成（fp16/bf16 都试过，bucket size 设了 500 MB）。PyTorch FSDP 顺利跑完，因为它在 layer 粒度包装 + multi-stream prefetch 对 PCIe-SHM 更友好。上面的显存数字是 DeepSpeed engine 初始化完后从 `nvidia-smi` 抓的。

**两组实验合起来说明三件事：**

1. **nanoMegatron ZeRO-2 吞吐追平 DeepSpeed ZeRO-2**（V100：31 vs 32 tok/s）。`zero.py` 的 `_GradBucket` 把 ~1957 个 per-param NCCL 调用合并成 ~15 个 bucket 级 AllReduce，和 DeepSpeed 的 flat-buffer reducer 同一水平。显存略高（27.6 vs 26.3 GB），因为 per-param ownership bucketing 有少量额外开销。

2. **EP-4 比 ZeRO 快 6×**（V100：207 vs 31 tok/s），TP-4 快 3×（106 tok/s）。这些策略每层只有 2 个 collective（EP: AllToAll，TP: AllReduce），远少于 ZeRO 的 per-bucket AllReduce + broadcast。对 MoE 模型来说，expert parallelism 是最优解。

3. **nanoMegatron ZeRO-3 比 PyTorch FSDP 慢 1.7×**（18 vs 32 tok/s），显存接近（20.7 vs 18.8 GB）。差距在 wrapping 粒度：我们 per-Linear AllGather（~1700 次/forward），FSDP per-DecoderLayer（~32 次/forward）。要追平需要做 layer-level `FlatParameter` —— 和 FSDP 的 production 实现一样的优化。

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
