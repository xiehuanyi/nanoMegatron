# Megatron Parity Tracker

目标不是只复现并行算法，而是在相同模型、batch、序列长度、精度、并行拓扑和
attention backend 下，让 nanoMegatron 达到 Megatron Core 的训练吞吐和峰值显存。

## 验收口径

- 速度：3 次独立运行的中位吞吐 `nano / Megatron >= 0.95`。
- 显存：每卡峰值 `nano / Megatron <= 1.05`，同时记录 allocated、reserved 和
  `nvidia-smi` 三种口径。
- 稳定性：稳态 step time 的变异系数不超过 3%，无 OOM、NaN 或 collective hang。
- 正确性：固定 seed 的 loss/梯度与参考实现在约定精度容差内；性能达标不能以
  改变模型语义、少算 token 或跳过 optimizer step 为代价。
- 所有结果必须记录 git revision、Megatron revision、GPU、driver、CUDA、PyTorch、
  Transformer Engine/NCCL 版本和完整命令。

状态：`DONE` 已实现并验证，`PARTIAL` 有实现但未达到生产语义/性能，`TODO` 未实现，
`N/A` 当前模型不适用。

## 特性清单

| ID | Megatron 特性 | nanoMegatron | 状态 | 性能/显存验收点 | 优先级 |
|---|---|---|---|---|---|
| M01 | Dense GPT/Qwen 模型 | Qwen3-0.6B 参数量、结构和 1024 个 shifted labels 对齐 | PARTIAL | 固定输入的 loss/梯度逐层对齐 | P0 |
| M02 | FP32/FP16/BF16 混合精度 | FP16 master weights；策略间不统一 | PARTIAL | loss scale、master grad/weight 布局对齐 | P0 |
| M03 | Tensor Parallel | Column/Row parallel，Phi-MoE 路径 | PARTIAL | TP=2/4 吞吐、显存、通信次数 | P0 |
| M04 | Sequence Parallel | 当前仅 stub/有限路径 | PARTIAL | TP+SP 激活显存和 numerics | P0 |
| M05 | Data Parallel DDP | 手写 all-reduce | DONE | bucket size、overlap 后达到 DP 基线 | P0 |
| M06 | Distributed Optimizer | dtype flat buffer + 等长 tensor shard + RS/AG | PARTIAL | D6 三次运行验收 | P0 |
| M07 | Pipeline Parallel | GPipe | PARTIAL | 1F1B、bubble、P2P overlap | P1 |
| M08 | Interleaved/virtual PP | 无 | TODO | VPP schedule 与 bubble | P1 |
| M09 | Context Parallel | 无 | TODO | 长序列 ring P2P/all-gather 路径 | P1 |
| M10 | Expert Parallel | AllToAll dispatch | PARTIAL | token permutation、grouped GEMM、负载均衡 | P1 |
| M11 | Expert Tensor Parallel | 无独立 ETP | TODO | EP/ETP folding 拓扑 | P2 |
| M12 | TP 通信计算重叠 | 同步 collective | TODO | `tp-comm-overlap` 同级隐藏率 | P0 |
| M13 | DP grad reduce overlap | post-accumulate hook 异步 ReduceScatter | PARTIAL | D1 backward/RS overlap | P0 |
| M14 | DP param gather overlap | 按 forward 顺序异步 AllGather，pre-hook 等待，支持跨 step | PARTIAL | 对齐 Megatron stream/event 调度 | P0 |
| M15 | PP 通信重叠 | 无 | TODO | send/recv 与计算重叠 | P1 |
| M16 | Contiguous param/grad buffers | 参数/梯度连续分桶，40M 元素越界后封桶，rank shard 等长 | PARTIAL | D4 已对齐为 11 buckets，待显存验收 | P0 |
| M17 | Activation recomputation | 整层 checkpoint | PARTIAL | full/selective 两档速度/显存曲线 | P0 |
| M18 | Selective recomputation | 无 | TODO | core-attn selective recompute | P1 |
| M19 | Activation CPU offload | 无 | TODO | 异步 D2H/H2D overlap | P2 |
| M20 | Attention backend | D4 对齐 local/unfused `baddbmm-softmax-bmm`；另有 SDPA | PARTIAL | A100/H100 再对标 TE/FlashAttention | P0 |
| M21 | RMSNorm | D4 使用 `torch.nn.RMSNorm`，与当前 Megatron local path 一致 | DONE | D4 `48977678` | P0 |
| M22 | Fused QKV/MLP GEMM | fused QKV 和 gate/up projection | DONE | D2 `48908022` | P0 |
| M23 | Fused RoPE | D4 缓存 cos/sin，仍是 PyTorch pointwise graph | PARTIAL | kernel 数量和 HBM traffic | P1 |
| M24 | Cross entropy | D4 custom autograd，FP32 in-place softmax，不保留返回 logits | PARTIAL | DP 已对齐，TP vocab-parallel backward 继续核对 | P0 |
| M25 | Fused optimizer | nano 使用 fused torch AdamW；Megatron D6 使用 TE FusedAdam | PARTIAL | optimizer step time、state memory | P0 |
| M26 | FP8 / Transformer Engine | 无 | TODO | H100/A100 可用矩阵分别验收 | P2 |
| M27 | CUDA Graph | 无 | TODO | steady-state launch overhead | P2 |
| M28 | MoE router aux/z loss | 无 | TODO | loss 与梯度正确性 | P1 |
| M29 | MoE capacity/drop policies | 无 | TODO | dispatch correctness 与负载 | P1 |
| M30 | Grouped GEMM for experts | Python expert loop | TODO | expert kernel 数和吞吐 | P1 |
| M31 | Distributed checkpoint | rank-0 torch save | PARTIAL | sharded save/load、并行重分片 | P1 |
| M32 | Async checkpoint | 无 | TODO | checkpoint stall time | P2 |
| M33 | Deterministic distributed RNG | 基础 seed，无 TP RNG tracker | PARTIAL | dropout/checkpoint 重算一致 | P1 |
| M34 | Gradient accumulation | 支持 | DONE | no_sync/通信频率对齐 | P0 |
| M35 | Gradient clipping/NaN detection | 部分策略支持 | PARTIAL | distributed norm 与 overflow 行为 | P1 |
| M36 | 数据预处理/packed sequence | 普通 HF DataLoader | TODO | padding 浪费、loader 吞吐 | P1 |
| M37 | MFU/通信/内存可观测性 | tok/s + peak allocated | PARTIAL | JSON、MFU、per-rank memory、NCCL trace | P0 |
| M38 | Fault tolerance/straggler detection | 无 | TODO | 大规模训练前再纳入 | P2 |

## 推进顺序

1. `M01/M02/M20/M21/M22/M24/M25/M37`：先把单卡 Qwen3-0.6B 的计算图、指标和
   kernel 基线做准，避免并行优化掩盖单卡问题。
2. `M05/M06/M13/M14/M16`：对标 DP + distributed optimizer 的连续 buffer 和 overlap。
3. `M03/M04/M12`：对标 TP + SP，逐层核对 collective 数量与 overlap。
4. `M07/M08/M15`：实现 1F1B/VPP，再扩到组合并行。
5. `M09/M10/M11/M28-M30`：长上下文和 MoE 独立矩阵。

每完成一行，都要把代码 revision 和对应实验 ID 回填到本表，不能只把“能跑”标成
`DONE`。

## 当前基线

单卡 `S0` 只保留为模型/kernel smoke test，不再用于判断 Megatron parity。正式基线从
两卡分布式实验开始：

| ID | 拓扑 | nano tok/s | Megatron tok/s | 速度比 | nano/Megatron 显存 | 当前结论 |
|---|---|---:|---:|---:|---:|---|
| D0 (旧实现) | DP2 + distributed optimizer | 3647.8 | 5818.2 | 0.627 | 1.147 | 两项均未达标 |
| D1 (flat RS/AG) | DP2 + distributed optimizer | 4770.0 | 5812.2 | 0.821 | 1.328 | 速度提升明显，显存回退，仍未达标 |
| D2* (fused projections/Adam) | DP2 + distributed optimizer | 4974.5 | 5813.2 | 0.856 | 1.272 | 单次探索，尚未正式验收 |
| D3* (+ param AG overlap) | DP2 + distributed optimizer | 5020.1 | 5815.9 | 0.863 | 1.272 | 单次探索，尚未正式验收 |
| D4 (compute graph parity) | DP2 + distributed optimizer | 8435.2 | 7610.8 | 1.108 | 1.017 | 三次中位数，速度和显存均达标 |
| D5 (identical fixed input) | DP2 + distributed optimizer | 8437.2 | 7729.2 | 1.092 | 1.017 | 三次中位数，消除输入行为差异后仍达标 |
| D6 (TE fused optimizer) | DP2 + distributed optimizer | 8436.0 | 8186.8 | 1.031 | 1.017 | 三次中位数，Megatron 启用 TE 后差距收窄到 3.1% |
| T0 | TP2 | 4860.1 | 4562.5 | 1.067 | 1.062 | 速度达标，显存未达标 |

以上均为 3 次独立运行的中位数。D0 定位出的整 bucket AllReduce、完整 parameter
round-robin owner 和串行 Broadcast 已在 D1 实现中替换为 dtype 连续 buffer、均匀
tensor shard、backward 异步 ReduceScatter 和 bucket AllGather。两 rank 的梯度累积及
Adam 更新已通过 reference 对比。D1 三次运行的中位吞吐从 3647.8 提升到 4770.0
tok/s（+30.8%），但仍只有 Megatron 的 82.1%；峰值显存从 13130 MiB 增至
15204 MiB。下一步必须拆分 forward/backward、ReduceScatter、Adam 和 AllGather
耗时，并核对 persistent grad/master-grad buffer 与激活保存量，不能把 D1 标为完成。

D2/D3 进一步融合 QKV、SwiGLU gate/up 和 AdamW，并按 forward 顺序发起参数
AllGather、由 module pre-hook 等待对应 bucket。单次吞吐分别达到 4974.5 和
5020.1 tok/s；这是有方向性的改善，但按验收规则仍需三次独立运行才能替换 D1。

D4 对齐此前最大的计算图差异：使用与 Megatron local/unfused 路径相同的 FP16
`baddbmm -> causal mask -> softmax -> bmm` attention，切换到
`torch.nn.RMSNorm`，缓存 RoPE，使用 custom in-place cross entropy，并在 benchmark
中不返回完整 logits。同时按 Megatron 的 40M 元素规则在加入参数后封桶，使 bucket
数从 13 降到 11。公平性方面，nano 现在也计算全部 1024 个 shifted labels，并在每步
执行 distributed grad norm/finite check；这两项此前都让 nano 的数字略占便宜。

D4 后仍未完全对齐的是 Megatron 的通信/多 tensor 执行细节：ReduceScatter 后除法、
model-dtype 到 FP32 main-grad copy、overflow/norm 和 optimizer 调度尚未融合；
AllGather 的依赖由 Python `Work.wait()` 驱动，而非完整的 stream/event 调度。D4
结果出来后，下一轮优先用分阶段 CUDA timing 判断瓶颈在 attention backward、grad
copy/norm、Adam 还是 param gather，避免继续凭感觉改动。

D4 三次运行 `48977678/48979015/48979016` 均位于 2× V100-SXM2 NVLink 节点。
三次中位数为 nano `8435.2 tok/s`、Megatron `7610.8 tok/s`，速度比 `1.108`；
SMI 峰值中位数为 `11710/11518 MiB`，显存比 `1.017`，两项均正式通过目标。当前
不能把 10.8% 优势简单归因于某个 kernel：nano 的训练 loop 和 module wrapper 更窄，
Megatron 还包含通用 TP layer、result validation、timer 和逐 step logging。下一轮
需用 staged CUDA timing/profiler 拆分后再下结论。

数值正确性另用相同权重和相同 shifted tokens 对比，并将 nano 的 QKV 权重重排为
Megatron GQA group layout。FP32 job `48980002` 的 logits/loss 完全一致，所有参数
gradient 最大绝对误差 `3.73e-8`、最小 cosine `0.99999988`。FP16 job `48980005`
的 loss 绝对误差 `6.48e-5`、logits 最大绝对误差 `9.77e-4`、gradient 最大绝对误差
`2.44e-4`、最小 cosine `0.99999905`，符合 FP16 舍入预期。

D4 的 nano 复用固定随机 batch，而 Megatron 每步推进 mock dataloader。D5 使用共同
的 rank-local CUDA generator，使两边生成并复用完全相同的 token/label tensor。隔离
三次运行 `48980514/48980551/48980552` 后，中位数为 nano
`8437.2 tok/s`、Megatron `7729.2 tok/s`，速度比 `1.092`，说明数据变化不是主要
原因。两边日志记录的 rank-0 token sum 均为 `77806310`，前八个 token 均为
`[144072, 109224, 139223, 44068, 82536, 147356, 12291, 89204]`，输入已逐值核对。

profiling job `48980515` 同时启用 Megatron timers 和两边的 PyTorch CUDA profiler。
三个 active steps 中，nano/Megatron 的累计 CUDA kernel time 为 `808.2/898.0 ms`；
Megatron 每步多 `763` 个 CUDA launch，多约 `29.9 ms` aggregate kernel time。
主要来源如下：

1. Megatron 当前环境没有 Apex/TE optimizer extension，fallback Adam inner step 为
   `28.7 ms`；nano 使用 fused `torch.optim.AdamW`，这是最大的单项差异。
2. Megatron 的通用 TP-capable linear/main-grad 路径增加 unscale、finite check、
   foreach update、norm、cat/copy 等 kernel。
3. Megatron 每步额外承担约 763 个小 kernel launch；三个 step 的
   `cudaLaunchKernel` CPU 时间比 nano 多 `24.7 ms`。
4. 两边主要 FP16 GEMM kernel 时间接近，因此优势不是少算 layer、hidden size 或
   token，而是 optimizer 和通用框架小 kernel/launch 开销。

D6 在相同 PyTorch 2.8.0+cu128 环境中安装 Megatron 锁定的 Transformer Engine
2.9.0。PyPI 的 core 库有 CUDA 12 wheel，但 PyTorch binding 需要本地编译；初次
编译分别因找不到 `cudnn.h` 和 `nccl.h` 失败。将
`site-packages/nvidia/*/include` 和 `site-packages/nvidia/*/lib` 加入编译/链接
路径后构建成功。运行日志确认实际后端为
`transformer_engine.pytorch.optimizers.fused_adam.FusedAdam`，multi-tensor kernel
来自 `transformer_engine_torch`，不再使用 torch/local fallback。

D6 三次运行 `48981508/48981561/48981562` 的中位数为 nano
`8436.0 tok/s`、Megatron `8186.8 tok/s`，速度比 `1.031`；峰值显存为
`11710/11510 MiB`，显存比 `1.017`。Megatron 相对 D5 提升 `5.9%`。其中
`48980514` 和 `48981561` 位于同一 `gpu212-02` 节点，Megatron 从
`7702.7` 提升到 `8186.8 tok/s`，排除了大部分跨节点差异。

D6 profiling job `48981563` 显示 Megatron steady optimizer inner step 从
`28.68 ms` 降到 `12.32 ms`，optimizer total 从 `44.46 ms` 降到 `25.38 ms`。
同一 profile job 内，Megatron 相对 nano 的额外 aggregate CUDA kernel time 为
`7.54 ms/step`，低于 D5 的 `29.94 ms/step`；额外 launch 数从 `763` 降到
`534/step`。剩余差距主要不在 Adam，而在通用 TP-capable module wrapper、
main-grad copy/check、RoPE pointwise graph 和 Python/launch 调度。

TE FusedAdam 与 fused torch AdamW 的独立正确性 job `48981695` 使用 1,000,003 个
FP32 参数、相同梯度和 D6 的 Adam 超参数运行 20 步。首步参数最大绝对误差
`3.73e-9`；20 步后参数最大绝对误差 `2.38e-7`、cosine
`0.9999999999999999`，optimizer state cosine 均高于
`0.99999999999999`。该误差属于不同 fused kernel 的 FP32 舍入差异。

## 参考口径

- Megatron Core advanced features:
  https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/index.html
- Megatron Core distributed optimizer:
  https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html
- Megatron-LM source: https://github.com/NVIDIA/Megatron-LM
- Qwen3-0.6B config: https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json
