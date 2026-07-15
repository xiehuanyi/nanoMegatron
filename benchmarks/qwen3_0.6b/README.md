# Qwen3-0.6B Benchmark Matrix

## 固定模型

Qwen3-0.6B dense：28 层，hidden 1024，FFN 3072，16 Q heads，8 KV heads，
head dim 128，vocab 151936，Q/K RMSNorm，RoPE base 1,000,000，tied embedding。
权重和 token 使用固定 seed 随机生成；性能基线不下载 checkpoint，也不包含 DataLoader。

## 实验矩阵

| ID | GPU | 精度/attention | 并行 | seq | micro/global batch | recompute | 目的 | 状态 |
|---|---|---|---|---:|---:|---|---|---|
| S0 | 1x V100 | FP16 / portable math | 无 | 1024 | 1 / 1 | off | 仅做模型和 kernel 预检，不算正式 Megatron 对标 | DONE |
| D0 | 2x V100 | FP16 / portable math | DP2 + distributed optimizer | 1024 | 1 / 2 | off | DP buffer、RS/AG 和 overlap | DONE 3/3 |
| D1 | 2x V100 | FP16 / portable math | DP2 + flat distributed optimizer | 1024 | 1 / 2 | off | D0 优化后重新验收 | DONE 3/3 |
| D2 | 2x V100 | FP16 / portable math | D1 + fused projections/Adam | 1024 | 1 / 2 | off | 减少 GEMM launch 和 saved input | EXPLORE 1/3 |
| D3 | 2x V100 | FP16 / portable math | D2 + param gather overlap | 1024 | 1 / 2 | off | AG 与下一轮 forward overlap | EXPLORE 1/3 |
| T0 | 2x V100 | FP16 / portable math | TP2 | 1024 | 1 / 1 | off | TP 参数/词表分片和通信 | DONE 3/3 |
| T1 | 2x V100 | FP16 / portable math | TP2 + SP | 1024 | 1 / 1 | off | Sequence Parallel 激活显存 | TODO |
| H0 | 2x A100 | BF16 / flash | TP2 + SP | 2048 | 1 / 1 | off | production kernel 基线 | TODO |
| H1 | 4x A100 | BF16 / flash | TP2 + DP2 | 4096 | 1 / 8 | selective | 组合并行 | TODO |

## 公平性约束

- 两端模型结构、有效 token、global batch、dtype、optimizer 超参和 optimizer step 数一致。
- B0 禁用 TE/fused kernels，定位框架与计算图差异；B1 起两端都使用各自最快的等价 kernel。
- warmup 至少 3 步且必须覆盖 Adam state 初始化；之后测 10 步，正式结果跑 3 次。
- 主指标为 end-to-end training tok/s 和每卡 peak HBM；辅助指标为 step time、MFU、
  allocated/reserved、NCCL bytes/calls 和 kernel breakdown。
- 不用单个短 step 下结论。结果表只填三次独立运行的中位数。

## 结果

| ID | Revision | GPU | nano tok/s | Megatron tok/s | 速度比 | nano peak MiB | Megatron peak MiB | 显存比 | 结论 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| S0 | d5c49a8 + working tree | 1x V100 PCIe 32GB | 3364.7 | 3559.3 | 0.945 | 16110 | 16610 | 0.970 | 只作预检，不作为项目结论 |
| D0 | d5c49a8 + working tree | 2x V100 PCIe 32GB | 3647.8 | 5818.2 | 0.627 | 13130 | 11446 | 1.147 | 速度、显存均未达标 |
| D1 | d5c49a8 + working tree | 2x V100 PCIe 32GB | 4770.0 | 5812.2 | 0.821 | 15204 | 11446 | 1.328 | 吞吐 +30.8%，显存回退，仍未达标 |
| D2* | d5c49a8 + working tree | 2x V100 PCIe 32GB | 4974.5 | 5813.2 | 0.856 | 14562 | 11446 | 1.272 | 单次探索结果 |
| D3* | d5c49a8 + working tree | 2x V100 PCIe 32GB | 5020.1 | 5815.9 | 0.863 | 14562 | 11446 | 1.272 | 单次探索结果 |
| T0 | d5c49a8 + working tree | 2x V100 PCIe 32GB | 4860.1 | 4562.5 | 1.067 | 8516 | 8016 | 1.062 | 速度达标，显存高 6.2% |

原始结果放在 `benchmark_logs/qwen3_0.6b/<job_id>/`。`summary.json` 是表格的唯一数据源，
原始 `.log`、显存采样和软件/硬件清单必须一并保留。

B0 参考版本固定为 Megatron Core `v0.15.3`
(`309ffca6a40553362d44fd17efc56b772fa1aa44`)；这是支持当前 Python 3.10
benchmark 环境的稳定版本。生产 kernel 实验会在独立 Python 3.12 + TE 环境中跟踪
当前 Megatron Core。

S0 为 3 次独立作业的中位数（jobs `48892019`、`48892133`、`48892134`）。
Megatron 端是 v0.15.3 local/unfused 路径，不含 Transformer Engine/Apex；因此它回答的是
“portable 实现的框架开销”，不是最终 production kernel 对比。第一轮 profiling 优先检查
QKV/MLP 两组分离 GEMM 和 Python fp16-to-fp32 optimizer copy。

D0/T0 才是正式分布式 baseline，表中为三次独立作业的中位数。D0 jobs：
`48895927`、`48896947`、`48896948`；T0 jobs：`48896936`、`48896945`、
`48896946`。测试节点两张 V100 之间是 PCIe `NODE`，没有 NVLink，两端使用完全
相同的拓扑。聚合数据分别在 `benchmark_logs/qwen3_0.6b/D0_summary.json` 和
`benchmark_logs/qwen3_0.6b/T0_summary.json`。

D1 jobs：`48899170`、`48901884`、`48901885`，聚合数据在
`benchmark_logs/qwen3_0.6b/D1_summary.json`。三次 nano step time 分别为
429.35、429.21、429.46 ms，波动很小；因此 82.1% 的差距是稳定的实现差距，
不是测量噪声。
