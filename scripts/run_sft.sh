#!/bin/bash
#SBATCH --job-name=nanomega_sft
#SBATCH --output=nanomega_sft_%j.log
#SBATCH --error=nanomega_sft_%j.err
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# ============================================================
# nanoMegatron SFT 实验：在 4×V100 上跑各种并行策略
# ============================================================

set -e

cd /ibex/project/c2334/huanyi/nanoMegatron

# 公共参数
CONFIG="configs/default.yaml"
NPROC=4

echo "=========================================="
echo "Experiment 1/6: DDP"
echo "=========================================="
torchrun --nproc_per_node=$NPROC scripts/train.py --config $CONFIG --strategy ddp 2>&1 | tee logs/ddp.log

echo "=========================================="
echo "Experiment 2/6: ZeRO-1"
echo "=========================================="
torchrun --nproc_per_node=$NPROC scripts/train.py --config $CONFIG --strategy zero1 2>&1 | tee logs/zero1.log

echo "=========================================="
echo "Experiment 3/6: ZeRO-2"
echo "=========================================="
torchrun --nproc_per_node=$NPROC scripts/train.py --config $CONFIG --strategy zero2 2>&1 | tee logs/zero2.log

echo "=========================================="
echo "Experiment 4/6: ZeRO-3"
echo "=========================================="
torchrun --nproc_per_node=$NPROC scripts/train.py --config $CONFIG --strategy zero3 2>&1 | tee logs/zero3.log

echo "=========================================="
echo "Experiment 5/6: Tensor Parallel (tp=4)"
echo "=========================================="
torchrun --nproc_per_node=$NPROC scripts/train.py --config $CONFIG --strategy tp --tp_size 4 2>&1 | tee logs/tp.log

echo "=========================================="
echo "Experiment 6/6: Expert Parallel (ep=4)"
echo "=========================================="
torchrun --nproc_per_node=$NPROC scripts/train.py --config $CONFIG --strategy ep --ep_size 4 2>&1 | tee logs/ep.log

echo "=========================================="
echo "All experiments finished!"
echo "=========================================="
