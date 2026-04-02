#!/bin/bash
#SBATCH --job-name=nanomega_sft
#SBATCH --output=nanomega_sft_%j.log
#SBATCH --error=nanomega_sft_%j.err
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set +e
cd /ibex/project/c2334/huanyi/nanoMegatron

source /ibex/user/xieh0a/miniconda3/etc/profile.d/conda.sh
conda activate /ibex/project/c2334/huanyi/conda_env/finetuning
pip install -q datasets safetensors pyyaml 2>/dev/null

export PYTHONPATH="/ibex/project/c2334/huanyi/nanoMegatron:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=300

CONFIG="configs/default.yaml"
NPROC=4

# ZeRO-1: 只分片 optimizer states（fp16 model ~19GB/GPU）
echo "=========================================="
echo "Experiment 1/2: ZeRO-1 (fp16 model)"
echo "=========================================="
torchrun --nproc_per_node=$NPROC scripts/train.py --config $CONFIG --strategy zero1 2>&1 | tee logs/zero1.log
echo "ZeRO-1 EXIT CODE: $?"

# ZeRO-3: 全分片 = FSDP 原理（fp16 model ~15GB/GPU）
echo "=========================================="
echo "Experiment 2/2: ZeRO-3 (fp16 model)"
echo "=========================================="
torchrun --nproc_per_node=$NPROC scripts/train.py --config $CONFIG --strategy zero3 2>&1 | tee logs/zero3.log
echo "ZeRO-3 EXIT CODE: $?"

echo "=========================================="
echo "All experiments finished!"
echo "=========================================="
