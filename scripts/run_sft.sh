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

echo "=========================================="
echo "Experiment 1/2: ZeRO-2 (fp16 model, ~19GB/GPU)"
echo "=========================================="
torchrun --nproc_per_node=$NPROC scripts/train.py --config $CONFIG --strategy zero2 2>&1 | tee logs/zero2.log
echo "ZeRO-2 EXIT CODE: $?"

echo "=========================================="
echo "Experiment 2/2: Expert Parallel (ep=4, fp16 model)"
echo "=========================================="
torchrun --nproc_per_node=$NPROC scripts/train.py --config $CONFIG --strategy ep --ep_size 4 2>&1 | tee logs/ep.log
echo "EP EXIT CODE: $?"

echo "=========================================="
echo "All experiments finished!"
echo "=========================================="
