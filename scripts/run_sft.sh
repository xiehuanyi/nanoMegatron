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
echo "Tensor Parallel (tp=4, fp32, ~20GB/GPU)"
echo "=========================================="
torchrun --nproc_per_node=$NPROC scripts/train.py --config $CONFIG --strategy tp --tp_size 4 2>&1 | tee logs/tp.log
echo "TP EXIT CODE: $?"

echo "=========================================="
echo "All experiments finished!"
echo "=========================================="
