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
export CUDA_DEVICE_MAX_CONNECTIONS=1  # 让 NCCL 和 compute 可以 overlap

CONFIG="configs/default.yaml"
NPROC=4

echo "=========================================="
echo "ZeRO-1"
echo "=========================================="
torchrun --nproc_per_node=$NPROC --master_port=29501 scripts/train.py --config $CONFIG --strategy zero1 2>&1 | tee logs/zero1.log
echo "ZeRO-1 EXIT: $?"
sleep 10

echo "=========================================="
echo "ZeRO-2"
echo "=========================================="
torchrun --nproc_per_node=$NPROC --master_port=29502 scripts/train.py --config $CONFIG --strategy zero2 2>&1 | tee logs/zero2.log
echo "ZeRO-2 EXIT: $?"
sleep 10

echo "=========================================="
echo "EP (AllToAll dispatch, NEW)"
echo "=========================================="
torchrun --nproc_per_node=$NPROC --master_port=29503 scripts/train.py --config $CONFIG --strategy ep --ep_size 4 2>&1 | tee logs/ep.log
echo "EP EXIT: $?"

echo "=========================================="
echo "All done!"
echo "=========================================="
