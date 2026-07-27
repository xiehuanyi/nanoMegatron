#!/bin/bash
# Two-rank Qwen3 context-parallel correctness and performance smoke test.
#
# Submit from the repository root:
#   sbatch scripts/run_qwen3_context_parallel.sh

#SBATCH --job-name=qwen3-cp2
#SBATCH --output=benchmark_logs/qwen3_0.6b/cp2_%j.out
#SBATCH --error=benchmark_logs/qwen3_0.6b/cp2_%j.err
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

REPO_DIR=${REPO_DIR:-/ibex/project/c2334/huanyi/nanoMegatron}
TORCHRUN=${TORCHRUN:-/ibex/project/c2334/huanyi/conda_env/finetuning/bin/torchrun}
OUT_DIR=${OUT_DIR:-benchmark_logs/qwen3_0.6b/cp2_${SLURM_JOB_ID:-local}}

cd "$REPO_DIR"
mkdir -p "$OUT_DIR"
export PYTHONPATH="$REPO_DIR"

"$TORCHRUN" --standalone --nproc_per_node=2 \
    scripts/check_qwen3_context_parallel.py

"$TORCHRUN" --standalone --nproc_per_node=2 \
    scripts/check_qwen3_context_parallel.py --dtype float16

"$TORCHRUN" --standalone --nproc_per_node=2 \
    scripts/benchmark_qwen3.py \
    --strategy cp \
    --seq-len 2048 \
    --warmup-steps 2 \
    --measure-steps 5 \
    --output "$OUT_DIR/nano.json"
