#!/bin/bash
# Portable Qwen3-0.6B baseline. Submit with: sbatch scripts/run_qwen3_baseline.sh

#SBATCH --job-name=qwen3-parity
#SBATCH --output=benchmark_logs/qwen3_0.6b/slurm_%j.out
#SBATCH --error=benchmark_logs/qwen3_0.6b/slurm_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00

set -euo pipefail

REPO_DIR=${REPO_DIR:-/ibex/project/c2334/huanyi/nanoMegatron}
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-/ibex/project/c2334/huanyi/Megatron-LM}
MEGATRON_LM_REV=${MEGATRON_LM_REV:-309ffca6a40553362d44fd17efc56b772fa1aa44}
PYTHON=${PYTHON:-/ibex/project/c2334/huanyi/conda_env/finetuning/bin/python}
CONFIG=${CONFIG:-configs/qwen3_0.6b_benchmark.yaml}
OUT_DIR=${OUT_DIR:-benchmark_logs/qwen3_0.6b/${SLURM_JOB_ID:-local}}

cd "$REPO_DIR"
mkdir -p "$OUT_DIR"
export PYTHONPATH="$REPO_DIR:$MEGATRON_LM_DIR"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=${CUDA_HOME:-/sw/rl9g/cuda/12.4.1/rl9_binary}
export PATH="$(dirname "$PYTHON"):$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29571}
export RANK=${RANK:-0}
export LOCAL_RANK=${LOCAL_RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

actual_megatron_rev=$(git -C "$MEGATRON_LM_DIR" rev-parse HEAD)
if [ "$actual_megatron_rev" != "$MEGATRON_LM_REV" ]; then
    echo "Megatron-LM revision mismatch: expected $MEGATRON_LM_REV, got $actual_megatron_rev" >&2
    exit 2
fi

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv > "$OUT_DIR/hardware.csv"
"$PYTHON" - <<'PY' > "$OUT_DIR/software.txt"
import torch
print("torch=" + torch.__version__)
print("cuda=" + str(torch.version.cuda))
PY
git -C "$REPO_DIR" rev-parse HEAD >> "$OUT_DIR/software.txt"
git -C "$MEGATRON_LM_DIR" rev-parse HEAD >> "$OUT_DIR/software.txt"

sample_memory() {
    local output=$1
    while true; do
        printf '%s,' "$(date +%s.%N)" >> "$output"
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$output"
        sleep 0.2
    done
}

run_sampled() {
    local memory_log=$1
    shift
    : > "$memory_log"
    sample_memory "$memory_log" &
    local sampler_pid=$!
    set +e
    "$@"
    local rc=$?
    set -e
    kill "$sampler_pid" 2>/dev/null || true
    wait "$sampler_pid" 2>/dev/null || true
    return "$rc"
}

run_sampled "$OUT_DIR/nano_memory.csv" \
    "$PYTHON" scripts/benchmark_qwen3.py --config "$CONFIG" \
    --output "$OUT_DIR/nano.json" 2>&1 | tee "$OUT_DIR/nano.log"

cd "$MEGATRON_LM_DIR"
# NullTokenizer reserves one EOD id, so 151935 produces Qwen's 151936 rows.
run_sampled "$REPO_DIR/$OUT_DIR/megatron_memory.csv" \
    "$PYTHON" pretrain_gpt.py \
    --use-mcore-models \
    --transformer-impl local \
    --attention-backend unfused \
    --num-layers 28 \
    --hidden-size 1024 \
    --ffn-hidden-size 3072 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 8 \
    --kv-channels 128 \
    --seq-length 1024 \
    --max-position-embeddings 40960 \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --qk-layernorm \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --no-persist-layer-norm \
    --no-gradient-accumulation-fusion \
    --swiglu \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-masked-softmax-fusion \
    --no-rope-fusion \
    --no-bias-swiglu-fusion \
    --micro-batch-size 1 \
    --global-batch-size 1 \
    --train-iters 13 \
    --lr-decay-iters 13 \
    --lr 1e-4 \
    --min-lr 1e-4 \
    --weight-decay 0.0 \
    --clip-grad 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --fp16 \
    --loss-scale 1.0 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 151935 \
    --make-vocab-size-divisible-by 128 \
    --data-cache-path "$REPO_DIR/$OUT_DIR/mcore_cache" \
    --num-workers 1 \
    --log-interval 1 \
    --eval-iters 0 \
    --eval-interval 1000 \
    --no-create-attention-mask-in-dataloader \
    --seed 1234 \
    2>&1 | tee "$REPO_DIR/$OUT_DIR/megatron.log"

cd "$REPO_DIR"
"$PYTHON" scripts/summarize_qwen3_baseline.py "$OUT_DIR"
