#!/bin/bash
# Qwen3-0.6B distributed-optimizer baseline on 2 GPUs.

#SBATCH --job-name=qwen3-dist
#SBATCH --output=benchmark_logs/qwen3_0.6b/slurm_dist_%j.out
#SBATCH --error=benchmark_logs/qwen3_0.6b/slurm_dist_%j.err
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:00:00

set -euo pipefail

REPO_DIR=${REPO_DIR:-/ibex/project/c2334/huanyi/nanoMegatron}
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-/ibex/project/c2334/huanyi/Megatron-LM}
MEGATRON_LM_REV=${MEGATRON_LM_REV:-309ffca6a40553362d44fd17efc56b772fa1aa44}
PYTHON=${PYTHON:-/ibex/project/c2334/huanyi/conda_env/finetuning/bin/python}
TORCHRUN=${TORCHRUN:-/ibex/project/c2334/huanyi/conda_env/finetuning/bin/torchrun}
CONFIG=${CONFIG:-configs/qwen3_0.6b_benchmark.yaml}
BENCHMARK_MODE=${BENCHMARK_MODE:-dp2}

case "$BENCHMARK_MODE" in
    dp2)
        NANO_STRATEGY=zero2
        GLOBAL_BATCH=2
        TP_SIZE=1
        GLOBAL_TOKENS=2048
        PROTOCOL=D5-dp2-identical-fixed-input
        MCORE_PARALLEL_ARGS=(--use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather)
        ;;
    tp2)
        NANO_STRATEGY=tp
        GLOBAL_BATCH=1
        TP_SIZE=2
        GLOBAL_TOKENS=1024
        PROTOCOL=T0-tp2
        MCORE_PARALLEL_ARGS=()
        ;;
    *)
        echo "Unknown BENCHMARK_MODE=$BENCHMARK_MODE" >&2
        exit 2
        ;;
esac

OUT_DIR=${OUT_DIR:-benchmark_logs/qwen3_0.6b/${BENCHMARK_MODE}_${SLURM_JOB_ID:-local}}
NANO_EXTRA_ARGS=()
if [[ ${CROSS_STEP_OVERLAP:-0} == 1 ]]; then
    NANO_EXTRA_ARGS+=(--cross-step-overlap)
fi
MCORE_TRAIN_ITERS=13
MCORE_LOG_INTERVAL=1
MCORE_PROFILE_ARGS=()
if [[ ${PROFILE:-0} == 1 ]]; then
    NANO_EXTRA_ARGS+=(
        --warmup-steps 10
        --measure-steps 5
        --profile-output "$OUT_DIR/nano_trace.json"
        --profile-step-start 10
        --profile-step-end 13
    )
    MCORE_TRAIN_ITERS=15
    MCORE_LOG_INTERVAL=5
    MCORE_PROFILE_ARGS=(
        --timing-log-level 2
        --timing-log-option minmax
        --no-barrier-with-level-1-timing
        --profile
        --use-pytorch-profiler
        --profile-step-start 10
        --profile-step-end 13
        --profile-ranks 0
        --tensorboard-dir "$REPO_DIR/$OUT_DIR/megatron_profile"
    )
fi

cd "$REPO_DIR"
mkdir -p "$OUT_DIR"
export PYTHONPATH="$REPO_DIR:$MEGATRON_LM_DIR"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=${CUDA_HOME:-/sw/rl9g/cuda/12.4.1/rl9_binary}
export PATH="$(dirname "$PYTHON"):$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.0}

actual_megatron_rev=$(git -C "$MEGATRON_LM_DIR" rev-parse HEAD)
test "$actual_megatron_rev" = "$MEGATRON_LM_REV"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv > "$OUT_DIR/hardware.csv"
nvidia-smi topo -m > "$OUT_DIR/topology.txt"

sample_memory() {
    local output=$1
    while true; do
        local timestamp
        timestamp=$(date +%s.%N)
        nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
            awk -v ts="$timestamp" -F, '{gsub(/ /, "", $0); print ts "," $0}' >> "$output"
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
    "$TORCHRUN" --standalone --nproc_per_node=2 scripts/benchmark_qwen3.py \
    --config "$CONFIG" --strategy "$NANO_STRATEGY" --output "$OUT_DIR/nano.json" \
    "${NANO_EXTRA_ARGS[@]}" \
    2>&1 | tee "$OUT_DIR/nano.log"

cd "$MEGATRON_LM_DIR"
run_sampled "$REPO_DIR/$OUT_DIR/megatron_memory.csv" \
    "$TORCHRUN" --standalone --nproc_per_node=2 \
    "$REPO_DIR/scripts/benchmark_megatron_qwen3.py" \
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
    --global-batch-size "$GLOBAL_BATCH" \
    --train-iters "$MCORE_TRAIN_ITERS" \
    --lr-decay-iters "$MCORE_TRAIN_ITERS" \
    --lr 1e-4 \
    --min-lr 1e-4 \
    --weight-decay 0.0 \
    --clip-grad 0.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --fp16 \
    --loss-scale 1.0 \
    --tensor-model-parallel-size "$TP_SIZE" \
    --pipeline-model-parallel-size 1 \
    "${MCORE_PARALLEL_ARGS[@]}" \
    --mock-data \
    --tokenizer-type NullTokenizer \
    --vocab-size 151935 \
    --make-vocab-size-divisible-by 1187 \
    --data-cache-path "$REPO_DIR/$OUT_DIR/mcore_cache" \
    --num-workers 1 \
    --log-interval "$MCORE_LOG_INTERVAL" \
    "${MCORE_PROFILE_ARGS[@]}" \
    --eval-iters 0 \
    --eval-interval 1000 \
    --no-create-attention-mask-in-dataloader \
    --seed 1234 \
    2>&1 | tee "$REPO_DIR/$OUT_DIR/megatron.log"

cd "$REPO_DIR"
if [[ ${PROFILE:-0} != 1 ]]; then
    "$PYTHON" scripts/summarize_qwen3_baseline.py "$OUT_DIR" \
        --global-tokens-per-step "$GLOBAL_TOKENS" --protocol "$PROTOCOL"
fi
