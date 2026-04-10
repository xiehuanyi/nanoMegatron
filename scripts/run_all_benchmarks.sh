#!/bin/bash
# 综合 benchmark 脚本：在 2× A5000 24GB 上跑所有策略
# 用法：bash scripts/run_all_benchmarks.sh

set -e
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=600
export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false

CONFIG=configs/benchmark.yaml
LOG_DIR=benchmark_logs
mkdir -p $LOG_DIR

echo "============================================================"
echo "nanoMegatron Benchmark Suite (2× A5000 24GB)"
echo "============================================================"

# ── 1. nanoMegatron ZeRO-1 (2 GPUs) ──
echo ""
echo "[1/7] nanoMegatron ZeRO-1 (2 GPUs, fp16)..."
timeout 300 torchrun --nproc_per_node=2 --master_port=29501 \
    scripts/train.py --config $CONFIG --strategy zero1 \
    2>&1 | tee $LOG_DIR/nano_zero1.log || echo "  → ZeRO-1 FAILED (likely OOM)"

# ── 2. nanoMegatron ZeRO-2 (2 GPUs) ──
echo ""
echo "[2/7] nanoMegatron ZeRO-2 (2 GPUs, fp16)..."
timeout 300 torchrun --nproc_per_node=2 --master_port=29502 \
    scripts/train.py --config $CONFIG --strategy zero2 \
    2>&1 | tee $LOG_DIR/nano_zero2.log || echo "  → ZeRO-2 FAILED (likely OOM)"

# ── 3. nanoMegatron ZeRO-3 (2 GPUs) ──
echo ""
echo "[3/7] nanoMegatron ZeRO-3 (2 GPUs, fp16)..."
timeout 300 torchrun --nproc_per_node=2 --master_port=29503 \
    scripts/train.py --config $CONFIG --strategy zero3 \
    2>&1 | tee $LOG_DIR/nano_zero3.log || echo "  → ZeRO-3 FAILED (likely OOM)"

# ── 4. nanoMegatron TP-2 (2 GPUs, fp32) ──
echo ""
echo "[4/7] nanoMegatron TP-2 (2 GPUs, fp32)..."
timeout 300 torchrun --nproc_per_node=2 --master_port=29504 \
    scripts/train.py --config $CONFIG --strategy tp --tp_size 2 \
    2>&1 | tee $LOG_DIR/nano_tp2.log || echo "  → TP-2 FAILED (likely OOM)"

# ── 5. DeepSpeed ZeRO-1 (2 GPUs) ──
echo ""
echo "[5/7] DeepSpeed ZeRO-1 (2 GPUs, fp16)..."
timeout 300 torchrun --nproc_per_node=2 --master_port=29505 \
    scripts/benchmark_deepspeed.py --stage 1 --config $CONFIG \
    2>&1 | tee $LOG_DIR/ds_zero1.log || echo "  → DS ZeRO-1 FAILED"

# ── 6. DeepSpeed ZeRO-2 (2 GPUs) ──
echo ""
echo "[6/7] DeepSpeed ZeRO-2 (2 GPUs, fp16)..."
timeout 300 torchrun --nproc_per_node=2 --master_port=29506 \
    scripts/benchmark_deepspeed.py --stage 2 --config $CONFIG \
    2>&1 | tee $LOG_DIR/ds_zero2.log || echo "  → DS ZeRO-2 FAILED"

# ── 7. DeepSpeed ZeRO-3 (2 GPUs) ──
echo ""
echo "[7/7] DeepSpeed ZeRO-3 (2 GPUs, fp16)..."
timeout 300 torchrun --nproc_per_node=2 --master_port=29507 \
    scripts/benchmark_deepspeed.py --stage 3 --config $CONFIG \
    2>&1 | tee $LOG_DIR/ds_zero3.log || echo "  → DS ZeRO-3 FAILED"

# ── 8. PyTorch FSDP (2 GPUs) ──
echo ""
echo "[8/8] PyTorch FSDP (2 GPUs, fp16)..."
timeout 300 torchrun --nproc_per_node=2 --master_port=29508 \
    scripts/benchmark_fsdp.py \
    2>&1 | tee $LOG_DIR/pt_fsdp.log || echo "  → PyTorch FSDP FAILED"

echo ""
echo "============================================================"
echo "All benchmarks completed. Logs in $LOG_DIR/"
echo "============================================================"

# 汇总结果
echo ""
echo "=== RESULTS SUMMARY ==="
for f in $LOG_DIR/*.log; do
    name=$(basename $f .log)
    result=$(grep -E "RESULT|mem [0-9]" $f 2>/dev/null | tail -1)
    if [ -n "$result" ]; then
        echo "  $name: $result"
    else
        oom=$(grep -i "out of memory\|CUDA OOM\|OutOfMemoryError" $f 2>/dev/null | head -1)
        if [ -n "$oom" ]; then
            echo "  $name: OOM"
        else
            echo "  $name: (check log)"
        fi
    fi
done
