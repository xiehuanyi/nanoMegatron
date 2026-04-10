#!/bin/bash
# 4 卡完整 benchmark：跑 nanoMegatron 全部策略
# 用完整 32 层模型 + 50 步

set +e   # 即使某个策略 OOM，也继续跑下一个
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=600
export NCCL_P2P_DISABLE=1
export TOKENIZERS_PARALLELISM=false

CONFIG=configs/benchmark_4gpu.yaml
LOG_DIR=benchmark_logs/4gpu
mkdir -p $LOG_DIR

run_one() {
    local name=$1
    local args=$2
    local logfile=$LOG_DIR/${name}.log
    echo ""
    echo "================================================================"
    echo "[$(date +%H:%M:%S)] Running $name ..."
    echo "================================================================"
    timeout 1200 torchrun --nproc_per_node=4 --master_port=29500 \
        scripts/train.py --config $CONFIG $args 2>&1 | tee $logfile
    local rc=$?
    if [ $rc -ne 0 ]; then
        if grep -qiE "out of memory|OutOfMemoryError|CUDA out" $logfile 2>/dev/null; then
            echo "[$name] → OOM"
        else
            echo "[$name] → FAILED (rc=$rc)"
        fi
    else
        echo "[$name] → OK"
    fi
}

run_one ddp       "--strategy ddp"
run_one zero1     "--strategy zero1"
run_one zero2     "--strategy zero2"
run_one zero3     "--strategy zero3"
run_one tp4       "--strategy tp --tp_size 4"
run_one ep4       "--strategy ep --ep_size 4"

echo ""
echo "================================================================"
echo "RESULTS SUMMARY"
echo "================================================================"
for f in $LOG_DIR/*.log; do
    name=$(basename $f .log)
    # 找最后一行 "step XX | loss ... | tok/s ... | mem ...GB"
    last_step=$(grep -E "^\s*step\s+[0-9]" $f 2>/dev/null | tail -1)
    peak=$(grep -E "Peak GPU memory" $f 2>/dev/null | tail -1)
    if [ -n "$last_step" ]; then
        echo "  $name: $last_step"
        if [ -n "$peak" ]; then
            echo "         $peak"
        fi
    elif grep -qiE "out of memory|OutOfMemoryError|CUDA out" $f 2>/dev/null; then
        echo "  $name: OOM"
    else
        echo "  $name: (check $f)"
    fi
done
