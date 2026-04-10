"""
TP 通信开销 profiling 脚本（轻量版）。
只计数 NCCL 调用次数，不做 synchronize（避免人为引入巨大 overhead）。
然后分别测量 TP 和 baseline 的实际吞吐，从差异推断通信开销。

用法：
  torchrun --nproc_per_node=2 scripts/profile_tp.py
"""

import os
import time
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from nano_megatron.utils import load_config, is_main_process
from nano_megatron.model import PhiMoEForCausalLM, load_hf_weights
from nano_megatron.data import create_dataloader
from nano_megatron.parallel import STRATEGIES


# ── 轻量 NCCL 计数器（不做 synchronize）──
_nccl_count = 0
_orig_all_reduce = dist.all_reduce
_orig_broadcast = dist.broadcast
_orig_all_gather = dist.all_gather
_orig_reduce_scatter = dist.reduce_scatter
_orig_reduce = dist.reduce


def _counting_all_reduce(*a, **kw):
    global _nccl_count; _nccl_count += 1; return _orig_all_reduce(*a, **kw)
def _counting_broadcast(*a, **kw):
    global _nccl_count; _nccl_count += 1; return _orig_broadcast(*a, **kw)
def _counting_all_gather(*a, **kw):
    global _nccl_count; _nccl_count += 1; return _orig_all_gather(*a, **kw)
def _counting_reduce_scatter(*a, **kw):
    global _nccl_count; _nccl_count += 1; return _orig_reduce_scatter(*a, **kw)
def _counting_reduce(*a, **kw):
    global _nccl_count; _nccl_count += 1; return _orig_reduce(*a, **kw)


def enable_counting():
    dist.all_reduce = _counting_all_reduce
    dist.broadcast = _counting_broadcast
    dist.all_gather = _counting_all_gather
    dist.reduce_scatter = _counting_reduce_scatter
    dist.reduce = _counting_reduce


def disable_counting():
    dist.all_reduce = _orig_all_reduce
    dist.broadcast = _orig_broadcast
    dist.all_gather = _orig_all_gather
    dist.reduce_scatter = _orig_reduce_scatter
    dist.reduce = _orig_reduce


def reset_count():
    global _nccl_count; _nccl_count = 0


def benchmark_forward_backward(model, train_loader, num_warmup=2, num_steps=5, dtype=torch.float16):
    """Benchmark forward+backward（无 optimizer）。"""
    local_rank = dist.get_rank()
    data_iter = iter(train_loader)

    # Warmup
    for _ in range(num_warmup):
        batch = next(data_iter)
        batch = {k: v.to(local_rank) for k, v in batch.items()}
        with torch.amp.autocast("cuda", dtype=dtype):
            output = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = output["loss"]
        loss.backward()
        for p in model.parameters():
            p.grad = None

    torch.cuda.synchronize()

    # 计数 + 计时
    enable_counting()
    reset_count()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    total_tokens = 0
    for _ in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch = {k: v.to(local_rank) for k, v in batch.items()}
        total_tokens += batch["input_ids"].numel()

        with torch.amp.autocast("cuda", dtype=dtype):
            output = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = output["loss"]
        loss.backward()
        for p in model.parameters():
            p.grad = None

    torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    disable_counting()

    return {
        "total_time": total_time,
        "nccl_calls": _nccl_count,
        "nccl_calls_per_step": _nccl_count / num_steps,
        "throughput": total_tokens / total_time,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
    }


def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    config = load_config("configs/benchmark.yaml")
    config.parallel.tp_size = 2

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main_process():
        print("=" * 70)
        print("TP Communication Profiling (lightweight, no synchronize)")
        print("=" * 70)

    results = []

    # ── 1. Profile TP-2 ──
    if is_main_process():
        print("\n[1/2] Profiling TP-2 (fp32 model + fp16 autocast)...")
    torch.cuda.reset_peak_memory_stats()

    model_tp = PhiMoEForCausalLM(config.model)
    load_hf_weights(model_tp, config.model.name)
    model_tp.model.enable_gradient_checkpointing()

    config.parallel.strategy = "tp"
    model_tp, _ = STRATEGIES["tp"](model_tp, config)

    train_loader = create_dataloader(tokenizer, config, split="train")
    tp_result = benchmark_forward_backward(model_tp, train_loader)
    tp_result["name"] = "TP-2 (fp32)"
    results.append(tp_result)

    if is_main_process():
        print(f"  TP-2: {tp_result['throughput']:.0f} tok/s | "
              f"{tp_result['nccl_calls_per_step']:.0f} NCCL calls/step | "
              f"mem {tp_result['peak_mem_gb']:.1f}GB")

    del model_tp
    torch.cuda.empty_cache()

    # ── 2. Profile baseline (fp16, no comm) ──
    if is_main_process():
        print("\n[2/2] Profiling baseline (fp16 model, no parallelism)...")
    torch.cuda.reset_peak_memory_stats()

    model_base = PhiMoEForCausalLM(config.model)
    load_hf_weights(model_base, config.model.name)
    model_base = model_base.half().to(local_rank)
    model_base.model.enable_gradient_checkpointing()

    train_loader2 = create_dataloader(tokenizer, config, split="train")
    base_result = benchmark_forward_backward(model_base, train_loader2)
    base_result["name"] = "Baseline (fp16)"
    results.append(base_result)

    if is_main_process():
        print(f"  Baseline: {base_result['throughput']:.0f} tok/s | "
              f"{base_result['nccl_calls_per_step']:.0f} NCCL calls/step | "
              f"mem {base_result['peak_mem_gb']:.1f}GB")

    del model_base
    torch.cuda.empty_cache()

    # ── 输出分析 ──
    if is_main_process():
        tp, base = results[0], results[1]
        slowdown = base["throughput"] / tp["throughput"] if tp["throughput"] > 0 else float("inf")

        print("\n" + "=" * 70)
        print("PROFILING RESULTS")
        print("=" * 70)
        print(f"\n{'Metric':<30} {'TP-2 (fp32)':<20} {'Baseline (fp16)':<20}")
        print("-" * 70)
        print(f"{'Throughput (tok/s)':<30} {tp['throughput']:<20.0f} {base['throughput']:<20.0f}")
        print(f"{'Peak memory (GB)':<30} {tp['peak_mem_gb']:<20.1f} {base['peak_mem_gb']:<20.1f}")
        print(f"{'NCCL calls/step':<30} {tp['nccl_calls_per_step']:<20.0f} {base['nccl_calls_per_step']:<20.0f}")
        print(f"{'Total time ({5} steps)':<30} {tp['total_time']:<20.2f} {base['total_time']:<20.2f}")
        print(f"\nTP-2 slowdown vs baseline: {slowdown:.2f}x")
        print(f"TP NCCL calls per step: {tp['nccl_calls_per_step']:.0f}")
        print(f"\n--- Root Cause Analysis ---")
        print(f"1. NCCL call count: {tp['nccl_calls_per_step']:.0f} collective ops per forward+backward")
        print(f"   - 32 layers × (attention AllReduce + 16 expert AllReduce + routing broadcast)")
        print(f"   - Gradient checkpointing doubles forward NCCL calls")
        print(f"2. Each NCCL call on PCIe has ~20-50μs latency overhead")
        print(f"3. TP fp32 model: 2x memory → 2x parameter I/O bandwidth")
        print(f"4. No compute-communication overlap (synchronous AllReduce)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
