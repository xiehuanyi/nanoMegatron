#!/usr/bin/env python3
"""Synthetic Qwen3 baseline for nanoMegatron, including DP/ZeRO runs."""

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from nano_megatron.parallel.zero import FP16OptimizerWrapper, ZeROOptimizer
from nano_megatron.qwen3 import Qwen3ForCausalLM
from nano_megatron.utils import load_config


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_is_dirty() -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/qwen3_0.6b_benchmark.yaml")
    parser.add_argument("--output", default="benchmark_logs/qwen3_0.6b/nano.json")
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--measure-steps", type=int)
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--cross-step-overlap", action="store_true")
    parser.add_argument("--profile-output")
    parser.add_argument("--profile-step-start", type=int, default=10)
    parser.add_argument("--profile-step-end", type=int, default=13)
    parser.add_argument(
        "--strategy",
        choices=("none", "ddp", "zero2", "tp", "cp"),
        default="none",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This is a GPU benchmark; CUDA is not available")

    config = load_config(args.config)
    bench = config.benchmark
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else bench.warmup_steps
    measure_steps = args.measure_steps if args.measure_steps is not None else bench.measure_steps
    seq_len = args.seq_len if args.seq_len is not None else bench.seq_len
    if bench.dtype != "float16":
        raise ValueError("The portable baseline currently requires dtype=float16")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    distributed = args.strategy != "none"
    if distributed:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1

    torch.manual_seed(bench.seed)
    torch.cuda.manual_seed_all(bench.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    device = torch.device("cuda", local_rank)
    model = Qwen3ForCausalLM(config.model).half()
    if args.strategy == "tp":
        from nano_megatron.parallel.qwen3_tensor_parallel import parallelize_qwen3

        model = parallelize_qwen3(model)
    elif args.strategy == "cp":
        from nano_megatron.parallel.context_parallel import parallelize_qwen3_context

        model = parallelize_qwen3_context(model)
    # TP replacement layers are constructed after the initial cast, so cast once more.
    model = model.half().to(device)
    if bench.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model.train()
    if args.strategy == "ddp":
        model = DistributedDataParallel(model, device_ids=[local_rank], bucket_cap_mb=25)
        optimizer = FP16OptimizerWrapper(
            model.parameters(), lr=bench.lr, weight_decay=bench.weight_decay
        )
    elif args.strategy == "zero2":
        optimizer = ZeROOptimizer(
            model, lr=bench.lr, weight_decay=bench.weight_decay, stage=2
        )
    else:
        optimizer = FP16OptimizerWrapper(
            model.parameters(), lr=bench.lr, weight_decay=bench.weight_decay
        )
    for group in optimizer.optimizer.param_groups:
        group["betas"] = (bench.adam_beta1, bench.adam_beta2)
        group["eps"] = bench.adam_eps

    input_seed = bench.seed if args.strategy in ("tp", "cp") else bench.seed + rank
    generator = torch.Generator(device=device).manual_seed(input_seed)
    token_stream = torch.randint(
        0,
        config.model.vocab_size,
        (bench.micro_batch_size, seq_len + 1),
        device=device,
        generator=generator,
    )
    tokens = token_stream[:, :-1].contiguous()
    labels = token_stream[:, 1:].contiguous()

    durations = []
    losses = []
    grad_norms = []
    timing_events = []
    total_steps = warmup_steps + measure_steps
    profiler = None
    if args.profile_output and rank == 0:
        profile_path = Path(args.profile_output)
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=max(args.profile_step_start - 1, 0),
                warmup=1 if args.profile_step_start > 0 else 0,
                active=args.profile_step_end - args.profile_step_start,
                repeat=1,
            ),
            on_trace_ready=lambda prof: prof.export_chrome_trace(str(profile_path)),
            record_shapes=True,
        )
        profiler.start()
    if distributed:
        dist.barrier()
    torch.cuda.synchronize()
    for step in range(total_steps):
        if distributed and not args.cross_step_overlap:
            dist.barrier()
        if args.cross_step_overlap:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            torch.cuda.synchronize()
            start = time.perf_counter()
        with torch.profiler.record_function("nano::zero_grad"):
            optimizer.zero_grad()
        with torch.profiler.record_function("nano::forward"):
            output = model(
                tokens,
                labels=labels,
                return_logits=False,
                labels_shifted=True,
            )
        with torch.profiler.record_function("nano::backward"):
            output["loss"].backward()
        if args.strategy == "tp":
            from nano_megatron.parallel.qwen3_tensor_parallel import sync_replicated_grads

            with torch.profiler.record_function("nano::tp_grad_sync"):
                sync_replicated_grads(model)
        elif args.strategy == "cp":
            from nano_megatron.parallel.context_parallel import (
                sync_context_parallel_grads,
            )

            with torch.profiler.record_function("nano::cp_grad_sync"):
                sync_context_parallel_grads(model)
        if args.strategy == "zero2":
            with torch.profiler.record_function("nano::grad_sync_copy_norm"):
                grad_norms.append(optimizer.grad_norm())
        with torch.profiler.record_function("nano::optimizer_step_param_gather"):
            optimizer.step()
        if args.cross_step_overlap:
            end_event.record()
            timing_events.append((start_event, end_event))
        else:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if distributed:
                elapsed_tensor = torch.tensor(elapsed, device=device)
                dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
                elapsed = elapsed_tensor.item()

        if step + 1 == warmup_steps:
            torch.cuda.reset_peak_memory_stats(device)
        if step >= warmup_steps:
            losses.append(output["loss"].item())
            if not args.cross_step_overlap:
                durations.append(elapsed)
        if rank == 0 and not args.cross_step_overlap:
            print(
                json.dumps(
                    {
                        "event": "step",
                        "step": step + 1,
                        "warmup": step < warmup_steps,
                        "loss": output["loss"].item(),
                        "elapsed_ms": elapsed * 1000,
                    }
                ),
                flush=True,
            )
        if profiler is not None:
            profiler.step()

    if args.strategy == "zero2":
        optimizer.finish_param_sync()
    if profiler is not None:
        profiler.stop()
    torch.cuda.synchronize()
    if args.cross_step_overlap:
        for step, (start_event, end_event) in enumerate(timing_events):
            elapsed = start_event.elapsed_time(end_event) / 1000
            if distributed:
                elapsed_tensor = torch.tensor(elapsed, device=device)
                dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
                elapsed = elapsed_tensor.item()
            if step >= warmup_steps:
                durations.append(elapsed)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "event": "step",
                            "step": step + 1,
                            "warmup": step < warmup_steps,
                            "loss": losses[max(0, step - warmup_steps)]
                            if step >= warmup_steps else None,
                            "elapsed_ms": elapsed * 1000,
                        }
                    ),
                    flush=True,
                )

    data_parallel_size = world_size if args.strategy in ("ddp", "zero2") else 1
    tokens_per_step = bench.micro_batch_size * seq_len * data_parallel_size
    mean_seconds = statistics.mean(durations)
    base_model = getattr(model, "module", model)
    local_peak_allocated = torch.cuda.max_memory_allocated(device) / 2**30
    local_peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30
    if distributed:
        peaks = [None] * world_size
        dist.all_gather_object(peaks, (local_peak_allocated, local_peak_reserved))
    else:
        peaks = [(local_peak_allocated, local_peak_reserved)]
    result = {
        "schema_version": 1,
        "engine": "nanoMegatron",
        "strategy": args.strategy,
        "world_size": world_size,
        "data_parallel_size": data_parallel_size,
        "context_parallel_size": world_size if args.strategy == "cp" else 1,
        "model": config.model.name,
        "parameter_count": base_model.parameter_count(),
        "dtype": bench.dtype,
        "attention_backend": config.model.attention_backend,
        "micro_batch_size": bench.micro_batch_size,
        "global_batch_size": bench.micro_batch_size * data_parallel_size,
        "seq_len": seq_len,
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "gradient_checkpointing": bench.gradient_checkpointing,
        "input_protocol": (
            "replicated fixed tokens, zigzag context shards"
            if args.strategy == "cp"
            else "rank-local fixed tokens, shared generator with Megatron"
        ),
        "rank0_input_sum": token_stream.sum().item() if rank == 0 else None,
        "rank0_input_prefix": token_stream[0, :8].tolist() if rank == 0 else None,
        "cross_step_overlap": args.cross_step_overlap,
        "mean_step_ms": mean_seconds * 1000,
        "median_step_ms": statistics.median(durations) * 1000,
        "tokens_per_second": tokens_per_step / mean_seconds,
        "peak_memory_allocated_gib": max(value[0] for value in peaks),
        "peak_memory_reserved_gib": max(value[1] for value in peaks),
        "per_rank_peak_memory_gib": peaks,
        "final_loss": losses[-1],
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "git_revision": _git_revision(),
        "git_dirty": _git_is_dirty(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "measured_step_ms": [round(value * 1000, 3) for value in durations],
    }
    if args.strategy == "zero2":
        result["distributed_optimizer"] = {
            "bucket_count": len(optimizer._grad_bucket.buckets),
            "local_shard_numel": optimizer.local_shard_numel,
            "padded_parameter_count": sum(
                bucket.padded_numel for bucket in optimizer._grad_bucket.buckets
            ),
            "gradient_communication_dtype": str(
                optimizer._grad_bucket.buckets[0].grad_buffer.dtype
            ).removeprefix("torch."),
            "gradient_collective": "reduce_scatter_tensor",
            "parameter_collective": "all_gather_into_tensor",
            "final_grad_norm": grad_norms[-1],
        }
    if rank == 0:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("RESULT " + json.dumps(result), flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
