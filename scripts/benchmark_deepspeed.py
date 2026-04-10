"""
DeepSpeed ZeRO 对比 benchmark。
用相同的 Phi-tiny-MoE 模型，对比 DeepSpeed 和 nanoMegatron 的 throughput / 显存。

用法：
  # ZeRO-1
  torchrun --nproc_per_node=2 scripts/benchmark_deepspeed.py --stage 1
  # ZeRO-2
  torchrun --nproc_per_node=2 scripts/benchmark_deepspeed.py --stage 2
  # ZeRO-3
  torchrun --nproc_per_node=2 scripts/benchmark_deepspeed.py --stage 3
"""

import argparse
import os
import time
import json
import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoTokenizer

from nano_megatron.utils import load_config, is_main_process
from nano_megatron.model import PhiMoEForCausalLM, load_hf_weights
from nano_megatron.data import create_dataloader


def get_ds_config(stage, lr, batch_size, grad_accum, dtype="bf16"):
    """生成 DeepSpeed 配置。bf16 比 fp16 快（不用 dynamic loss scale 探测）。"""
    use_bf16 = (dtype == "bf16")
    config = {
        "train_batch_size": batch_size * grad_accum * dist.get_world_size(),
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "steps_per_print": 9999,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": lr,
                "warmup_num_steps": 2,
                "total_num_steps": 10,
            }
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": stage,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "bf16": {"enabled": use_bf16},
        "fp16": {"enabled": not use_bf16, "loss_scale": 128, "initial_scale_power": 7} if not use_bf16 else {"enabled": False},
        "wall_clock_breakdown": False,
    }

    if stage == 3:
        config["zero_optimization"].update({
            "stage3_param_persistence_threshold": 1e4,
            "stage3_max_live_parameters": 3e7,
            "stage3_prefetch_bucket_size": 3e7,
            "stage3_max_reuse_distance": 3e7,
        })

    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--config", type=str, default="configs/benchmark.yaml")
    parser.add_argument("--max_steps", type=int, default=50)
    args = parser.parse_args()

    # DeepSpeed 自己初始化分布式
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    config = load_config(args.config)

    if is_main_process():
        print(f"[BENCHMARK] DeepSpeed ZeRO-{args.stage} on {dist.get_world_size()} GPUs")

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = PhiMoEForCausalLM(config.model)
    load_hf_weights(model, config.model.name)

    # gradient checkpointing
    model.model.enable_gradient_checkpointing()

    # DeepSpeed 配置
    ds_config = get_ds_config(
        stage=args.stage,
        lr=config.training.lr,
        batch_size=config.data.batch_size,
        grad_accum=config.training.grad_accum_steps,
    )

    # 初始化 DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )

    # 数据
    train_loader = create_dataloader(tokenizer, config, split="train")
    data_iter = iter(train_loader)

    # 训练循环
    # DeepSpeed 的 step() 内部计数 gradient_accumulation_steps，每 N 次 backward 后才真正更新
    model_engine.train()
    torch.cuda.reset_peak_memory_stats()
    grad_accum = config.training.grad_accum_steps

    if is_main_process():
        print(f"[BENCHMARK] Starting {args.max_steps} steps (grad_accum={grad_accum})...")

    log_interval = max(1, args.max_steps // 5)
    warmup_steps = max(2, args.max_steps // 5)

    throughputs = []
    for step in range(1, args.max_steps + 1):
        t0 = time.time()
        total_loss = 0.0

        for micro in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = {k: v.to(local_rank) for k, v in batch.items()}
            output = model_engine(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = output["loss"]
            model_engine.backward(loss)
            model_engine.step()  # DS 内部计数，第 grad_accum 次才真正 step
            total_loss += loss.item()

        dt = time.time() - t0
        tokens_per_sec = batch["input_ids"].numel() * grad_accum / dt

        if step > warmup_steps:
            throughputs.append(tokens_per_sec)

        if (step % log_interval == 0 or step == 1) and is_main_process():
            mem = torch.cuda.max_memory_allocated() / 1e9
            avg_loss = total_loss / grad_accum
            print(f"  step {step:5d} | loss {avg_loss:.4f} | tok/s {tokens_per_sec:.0f} | mem {mem:.1f}GB", flush=True)

    if is_main_process():
        mem = torch.cuda.max_memory_allocated() / 1e9
        avg_tps = sum(throughputs) / len(throughputs) if throughputs else 0
        print(f"\n[RESULT] DeepSpeed ZeRO-{args.stage}: avg {avg_tps:.0f} tok/s | peak mem {mem:.1f}GB")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
