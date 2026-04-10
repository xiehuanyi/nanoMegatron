"""
PyTorch 原生 FSDP 对比 benchmark。

用法：
  torchrun --nproc_per_node=2 scripts/benchmark_fsdp.py
"""

import os
import time
import functools
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoTokenizer

from nano_megatron.utils import load_config, is_main_process
from nano_megatron.model import PhiMoEForCausalLM, PhiMoEDecoderLayer, load_hf_weights
from nano_megatron.data import create_dataloader


def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/benchmark.yaml")
    parser.add_argument("--max_steps", type=int, default=50)
    args = parser.parse_args()
    config = load_config(args.config)
    max_steps = args.max_steps

    if is_main_process():
        print(f"[BENCHMARK] PyTorch FSDP on {dist.get_world_size()} GPUs")

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = PhiMoEForCausalLM(config.model)
    load_hf_weights(model, config.model.name)
    model.model.enable_gradient_checkpointing()

    # FSDP 包装：按 DecoderLayer 分片
    mp_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={PhiMoEDecoderLayer},
    )

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        device_id=local_rank,
        use_orig_params=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=0.01)

    # 数据
    train_loader = create_dataloader(tokenizer, config, split="train")
    data_iter = iter(train_loader)

    model.train()
    torch.cuda.reset_peak_memory_stats()

    if is_main_process():
        print(f"[BENCHMARK] Starting {max_steps} steps...")

    log_interval = max(1, max_steps // 5)
    warmup_steps = max(2, max_steps // 5)

    throughputs = []
    for step in range(1, max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch = {k: v.to(local_rank) for k, v in batch.items()}
        t0 = time.time()

        total_loss = 0.0
        for _ in range(config.training.grad_accum_steps):
            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(input_ids=batch["input_ids"], labels=batch["labels"])
                loss = output["loss"] / config.training.grad_accum_steps
            loss.backward()
            total_loss += loss.item() * config.training.grad_accum_steps

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        dt = time.time() - t0
        tokens_per_sec = batch["input_ids"].numel() * config.training.grad_accum_steps / dt

        if step > warmup_steps:
            throughputs.append(tokens_per_sec)

        if (step % log_interval == 0 or step == 1) and is_main_process():
            mem = torch.cuda.max_memory_allocated() / 1e9
            avg_loss = total_loss / config.training.grad_accum_steps
            print(f"  step {step:5d} | loss {avg_loss:.4f} | tok/s {tokens_per_sec:.0f} | mem {mem:.1f}GB", flush=True)

    if is_main_process():
        mem = torch.cuda.max_memory_allocated() / 1e9
        avg_tps = sum(throughputs) / len(throughputs) if throughputs else 0
        print(f"\n[RESULT] PyTorch FSDP: avg {avg_tps:.0f} tok/s | peak mem {mem:.1f}GB")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
