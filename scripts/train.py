"""
训练入口脚本。

用法：
  # 单卡
  python scripts/train.py --config configs/default.yaml

  # 多卡 DDP
  torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml

  # ZeRO-2
  torchrun --nproc_per_node=4 scripts/train.py --config configs/default.yaml --strategy zero2

  # Tensor Parallel (2 卡)
  torchrun --nproc_per_node=2 scripts/train.py --config configs/default.yaml --strategy tp --tp_size 2
"""

import argparse
import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from nano_megatron.utils import load_config, is_main_process
from nano_megatron.model import PhiMoEForCausalLM, load_hf_weights
from nano_megatron.data import create_dataloader
from nano_megatron.trainer import Trainer
from nano_megatron.parallel import STRATEGIES


def main():
    parser = argparse.ArgumentParser(description="nanoMegatron 训练脚本")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--strategy", type=str, default=None, help="覆盖 config 中的并行策略")
    parser.add_argument("--tp_size", type=int, default=None)
    parser.add_argument("--pp_size", type=int, default=None)
    parser.add_argument("--ep_size", type=int, default=None)
    args = parser.parse_args()

    # ── 加载配置 ──
    config = load_config(args.config)
    if args.strategy:
        config.parallel.strategy = args.strategy
    if args.tp_size:
        config.parallel.tp_size = args.tp_size
    if args.pp_size:
        config.parallel.pp_size = args.pp_size
    if args.ep_size:
        config.parallel.ep_size = args.ep_size

    # ── 初始化分布式（多卡时自动启用）──
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    if is_main_process():
        print(f"[CONFIG] Strategy: {config.parallel.strategy}")
        print(f"[CONFIG] Model: {config.model.name}")

    # ── 加载 tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 构建模型 + 加载预训练权重 ──
    if is_main_process():
        print("[MODEL] Building model...")
    model = PhiMoEForCausalLM(config.model)
    load_hf_weights(model, config.model.name)

    # ── 梯度检查点（省显存）──
    if getattr(config.training, "gradient_checkpointing", False):
        model.model.enable_gradient_checkpointing()
        if is_main_process():
            print("[CONFIG] Gradient checkpointing enabled")

    # ── 应用并行策略 ──
    strategy = config.parallel.strategy
    custom_optimizer = None
    train_step_fn = None

    if strategy in STRATEGIES and dist.is_initialized():
        setup_fn = STRATEGIES[strategy]
        result = setup_fn(model, config)
        model, extra = result
        # extra 可能是自定义 optimizer 或 train_step_fn
        if callable(extra):
            train_step_fn = extra
        elif extra is not None:
            custom_optimizer = extra
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    # ── 创建优化器和调度器 ──
    if custom_optimizer:
        optimizer = custom_optimizer
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )

    # 线性 warmup + cosine decay
    from torch.optim.lr_scheduler import LambdaLR
    warmup = config.training.warmup_steps
    max_steps = config.training.max_steps

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        import math
        progress = (step - warmup) / max(1, max_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    # ZeROOptimizer 不是 torch.optim.Optimizer 子类，需要对内部 optimizer 挂 scheduler
    inner_opt = getattr(optimizer, "optimizer", optimizer)
    scheduler = LambdaLR(inner_opt, lr_lambda)

    # ── 创建数据加载器 ──
    train_loader = create_dataloader(tokenizer, config, split="train")
    eval_loader = create_dataloader(tokenizer, config, split="test")

    # ── 开始训练 ──
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        eval_loader=eval_loader,
        tokenizer=tokenizer,
        config=config,
        train_step_fn=train_step_fn,
    )
    trainer.train()

    # ── 清理 ──
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
