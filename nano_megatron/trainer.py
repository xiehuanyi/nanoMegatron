"""
训练循环。

设计原则：Trainer 只负责循环逻辑（前向、反向、优化、日志、checkpoint），
并行策略的差异通过外部传入的 model/optimizer/train_step_fn 来体现。
"""

import os
import time
import torch
import torch.nn as nn
from nano_megatron.utils import is_main_process
from nano_megatron.metrics import MetricsTracker
from nano_megatron.evaluate import evaluate_gsm8k


class Trainer:
    def __init__(self, model, optimizer, scheduler, train_loader, eval_loader,
                 tokenizer, config, train_step_fn=None):
        """
        Args:
            model: 模型（可能已被并行策略包装过）
            optimizer: 优化器（可能是 ZeRO 包装过的）
            scheduler: 学习率调度器
            train_loader: 训练数据
            eval_loader: 评估数据
            tokenizer: 分词器（评估生成时用）
            config: 全局配置
            train_step_fn: 自定义 train step（PP 等策略需要），默认 None 用标准流程
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.tokenizer = tokenizer
        self.config = config
        self.train_step_fn = train_step_fn

        self.device = next(model.parameters()).device
        self.metrics = MetricsTracker()

        # 混合精度
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.dtype = dtype_map.get(config.training.dtype, torch.bfloat16)

        # GradScaler 只在 fp32 模型 + fp16 autocast 场景下使用
        # fp16 模型的梯度已经是 fp16，GradScaler 不能 unscale fp16 梯度
        model_is_fp16 = next(model.parameters()).dtype == torch.float16
        is_standard_opt = isinstance(optimizer, torch.optim.Optimizer)
        self.use_scaler = (self.dtype == torch.float16) and is_standard_opt and not model_is_fp16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)

    def train(self):
        cfg = self.config.training
        self.model.train()
        data_iter = iter(self.train_loader)
        global_step = 0

        if is_main_process():
            print(f"[TRAIN] Starting training for {cfg.max_steps} steps...")

        while global_step < cfg.max_steps:
            # 取 batch（循环遍历数据）
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            batch = {k: v.to(self.device) for k, v in batch.items()}
            t0 = time.time()

            # 梯度累积
            total_loss = 0.0
            for micro_step in range(cfg.grad_accum_steps):
                loss = self._train_step(batch)
                total_loss += loss
            avg_loss = total_loss / cfg.grad_accum_steps

            # 梯度裁剪 + 优化器更新
            # ZeRO-2 用 backward hook 释放了非 owner 的梯度，nn.utils.clip_grad_norm_
            # 在不同 rank 上算出的 norm 不一致，必须用 ZeROOptimizer.clip_grad_norm
            has_dist_clip = hasattr(self.optimizer, "clip_grad_norm") and \
                            getattr(self.optimizer, "stage", None) == 2
            if self.use_scaler:
                self.scaler.unscale_(self.optimizer)
                if has_dist_clip:
                    self.optimizer.clip_grad_norm(cfg.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if has_dist_clip:
                    self.optimizer.clip_grad_norm(cfg.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler:
                self.scheduler.step()

            dt = time.time() - t0
            global_step += 1

            # 日志
            if global_step % cfg.log_interval == 0 and is_main_process():
                tokens_per_sec = (batch["input_ids"].numel() * cfg.grad_accum_steps) / dt
                self.metrics.update(loss=avg_loss, tokens_per_sec=tokens_per_sec)
                inner_opt = getattr(self.optimizer, "optimizer", self.optimizer)
                lr = inner_opt.param_groups[0]["lr"]
                print(f"  step {global_step:5d} | loss {avg_loss:.4f} | "
                      f"lr {lr:.2e} | tok/s {tokens_per_sec:.0f} | "
                      f"mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB")

            # 评估
            if global_step % cfg.eval_interval == 0 and is_main_process():
                acc = evaluate_gsm8k(self.model, self.tokenizer, self.config, self.device)
                print(f"  [EVAL] step {global_step} | GSM8k accuracy: {acc:.2%}")
                self.model.train()

            # 保存 checkpoint
            if global_step % cfg.save_interval == 0 and is_main_process():
                self._save_checkpoint(global_step)

        if is_main_process():
            print("[TRAIN] Training finished.")
            self.metrics.summary()

    def _train_step(self, batch):
        """单步训练（支持自定义 train_step_fn 和默认 AMP 流程）。"""
        if self.train_step_fn:
            return self.train_step_fn(self.model, batch, self.optimizer, self.scaler)

        with torch.amp.autocast("cuda", dtype=self.dtype):
            output = self.model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = output["loss"] / self.config.training.grad_accum_steps

        # Debug: 首次检查 forward 是否已经 NaN
        if not hasattr(self, "_debug_checked"):
            self._debug_checked = True
            if is_main_process():
                logits = output["logits"]
                print(f"  [DEBUG] First forward: loss={loss.item():.4f}, "
                      f"logits_max={logits.abs().max().item():.1f}, "
                      f"logits_nan={torch.isnan(logits).any().item()}")

        if self.use_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        return loss.item() * self.config.training.grad_accum_steps

    def _save_checkpoint(self, step):
        os.makedirs(self.config.training.output_dir, exist_ok=True)
        path = os.path.join(self.config.training.output_dir, f"checkpoint_{step}.pt")
        # 如果是 DDP 包装的模型，取内部的 module
        model_to_save = getattr(self.model, "module", self.model)
        torch.save({"step": step, "model": model_to_save.state_dict()}, path)
        print(f"  [SAVE] Checkpoint saved to {path}")
