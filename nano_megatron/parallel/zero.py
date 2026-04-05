"""
ZeRO（Zero Redundancy Optimizer）—— 通过分片消除冗余显存占用。

┌─────────┬──────────────────────────┬─────────────────────┐
│  Stage  │  每个 GPU 存什么          │  通信模式            │
├─────────┼──────────────────────────┼─────────────────────┤
│ ZeRO-1  │ 完整参数 + 完整梯度       │ AllReduce 梯度       │
│         │ + 1/N optimizer states   │ → 各自更新本地 shard │
│         │                          │ → Broadcast 参数     │
├─────────┼──────────────────────────┼─────────────────────┤
│ ZeRO-2  │ 完整参数 + 1/N 梯度      │ Reduce 梯度到 owner  │
│         │ + 1/N optimizer states   │ → 更新 → Broadcast   │
└─────────┴──────────────────────────┴─────────────────────┘

关于生产框架的优化（DeepSpeed / Megatron）：
- **Flat buffer**：把所有参数 flatten 成一个大 tensor，一次 NCCL 调用完成通信
  → 省 ~4 ms latency per step（200 次 NCCL launch → 1 次）
- **Backward hook overlap**：梯度一算完就 async reduce，和后续 backward 计算重叠
- 但 flat buffer 需要额外 7-8 GB 显存（完整 flat grads），V100 32GB 放不下 3.8B 模型

这里用 per-param 实现（省显存但慢）。切换到 flat buffer 需要 A100 80GB。
"""

import torch
import torch.distributed as dist


class ZeROOptimizer:
    """ZeRO Stage 1/2 优化器（per-param 实现，兼容 V100 32GB 显存）。"""

    def __init__(self, model, lr: float, weight_decay: float, stage: int):
        self.stage = stage
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.all_params = [p for p in model.parameters() if p.requires_grad]

        # round-robin 分配参数给各 rank
        self.local_params = [p for i, p in enumerate(self.all_params)
                             if i % self.world_size == self.rank]

        # fp32 参数副本（只为本地参数）→ Adam 在 fp32 上更新 → 同步回 fp16
        self.fp32_copies = [p.data.float().clone().requires_grad_(True) for p in self.local_params]
        self.optimizer = torch.optim.AdamW(self.fp32_copies, lr=lr, weight_decay=weight_decay)
        self.param_groups = self.optimizer.param_groups

    def _ensure_grads(self):
        """确保所有参数都有梯度张量（为零也行），否则 collective 会 rank 不同步死锁。"""
        for p in self.all_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)

    def _copy_fp16_grads_to_fp32(self):
        """增量拷贝 fp16 梯度到 fp32（立即释放 fp16 梯度省显存）。"""
        for fp32_p, fp16_p in zip(self.fp32_copies, self.local_params):
            if fp16_p.grad is not None:
                fp32_p.grad = fp16_p.grad.float()
                fp16_p.grad = None

    def _sync_fp32_to_fp16(self):
        """把 fp32 更新后的参数同步回 fp16 模型参数。"""
        for fp32_p, fp16_p in zip(self.fp32_copies, self.local_params):
            fp16_p.data.copy_(fp32_p.data)

    def step(self):
        self._ensure_grads()

        if self.stage == 1:
            # ZeRO-1: AllReduce 所有梯度（每 rank 都有完整梯度）
            for p in self.all_params:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        else:  # stage == 2
            # ZeRO-2: Reduce 梯度到 owner rank（非 owner 不保留梯度）
            for i, p in enumerate(self.all_params):
                owner = i % self.world_size
                dist.reduce(p.grad, dst=owner, op=dist.ReduceOp.AVG)

        # 释放非本地梯度省显存（ZeRO-1 和 2 都需要，只为本地 shard 做 optimizer step）
        for i, p in enumerate(self.all_params):
            if i % self.world_size != self.rank:
                p.grad = None

        # fp32 Adam 更新本地 shard
        self._copy_fp16_grads_to_fp32()
        self.optimizer.step()
        self._sync_fp32_to_fp16()

        # Broadcast 更新后的参数（每个参数的 owner 把自己更新的参数发给其他 rank）
        for i, p in enumerate(self.all_params):
            owner = i % self.world_size
            dist.broadcast(p.data, src=owner)

    def zero_grad(self):
        self.optimizer.zero_grad()
        for p in self.all_params:
            p.grad = None


def setup_zero(model, config, stage: int):
    """用 ZeRO 包装模型。"""
    local_rank = dist.get_rank()
    model = model.to(local_rank)

    optimizer = ZeROOptimizer(
        model,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        stage=stage,
    )
    return model, optimizer


# ============================================================
# FP16 Optimizer Wrapper (其他策略用)
# ============================================================

class FP16OptimizerWrapper:
    """fp16 模型的 fp32 优化器包装。

    问题：fp16 参数直接用 Adam 会导致 grad² 溢出 → NaN。
    方案：维护一份 fp32 参数副本做 Adam 更新，每步同步回 fp16。
    """

    def __init__(self, params, lr: float, weight_decay: float):
        self.fp16_params = list(params)
        self.fp32_params = [p.data.float().clone().requires_grad_(True) for p in self.fp16_params]
        self.optimizer = torch.optim.AdamW(self.fp32_params, lr=lr, weight_decay=weight_decay)
        self.param_groups = self.optimizer.param_groups

    def step(self):
        for fp32_p, fp16_p in zip(self.fp32_params, self.fp16_params):
            if fp16_p.grad is not None:
                fp32_p.grad = fp16_p.grad.float()
                fp16_p.grad = None
        self.optimizer.step()
        for fp32_p, fp16_p in zip(self.fp32_params, self.fp16_params):
            fp16_p.data.copy_(fp32_p.data)

    def zero_grad(self):
        self.optimizer.zero_grad()
        for p in self.fp16_params:
            p.grad = None
