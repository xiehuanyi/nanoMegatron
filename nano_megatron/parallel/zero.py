"""
ZeRO（Zero Redundancy Optimizer）—— 通过分片消除冗余显存占用。

┌─────────┬──────────────────────────┬─────────────────────┐
│  Stage  │  每个 GPU 存什么          │  通信模式            │
├─────────┼──────────────────────────┼─────────────────────┤
│ ZeRO-1  │ 完整参数 + 完整梯度       │ AllReduce 梯度       │
│         │ + 1/N optimizer states   │ → 各自更新本地 shard │
│         │                          │ → Broadcast 参数     │
├─────────┼──────────────────────────┼─────────────────────┤
│ ZeRO-2  │ 完整参数 + 1/N 梯度      │ post-accum hook      │
│         │ + 1/N optimizer states   │ → Reduce 到 owner    │
│         │                          │ → 立即 free 非 owner │
└─────────┴──────────────────────────┴─────────────────────┘

关于生产框架的优化（DeepSpeed / Megatron）：
- **Flat buffer**：把所有参数 flatten 成一个大 tensor，一次 NCCL 调用完成通信
  → 省 ~4 ms latency per step（200 次 NCCL launch → 1 次）
- **Backward hook overlap**：梯度一算完就 async reduce，和后续 backward 计算重叠
- 但 flat buffer 需要额外 7-8 GB 显存（完整 flat grads），V100 32GB 放不下 3.8B 模型

这里用 per-param 实现（省显存但慢）。切换到 flat buffer 需要 A100 80GB。

ZeRO-2 backward hook 修复（v2）：
- 朴素 ZeRO-2：在 step() 里 reduce-then-free，但 max_memory_allocated 的 peak
  发生在 backward 结束时（此时所有梯度都还在），所以 ZeRO-2 和 ZeRO-1 显存相同。
- 修复：用 register_post_accumulate_grad_hook 在 backward 期间增量 reduce-and-free，
  使每个梯度算完立即被 reduce 到 owner，非 owner 立刻释放显存。
  这样 backward 期间最多只有 1 个完整梯度活着（而不是全部）。
- 副作用：grad_clip_norm 必须用分布式版本（ZeROOptimizer.clip_grad_norm）。
"""

import torch
import torch.distributed as dist


class ZeROOptimizer:
    """ZeRO Stage 1/2 优化器（per-param 实现，兼容 V100 32GB 显存）。

    ZeRO-2 v2: 用 backward hook 在 backward 期间增量 reduce-and-free，
    实际降低 backward 阶段的 peak 显存。
    """

    def __init__(self, model, lr: float, weight_decay: float, stage: int):
        self.stage = stage
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.all_params = [p for p in model.parameters() if p.requires_grad]
        self.param_to_owner = {id(p): (i % self.world_size) for i, p in enumerate(self.all_params)}

        # round-robin 分配参数给各 rank
        self.local_params = [p for i, p in enumerate(self.all_params)
                             if i % self.world_size == self.rank]

        # fp32 参数副本（只为本地参数）→ Adam 在 fp32 上更新 → 同步回 fp16
        self.fp32_copies = [p.data.float().clone().requires_grad_(True) for p in self.local_params]
        self.optimizer = torch.optim.AdamW(self.fp32_copies, lr=lr, weight_decay=weight_decay)
        self.param_groups = self.optimizer.param_groups

        # ── ZeRO-2 v2: 注册 backward hook 增量 reduce ──
        if stage == 2:
            for p in self.all_params:
                owner = self.param_to_owner[id(p)]
                p.register_post_accumulate_grad_hook(self._make_zero2_hook(owner))

    def _make_zero2_hook(self, owner):
        """每个参数的 hook：在 backward 累积完梯度后立即 Reduce-and-free。

        这是 ZeRO-2 真正省显存的关键：让 backward 期间最多只有少数几个梯度活着。
        """
        def hook(p):
            # SUM 到 owner（NCCL 普遍支持 SUM；后面在 step() 里手动除以 world_size 算 AVG）
            dist.reduce(p.grad, dst=owner, op=dist.ReduceOp.SUM)
            if dist.get_rank() != owner:
                # 非 owner 立即释放梯度，省显存（这是 v2 的核心改动！）
                p.grad = None
        return hook

    def _ensure_grads(self):
        """确保所有参数都有梯度张量。仅 ZeRO-1 在 step 前需要。"""
        for p in self.all_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)

    def _copy_fp16_grads_to_fp32(self):
        """增量拷贝 fp16 梯度到 fp32（立即释放 fp16 梯度省显存）。

        ZeRO-2 用 SUM hook，需要除以 world_size 得到 AVG（与 ZeRO-1 的 AllReduce AVG 一致）。
        """
        div = self.world_size if self.stage == 2 else 1
        for fp32_p, fp16_p in zip(self.fp32_copies, self.local_params):
            if fp16_p.grad is not None:
                fp32_p.grad = fp16_p.grad.float() / div
                fp16_p.grad = None

    def _sync_fp32_to_fp16(self):
        """把 fp32 更新后的参数同步回 fp16 模型参数。"""
        for fp32_p, fp16_p in zip(self.fp32_copies, self.local_params):
            fp16_p.data.copy_(fp32_p.data)

    def clip_grad_norm(self, max_norm: float):
        """分布式 grad clipping（ZeRO-2 专用，因为非 owner 的梯度被 hook free 了）。

        每个 rank 算自己 owner 参数的 norm²，AllReduce 后得到全局 norm。
        然后用全局 norm 计算 clip coefficient，应用到各 rank 自己的 owner 参数上。
        """
        device = next(p for p in self.local_params if p.grad is not None).grad.device
        local_norm_sq = torch.zeros(1, device=device, dtype=torch.float32)
        for p in self.local_params:
            if p.grad is not None:
                # ZeRO-2 hook 用 SUM，还没除 world_size，所以 norm 也要相应缩放
                # 直接除以 world_size 把 SUM 转成 AVG 再算 norm
                g = p.grad.float() / self.world_size
                local_norm_sq += g.pow(2).sum()

        dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM)
        global_norm = local_norm_sq.sqrt().item()

        if global_norm > max_norm and global_norm > 0:
            clip_coef = max_norm / global_norm
            for p in self.local_params:
                if p.grad is not None:
                    p.grad.mul_(clip_coef)
        return global_norm

    def step(self):
        if self.stage == 1:
            # ZeRO-1: 朴素方式，step() 里 AllReduce 所有梯度
            self._ensure_grads()
            for p in self.all_params:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            # 释放非本地梯度
            for i, p in enumerate(self.all_params):
                if i % self.world_size != self.rank:
                    p.grad = None
        else:
            # ZeRO-2 v2: 梯度已被 hook reduce 到 owner，非 owner 已 free
            # 这里什么都不需要做（owner 的 grad 直接进入下面的 fp32 拷贝）
            pass

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
