"""
ZeRO（Zero Redundancy Optimizer）—— 通过分片消除冗余显存占用。

三个阶段，逐步减少每个 GPU 的显存需求：

┌─────────┬──────────────────────────┬─────────────────────┐
│  Stage   │  每个 GPU 存什么          │  省了什么            │
├─────────┼──────────────────────────┼─────────────────────┤
│ ZeRO-1  │ 完整参数 + 完整梯度       │ Optimizer States     │
│         │ + 1/N optimizer states   │ 省 N 倍             │
├─────────┼──────────────────────────┼─────────────────────┤
│ ZeRO-2  │ 完整参数 + 1/N 梯度      │ + Gradients          │
│         │ + 1/N optimizer states   │ 省更多               │
├─────────┼──────────────────────────┼─────────────────────┤
│ ZeRO-3  │ 1/N 参数 + 1/N 梯度     │ + Parameters         │
│         │ + 1/N optimizer states   │ 全省（= FSDP 原理）   │
└─────────┴──────────────────────────┴─────────────────────┘

注：ZeRO-3 的核心思路就是 PyTorch FSDP 的实现原理。
"""

import torch
import torch.distributed as dist


class ZeROOptimizer:
    """手写 ZeRO 优化器包装器，支持 Stage 1/2/3。

    核心思路：把所有参数按 index 分配给各个 rank。
    - 第 i 个参数归 rank (i % world_size) 管理。
    - 每个 rank 只为自己管理的参数维护 optimizer state。
    """

    def __init__(self, model, lr: float, weight_decay: float, stage: int):
        self.stage = stage
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.all_params = list(model.parameters())

        # 按 round-robin 分配参数给各 rank
        self.local_params = [p for i, p in enumerate(self.all_params) if i % self.world_size == self.rank]
        self.param_to_rank = {id(p): i % self.world_size for i, p in enumerate(self.all_params)}

        # 混合精度：模型可能是 fp16，但 optimizer 必须用 fp32（防止 grad² 溢出）
        # 为本地参数创建 fp32 副本，Adam 在 fp32 上更新，再同步回 fp16 参数
        self.fp32_copies = [p.data.float().clone() for p in self.local_params]
        for fp32_p in self.fp32_copies:
            fp32_p.requires_grad_(True)

        self.optimizer = torch.optim.AdamW(self.fp32_copies, lr=lr, weight_decay=weight_decay)

        # ZeRO-3：保存参数分片，注册 forward/backward hook
        if stage == 3:
            self._setup_stage3(model)

        # 兼容 Trainer 的 param_groups 访问
        self.param_groups = self.optimizer.param_groups

    # ── Stage 1: 分片 optimizer states ──────────────────────
    # 所有 rank 都有完整参数和梯度，只是 optimizer state 分片

    # ── Stage 2: + 分片 gradients ───────────────────────────
    # 用 Reduce（而非 AllReduce）把梯度只发给负责的 rank

    # ── Stage 3: + 分片 parameters ──────────────────────────
    # forward 前 AllGather 参数，backward 后释放

    def step(self):
        """执行一步优化。不同 stage 的区别在于梯度同步方式。"""
        if self.stage == 1:
            self._step_stage1()
        elif self.stage == 2:
            self._step_stage2()
        elif self.stage == 3:
            self._step_stage3()

    def _ensure_grads(self):
        """确保所有参数都有梯度张量（即使是零），否则 collective 会 rank 不同步导致死锁。"""
        for p in self.all_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)

    def _copy_grads_to_fp32(self):
        """把 fp16 模型梯度拷贝到 fp32 副本，供 Adam 使用。"""
        for fp32_p, p in zip(self.fp32_copies, self.local_params):
            if p.grad is not None:
                fp32_p.grad = p.grad.float()
            else:
                fp32_p.grad = torch.zeros_like(fp32_p)

    def _copy_fp32_to_model(self):
        """把 fp32 更新后的参数同步回 fp16 模型参数。"""
        for fp32_p, p in zip(self.fp32_copies, self.local_params):
            p.data.copy_(fp32_p.data)

    def _step_stage1(self):
        """ZeRO-1: AllReduce 梯度 → 本地更新 → Broadcast 参数。"""
        self._ensure_grads()

        # 1) AllReduce 所有梯度
        for p in self.all_params:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        # 2) 释放非本地梯度（省显存）
        for i, p in enumerate(self.all_params):
            if i % self.world_size != self.rank:
                p.grad = None

        # 3) fp32 Adam 更新，释放 fp16 梯度省显存
        self._copy_grads_to_fp32()
        for p in self.local_params:
            p.grad = None
        self.optimizer.step()
        self._copy_fp32_to_model()

        # 4) Broadcast 更新后的参数
        for i, p in enumerate(self.all_params):
            owner = i % self.world_size
            dist.broadcast(p.data, src=owner)

    def _step_stage2(self):
        """ZeRO-2: Reduce 梯度到 owner → 本地更新 → Broadcast 参数。

        和 Stage 1 的区别：不用 AllReduce，用 Reduce 把梯度只发给负责的 rank。
        这样非 owner rank 不需要存这个参数的梯度。

        显存优化：reduce 后立即释放非本地梯度，再做 fp32 优化器更新。
        """
        self._ensure_grads()

        # 1) Reduce 梯度到各参数的 owner rank
        for i, p in enumerate(self.all_params):
            owner = i % self.world_size
            dist.reduce(p.grad, dst=owner, op=dist.ReduceOp.AVG)

        # 2) 立即释放非本地梯度（省 75% 梯度显存）
        for i, p in enumerate(self.all_params):
            if i % self.world_size != self.rank:
                p.grad = None

        # 3) fp32 Adam 更新 → 同步回 fp16，然后释放 fp16 本地梯度
        self._copy_grads_to_fp32()
        for p in self.local_params:
            p.grad = None  # 释放 fp16 梯度，fp32 副本已有
        self.optimizer.step()
        self._copy_fp32_to_model()

        # 4) Broadcast 更新后的参数
        for i, p in enumerate(self.all_params):
            owner = i % self.world_size
            dist.broadcast(p.data, src=owner)

    def _step_stage3(self):
        """ZeRO-3: 梯度 Reduce → 更新本地分片。
        参数的 AllGather/释放在 forward hook 中处理。
        """
        self._ensure_grads()

        # 1) Reduce 梯度到 owner
        for i, p in enumerate(self.all_params):
            owner = i % self.world_size
            dist.reduce(p.grad, dst=owner, op=dist.ReduceOp.AVG)

        # 2) 释放非本地梯度
        for i, p in enumerate(self.all_params):
            if i % self.world_size != self.rank:
                p.grad = None

        # 3) fp32 Adam 更新 → 同步回 fp16
        self._copy_grads_to_fp32()
        for p in self.local_params:
            p.grad = None
        self.optimizer.step()
        self._copy_fp32_to_model()

        # 3) 把更新后的参数存回分片
        for i, p in enumerate(self.all_params):
            if i % self.world_size == self.rank:
                self.param_shards[id(p)].copy_(p.data)

        # 4) 释放完整参数，恢复分片状态
        for i, p in enumerate(self.all_params):
            owner = i % self.world_size
            if owner == self.rank:
                p.data = self.param_shards[id(p)]
            else:
                p.data = torch.empty(0, device=p.device)  # 释放非本地参数
                p.grad = None

    def _setup_stage3(self, model):
        """ZeRO-3 初始化：保存参数分片，注册 hook。

        思路和 FSDP 一样：
        - 平时每个 rank 只存自己负责的参数分片
        - forward 前 AllGather 拿到完整参数
        - forward/backward 完了释放完整参数
        """
        self.param_shards = {}

        # 保存每个本地参数的副本作为分片
        for i, p in enumerate(self.all_params):
            if i % self.world_size == self.rank:
                self.param_shards[id(p)] = p.data.clone()

        # 为每个子模块注册 forward pre-hook：forward 前 AllGather 参数
        for module in model.modules():
            module.register_forward_pre_hook(self._pre_forward_hook)

    def _pre_forward_hook(self, module, input):
        """Forward 前：AllGather 该模块的参数。"""
        for p in module.parameters(recurse=False):
            p_id = id(p)
            owner = self.param_to_rank.get(p_id, 0)
            # 如果参数已被释放，需要重新分配空间
            if p.data.numel() == 0:
                p.data = torch.empty_like(self.param_shards.get(p_id, p.data))
            dist.broadcast(p.data, src=owner)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()


def setup_zero(model, config, stage: int):
    """用 ZeRO 包装模型。

    Returns:
        (model, None)  —— 模型不需要额外包装，用自定义 optimizer 即可
    """
    local_rank = dist.get_rank()
    model = model.to(local_rank)

    # 替换默认优化器为 ZeRO 优化器
    # 注意：返回的 optimizer 是 ZeROOptimizer，Trainer 会调用它的 step()/zero_grad()
    optimizer = ZeROOptimizer(
        model,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        stage=stage,
    )

    return model, optimizer


class FP16OptimizerWrapper:
    """fp16 模型的 fp32 优化器包装。

    问题：fp16 参数直接用 Adam 会导致 grad² 溢出（256² > 65504）→ NaN。
    方案：维护一份 fp32 参数副本做 Adam 更新，每步同步回 fp16。

    这就是经典的 "mixed precision training" 中 optimizer 的做法。
    """

    def __init__(self, params, lr: float, weight_decay: float):
        self.fp16_params = list(params)
        # 创建 fp32 参数副本
        self.fp32_params = [p.data.float().clone().requires_grad_(True) for p in self.fp16_params]
        self.optimizer = torch.optim.AdamW(self.fp32_params, lr=lr, weight_decay=weight_decay)
        self.param_groups = self.optimizer.param_groups

    def step(self):
        # 1) 逐参数拷贝 fp16 梯度到 fp32，立即释放 fp16 梯度（省显存）
        for fp32_p, fp16_p in zip(self.fp32_params, self.fp16_params):
            if fp16_p.grad is not None:
                fp32_p.grad = fp16_p.grad.float()
                fp16_p.grad = None  # 立即释放，避免 fp16 + fp32 梯度同时占用显存

        # 2) fp32 Adam 更新
        self.optimizer.step()

        # 3) 同步回 fp16 模型参数
        for fp32_p, fp16_p in zip(self.fp32_params, self.fp16_params):
            fp16_p.data.copy_(fp32_p.data)

    def zero_grad(self):
        self.optimizer.zero_grad()
        for p in self.fp16_params:
            p.grad = None
