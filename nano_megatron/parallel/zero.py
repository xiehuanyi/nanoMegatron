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
        # 例如 4 卡 8 个参数：rank0 管 [0,4], rank1 管 [1,5], rank2 管 [2,6], rank3 管 [3,7]
        self.local_params = [p for i, p in enumerate(self.all_params) if i % self.world_size == self.rank]
        self.param_to_rank = {id(p): i % self.world_size for i, p in enumerate(self.all_params)}

        # 只为本地参数创建 optimizer（节省 optimizer state 内存）
        self.optimizer = torch.optim.AdamW(self.local_params, lr=lr, weight_decay=weight_decay)

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

    def _step_stage1(self):
        """ZeRO-1: AllReduce 梯度 → 本地更新 → Broadcast 参数。"""
        # 1) AllReduce 所有梯度（和 DDP 一样，所有 rank 得到相同的平均梯度）
        for p in self.all_params:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        # 2) 只更新本地负责的参数
        self.optimizer.step()

        # 3) Broadcast 更新后的参数（各 rank 把自己更新的参数广播给其他人）
        for i, p in enumerate(self.all_params):
            owner = i % self.world_size
            dist.broadcast(p.data, src=owner)

    def _step_stage2(self):
        """ZeRO-2: Reduce 梯度到 owner → 本地更新 → Broadcast 参数。

        和 Stage 1 的区别：不用 AllReduce，用 Reduce 把梯度只发给负责的 rank。
        这样非 owner rank 不需要存这个参数的梯度。
        """
        # 1) Reduce 梯度到各参数的 owner rank
        for i, p in enumerate(self.all_params):
            if p.grad is not None:
                owner = i % self.world_size
                dist.reduce(p.grad, dst=owner, op=dist.ReduceOp.AVG)

        # 2) 只更新本地负责的参数
        self.optimizer.step()

        # 3) Broadcast 更新后的参数
        for i, p in enumerate(self.all_params):
            owner = i % self.world_size
            dist.broadcast(p.data, src=owner)

        # 4) 释放非 owner 的梯度（省显存）
        for i, p in enumerate(self.all_params):
            if i % self.world_size != self.rank:
                p.grad = None

    def _step_stage3(self):
        """ZeRO-3: 梯度 Reduce → 更新本地分片。
        参数的 AllGather/释放在 forward hook 中处理。
        """
        # 1) Reduce 梯度到 owner
        for i, p in enumerate(self.all_params):
            if p.grad is not None:
                owner = i % self.world_size
                dist.reduce(p.grad, dst=owner, op=dist.ReduceOp.AVG)

        # 2) 更新本地参数分片
        self.optimizer.step()

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
