"""
ZeRO-3 / FSDP —— 参数、梯度、优化器状态全部分片。

实现思路（参考 Tiny-FSDP）：
不用 hook（backward 顺序不确定 → NCCL 死锁），而是：
1. 替换 nn.Linear / nn.Embedding / RMSNorm 为 FSDP 版本
2. AllGather / ReduceScatter 放在 autograd.Function 里
3. 因为所有 rank 的计算图拓扑一致，backward 顺序天然一致 → 不会死锁

参数分片方式：沿 dim 0 切分，每个 rank 存 1/N 的参数。

数据流（以 Linear 为例）：
  forward:  AllGather 拿到完整权重 → matmul → 缓存完整权重给 backward
  backward: 用缓存的完整权重算梯度 → ReduceScatter 梯度 → 释放缓存

MoE expert 分片修复（v2）：
- 朴素实现跳过了 experts 的分片（experts 占 ~74% 参数 → ZeRO-3 几乎没节省）
- 跳过原因：MoE routing 让不同 rank 跳过不同 expert → AllGather 错位 → 死锁
- 修复：patch MoE forward 让 **所有** rank 始终调用 **所有** expert（即使 mask 为空），
  这样 AllGather 调用顺序在所有 rank 上一致 → 不死锁
- 空 mask 的 expert 调用：F.linear([0, in], W) 返回 [0, out]，AllGather 照常进行
"""

import types
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ============================================================
# 分片工具
# ============================================================

def shard_tensor(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """沿 dim 0 切出属于 rank 的分片。"""
    dim0 = tensor.shape[0]
    shard_size = math.ceil(dim0 / world_size)
    start = rank * shard_size
    end = min(start + shard_size, dim0)
    return tensor[start:end].contiguous().clone()


def gather_tensor(shard: torch.Tensor, full_shape: torch.Size, group=None) -> torch.Tensor:
    """AllGather：从各 rank 收集分片，拼成完整 tensor。"""
    world_size = dist.get_world_size(group)
    # 各 rank 的 shard 大小可能不同（最后一个 rank 可能更小）
    max_shard = math.ceil(full_shape[0] / world_size)
    # Pad shard to max_shard size for all_gather
    padded = torch.zeros(max_shard, *shard.shape[1:], dtype=shard.dtype, device=shard.device)
    padded[:shard.shape[0]] = shard
    # AllGather
    gathered = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded, group=group)
    # Cat and trim to full_shape
    full = torch.cat(gathered, dim=0)[:full_shape[0]]
    return full


def scatter_grad(grad: torch.Tensor, rank: int, world_size: int, group=None) -> torch.Tensor:
    """ReduceScatter：梯度沿 dim 0 切分并求和，每个 rank 只保留自己的分片。"""
    dim0 = grad.shape[0]
    shard_size = math.ceil(dim0 / world_size)
    # Pad to even split
    padded_size = shard_size * world_size
    if dim0 < padded_size:
        pad = torch.zeros(padded_size - dim0, *grad.shape[1:], dtype=grad.dtype, device=grad.device)
        grad_padded = torch.cat([grad, pad], dim=0)
    else:
        grad_padded = grad
    # Split into chunks
    chunks = list(grad_padded.chunk(world_size, dim=0))
    output = torch.empty_like(chunks[0])
    dist.reduce_scatter(output, chunks, group=group)
    # Trim if last rank's shard is smaller
    actual_size = min(shard_size, dim0 - rank * shard_size)
    return output[:actual_size].contiguous()


# ============================================================
# FSDP autograd.Function — 核心：所有通信在这里发生
# ============================================================

class _FSDPLinearFunc(torch.autograd.Function):
    """FSDP 版 Linear 的 forward/backward。

    forward: AllGather 权重 → matmul → 缓存完整权重
    backward: 用缓存权重算梯度 → ReduceScatter 权重梯度
    """

    @staticmethod
    def forward(ctx, x, weight_shard, bias, full_weight_shape, group):
        # AllGather 拿到完整权重
        full_weight = gather_tensor(weight_shard, full_weight_shape, group)
        # 缓存给 backward 用
        ctx.save_for_backward(x, full_weight)
        ctx.bias_exists = bias is not None
        ctx.full_weight_shape = full_weight_shape
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)
        # Y = X @ W^T + b
        output = F.linear(x, full_weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, full_weight = ctx.saved_tensors

        # 输入的梯度：dX = dY @ W
        grad_x = grad_output.matmul(full_weight)

        # 权重的梯度：dW = dY^T @ X（结果是完整 dW，需要 ReduceScatter 切成分片）
        grad_weight_full = grad_output.reshape(-1, grad_output.shape[-1]).t().matmul(
            x.reshape(-1, x.shape[-1])
        )
        # ReduceScatter：每个 rank 只保留自己分片的梯度
        grad_weight_shard = scatter_grad(grad_weight_full, ctx.rank, ctx.world_size, ctx.group)

        # bias 梯度（不分片，AllReduce 求平均）
        grad_bias = None
        if ctx.bias_exists:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
            dist.all_reduce(grad_bias, op=dist.ReduceOp.AVG, group=ctx.group)

        return grad_x, grad_weight_shard, grad_bias, None, None


class _FSDPEmbeddingFunc(torch.autograd.Function):
    """FSDP 版 Embedding 的 forward/backward。"""

    @staticmethod
    def forward(ctx, input_ids, weight_shard, full_weight_shape, group):
        full_weight = gather_tensor(weight_shard, full_weight_shape, group)
        ctx.save_for_backward(input_ids, full_weight)
        ctx.full_weight_shape = full_weight_shape
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)
        return F.embedding(input_ids, full_weight)

    @staticmethod
    def backward(ctx, grad_output):
        input_ids, full_weight = ctx.saved_tensors
        # Embedding 梯度是稀疏的，先转成 dense
        grad_weight_full = torch.zeros_like(full_weight)
        grad_weight_full.index_add_(0, input_ids.reshape(-1),
                                     grad_output.reshape(-1, grad_output.shape[-1]))
        grad_weight_shard = scatter_grad(grad_weight_full, ctx.rank, ctx.world_size, ctx.group)
        return None, grad_weight_shard, None, None


# ============================================================
# FSDP 模块替换
# ============================================================

class FSDPLinear(nn.Module):
    """FSDP 版 Linear：只存 1/N 的权重，forward 时 AllGather。"""

    def __init__(self, weight_shard, bias, full_weight_shape, group):
        super().__init__()
        self.weight = nn.Parameter(weight_shard)
        self.bias = nn.Parameter(bias) if bias is not None else None
        self.full_weight_shape = full_weight_shape
        self.group = group

    def forward(self, x):
        return _FSDPLinearFunc.apply(x, self.weight, self.bias, self.full_weight_shape, self.group)


class FSDPEmbedding(nn.Module):
    """FSDP 版 Embedding：只存 1/N 的嵌入表。"""

    def __init__(self, weight_shard, full_weight_shape, group):
        super().__init__()
        self.weight = nn.Parameter(weight_shard)
        self.full_weight_shape = full_weight_shape
        self.group = group

    def forward(self, input_ids):
        return _FSDPEmbeddingFunc.apply(input_ids, self.weight, self.full_weight_shape, self.group)


# ============================================================
# 模型包装：递归替换所有 Linear / Embedding
# ============================================================

def fsdp_wrap_module(model, group):
    """递归替换模型中的 Linear 和 Embedding 为 FSDP 版本（包括 MoE experts）。

    v2：现在也分片 MoE experts。前提是 MoE forward 已经被 patch 成
    "所有 rank 始终调用所有 expert"（见 _patch_moe_for_fsdp），避免 AllGather 错位。
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            full_shape = child.weight.shape
            w_shard = shard_tensor(child.weight.data, rank, world_size)
            bias = child.bias.data.clone() if child.bias is not None else None
            fsdp_mod = FSDPLinear(w_shard, bias, full_shape, group)
            setattr(model, name, fsdp_mod)

        elif isinstance(child, nn.Embedding):
            full_shape = child.weight.shape
            w_shard = shard_tensor(child.weight.data, rank, world_size)
            fsdp_mod = FSDPEmbedding(w_shard, full_shape, group)
            setattr(model, name, fsdp_mod)

        else:
            fsdp_wrap_module(child, group)


def _patch_moe_for_fsdp(moe):
    """让 MoE forward 始终调用所有 expert（即使 mask 为空）。

    原版 MoE forward 用 `if mask.any():` 跳过未激活的 expert。在 FSDP 下这会导致
    不同 rank 跳过不同 expert（因为各 rank 的 routing 不同）→ FSDPLinear 的 AllGather
    在不同 rank 上调用次数不同 → 死锁。

    修复：去掉 `if mask.any():` 检查，所有 expert 始终调用（空输入时 F.linear 返回空）。
    所有 rank 按相同顺序调用 → AllGather 对齐 → 不死锁。
    """
    def patched_forward(self, x):
        B, L, D = x.shape
        x_flat = x.view(-1, D)

        router_logits = self.gate(x_flat)
        topk_weights, topk_indices = torch.topk(
            router_logits.float(), self.num_experts_per_tok, dim=-1
        )
        topk_weights = F.softmax(topk_weights, dim=-1).to(x.dtype)

        output = torch.zeros_like(x_flat)
        for i in range(self.num_experts_per_tok):
            expert_idx = topk_indices[:, i]
            weight = topk_weights[:, i]
            for e_id in range(len(self.experts)):
                mask = (expert_idx == e_id)
                # 关键：始终调用 expert（即使 mask 全 False），保证 AllGather 顺序一致
                expert_input = x_flat[mask]  # 可能是 [0, D]
                expert_output = self.experts[e_id](expert_input)  # FSDPLinear 触发 AllGather
                if expert_input.shape[0] > 0:
                    output[mask] += weight[mask].unsqueeze(-1) * expert_output

        return output.view(B, L, D), router_logits

    moe.forward = types.MethodType(patched_forward, moe)


class FSDPMixedOptimizer:
    """FSDP 混合精度优化器。

    v2：通过 module class 检测分片参数（之前用 name pattern，会漏掉 experts 内的 Linear）。
    现在 experts 也被分片了，所以"非分片"参数只剩 RMSNorm（很小，fp32 副本毫无压力）。
    全部参数走 fp32 Adam。
    """

    def __init__(self, model, lr, weight_decay):
        # 通过 module 类型检测分片参数
        sharded_param_ids = set()
        for module in model.modules():
            if isinstance(module, (FSDPLinear, FSDPEmbedding)):
                sharded_param_ids.add(id(module.weight))
                if hasattr(module, "bias") and module.bias is not None:
                    sharded_param_ids.add(id(module.bias))

        sharded_params = []
        other_params = []
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if id(p) in sharded_param_ids:
                sharded_params.append(p)
            else:
                other_params.append(p)

        # 分片参数：fp32 Adam
        self.sharded_fp16 = sharded_params
        self.sharded_fp32 = [p.data.float().clone().requires_grad_(True) for p in sharded_params]
        self.sharded_opt = torch.optim.AdamW(self.sharded_fp32, lr=lr, weight_decay=weight_decay)

        # 非分片参数（v2 修复后只剩 RMSNorm 的 weight/bias 等小张量）：也用 fp32 Adam
        # （之前用 SGD 是因为非分片参数太大，fp32 副本会 OOM；现在 experts 已分片，问题消失）
        self.other_fp16 = other_params
        self.other_fp32 = [p.data.float().clone().requires_grad_(True) for p in other_params]
        self.other_opt = torch.optim.AdamW(self.other_fp32, lr=lr, weight_decay=weight_decay) \
            if other_params else None

        self.param_groups = self.sharded_opt.param_groups + (
            self.other_opt.param_groups if self.other_opt else []
        )

    def step(self):
        # 分片参数
        for fp32_p, fp16_p in zip(self.sharded_fp32, self.sharded_fp16):
            if fp16_p.grad is not None:
                fp32_p.grad = fp16_p.grad.float()
                fp16_p.grad = None
        self.sharded_opt.step()
        for fp32_p, fp16_p in zip(self.sharded_fp32, self.sharded_fp16):
            fp16_p.data.copy_(fp32_p.data)

        # 非分片参数
        if self.other_opt is not None:
            for fp32_p, fp16_p in zip(self.other_fp32, self.other_fp16):
                if fp16_p.grad is not None:
                    fp32_p.grad = fp16_p.grad.float()
                    fp16_p.grad = None
            self.other_opt.step()
            for fp32_p, fp16_p in zip(self.other_fp32, self.other_fp16):
                fp16_p.data.copy_(fp32_p.data)

    def zero_grad(self):
        self.sharded_opt.zero_grad()
        if self.other_opt is not None:
            self.other_opt.zero_grad()


def setup_fsdp(model, config):
    """用 FSDP（ZeRO-3）包装模型（v2：experts 也被分片）。

    Returns:
        (model, FSDPMixedOptimizer)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    model = model.to(rank)

    # 创建进程组
    group = dist.new_group(list(range(world_size)))

    # ── v2 修复：先 patch MoE forward 让所有 rank 始终调用所有 expert ──
    # 这是 expert 分片的前提（避免 AllGather 错位）
    base = getattr(model, "module", model)
    for layer in base.model.layers:
        _patch_moe_for_fsdp(layer.block_sparse_moe)

    # 递归替换 Linear / Embedding 为 FSDP 版本（包括 experts 内部的 w1/w2/w3）
    fsdp_wrap_module(model, group)

    # 返回混合精度优化器
    optimizer = FSDPMixedOptimizer(model, lr=config.training.lr,
                                    weight_decay=config.training.weight_decay)
    return model, optimizer
