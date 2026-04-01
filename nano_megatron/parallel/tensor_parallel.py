"""
Tensor Parallel（TP）—— 把单个层的权重切分到多个 GPU。

核心组件：
1. ColumnParallelLinear：沿输出维度切分权重 → 每个 GPU 算一部分输出
2. RowParallelLinear：沿输入维度切分权重 → 每个 GPU 算一部分，然后 AllReduce

应用到 Attention 和 MoE：
┌─────────────────────────────────────────────────────────┐
│  Attention:                                              │
│    Q, K, V proj → ColumnParallel（按 head 切分）          │
│    O proj       → RowParallel（结果 AllReduce）           │
│                                                          │
│  MoE Expert:                                             │
│    w1, w3 → ColumnParallel（沿 intermediate_size 切）     │
│    w2     → RowParallel（结果 AllReduce）                 │
└─────────────────────────────────────────────────────────┘

这样每个 GPU 只存 1/tp_size 的 Attention head 和 1/tp_size 的 FFN 宽度。
"""

import torch
import torch.nn as nn
import torch.distributed as dist


# ============================================================
# 通信原语
# ============================================================

class _AllReduceFunc(torch.autograd.Function):
    """前向 AllReduce，反向直通（identity）。
    用在 RowParallel 的输出：前向需要聚合各 GPU 的部分结果。"""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        dist.all_reduce(x, group=group)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None  # 反向时梯度直接传过去，因为 AllReduce 对梯度是等价的


class _SplitFunc(torch.autograd.Function):
    """前向 identity，反向 AllReduce。
    用在 ColumnParallel 的输入：前向各 GPU 用相同输入，反向需要聚合梯度。"""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


# ============================================================
# 并行 Linear 层
# ============================================================

class ColumnParallelLinear(nn.Module):
    """沿输出维度切分的 Linear。

    原始: Y = XW^T + b，W: [out, in]
    切分: 每个 GPU 存 W_i: [out/tp, in]，算 Y_i = XW_i^T + b_i
    结果: Y_i 是完整输出的一段（沿 last dim 切分）

    典型用法：Attention 的 Q/K/V 投影（按 head 切），MoE 的 w1/w3。
    """

    def __init__(self, in_features, out_features, bias=True, tp_group=None):
        super().__init__()
        self.tp_group = tp_group
        tp_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)

        assert out_features % tp_size == 0, f"out_features({out_features}) 必须能被 tp_size({tp_size}) 整除"
        self.out_per_rank = out_features // tp_size

        self.linear = nn.Linear(in_features, self.out_per_rank, bias=bias)

    def forward(self, x):
        # 输入不需要切：所有 GPU 拿到相同的 x（反向时 AllReduce 梯度）
        x = _SplitFunc.apply(x, self.tp_group)
        return self.linear(x)

    def load_weight_shard(self, full_weight, full_bias=None):
        """从完整权重中取出属于本 rank 的切片。"""
        tp_rank = dist.get_rank(self.tp_group)
        start = tp_rank * self.out_per_rank
        end = start + self.out_per_rank
        self.linear.weight.data.copy_(full_weight[start:end])
        if full_bias is not None and self.linear.bias is not None:
            self.linear.bias.data.copy_(full_bias[start:end])


class RowParallelLinear(nn.Module):
    """沿输入维度切分的 Linear。

    原始: Y = XW^T + b，W: [out, in]
    切分: 每个 GPU 存 W_i: [out, in/tp]，输入也切分 X_i: [..., in/tp]
    结果: Y_i = X_i @ W_i^T，然后 AllReduce 求和得到完整 Y

    典型用法：Attention 的 O 投影，MoE 的 w2。
    """

    def __init__(self, in_features, out_features, bias=True, tp_group=None):
        super().__init__()
        self.tp_group = tp_group
        tp_size = dist.get_world_size(tp_group)

        assert in_features % tp_size == 0
        self.in_per_rank = in_features // tp_size

        self.linear = nn.Linear(self.in_per_rank, out_features, bias=bias)

    def forward(self, x):
        # x 已经是切分过的（来自上游 ColumnParallel 的输出）
        y = self.linear(x)
        # AllReduce 聚合各 GPU 的部分结果
        return _AllReduceFunc.apply(y, self.tp_group)

    def load_weight_shard(self, full_weight, full_bias=None):
        """从完整权重中取出属于本 rank 的切片。"""
        tp_rank = dist.get_rank(self.tp_group)
        start = tp_rank * self.in_per_rank
        end = start + self.in_per_rank
        self.linear.weight.data.copy_(full_weight[:, start:end])
        if full_bias is not None and self.linear.bias is not None:
            self.linear.bias.data.copy_(full_bias)  # bias 不切分，每个 rank 都有完整的


# ============================================================
# 模型切分
# ============================================================

def tp_parallelize_attention(attn, tp_group):
    """把 Attention 的投影层替换为 TP 版本。

    Q, K, V → ColumnParallel（按 head 数切分输出维度）
    O       → RowParallel（把各 GPU 的部分结果聚合）
    """
    # 保存原始权重
    q_w, q_b = attn.q_proj.weight.data, attn.q_proj.bias.data
    k_w, k_b = attn.k_proj.weight.data, attn.k_proj.bias.data
    v_w, v_b = attn.v_proj.weight.data, attn.v_proj.bias.data
    o_w, o_b = attn.o_proj.weight.data, attn.o_proj.bias.data

    in_dim = q_w.shape[1]
    q_out = q_w.shape[0]
    k_out = k_w.shape[0]
    v_out = v_w.shape[0]
    o_out = o_w.shape[0]

    # 替换为并行版本
    attn.q_proj = ColumnParallelLinear(in_dim, q_out, bias=True, tp_group=tp_group)
    attn.k_proj = ColumnParallelLinear(in_dim, k_out, bias=True, tp_group=tp_group)
    attn.v_proj = ColumnParallelLinear(in_dim, v_out, bias=True, tp_group=tp_group)
    attn.o_proj = RowParallelLinear(q_out, o_out, bias=True, tp_group=tp_group)

    # 加载对应分片的权重
    attn.q_proj.load_weight_shard(q_w, q_b)
    attn.k_proj.load_weight_shard(k_w, k_b)
    attn.v_proj.load_weight_shard(v_w, v_b)
    attn.o_proj.load_weight_shard(o_w, o_b)

    # 更新 head 数为本 rank 的 head 数
    tp_size = dist.get_world_size(tp_group)
    attn.num_heads = attn.num_heads // tp_size
    attn.num_kv_heads = attn.num_kv_heads // tp_size


def tp_parallelize_expert(expert, tp_group):
    """把单个 Expert 的 FFN 替换为 TP 版本。

    w1, w3 → ColumnParallel（沿 intermediate_size 切）
    w2     → RowParallel（结果 AllReduce）
    """
    w1_w = expert.w1.weight.data
    w2_w = expert.w2.weight.data
    w3_w = expert.w3.weight.data

    in_dim = w1_w.shape[1]     # hidden_size
    mid_dim = w1_w.shape[0]    # intermediate_size
    out_dim = w2_w.shape[0]    # hidden_size

    expert.w1 = ColumnParallelLinear(in_dim, mid_dim, bias=False, tp_group=tp_group)
    expert.w3 = ColumnParallelLinear(in_dim, mid_dim, bias=False, tp_group=tp_group)
    expert.w2 = RowParallelLinear(mid_dim, out_dim, bias=False, tp_group=tp_group)

    expert.w1.load_weight_shard(w1_w)
    expert.w3.load_weight_shard(w3_w)
    expert.w2.load_weight_shard(w2_w)


def setup_tp(model, config):
    """对模型施加 Tensor Parallel。

    Returns:
        (model, None)
    """
    tp_size = config.parallel.tp_size
    rank = dist.get_rank()

    # 创建 TP 进程组（相邻 tp_size 个 rank 为一组）
    tp_groups = []
    for i in range(0, dist.get_world_size(), tp_size):
        ranks = list(range(i, i + tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            tp_group = group
        tp_groups.append(group)

    model = model.to(rank)

    # 替换每一层的 Attention 和 Expert
    base = getattr(model, "module", model)  # 兼容 DDP 包装
    for layer in base.model.layers:
        tp_parallelize_attention(layer.self_attn, tp_group)
        for expert in layer.block_sparse_moe.experts:
            tp_parallelize_expert(expert, tp_group)

    return model, None
