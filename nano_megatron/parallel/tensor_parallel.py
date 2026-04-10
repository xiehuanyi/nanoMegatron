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

通信合并优化（v2）：
- 朴素实现：每个 ColumnParallel 的 _SplitFunc 和每个 RowParallel 的 _AllReduceFunc 都
  会触发一次 NCCL 调用。32 层 × (3 attn + 16×3 expert) = 1632 次/forward+backward。
- 优化：Q/K/V 共用同一个输入 x，在 attention 入口做 1 次 SplitFunc，Q/K/V 跳过自己的；
  MoE 的 16 个 expert 累积 partial output，最后做 1 次 AllReduce。
- 结果：从 ~2158 次 NCCL/step 降到 ~250 次/step（约 8.6x 减少）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

    skip_split: 若上层已经统一做了 _SplitFunc（例如 Attention 入口），
                这里就不要重复做，避免 backward 时重复 AllReduce。
    """

    def __init__(self, in_features, out_features, bias=True, tp_group=None, device=None):
        super().__init__()
        self.tp_group = tp_group
        self.skip_split = False  # 由上层（如 attention pre-hook 或 MoE wrapper）控制
        tp_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)

        assert out_features % tp_size == 0, f"out_features({out_features}) 必须能被 tp_size({tp_size}) 整除"
        self.out_per_rank = out_features // tp_size

        self.linear = nn.Linear(in_features, self.out_per_rank, bias=bias, device=device)

    def forward(self, x):
        # 输入不需要切：所有 GPU 拿到相同的 x（反向时 AllReduce 梯度）
        if not self.skip_split:
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

    skip_reduce: 若上层会统一对 partial output 做 AllReduce（例如 MoE wrapper
                 累积所有 expert 的 partial 后再 AllReduce 一次），这里就不要重复做。
    """

    def __init__(self, in_features, out_features, bias=True, tp_group=None, device=None):
        super().__init__()
        self.tp_group = tp_group
        self.skip_reduce = False  # 由上层（如 MoE wrapper）控制
        tp_size = dist.get_world_size(tp_group)

        assert in_features % tp_size == 0
        self.in_per_rank = in_features // tp_size

        self.linear = nn.Linear(self.in_per_rank, out_features, bias=bias, device=device)

    def forward(self, x):
        # x 已经是切分过的（来自上游 ColumnParallel 的输出）
        y = self.linear(x)
        # AllReduce 聚合各 GPU 的部分结果（除非上层会统一做）
        if not self.skip_reduce:
            y = _AllReduceFunc.apply(y, self.tp_group)
        return y

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

    优化：Q/K/V 共用同一个输入 x，原本会触发 3 次 _SplitFunc backward AllReduce。
    用 forward pre-hook 在 attention 入口做 1 次 _SplitFunc，Q/K/V 跳过自己的，
    省 2 次 NCCL/层 backward。
    """
    # 保存原始权重（在原来的 device 上）
    device = attn.q_proj.weight.device
    q_w, q_b = attn.q_proj.weight.data, attn.q_proj.bias.data
    k_w, k_b = attn.k_proj.weight.data, attn.k_proj.bias.data
    v_w, v_b = attn.v_proj.weight.data, attn.v_proj.bias.data
    o_w, o_b = attn.o_proj.weight.data, attn.o_proj.bias.data

    in_dim = q_w.shape[1]
    q_out = q_w.shape[0]
    k_out = k_w.shape[0]
    v_out = v_w.shape[0]
    o_out = o_w.shape[0]

    # 替换为并行版本（在同一 device 上创建）
    attn.q_proj = ColumnParallelLinear(in_dim, q_out, bias=True, tp_group=tp_group, device=device)
    attn.k_proj = ColumnParallelLinear(in_dim, k_out, bias=True, tp_group=tp_group, device=device)
    attn.v_proj = ColumnParallelLinear(in_dim, v_out, bias=True, tp_group=tp_group, device=device)
    attn.o_proj = RowParallelLinear(q_out, o_out, bias=True, tp_group=tp_group, device=device)

    # 加载对应分片的权重
    attn.q_proj.load_weight_shard(q_w, q_b)
    attn.k_proj.load_weight_shard(k_w, k_b)
    attn.v_proj.load_weight_shard(v_w, v_b)
    attn.o_proj.load_weight_shard(o_w, o_b)

    # ── 关键优化：合并 Q/K/V 的 SplitFunc ──
    # 让 Q/K/V 跳过自己的 _SplitFunc（forward identity, backward AllReduce）
    attn.q_proj.skip_split = True
    attn.k_proj.skip_split = True
    attn.v_proj.skip_split = True
    attn.tp_group = tp_group

    # forward pre-hook：在 attention 入口对 x 做 1 次 _SplitFunc
    # backward 时所有 Q/K/V 的 dL/dx 在 autograd 里 sum，然后 SplitFunc 反向只 AllReduce 1 次
    def _split_input_hook(module, args):
        # args = (x, cos, sin)
        x = args[0]
        x = _SplitFunc.apply(x, module.tp_group)
        return (x,) + args[1:]

    attn.register_forward_pre_hook(_split_input_hook)

    # 更新 head 数为本 rank 的 head 数
    tp_size = dist.get_world_size(tp_group)
    attn.num_heads = attn.num_heads // tp_size
    attn.num_kv_heads = attn.num_kv_heads // tp_size
    attn.num_kv_groups = attn.num_heads // attn.num_kv_heads


def tp_parallelize_expert(expert, tp_group):
    """把单个 Expert 的 FFN 替换为 TP 版本。

    w1, w3 → ColumnParallel（沿 intermediate_size 切）
    w2     → RowParallel（结果 AllReduce）
    """
    device = expert.w1.weight.device
    w1_w = expert.w1.weight.data
    w2_w = expert.w2.weight.data
    w3_w = expert.w3.weight.data

    in_dim = w1_w.shape[1]     # hidden_size
    mid_dim = w1_w.shape[0]    # intermediate_size
    out_dim = w2_w.shape[0]    # hidden_size

    expert.w1 = ColumnParallelLinear(in_dim, mid_dim, bias=False, tp_group=tp_group, device=device)
    expert.w3 = ColumnParallelLinear(in_dim, mid_dim, bias=False, tp_group=tp_group, device=device)
    expert.w2 = RowParallelLinear(mid_dim, out_dim, bias=False, tp_group=tp_group, device=device)

    expert.w1.load_weight_shard(w1_w)
    expert.w3.load_weight_shard(w3_w)
    expert.w2.load_weight_shard(w2_w)


# ============================================================
# TP MoE Wrapper —— 合并 16 个 expert 的 AllReduce
# ============================================================

class TPMoEWrapper(nn.Module):
    """Wraps PhiMoESparseMoE with single SplitFunc + single AllReduce per layer.

    朴素 TP 实现：每个 expert 的 w2（RowParallel）做 1 次 AllReduce → 16 次/层。
    优化：让 expert 的 w2 跳过 AllReduce，所有 expert 的 partial output 累积到一个
    tensor 上，最后做 1 次 AllReduce。

    数学正确性：
        original: y = sum_i (RowParallel_AllReduce(expert_i(x_i)))
        new:      y_partial = sum_i (expert_i_partial(x_i))  ← 不 AllReduce
                  y = AllReduce(y_partial)
    AllReduce SUM 是线性的，对求和可交换，所以两者等价。

    注意：gate（router）NOT TP-parallelized，所以不能在 MoE 入口统一 SplitFunc
    （否则 gate 的梯度会被 AllReduce SUM 多算 N 倍）。这里只对 expert 路径做 SplitFunc。
    """

    def __init__(self, original_moe, tp_group):
        super().__init__()
        # 直接持有原 MoE 的子模块（gate, experts），让 nn.Module 自动注册参数
        self.gate = original_moe.gate
        self.experts = original_moe.experts
        self.num_experts_per_tok = original_moe.num_experts_per_tok
        self.tp_group = tp_group
        # AllReduce 浮点非结合性 → routing 可能微差 → broadcast rank 0 的决策
        self._sync_routing = True

    def forward(self, x):
        B, L, D = x.shape
        x_flat = x.view(-1, D)

        # ── 1. Gate（不参与 TP）──
        # 各 rank 用相同 x_flat 算相同 router_logits
        router_logits = self.gate(x_flat)
        topk_weights, topk_indices = torch.topk(
            router_logits.float(), self.num_experts_per_tok, dim=-1
        )
        topk_weights = F.softmax(topk_weights, dim=-1).to(x.dtype)

        # 浮点非结合性安全：广播 rank 0 的路由决策
        if self._sync_routing:
            dist.broadcast(topk_indices, src=0)
            dist.broadcast(topk_weights, src=0)

        # ── 2. 对 expert 路径做 SplitFunc ──
        # 这样 expert 的 backward 梯度（dL/dx_for_experts）会被 1 次 AllReduce SUM 起来
        # gate 的 dL/dx_flat 不走这条路，所以不会被错误地放大
        x_for_experts = _SplitFunc.apply(x_flat, self.tp_group)

        # ── 3. 跑所有 expert（partial output 累积，无 AllReduce）──
        output = torch.zeros_like(x_for_experts)
        for i in range(self.num_experts_per_tok):
            expert_idx = topk_indices[:, i]
            weight = topk_weights[:, i]
            for e_id in range(len(self.experts)):
                mask = (expert_idx == e_id)
                if mask.any():
                    expert_input = x_for_experts[mask]
                    expert_output = self.experts[e_id](expert_input)  # skip_split + skip_reduce
                    output[mask] += weight[mask].unsqueeze(-1) * expert_output

        # ── 4. 整层 1 次 AllReduce ──
        output = _AllReduceFunc.apply(output, self.tp_group)
        return output.view(B, L, D), router_logits


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

    # 对 Attention 和 Expert FFN 做 TP
    # 注意：TP 要求所有 rank 输入相同数据（不用 DistributedSampler）
    base = getattr(model, "module", model)
    for layer in base.model.layers:
        # Attention：合并 Q/K/V 的 SplitFunc（pre-hook 实现）
        tp_parallelize_attention(layer.self_attn, tp_group)

        # Expert FFN：替换 Linear 为 TP 版本，并设置 skip 标志
        # 让 expert 的 w1/w3 跳过 SplitFunc，w2 跳过 AllReduce
        # 这样 TPMoEWrapper 可以在外层统一做 1 次 SplitFunc + 1 次 AllReduce
        for expert in layer.block_sparse_moe.experts:
            tp_parallelize_expert(expert, tp_group)
            expert.w1.skip_split = True
            expert.w3.skip_split = True
            expert.w2.skip_reduce = True

        # 用 TPMoEWrapper 替换原 MoE，统一做 1 次 SplitFunc + 1 次 AllReduce
        layer.block_sparse_moe = TPMoEWrapper(layer.block_sparse_moe, tp_group)

    return model, None
