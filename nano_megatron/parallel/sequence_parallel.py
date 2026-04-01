"""
Sequence Parallel（SP）—— 在 TP 基础上，对 LayerNorm/Dropout 沿 sequence 维度切分。

核心思想：
TP 已经把 Attention/FFN 的计算切分了，但 LayerNorm 和 Dropout 仍然在每个 GPU 上
对完整序列计算（冗余）。SP 把这些操作也切分到 sequence 维度。

数据流：

  输入 [B, L, D]
       │
       ▼ scatter（沿 seq 维度切分）
  [B, L/tp, D] ← 每个 GPU 只拿 1/tp 的序列
       │
       ▼ LayerNorm（只算本地 seq 段）
       │
       ▼ AllGather（恢复完整序列）
  [B, L, D]
       │
       ▼ Attention / MoE（TP 切分）
       │
       ▼ ReduceScatter（结果沿 seq 维度切回来）
  [B, L/tp, D]
       │
       ▼ LayerNorm（只算本地 seq 段）
       ...

好处：LayerNorm 的激活值显存从 O(L) 降到 O(L/tp)。
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from nano_megatron.parallel.tensor_parallel import setup_tp


# ============================================================
# 通信原语
# ============================================================

class _AllGatherFunc(torch.autograd.Function):
    """前向 AllGather（沿 seq 维度拼接），反向 ReduceScatter。"""

    @staticmethod
    def forward(ctx, x, group, dim=1):
        ctx.group = group
        ctx.dim = dim
        world_size = dist.get_world_size(group)
        # 收集各 rank 的 x，沿 dim 拼接
        gather_list = [torch.empty_like(x) for _ in range(world_size)]
        dist.all_gather(gather_list, x, group=group)
        return torch.cat(gather_list, dim=dim)

    @staticmethod
    def backward(ctx, grad):
        # 反向：ReduceScatter（沿 dim 切分并求和）
        group = ctx.group
        dim = ctx.dim
        world_size = dist.get_world_size(group)
        # 把梯度沿 dim 切成 world_size 份
        chunks = grad.chunk(world_size, dim=dim)
        # ReduceScatter：每个 rank 得到自己那份的聚合梯度
        output = torch.empty_like(chunks[0])
        dist.reduce_scatter(output, list(chunks), group=group)
        return output, None, None


class _ReduceScatterFunc(torch.autograd.Function):
    """前向 ReduceScatter（沿 seq 维度切分并求和），反向 AllGather。"""

    @staticmethod
    def forward(ctx, x, group, dim=1):
        ctx.group = group
        ctx.dim = dim
        world_size = dist.get_world_size(group)
        chunks = x.chunk(world_size, dim=dim)
        output = torch.empty_like(chunks[0])
        dist.reduce_scatter(output, list(chunks), group=group)
        return output

    @staticmethod
    def backward(ctx, grad):
        group = ctx.group
        dim = ctx.dim
        world_size = dist.get_world_size(group)
        gather_list = [torch.empty_like(grad) for _ in range(world_size)]
        dist.all_gather(gather_list, grad, group=group)
        return torch.cat(gather_list, dim=ctx.dim), None, None


# ============================================================
# SP 包装的 LayerNorm
# ============================================================

class SPLayerNorm(nn.Module):
    """Sequence Parallel 版 LayerNorm。

    输入是 [B, L/tp, D]（已沿 seq 维度切分），
    LayerNorm 在 D 维度归一化，不需要跨 seq 的信息，所以可以直接算。
    """

    def __init__(self, norm_module: nn.Module, tp_group):
        super().__init__()
        self.norm = norm_module  # 复用原来的 RMSNorm
        self.tp_group = tp_group

    def forward(self, x):
        # x: [B, L/tp, D] → 直接对 D 维度归一化
        return self.norm(x)


# ============================================================
# SP Decoder Layer 包装
# ============================================================

class SPDecoderLayer(nn.Module):
    """在 TP 层外面包一层 SP 的 AllGather/ReduceScatter。"""

    def __init__(self, layer, tp_group):
        super().__init__()
        self.layer = layer
        self.tp_group = tp_group
        # 替换 LayerNorm 为 SP 版（其实 RMSNorm 本身不需要改，这里是标记语义）
        self.layer.input_layernorm = SPLayerNorm(layer.input_layernorm, tp_group)
        self.layer.post_attention_layernorm = SPLayerNorm(layer.post_attention_layernorm, tp_group)

    def forward(self, x, cos, sin):
        # x 输入是 [B, L/tp, D]

        # ── Pre-Norm Attention ──
        residual = x
        x = self.layer.input_layernorm(x)       # [B, L/tp, D] LayerNorm
        x = _AllGatherFunc.apply(x, self.tp_group)  # [B, L, D] 恢复完整序列给 Attention
        x = self.layer.self_attn(x, cos, sin)   # TP Attention
        x = _ReduceScatterFunc.apply(x, self.tp_group)  # [B, L/tp, D] 切回来
        x = residual + x

        # ── Pre-Norm MoE ──
        residual = x
        x = self.layer.post_attention_layernorm(x)  # [B, L/tp, D]
        x = _AllGatherFunc.apply(x, self.tp_group)  # [B, L, D]
        x, router_logits = self.layer.block_sparse_moe(x)
        x = _ReduceScatterFunc.apply(x, self.tp_group)  # [B, L/tp, D]
        x = residual + x

        return x, router_logits


# ============================================================
# Setup
# ============================================================

def setup_sp(model, config):
    """在 TP 基础上增加 Sequence Parallel。

    Returns:
        (model, None)
    """
    # 先应用 TP
    model, _ = setup_tp(model, config)

    tp_size = config.parallel.tp_size
    rank = dist.get_rank()

    # 创建 TP 进程组
    for i in range(0, dist.get_world_size(), tp_size):
        ranks = list(range(i, i + tp_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            tp_group = group

    # 把每个 DecoderLayer 包装成 SPDecoderLayer
    base = getattr(model, "module", model)
    for idx, layer in enumerate(base.model.layers):
        base.model.layers[idx] = SPDecoderLayer(layer, tp_group)

    return model, None
