"""
Expert Parallel（EP）—— 把不同 Expert 分配到不同 GPU。

原理：
MoE 模型有 16 个 Expert，如果用 4 张 GPU 做 EP，每张卡只放 4 个 Expert。
Router 决定每个 token 去哪两个 Expert 后，用 AllReduce 汇总各卡的结果。

数据流（4 卡 EP，16 个 Expert，Top-2 路由）：

  所有 GPU 看到相同的 token（不做数据并行）
       │
       ▼ Router 计算路由（replicated gate，结果一致）
       │
  每个 GPU 只算属于自己的 4 个 Expert
       │
       ▼ AllReduce（把各 GPU 的部分结果求和）
       │
  每个 GPU 拿到完整的 MoE 输出

好处：Expert 参数分散，单卡显存需求降低。
代价：每层一次 AllReduce 通信。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ============================================================
# AllReduce with autograd support
# ============================================================

class _EPAllReduceFunc(torch.autograd.Function):
    """前向 AllReduce，反向也 AllReduce。

    EP 场景：前向时各 rank 算了部分 expert 的输出，AllReduce 求和得到完整结果。
    反向时，梯度也需要 AllReduce（因为每个 rank 的 expert 都需要收到完整梯度）。
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        # 拷贝一份再 AllReduce，避免 in-place 操作影响 autograd
        output = x.clone()
        dist.all_reduce(output, group=group)
        return output

    @staticmethod
    def backward(ctx, grad):
        # 反向：每个 rank 的 expert 需要对应的梯度，也是 AllReduce
        grad_out = grad.clone()
        dist.all_reduce(grad_out, group=ctx.group)
        return grad_out, None


# ============================================================
# EP MoE Layer
# ============================================================

class EPSparseMoE(nn.Module):
    """Expert Parallel 版 MoE 层。

    每个 GPU 只持有 num_experts/ep_size 个 Expert。
    所有 GPU 看到相同数据，用相同 gate 做路由（结果一致），
    各自算本地 expert 的输出，最后 AllReduce 汇总。
    """

    def __init__(self, original_moe, ep_group):
        super().__init__()
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group)
        self.ep_rank = dist.get_rank(ep_group)
        self.num_experts = len(original_moe.experts)
        self.num_experts_per_tok = original_moe.num_experts_per_tok
        self.experts_per_rank = self.num_experts // self.ep_size

        # Router（gate）在所有 GPU 上复制
        self.gate = original_moe.gate

        # 只保留属于本 rank 的 expert
        start = self.ep_rank * self.experts_per_rank
        end = start + self.experts_per_rank
        self.local_experts = nn.ModuleList(list(original_moe.experts[start:end]))
        self.expert_start_idx = start

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        x_flat = x.view(-1, D)  # [N, D]

        # ── Step 1: 路由（所有 rank 结果一致，因为 gate 和输入相同）──
        router_logits = self.gate(x_flat)  # [N, num_experts]
        topk_weights, topk_indices = torch.topk(router_logits.float(), self.num_experts_per_tok, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1).to(x.dtype)

        # 广播路由决策，确保所有 rank 完全一致（防止浮点误差导致路由不同）
        dist.broadcast(topk_indices, src=0, group=self.ep_group)
        dist.broadcast(topk_weights, src=0, group=self.ep_group)

        # ── Step 2: 每个 rank 只算本地 expert ──
        output = torch.zeros_like(x_flat)

        for i in range(self.num_experts_per_tok):
            expert_idx = topk_indices[:, i]     # [N]
            weight = topk_weights[:, i]         # [N]

            for local_e in range(self.experts_per_rank):
                global_e = self.expert_start_idx + local_e
                mask = (expert_idx == global_e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.local_experts[local_e](expert_input)
                    output[mask] += weight[mask].unsqueeze(-1) * expert_output

        # ── Step 3: AllReduce 合并各 rank 的部分结果（带 autograd 支持）──
        output = _EPAllReduceFunc.apply(output, self.ep_group)

        return output.view(B, L, D), router_logits


def setup_ep(model, config):
    """对模型施加 Expert Parallel。

    Returns:
        (model, None)
    """
    ep_size = config.parallel.ep_size
    rank = dist.get_rank()

    # 创建 EP 进程组
    for i in range(0, dist.get_world_size(), ep_size):
        ranks = list(range(i, i + ep_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            ep_group = group

    model = model.to(rank)

    # 替换每层的 MoE 为 EP 版本
    base = getattr(model, "module", model)
    for layer in base.model.layers:
        original_moe = layer.block_sparse_moe
        layer.block_sparse_moe = EPSparseMoE(original_moe, ep_group)

    return model, None
