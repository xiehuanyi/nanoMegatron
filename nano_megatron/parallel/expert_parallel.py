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
    """前向 AllReduce SUM，反向直通（identity）。

    EP 场景：
    - 前向：各 rank 的 partial output（只有本地 expert 部分非零）AllReduce SUM → 完整输出
    - 反向：y = sum(x_i) 的梯度是 dx_i = dy（直接传过去，不需要再聚合）

    注意：如果反向也做 AllReduce，梯度会被放大 N 倍（N=world_size），导致梯度爆炸 → NaN！
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        output = x.clone()
        dist.all_reduce(output, group=group)
        return output

    @staticmethod
    def backward(ctx, grad):
        # Identity：每个 rank 的梯度直接传回本地 expert
        # 非本地 expert 对应的梯度位置自然为 0（因为 forward 中那些位置的 x_i 是 0）
        return grad, None


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

        # EP 不需要广播路由：所有 rank 输入相同数据 + 相同 gate，结果 bitwise 一致
        # （不像 TP，EP 没有 AllReduce 导致的浮点误差）

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


class EPMixedOptimizer:
    """EP 专用混合精度优化器。

    问题：EP 的 non-expert 参数（attention 等）是冗余复制的，占大部分显存。
    如果全部用 fp32 Adam，V100 32GB 放不下。

    方案：
    - Local expert 参数：fp32 Adam（expert 的 w1/w2/w3 梯度可能较大，需要 fp32 精度）
    - Non-expert 参数（attention, embed, norm）：fp16 Adam（梯度通常较小）

    这样省了 non-expert 参数的 fp32 副本 + fp32 Adam 开销。
    """

    def __init__(self, model, lr, weight_decay):
        # 分离 expert 和 non-expert 参数
        expert_params = []
        other_params = []
        for name, p in model.named_parameters():
            if "local_experts" in name:
                expert_params.append(p)
            else:
                other_params.append(p)

        # Expert: fp32 optimizer
        self.expert_fp16 = expert_params
        self.expert_fp32 = [p.data.float().clone().requires_grad_(True) for p in expert_params]
        self.expert_opt = torch.optim.AdamW(self.expert_fp32, lr=lr, weight_decay=weight_decay)

        # Non-expert: fp16 SGD + momentum（Adam 的 exp_avg_sq 在 fp16 下会溢出导致 NaN）
        # SGD 只有 momentum buffer（一阶），不会有 grad² 溢出问题
        self.other_opt = torch.optim.SGD(other_params, lr=lr, momentum=0.9, weight_decay=weight_decay)

        self.param_groups = self.expert_opt.param_groups + self.other_opt.param_groups

    def step(self):
        # Expert: fp16 grad → fp32 → Adam → sync back
        for fp32_p, fp16_p in zip(self.expert_fp32, self.expert_fp16):
            if fp16_p.grad is not None:
                fp32_p.grad = fp16_p.grad.float()
                fp16_p.grad = None
        self.expert_opt.step()
        for fp32_p, fp16_p in zip(self.expert_fp32, self.expert_fp16):
            fp16_p.data.copy_(fp32_p.data)

        # Non-expert: fp16 SGD（没有二阶矩，不会溢出）
        self.other_opt.step()

    def zero_grad(self):
        self.expert_opt.zero_grad()
        self.other_opt.zero_grad()


def setup_ep(model, config):
    """对模型施加 Expert Parallel。

    Returns:
        (model, EPMixedOptimizer)
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

    # 返回自定义混合精度优化器
    optimizer = EPMixedOptimizer(
        model, lr=config.training.lr, weight_decay=config.training.weight_decay,
    )
    return model, optimizer
