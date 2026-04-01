"""
Expert Parallel（EP）—— 把不同 Expert 分配到不同 GPU。

原理：
MoE 模型有 16 个 Expert，如果用 4 张 GPU 做 EP，每张卡只放 4 个 Expert。
Router 决定每个 token 去哪两个 Expert 后，用 AlltoAll 通信把 token 发到对应卡上。

数据流（4 卡 EP，16 个 Expert，Top-2 路由）：

  所有 GPU 都有完整 token
       │
       ▼ Router 计算路由分配
       │
       ▼ AlltoAll（把 token 发到 expert 所在的 GPU）
       │
  每个 GPU 只算自己 4 个 Expert 的 token
       │
       ▼ AlltoAll（把结果发回原来的 GPU）
       │
  每个 GPU 拿到自己 token 的最终结果

好处：Expert 参数分散，单卡显存需求降低。
代价：AlltoAll 通信开销（两次）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


class EPSparseMoE(nn.Module):
    """Expert Parallel 版 MoE 层。

    每个 GPU 只持有 num_experts/ep_size 个 Expert。
    Token 通过 AlltoAll 发送到目标 Expert 所在的 GPU。
    """

    def __init__(self, original_moe, ep_group):
        super().__init__()
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group)
        self.ep_rank = dist.get_rank(ep_group)
        self.num_experts = len(original_moe.experts)
        self.num_experts_per_tok = original_moe.num_experts_per_tok
        self.experts_per_rank = self.num_experts // self.ep_size

        # Router（gate）在所有 GPU 上复制，因为所有 token 都需要路由决策
        self.gate = original_moe.gate

        # 只保留属于本 rank 的 expert
        start = self.ep_rank * self.experts_per_rank
        end = start + self.experts_per_rank
        self.local_experts = nn.ModuleList(list(original_moe.experts[start:end]))
        self.expert_start_idx = start

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        x_flat = x.view(-1, D)  # [N, D]，N = B*L
        N = x_flat.shape[0]

        # ── Step 1: 所有 GPU 计算路由 ──
        router_logits = self.gate(x_flat)  # [N, num_experts]
        topk_weights, topk_indices = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)  # [N, 2]

        # ── Step 2: AlltoAll 把 token 发到 expert 所在的 GPU ──
        # 统计每个 rank 需要接收多少 token
        # expert_id -> rank: expert_id // experts_per_rank
        expert_to_rank = topk_indices // self.experts_per_rank  # [N, 2]

        # 对每个 expert 收集要处理的 token
        send_counts = torch.zeros(self.ep_size, dtype=torch.long, device=x.device)
        for r in range(self.ep_size):
            send_counts[r] = (expert_to_rank == r).sum()

        # AlltoAll 交换 count 信息
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)

        # 简化版 AlltoAll：按 expert 分组发送 token
        # 生产环境会用 torch.distributed.all_to_all，这里用更清晰的循环实现
        output = torch.zeros_like(x_flat)

        for i in range(self.num_experts_per_tok):
            expert_idx = topk_indices[:, i]     # [N]
            weight = topk_weights[:, i]         # [N]

            # 只处理属于本 rank 的 expert
            for local_e in range(self.experts_per_rank):
                global_e = self.expert_start_idx + local_e
                mask = (expert_idx == global_e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.local_experts[local_e](expert_input)
                    output[mask] += weight[mask].unsqueeze(-1) * expert_output

        # AllReduce 合并各 GPU 的部分结果
        # （因为每个 GPU 只算了部分 expert 的输出，需要求和）
        dist.all_reduce(output, group=self.ep_group)

        return output.view(B, L, D), router_logits


def setup_ep(model, config):
    """对模型施加 Expert Parallel：把每层的 MoE 替换为 EP 版本。

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
