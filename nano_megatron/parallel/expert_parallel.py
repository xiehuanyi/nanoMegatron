"""
Expert Parallel（EP）—— 把不同 Expert 分配到不同 GPU，用 AllToAll 分发 token。

原理（DeepSpeed-MoE / Megatron-MoE 风格）：
MoE 有 16 个 Expert，4 卡 EP 则每卡放 4 个 Expert。每个 rank 只处理自己数据的路由，
然后通过 AllToAll 把 token 发送到 expert 所在的 rank 计算，最后 AllToAll 拿回结果。

数据流（4 卡 EP）：

  每个 rank 只看到自己的数据 slice（data parallel over EP ranks）
       │
       ▼ Router 路由本地 token
       │
       ▼ Permute：按目标 expert 对 token 排序
       │
       ▼ AllToAll #1（dispatch）：每个 token 只发给目标 expert 所在 rank
       │
       ▼ 本地 expert 计算（只算自己负责的 expert）
       │
       ▼ AllToAll #2（combine）：结果发回原 rank
       │
       ▼ Unpermute + 加权求和
       │
       ▼ 各 rank 拿到自己数据的完整输出

对比旧版 AllReduce SUM 的做法：
- 旧版：所有 rank 对所有 token 算 router + 16 个 expert 全部冗余（只贡献自己的部分）→ AllReduce
- 新版：每个 rank 只算自己数据的 router + 只跑本地 expert 处理分发来的 token
- 通信量：从 O(B·S·D) 降到 O(top_k·B·S·D/ep_size)
- 计算量：每个 expert GEMM 只执行一次（无冗余）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ============================================================
# AllToAll 带 autograd 支持
# ============================================================

class _AllToAllFunc(torch.autograd.Function):
    """AllToAll 的 autograd 包装。前向 AllToAll，反向也 AllToAll（对称）。"""

    @staticmethod
    def forward(ctx, x, output_split_sizes, input_split_sizes, group):
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group

        world_size = dist.get_world_size(group)
        total_out = sum(output_split_sizes)
        output = torch.empty(total_out, *x.shape[1:], dtype=x.dtype, device=x.device)

        dist.all_to_all_single(
            output, x.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad):
        # 反向：交换 input/output 分片大小即可
        total_out = sum(ctx.input_split_sizes)
        grad_in = torch.empty(total_out, *grad.shape[1:], dtype=grad.dtype, device=grad.device)
        dist.all_to_all_single(
            grad_in, grad.contiguous(),
            output_split_sizes=ctx.input_split_sizes,
            input_split_sizes=ctx.output_split_sizes,
            group=ctx.group,
        )
        return grad_in, None, None, None


def all_to_all(x, output_split_sizes, input_split_sizes, group):
    return _AllToAllFunc.apply(x, output_split_sizes, input_split_sizes, group)


# ============================================================
# EP MoE Layer
# ============================================================

class EPSparseMoE(nn.Module):
    """Expert Parallel 版 MoE 层（AllToAll 实现）。"""

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

        # 只保留本地 expert
        start = self.ep_rank * self.experts_per_rank
        end = start + self.experts_per_rank
        self.local_experts = nn.ModuleList(list(original_moe.experts[start:end]))
        self.expert_start_idx = start

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        x_flat = x.view(-1, D)        # [N, D], N = B*L
        N = x_flat.shape[0]

        # ── Step 1: 本地路由 ──
        router_logits = self.gate(x_flat)                                          # [N, E]
        topk_w, topk_i = torch.topk(router_logits.float(), self.num_experts_per_tok, dim=-1)
        topk_w = F.softmax(topk_w, dim=-1).to(x.dtype)                             # [N, K]

        # 展开成 [N*K] 的路由表：每个 token × K 次选择 = N*K 次分发
        flat_expert_idx = topk_i.reshape(-1)          # [N*K]  每次分发的 expert id
        flat_weight = topk_w.reshape(-1)              # [N*K]  对应的权重
        # 对应原始 token 位置（每个 token 重复 K 次）
        flat_token_idx = torch.arange(N, device=x.device).repeat_interleave(self.num_experts_per_tok)

        # ── Step 2: 按目标 rank 对分发排序（permute）──
        # expert_id → rank: expert_id // experts_per_rank
        dst_rank = flat_expert_idx // self.experts_per_rank                        # [N*K]
        sort_order = torch.argsort(dst_rank)                                       # 稳定排序索引

        sorted_expert = flat_expert_idx[sort_order]
        sorted_token_idx = flat_token_idx[sort_order]
        sorted_weight = flat_weight[sort_order]
        sorted_x = x_flat[sorted_token_idx]                                        # [N*K, D] 按 dst rank 排好的 token

        # 统计每个目标 rank 发送多少个 token
        input_splits = torch.bincount(dst_rank, minlength=self.ep_size).tolist()

        # 交换 split sizes 让对方知道会收到多少
        input_splits_tensor = torch.tensor(input_splits, dtype=torch.long, device=x.device)
        output_splits_tensor = torch.empty_like(input_splits_tensor)
        dist.all_to_all_single(output_splits_tensor, input_splits_tensor, group=self.ep_group)
        output_splits = output_splits_tensor.tolist()

        # ── Step 3: AllToAll #1 — dispatch token 到目标 rank ──
        recv_x = all_to_all(sorted_x, output_splits, input_splits, self.ep_group)
        recv_expert = all_to_all(sorted_expert, output_splits, input_splits, self.ep_group)

        # ── Step 4: 本地 expert 计算 ──
        # recv_expert 是 global expert id，转成本地 id
        local_expert_id = recv_expert - self.expert_start_idx                      # [M]，M = 收到的 token 数
        recv_out = torch.zeros_like(recv_x)
        for e in range(self.experts_per_rank):
            mask = (local_expert_id == e)
            if mask.any():
                recv_out[mask] = self.local_experts[e](recv_x[mask])

        # ── Step 5: AllToAll #2 — 把结果发回原 rank ──
        sent_back = all_to_all(recv_out, input_splits, output_splits, self.ep_group)

        # ── Step 6: unpermute + 加权求和 ──
        # sent_back 的顺序和 sorted_x 一致；先乘权重，再还原到原 token 位置
        weighted = sent_back * sorted_weight.unsqueeze(-1)

        # 用 scatter_add 把 N*K 个加权结果累加回 N 个 token
        output = torch.zeros_like(x_flat)
        output.index_add_(0, sorted_token_idx, weighted)

        return output.view(B, L, D), router_logits


# ============================================================
# 优化器：分片参数 fp32 Adam + 非分片 fp16 SGD
# ============================================================

class EPMixedOptimizer:
    """EP 专用混合精度优化器。

    - Local expert 参数：fp32 Adam（expert 梯度较大，需 fp32 精度）
    - Non-expert 参数（attention, embed）：fp16 SGD（梯度小，省显存）
    """

    def __init__(self, model, lr, weight_decay):
        expert_params, other_params = [], []
        for name, p in model.named_parameters():
            if "local_experts" in name:
                expert_params.append(p)
            else:
                other_params.append(p)

        self.expert_fp16 = expert_params
        self.expert_fp32 = [p.data.float().clone().requires_grad_(True) for p in expert_params]
        self.expert_opt = torch.optim.AdamW(self.expert_fp32, lr=lr, weight_decay=weight_decay)
        self.other_opt = torch.optim.SGD(other_params, lr=lr, momentum=0.9, weight_decay=weight_decay)

        self.param_groups = self.expert_opt.param_groups + self.other_opt.param_groups

    def step(self):
        for fp32_p, fp16_p in zip(self.expert_fp32, self.expert_fp16):
            if fp16_p.grad is not None:
                fp32_p.grad = fp16_p.grad.float()
                fp16_p.grad = None
        self.expert_opt.step()
        for fp32_p, fp16_p in zip(self.expert_fp32, self.expert_fp16):
            fp16_p.data.copy_(fp32_p.data)

        self.other_opt.step()

    def zero_grad(self):
        self.expert_opt.zero_grad()
        self.other_opt.zero_grad()


def setup_ep(model, config):
    """对模型施加 Expert Parallel（AllToAll 版本）。"""
    ep_size = config.parallel.ep_size
    rank = dist.get_rank()

    for i in range(0, dist.get_world_size(), ep_size):
        ranks = list(range(i, i + ep_size))
        group = dist.new_group(ranks)
        if rank in ranks:
            ep_group = group

    model = model.to(rank)

    base = getattr(model, "module", model)
    for layer in base.model.layers:
        original_moe = layer.block_sparse_moe
        layer.block_sparse_moe = EPSparseMoE(original_moe, ep_group)

    optimizer = EPMixedOptimizer(model, lr=config.training.lr,
                                 weight_decay=config.training.weight_decay)
    return model, optimizer
