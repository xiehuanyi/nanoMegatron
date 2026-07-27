"""Correctness-first context parallelism for the dense Qwen3 path.

The implementation mirrors Megatron's all-gather CP variant: queries remain
sequence-sharded, while keys and values are gathered for attention. Backward
uses ReduceScatter so every rank receives the complete gradient for its local
K/V activations.
"""

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F


def context_parallel_indices(
    seq_len: int,
    rank: int,
    world_size: int,
    device=None,
) -> torch.Tensor:
    """Return Megatron-style zigzag token indices owned by one CP rank."""
    if world_size < 1:
        raise ValueError(f"context parallel size must be positive, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"context parallel rank {rank} is outside [0, {world_size})")
    chunks = 2 * world_size
    if seq_len % chunks:
        raise ValueError(
            f"sequence length {seq_len} must be divisible by 2 * "
            f"context parallel size ({chunks})"
        )

    chunk_len = seq_len // chunks
    first = torch.arange(
        rank * chunk_len,
        (rank + 1) * chunk_len,
        device=device,
        dtype=torch.long,
    )
    mirror_rank = chunks - rank - 1
    second = torch.arange(
        mirror_rank * chunk_len,
        (mirror_rank + 1) * chunk_len,
        device=device,
        dtype=torch.long,
    )
    return torch.cat((first, second))


def context_parallel_global_indices(
    seq_len: int,
    world_size: int,
    device=None,
) -> torch.Tensor:
    """Return the rank-major order produced by the K/V AllGather."""
    return torch.cat(
        [
            context_parallel_indices(seq_len, rank, world_size, device=device)
            for rank in range(world_size)
        ]
    )


class _AllGatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, dim: int, group):
        world_size = dist.get_world_size(group=group)
        ctx.dim = dim
        ctx.group = group
        ctx.world_size = world_size
        if world_size == 1:
            return tensor

        sequence_first = tensor.movedim(dim, 0).contiguous()
        output_shape = (sequence_first.shape[0] * world_size, *sequence_first.shape[1:])
        gathered = torch.empty(
            output_shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        dist.all_gather_into_tensor(gathered, sequence_first, group=group)
        return gathered.movedim(0, dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.world_size == 1:
            return grad_output, None, None

        sequence_first = grad_output.movedim(ctx.dim, 0).contiguous()
        local_shape = (
            sequence_first.shape[0] // ctx.world_size,
            *sequence_first.shape[1:],
        )
        local_grad = torch.empty(
            local_shape,
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        dist.reduce_scatter_tensor(
            local_grad,
            sequence_first,
            op=dist.ReduceOp.SUM,
            group=ctx.group,
        )
        return local_grad.movedim(0, ctx.dim), None, None


def gather_context_parallel_tensor(
    tensor: torch.Tensor,
    dim: int,
    group=None,
) -> torch.Tensor:
    return _AllGatherSequence.apply(tensor, dim, group)


def _math_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed: torch.Tensor,
) -> torch.Tensor:
    batch, num_heads, query_len, head_dim = query.shape
    key_len = key.shape[2]
    flat_query = query.reshape(batch * num_heads, query_len, head_dim)
    flat_key = key.reshape(batch * num_heads, key_len, head_dim)
    flat_value = value.reshape(batch * num_heads, key_len, head_dim)
    scores = torch.baddbmm(
        torch.empty(
            batch * num_heads,
            query_len,
            key_len,
            device=query.device,
            dtype=query.dtype,
        ),
        flat_query,
        flat_key.transpose(1, 2),
        beta=0.0,
        alpha=1.0 / math.sqrt(head_dim),
    ).view(batch, num_heads, query_len, key_len)
    scores.masked_fill_(~allowed, -10000.0)
    probabilities = torch.softmax(scores, dim=-1)
    context = torch.bmm(
        probabilities.view(batch * num_heads, query_len, key_len),
        flat_value,
    )
    return context.view(batch, num_heads, query_len, head_dim)


def context_parallel_attention(
    query: torch.Tensor,
    local_key: torch.Tensor,
    local_value: torch.Tensor,
    query_positions: torch.Tensor,
    gathered_key_positions: torch.Tensor,
    group=None,
    backend: str = "sdpa",
    num_kv_groups: int = 1,
) -> torch.Tensor:
    """Compute causal attention for local queries against globally gathered K/V."""
    key = gather_context_parallel_tensor(local_key, dim=2, group=group)
    value = gather_context_parallel_tensor(local_value, dim=2, group=group)
    key = key.repeat_interleave(num_kv_groups, dim=1)
    value = value.repeat_interleave(num_kv_groups, dim=1)
    allowed = (
        gathered_key_positions.view(1, 1, 1, -1)
        <= query_positions.view(1, 1, -1, 1)
    )

    if backend == "megatron_math":
        return _math_attention(query, key, value, allowed)
    if backend == "math":
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=allowed
            )
    return F.scaled_dot_product_attention(query, key, value, attn_mask=allowed)


def context_parallel_loss(
    local_loss_sum: torch.Tensor,
    local_token_count: int,
    group=None,
) -> torch.Tensor:
    """Return the global mean loss while retaining only the local autograd edge."""
    global_loss_sum = local_loss_sum.detach().clone()
    dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM, group=group)
    token_count = torch.tensor(
        local_token_count,
        dtype=torch.long,
        device=local_loss_sum.device,
    )
    dist.all_reduce(token_count, op=dist.ReduceOp.SUM, group=group)
    return (
        local_loss_sum + global_loss_sum - local_loss_sum.detach()
    ) / token_count


def parallelize_qwen3_context(model, process_group=None):
    """Enable CP across the supplied group; the input sequence stays replicated."""
    if not dist.is_initialized():
        raise RuntimeError("context parallelism requires an initialized process group")
    if getattr(model, "_tp_vocab", False):
        raise NotImplementedError("TP + CP composition is not implemented yet")

    process_group = process_group or dist.group.WORLD
    world_size = dist.get_world_size(group=process_group)
    rank = dist.get_rank(group=process_group)
    if world_size < 2:
        raise ValueError("context parallelism requires at least two ranks")
    model._cp_group = process_group
    model._cp_size = world_size
    model._cp_rank = rank
    return model


@torch.no_grad()
def sync_context_parallel_grads(
    model,
    process_group=None,
    bucket_size: int = 20_000_000,
):
    """Sum replicated parameter gradients across CP ranks in bounded buffers."""
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    bucket = []
    bucket_numel = 0

    def flush():
        nonlocal bucket, bucket_numel
        if not bucket:
            return
        if len(bucket) == 1 and bucket[0].grad is not None:
            dist.all_reduce(
                bucket[0].grad,
                op=dist.ReduceOp.SUM,
                group=process_group,
            )
            bucket = []
            bucket_numel = 0
            return
        flat = torch.cat(
            [
                parameter.grad.view(-1)
                if parameter.grad is not None
                else torch.zeros(
                    parameter.numel(),
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
                for parameter in bucket
            ]
        )
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=process_group)
        offset = 0
        for parameter in bucket:
            numel = parameter.numel()
            if parameter.grad is None:
                parameter.grad = torch.empty_like(parameter)
            parameter.grad.copy_(flat[offset : offset + numel].view_as(parameter))
            offset += numel
        bucket = []
        bucket_numel = 0

    for parameter in params:
        different_storage = bucket and (
            parameter.dtype != bucket[0].dtype
            or parameter.device != bucket[0].device
        )
        if different_storage or (bucket and bucket_numel + parameter.numel() > bucket_size):
            flush()
        bucket.append(parameter)
        bucket_numel += parameter.numel()
    flush()
