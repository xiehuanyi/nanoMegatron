"""Tensor parallel sharding for the dense Qwen3 benchmark model."""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from nano_megatron.parallel.tensor_parallel import (
    ColumnParallelLinear,
    RowParallelLinear,
    _AllReduceFunc,
    _SplitFunc,
    tp_parallelize_attention,
)


class VocabParallelEmbedding(nn.Module):
    def __init__(self, embedding: nn.Embedding, tp_group):
        super().__init__()
        self.tp_group = tp_group
        self.tp_size = dist.get_world_size(tp_group)
        self.tp_rank = dist.get_rank(tp_group)
        if embedding.num_embeddings % self.tp_size:
            raise ValueError("vocab size must be divisible by TP size")
        self.vocab_per_rank = embedding.num_embeddings // self.tp_size
        self.vocab_start = self.tp_rank * self.vocab_per_rank
        self.vocab_end = self.vocab_start + self.vocab_per_rank
        shard = embedding.weight.data[self.vocab_start : self.vocab_end].clone()
        self.weight = nn.Parameter(shard)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outside = (input_ids < self.vocab_start) | (input_ids >= self.vocab_end)
        local_ids = (input_ids - self.vocab_start).masked_fill(outside, 0)
        output = F.embedding(local_ids, self.weight)
        output = output.masked_fill(outside.unsqueeze(-1), 0)
        return _AllReduceFunc.apply(output, self.tp_group)


def vocab_parallel_cross_entropy(
    hidden: torch.Tensor,
    local_weight: torch.Tensor,
    labels: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
    tp_group,
):
    # Hidden is replicated; its gradient is the sum of contributions from all vocab shards.
    hidden = _SplitFunc.apply(hidden, tp_group)
    local_logits = F.linear(hidden[:, :-1], local_weight).float()
    targets = labels[:, 1:]

    local_max = local_logits.detach().amax(dim=-1)
    dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=tp_group)
    shifted = local_logits - local_max.unsqueeze(-1)
    local_exp_sum = shifted.exp().sum(dim=-1)
    exp_sum = local_exp_sum.detach().clone()
    dist.all_reduce(exp_sum, op=dist.ReduceOp.SUM, group=tp_group)
    # Preserve the global value while autograd only differentiates the local shard.
    exp_sum = exp_sum + local_exp_sum - local_exp_sum.detach()

    target_is_local = (targets >= vocab_start) & (targets < vocab_end)
    local_targets = (targets - vocab_start).masked_fill(~target_is_local, 0)
    local_target_logits = local_logits.gather(-1, local_targets.unsqueeze(-1)).squeeze(-1)
    local_target_logits = local_target_logits * target_is_local
    target_logits = local_target_logits.detach().clone()
    dist.all_reduce(target_logits, op=dist.ReduceOp.SUM, group=tp_group)
    target_logits = target_logits + local_target_logits - local_target_logits.detach()
    loss = local_max + exp_sum.log() - target_logits
    return loss.mean(), local_logits


def _parallelize_mlp(mlp, tp_group):
    if hasattr(mlp, "gate_up_proj"):
        gate_weight, up_weight = mlp.gate_up_proj.weight.data.chunk(2, dim=0)
        del mlp.gate_up_proj
    else:
        gate_weight = mlp.gate_proj.weight.data
        up_weight = mlp.up_proj.weight.data
    down_weight = mlp.down_proj.weight.data
    hidden_size = gate_weight.shape[1]
    intermediate_size = gate_weight.shape[0]

    mlp.gate_proj = ColumnParallelLinear(
        hidden_size, intermediate_size, bias=False, tp_group=tp_group
    )
    mlp.up_proj = ColumnParallelLinear(
        hidden_size, intermediate_size, bias=False, tp_group=tp_group
    )
    mlp.down_proj = RowParallelLinear(
        intermediate_size, hidden_size, bias=False, tp_group=tp_group
    )
    mlp.gate_proj.load_weight_shard(gate_weight)
    mlp.up_proj.load_weight_shard(up_weight)
    mlp.down_proj.load_weight_shard(down_weight)

    mlp.gate_proj.skip_split = True
    mlp.up_proj.skip_split = True
    mlp.tp_group = tp_group

    def split_input(module, args):
        return (_SplitFunc.apply(args[0], module.tp_group),)

    mlp.register_forward_pre_hook(split_input)


def parallelize_qwen3(model, tp_group=None):
    """Shard attention, MLP, embedding, and LM head across the TP group."""
    tp_group = tp_group or dist.group.WORLD
    model._global_parameter_count = model.parameter_count()
    for layer in model.layers:
        tp_parallelize_attention(layer.self_attn, tp_group)
        _parallelize_mlp(layer.mlp, tp_group)

    model.embed_tokens = VocabParallelEmbedding(model.embed_tokens, tp_group)
    model._tp_vocab = True
    model._tp_group = tp_group

    # Norm weights are replicated. Coalesce their gradient synchronization after backward.
    model._tp_replicated_params = []
    for name, parameter in model.named_parameters():
        if "norm" in name:
            model._tp_replicated_params.append(parameter)
    return model


def sync_replicated_grads(model):
    params = [p for p in model._tp_replicated_params if p.grad is not None]
    if not params:
        return
    flat = torch.cat([p.grad.reshape(-1) for p in params])
    dist.all_reduce(flat, group=model._tp_group)
    offset = 0
    for parameter in params:
        count = parameter.numel()
        parameter.grad.copy_(flat[offset : offset + count].view_as(parameter))
        offset += count
