"""
Pipeline Parallel（PP）—— 把模型的不同层放到不同 GPU，用 micro-batch 流水线。

原理（GPipe 风格）：
1. 把 32 层 Transformer 均匀分到 pp_size 个 GPU（stage）
2. 把一个 batch 切成 M 个 micro-batch
3. 按流水线调度：stage 0 算完 micro-batch 0 后传给 stage 1，
   然后 stage 0 继续算 micro-batch 1，以此类推
4. 所有 micro-batch 前向完成后，反向传播（逆序）

时间线（4 个 micro-batch，2 个 stage）：

  时间 →    t0    t1    t2    t3    t4    t5    t6    t7
  Stage 0:  F0    F1    F2    F3    B3    B2    B1    B0
  Stage 1:        F0    F1    F2    F3    B3    B2    B1

  F = Forward, B = Backward

好处：模型参数分散在多个 GPU，单卡显存需求降低。
代价：存在 pipeline bubble（空闲时间），micro-batch 越多 bubble 越小。
"""

import torch
import torch.nn as nn
import torch.distributed as dist


class PipelineStage(nn.Module):
    """一个 pipeline stage，包含模型的一部分层。"""

    def __init__(self, layers, is_first: bool, is_last: bool, embed=None, norm=None, lm_head=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.is_first = is_first
        self.is_last = is_last

        # 第一个 stage 持有 embedding，最后一个 stage 持有 norm + lm_head
        self.embed = embed
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, x_or_ids, cos, sin, labels=None):
        if self.is_first:
            x = self.embed(x_or_ids)  # token ids → embeddings
        else:
            x = x_or_ids  # 接收上一个 stage 传来的 hidden states

        for layer in self.layers:
            x, _ = layer(x, cos, sin)

        if self.is_last:
            x = self.norm(x)
            logits = self.lm_head(x)
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            return {"logits": logits, "loss": loss, "router_logits": []}
        return x


def split_model_into_stages(model, pp_size: int, rank: int) -> PipelineStage:
    """把模型切成 pp_size 个 stage，返回当前 rank 对应的 stage。"""
    base = getattr(model, "module", model)
    all_layers = list(base.model.layers)
    num_layers = len(all_layers)

    # 均匀分配层
    layers_per_stage = num_layers // pp_size
    remainder = num_layers % pp_size

    # 前 remainder 个 stage 多分一层
    starts = []
    offset = 0
    for i in range(pp_size):
        count = layers_per_stage + (1 if i < remainder else 0)
        starts.append((offset, offset + count))
        offset += count

    start, end = starts[rank]
    my_layers = all_layers[start:end]

    is_first = (rank == 0)
    is_last = (rank == pp_size - 1)

    return PipelineStage(
        layers=my_layers,
        is_first=is_first,
        is_last=is_last,
        embed=base.model.embed_tokens if is_first else None,
        norm=base.model.norm if is_last else None,
        lm_head=base.lm_head if is_last else None,
    )


def pp_train_step(model, batch, optimizer, scaler):
    """GPipe 风格的流水线训练步。

    model 是 PipelineStage，只包含当前 rank 的层。
    """
    pp_rank = dist.get_rank()
    pp_size = dist.get_world_size()
    num_micro_batches = 4  # 可配置

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    B = input_ids.shape[0]
    micro_B = B // num_micro_batches

    # 预计算 RoPE
    from nano_megatron.model import build_rope_cache
    L = input_ids.shape[1]
    device = input_ids.device
    cos, sin = build_rope_cache(L, 128, 10000.0, device)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # ── Forward pass：逐个 micro-batch ──
    losses = []
    hidden_states = []  # 保存中间结果用于 backward

    for m in range(num_micro_batches):
        mb_start = m * micro_B
        mb_end = mb_start + micro_B
        mb_ids = input_ids[mb_start:mb_end]
        mb_labels = labels[mb_start:mb_end]

        if model.is_first:
            # 第一个 stage：从 token ids 开始
            inp = mb_ids
        else:
            # 非第一个 stage：从上一个 stage 接收 hidden states
            inp = torch.empty(micro_B, L, model.layers[0].input_layernorm.norm.weight.shape[0],
                              device=device, dtype=torch.float32)
            dist.recv(inp, src=pp_rank - 1)
            inp.requires_grad_(True)

        output = model(inp, cos, sin, labels=mb_labels if model.is_last else None)

        if model.is_last:
            losses.append(output["loss"])
        else:
            # 非最后一个 stage：把 hidden states 发给下一个 stage
            dist.send(output.detach(), dst=pp_rank + 1)

        hidden_states.append((inp, output))

    # ── Backward pass：逆序 micro-batch ──
    for m in reversed(range(num_micro_batches)):
        inp, output = hidden_states[m]

        if model.is_last:
            output["loss"].backward()
        else:
            # 从下一个 stage 接收梯度
            grad = torch.empty_like(output)
            dist.recv(grad, src=pp_rank + 1)
            output.backward(grad)

        if not model.is_first and inp.grad is not None:
            # 把梯度传给上一个 stage
            dist.send(inp.grad, dst=pp_rank - 1)

    # 返回平均 loss
    if losses:
        return sum(l.item() for l in losses) / len(losses)
    return 0.0


def setup_pp(model, config):
    """对模型施加 Pipeline Parallel。

    Returns:
        (stage_model, pp_train_step)  —— 返回自定义 train_step
    """
    pp_size = config.parallel.pp_size
    rank = dist.get_rank()

    # 先把完整模型加载到 CPU，然后只把本 stage 的部分移到 GPU
    stage = split_model_into_stages(model, pp_size, rank)
    stage = stage.to(rank)

    return stage, pp_train_step
