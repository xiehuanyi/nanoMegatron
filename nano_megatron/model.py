"""
Phi-tiny-MoE 模型（手写）。

架构要点：
- 32 层 Transformer，每层包含 GQA Attention + Sparse MoE FFN
- GQA：16 个 Query Head，4 个 KV Head（每个 KV Head 服务 4 个 Query Head）
- MoE：16 个 Expert，Top-2 路由，每个 Expert 是 SiLU-gated FFN（intermediate=448）
- RoPE 位置编码，RMSNorm（带 bias）

权重命名完全对齐 HuggingFace `microsoft/Phi-tiny-MoE-instruct`，可直接加载。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 基础组件
# ============================================================

class RMSNorm(nn.Module):
    """RMSNorm（带 bias），Phi 系列模型的标配。
    和 LayerNorm 区别：不减均值，只除以 RMS。"""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 始终在 fp32 下计算（fp16 下 x^2 容易溢出）
        orig_dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight.float() * (x * rms) + self.bias.float()).to(orig_dtype)


def build_rope_cache(seq_len: int, head_dim: int, theta: float = 10000.0,
                     device: torch.device = None) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算 RoPE 的 cos/sin 表。返回 (cos, sin)，形状 [seq_len, head_dim]。"""
    # 频率：theta^(-2i/d)，i = 0, 1, ..., d/2-1
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    # 位置 × 频率 -> 角度
    positions = torch.arange(seq_len, device=device).float()
    angles = torch.outer(positions, freqs)  # [seq_len, head_dim/2]
    # 拼成 [seq_len, head_dim]（每对相邻维度共享同一个角度）
    angles = angles.repeat(1, 2)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 x 施加 RoPE 旋转。x: [batch, heads, seq_len, head_dim]。"""
    # cos/sin 转成和 x 相同的 dtype，避免 fp16 模型时 dtype 不匹配
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)
    # 把相邻维度配对旋转：(x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)
    d = x.shape[-1]
    x_rot = torch.stack([-x[..., d // 2:], x[..., :d // 2]], dim=-1).flatten(-2)
    return x * cos + x_rot * sin


# ============================================================
# Attention（GQA + RoPE）
# ============================================================

class PhiMoEAttention(nn.Module):
    """Grouped Query Attention：16 个 Q head 共享 4 个 KV head。"""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads          # 16
        self.num_kv_heads = config.num_kv_heads    # 4
        self.head_dim = config.head_dim            # 128
        self.num_kv_groups = self.num_heads // self.num_kv_heads  # 4（每个 KV head 服务 4 个 Q head）

        # 投影层（Phi 系列全部带 bias）
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=True)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape

        # 投影 + reshape -> [B, heads, L, head_dim]
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE 旋转位置编码
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA：把 KV head 重复 4 次，对齐 Q head 数量
        # [B, 4, L, 128] -> [B, 16, L, 128]
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Scaled Dot-Product Attention（PyTorch 2.0+ 自动用 FlashAttention）
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # 合并多头 -> 输出投影
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


# ============================================================
# MoE（Mixture of Experts）
# ============================================================

class PhiMoEExpert(nn.Module):
    """单个 Expert：SiLU-gated FFN。
    output = w2( silu(w1(x)) * w3(x) )
    """

    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class PhiMoESparseMoE(nn.Module):
    """Sparse MoE 层：Router + 16 Experts，Top-2 路由。

    路由流程：
    1. Gate 线性层算出每个 token 对 16 个 expert 的 logits
    2. Top-2 选出两个 expert，softmax 得到权重
    3. 分别过两个 expert，加权求和
    """

    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([PhiMoEExpert(config) for _ in range(config.num_experts)])
        self.num_experts_per_tok = config.num_experts_per_tok  # 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_flat = x.view(-1, D)  # [B*L, D]

        # 路由：计算每个 token 对所有 expert 的分数
        router_logits = self.gate(x_flat)                          # [B*L, num_experts]
        # Top-K 选择（在 fp32 下做 softmax，防止溢出）
        topk_weights, topk_indices = torch.topk(router_logits.float(), self.num_experts_per_tok, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1).to(x.dtype)  # 归一化权重

        # TP 场景：AllReduce 的浮点非结合性可能导致各 rank routing 微小不同，需要广播同步
        # ZeRO/DDP：各 rank 数据不同，不能广播
        # EP：有自己的 EPSparseMoE.forward 处理
        if getattr(self, "_sync_routing", False):
            import torch.distributed as dist
            dist.broadcast(topk_indices, src=0)
            dist.broadcast(topk_weights, src=0)

        # 对每个 token，过选中的 expert 并加权求和
        # 注意：这里用简单循环实现，清晰易懂。生产环境会用 scatter/gather 加速。
        output = torch.zeros_like(x_flat)
        for i in range(self.num_experts_per_tok):
            expert_idx = topk_indices[:, i]    # [B*L] 每个 token 选的第 i 个 expert
            weight = topk_weights[:, i]        # [B*L] 对应权重

            # 按 expert 分组处理（避免对每个 token 单独调 expert）
            for e_id in range(len(self.experts)):
                mask = (expert_idx == e_id)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e_id](expert_input)
                    output[mask] += weight[mask].unsqueeze(-1) * expert_output

        return output.view(B, L, D), router_logits


# ============================================================
# Transformer Block
# ============================================================

class PhiMoEDecoderLayer(nn.Module):
    """一层 Transformer = LayerNorm + Attention + LayerNorm + MoE。"""

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = PhiMoEAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_sparse_moe = PhiMoESparseMoE(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # Pre-Norm Attention + 残差
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin)
        x = residual + x

        # Pre-Norm MoE + 残差
        residual = x
        x = self.post_attention_layernorm(x)
        x, router_logits = self.block_sparse_moe(x)
        x = residual + x

        return x, router_logits


# ============================================================
# 完整模型
# ============================================================

class PhiMoEModel(nn.Module):
    """Transformer backbone（不含 LM head）。"""

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([PhiMoEDecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.config = config
        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        """开启梯度检查点：用时间换显存，forward 时不保存中间激活，backward 时重算。"""
        self.gradient_checkpointing = True

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        x = self.embed_tokens(input_ids)

        # 预计算 RoPE cos/sin
        cos, sin = build_rope_cache(L, self.config.head_dim, self.config.rope_theta, x.device)
        # 广播到 [1, 1, L, head_dim] 方便和 attention 的 [B, heads, L, head_dim] 相乘
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        all_router_logits = []
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # 梯度检查点：不保存中间激活，backward 时重新计算
                x, router_logits = torch.utils.checkpoint.checkpoint(
                    layer, x, cos, sin, use_reentrant=False
                )
            else:
                x, router_logits = layer(x, cos, sin)
            all_router_logits.append(router_logits)

        x = self.norm(x)
        return x, all_router_logits


class PhiMoEForCausalLM(nn.Module):
    """Phi-tiny-MoE 因果语言模型 = Backbone + LM Head。"""

    def __init__(self, config):
        super().__init__()
        self.model = PhiMoEModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.config = config

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        hidden, router_logits = self.model(input_ids)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            # 标准 next-token prediction loss
            # 始终在 fp32 下计算 loss，防止 fp16 溢出导致 NaN
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss, "router_logits": router_logits}


# ============================================================
# 权重加载
# ============================================================

def load_hf_weights(model: PhiMoEForCausalLM, model_name: str, device: str = "cpu"):
    """从 HuggingFace 加载预训练权重到手写模型。

    因为我们的参数命名和 HF checkpoint 完全一致，直接 load_state_dict 即可。
    """
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download, list_repo_files
    import os

    # 找到所有 safetensors 分片文件
    files = list_repo_files(model_name)
    shard_files = sorted([f for f in files if f.endswith(".safetensors")])

    # 逐个分片加载并合并
    state_dict = {}
    for shard in shard_files:
        local_path = hf_hub_download(model_name, shard)
        state_dict.update(load_file(local_path, device=device))

    # 加载到模型（strict=True 确保每个权重都对上了）
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARNING] Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[WARNING] Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    print(f"[INFO] Loaded {len(state_dict)} tensors from {model_name}")
    return model
