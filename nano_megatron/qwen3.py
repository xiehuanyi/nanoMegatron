"""Minimal dense Qwen3 model used by the Megatron parity benchmark."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def _rope(seq_len: int, head_dim: int, theta: float, device: torch.device):
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), inv_freq)
    angles = torch.cat((freqs, freqs), dim=-1)
    return angles.cos()[None, None], angles.sin()[None, None]


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos.to(x.dtype) + rotated * sin.to(x.dtype)


class Qwen3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.attention_backend = getattr(config, "attention_backend", "sdpa")

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        # Megatron uses one linear_qkv GEMM. Besides reducing launches, this
        # makes autograd save the shared input once rather than three times.
        self.qkv_proj = nn.Linear(
            config.hidden_size, q_size + 2 * kv_size, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        if hasattr(self, "qkv_proj"):
            q_size = self.num_heads * self.head_dim
            kv_size = self.num_kv_heads * self.head_dim
            q, k, v = self.qkv_proj(x).split((q_size, kv_size, kv_size), dim=-1)
        else:
            # Tensor parallelism replaces the fused projection with three
            # independently sharded projections.
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)

        # The portable baseline uses the math SDPA path on both V100 and A100.
        # The production baseline switches this to the fastest available SDPA kernel.
        if self.attention_backend == "math":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            with sdpa_kernel(SDPBackend.MATH):
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Same layout as Megatron's SwiGLU linear_fc1: [gate; up].
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "gate_up_proj"):
            gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        else:
            gate, up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class Qwen3ForCausalLM(nn.Module):
    """Qwen3-0.6B-compatible dense decoder with tied embeddings."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None):
        x = self.embed_tokens(input_ids)
        cos, sin = _rope(
            input_ids.shape[1], self.config.head_dim, self.config.rope_theta, input_ids.device
        )
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, cos, sin, use_reentrant=False)
            else:
                x = layer(x, cos, sin)
        x = self.norm(x)
        if getattr(self, "_tp_vocab", False) and labels is not None:
            from nano_megatron.parallel.qwen3_tensor_parallel import vocab_parallel_cross_entropy

            loss, logits = vocab_parallel_cross_entropy(
                x,
                self.embed_tokens.weight,
                labels,
                self.embed_tokens.vocab_start,
                self.embed_tokens.vocab_end,
                self._tp_group,
            )
            return {"logits": logits, "loss": loss}

        # Qwen3 ties the LM head to the token embedding matrix.
        logits = F.linear(x, self.embed_tokens.weight)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1)
            )
        return {"logits": logits, "loss": loss}

    def parameter_count(self) -> int:
        return getattr(
            self, "_global_parameter_count", sum(parameter.numel() for parameter in self.parameters())
        )
