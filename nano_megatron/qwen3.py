"""Minimal dense Qwen3 model used by the Megatron parity benchmark."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Qwen3RMSNorm(nn.RMSNorm):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__(hidden_size, eps=eps)


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cache = {}

    def _apply(self, fn):
        self._cache.clear()
        return super()._apply(fn)

    def forward(self, seq_len: int, dtype: torch.dtype):
        key = (seq_len, dtype, self.inv_freq.device)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        positions = torch.arange(
            seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(positions, self.inv_freq)
        angles = torch.cat((freqs, freqs), dim=-1)
        result = (
            angles.cos().to(dtype)[None, None],
            angles.sin().to(dtype)[None, None],
        )
        self._cache[key] = result
        return result


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin


_CAUSAL_MASK_CACHE = {}


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    key = (seq_len, device)
    mask = _CAUSAL_MASK_CACHE.get(key)
    if mask is None:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )
        _CAUSAL_MASK_CACHE[key] = mask
    return mask


def _megatron_math_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    batch, num_heads, seq_len, head_dim = query.shape
    query = query.reshape(batch * num_heads, seq_len, head_dim)
    key = key.reshape(batch * num_heads, seq_len, head_dim)
    value = value.reshape(batch * num_heads, seq_len, head_dim)

    scores = torch.baddbmm(
        torch.empty(
            batch * num_heads,
            seq_len,
            seq_len,
            device=query.device,
            dtype=query.dtype,
        ),
        query,
        key.transpose(1, 2),
        beta=0.0,
        alpha=1.0 / math.sqrt(head_dim),
    )
    scores = scores.view(batch, num_heads, seq_len, seq_len)
    scores.masked_fill_(_causal_mask(seq_len, query.device), -10000.0)
    probabilities = torch.softmax(scores, dim=-1)
    context = torch.bmm(
        probabilities.view(batch * num_heads, seq_len, seq_len), value
    )
    return context.view(batch, num_heads, seq_len, head_dim)


class _CausalCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits: torch.Tensor, targets: torch.Tensor):
        original_dtype = logits.dtype
        probabilities = logits.float()
        if probabilities.data_ptr() == logits.data_ptr():
            probabilities = probabilities.clone()

        flat_probabilities = probabilities.view(-1, probabilities.shape[-1])
        flat_targets = targets.reshape(-1)
        max_logits = flat_probabilities.amax(dim=-1, keepdim=True)
        flat_probabilities.sub_(max_logits)
        target_logits = flat_probabilities.gather(
            -1, flat_targets.unsqueeze(-1)
        ).squeeze(-1)
        flat_probabilities.exp_()
        sum_exp = flat_probabilities.sum(dim=-1)
        loss = (sum_exp.log() - target_logits).mean()
        flat_probabilities.div_(sum_exp.unsqueeze(-1))

        ctx.save_for_backward(probabilities, flat_targets)
        ctx.original_dtype = original_dtype
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        probabilities, flat_targets = ctx.saved_tensors
        grad_logits = probabilities.view(-1, probabilities.shape[-1])
        rows = torch.arange(flat_targets.numel(), device=flat_targets.device)
        grad_logits[rows, flat_targets] -= 1.0
        grad_logits.mul_(grad_output / flat_targets.numel())
        return probabilities.to(ctx.original_dtype), None


def causal_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return _CausalCrossEntropy.apply(logits, targets)


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

        if self.attention_backend == "megatron_math":
            out = _megatron_math_attention(q, k, v)
        elif self.attention_backend == "math":
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
        self.rotary_emb = Qwen3RotaryEmbedding(config.head_dim, config.rope_theta)
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
        return_logits: bool = True,
        labels_shifted: bool = False,
    ):
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(input_ids.shape[1], x.dtype)
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
                labels_shifted=labels_shifted,
            )
            return {"logits": logits if return_logits else None, "loss": loss}

        # Qwen3 ties the LM head to the token embedding matrix.
        loss = None
        if labels is not None and not return_logits:
            loss_hidden = x if labels_shifted else x[:, :-1]
            loss_targets = labels if labels_shifted else labels[:, 1:]
            loss_logits = F.linear(loss_hidden, self.embed_tokens.weight)
            loss = causal_cross_entropy(loss_logits, loss_targets)
            logits = None
        else:
            logits = F.linear(x, self.embed_tokens.weight)
            if labels is not None:
                loss_logits = logits if labels_shifted else logits[:, :-1].contiguous()
                loss_targets = labels if labels_shifted else labels[:, 1:].contiguous()
                loss = causal_cross_entropy(
                    loss_logits, loss_targets
                )
        return {"logits": logits, "loss": loss}

    def parameter_count(self) -> int:
        return getattr(
            self, "_global_parameter_count", sum(parameter.numel() for parameter in self.parameters())
        )
