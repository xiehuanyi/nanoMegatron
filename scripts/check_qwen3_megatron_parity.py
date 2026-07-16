#!/usr/bin/env python3
"""Compare nanoMegatron and Megatron Core on identical weights and tokens."""

import argparse
import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nano_megatron.qwen3 import Qwen3ForCausalLM


def _megatron_qkv(weight, num_heads, num_kv_heads, head_dim):
    q_size = num_heads * head_dim
    kv_size = num_kv_heads * head_dim
    q, k, v = weight.split((q_size, kv_size, kv_size), dim=0)
    q = q.view(num_kv_heads, num_heads // num_kv_heads, head_dim, -1)
    k = k.view(num_kv_heads, 1, head_dim, -1)
    v = v.view(num_kv_heads, 1, head_dim, -1)
    return torch.cat((q, k, v), dim=1).reshape_as(weight)


def _nano_qkv(weight, num_heads, num_kv_heads, head_dim):
    hidden_size = weight.shape[1]
    grouped = weight.view(
        num_kv_heads, num_heads // num_kv_heads + 2, head_dim, hidden_size
    )
    queries = grouped[:, : num_heads // num_kv_heads].reshape(
        num_heads * head_dim, hidden_size
    )
    keys = grouped[:, -2].reshape(num_kv_heads * head_dim, hidden_size)
    values = grouped[:, -1].reshape(num_kv_heads * head_dim, hidden_size)
    return torch.cat((queries, keys, values), dim=0)


def _copy_weights(nano, megatron):
    with torch.no_grad():
        megatron.embedding.word_embeddings.weight.copy_(nano.embed_tokens.weight)
        megatron.decoder.final_layernorm.weight.copy_(nano.norm.weight)
        for nano_layer, mega_layer in zip(nano.layers, megatron.decoder.layers):
            mega_layer.input_layernorm.weight.copy_(nano_layer.input_layernorm.weight)
            mega_layer.pre_mlp_layernorm.weight.copy_(
                nano_layer.post_attention_layernorm.weight
            )
            mega_layer.self_attention.linear_qkv.weight.copy_(
                _megatron_qkv(
                    nano_layer.self_attn.qkv_proj.weight,
                    nano_layer.self_attn.num_heads,
                    nano_layer.self_attn.num_kv_heads,
                    nano_layer.self_attn.head_dim,
                )
            )
            mega_layer.self_attention.linear_proj.weight.copy_(
                nano_layer.self_attn.o_proj.weight
            )
            mega_layer.self_attention.q_layernorm.weight.copy_(
                nano_layer.self_attn.q_norm.weight
            )
            mega_layer.self_attention.k_layernorm.weight.copy_(
                nano_layer.self_attn.k_norm.weight
            )
            mega_layer.mlp.linear_fc1.weight.copy_(nano_layer.mlp.gate_up_proj.weight)
            mega_layer.mlp.linear_fc2.weight.copy_(nano_layer.mlp.down_proj.weight)


def _compare_gradients(nano, megatron):
    comparisons = [
        ("embedding", nano.embed_tokens.weight.grad, megatron.embedding.word_embeddings.weight.grad),
        ("final_norm", nano.norm.weight.grad, megatron.decoder.final_layernorm.weight.grad),
    ]
    for index, (nano_layer, mega_layer) in enumerate(
        zip(nano.layers, megatron.decoder.layers)
    ):
        prefix = f"layer_{index}"
        comparisons.extend(
            [
                (
                    f"{prefix}.input_norm",
                    nano_layer.input_layernorm.weight.grad,
                    mega_layer.input_layernorm.weight.grad,
                ),
                (
                    f"{prefix}.post_attention_norm",
                    nano_layer.post_attention_layernorm.weight.grad,
                    mega_layer.pre_mlp_layernorm.weight.grad,
                ),
                (
                    f"{prefix}.qkv",
                    nano_layer.self_attn.qkv_proj.weight.grad,
                    _nano_qkv(
                        mega_layer.self_attention.linear_qkv.weight.grad,
                        nano_layer.self_attn.num_heads,
                        nano_layer.self_attn.num_kv_heads,
                        nano_layer.self_attn.head_dim,
                    ),
                ),
                (
                    f"{prefix}.attention_output",
                    nano_layer.self_attn.o_proj.weight.grad,
                    mega_layer.self_attention.linear_proj.weight.grad,
                ),
                (
                    f"{prefix}.q_norm",
                    nano_layer.self_attn.q_norm.weight.grad,
                    mega_layer.self_attention.q_layernorm.weight.grad,
                ),
                (
                    f"{prefix}.k_norm",
                    nano_layer.self_attn.k_norm.weight.grad,
                    mega_layer.self_attention.k_layernorm.weight.grad,
                ),
                (
                    f"{prefix}.mlp_input",
                    nano_layer.mlp.gate_up_proj.weight.grad,
                    mega_layer.mlp.linear_fc1.weight.grad,
                ),
                (
                    f"{prefix}.mlp_output",
                    nano_layer.mlp.down_proj.weight.grad,
                    mega_layer.mlp.linear_fc2.weight.grad,
                ),
            ]
        )

    results = {}
    for name, actual, expected in comparisons:
        diff = (actual.float() - expected.float()).abs()
        denominator = expected.float().abs().clamp_min(1e-8)
        results[name] = {
            "max_abs": diff.max().item(),
            "max_rel": (diff / denominator).max().item(),
            "cosine": F.cosine_similarity(
                actual.float().reshape(1, -1), expected.float().reshape(1, -1)
            ).item(),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))

    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig

    parallel_state.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    shape = dict(
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=256,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
    )
    nano_config = SimpleNamespace(**shape, attention_backend="megatron_math")
    mega_config = TransformerConfig(
        num_layers=shape["num_layers"],
        hidden_size=shape["hidden_size"],
        num_attention_heads=shape["num_heads"],
        num_query_groups=shape["num_kv_heads"],
        kv_channels=shape["head_dim"],
        ffn_hidden_size=shape["intermediate_size"],
        normalization="RMSNorm",
        layernorm_epsilon=shape["rms_norm_eps"],
        qk_layernorm=True,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        attention_softmax_in_fp32=False,
        masked_softmax_fusion=False,
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        apply_rope_fusion=False,
        cross_entropy_loss_fusion=False,
        attention_backend=AttnBackend.local,
        params_dtype=dtype,
        use_cpu_initialization=False,
    )
    mega_config.position_embedding_type = "rope"
    mega_config.rotary_base = shape["rope_theta"]

    nano = Qwen3ForCausalLM(nano_config).to(device="cuda", dtype=dtype)
    megatron = GPTModel(
        config=mega_config,
        transformer_layer_spec=get_gpt_layer_local_spec(
            qk_layernorm=True, normalization="RMSNorm"
        ),
        vocab_size=shape["vocab_size"],
        max_sequence_length=32,
        parallel_output=True,
        share_embeddings_and_output_weights=True,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=shape["rope_theta"],
    ).to(device="cuda", dtype=dtype)
    _copy_weights(nano, megatron)

    token_stream = torch.randint(0, shape["vocab_size"], (2, 17), device="cuda")
    tokens = token_stream[:, :-1].contiguous()
    labels = token_stream[:, 1:].contiguous()
    positions = torch.arange(tokens.shape[1], device="cuda").expand_as(tokens)

    nano_logits = nano(tokens)["logits"]
    mega_logits = megatron(tokens, positions, None)
    logit_diff = (nano_logits.float() - mega_logits.float()).abs()

    nano_loss = nano(
        tokens, labels=labels, labels_shifted=True, return_logits=False
    )["loss"]
    mega_loss = megatron(tokens, positions, None, labels=labels).mean()
    nano_loss.backward()
    mega_loss.backward()

    gradient_results = _compare_gradients(nano, megatron)
    print(
        json.dumps(
            {
                "dtype": args.dtype,
                "nano_loss": nano_loss.item(),
                "megatron_loss": mega_loss.item(),
                "loss_abs_diff": abs(nano_loss.item() - mega_loss.item()),
                "logits_max_abs": logit_diff.max().item(),
                "logits_mean_abs": logit_diff.mean().item(),
                "gradient_max_abs": max(
                    result["max_abs"] for result in gradient_results.values()
                ),
                "gradient_min_cosine": min(
                    result["cosine"] for result in gradient_results.values()
                ),
                "gradients": gradient_results,
            }
        ),
        flush=True,
    )

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
