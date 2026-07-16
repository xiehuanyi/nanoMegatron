from types import SimpleNamespace

import torch

from nano_megatron.qwen3 import Qwen3ForCausalLM, causal_cross_entropy
from nano_megatron.parallel.context_parallel import (
    context_parallel_global_indices,
    context_parallel_indices,
)
from nano_megatron.utils import load_config


def test_qwen3_tiny_forward_backward():
    config = SimpleNamespace(
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=128,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        attention_backend="sdpa",
    )
    model = Qwen3ForCausalLM(config)
    assert hasattr(model.layers[0].self_attn, "qkv_proj")
    assert hasattr(model.layers[0].mlp, "gate_up_proj")
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    output = model(tokens, labels=tokens)
    assert output["logits"].shape == (2, 8, config.vocab_size)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()


def test_qwen3_0_6b_parameter_count():
    config = load_config("configs/qwen3_0.6b_benchmark.yaml")
    with torch.device("meta"):
        model = Qwen3ForCausalLM(config.model)
    assert model.parameter_count() == 596_049_920


def test_causal_cross_entropy_matches_pytorch():
    torch.manual_seed(1234)
    logits = torch.randn(2, 7, 31, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    targets = torch.randint(0, logits.shape[-1], logits.shape[:-1])

    loss = causal_cross_entropy(logits, targets)
    reference_loss = torch.nn.functional.cross_entropy(
        reference_logits.view(-1, reference_logits.shape[-1]), targets.view(-1)
    )
    torch.testing.assert_close(loss, reference_loss)

    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def test_causal_cross_entropy_sum_matches_pytorch():
    torch.manual_seed(1234)
    logits = torch.randn(2, 7, 31, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    targets = torch.randint(0, logits.shape[-1], logits.shape[:-1])

    loss = causal_cross_entropy(logits, targets, reduction="sum")
    reference_loss = torch.nn.functional.cross_entropy(
        reference_logits.view(-1, reference_logits.shape[-1]),
        targets.view(-1),
        reduction="sum",
    )
    torch.testing.assert_close(loss, reference_loss)

    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad)


def test_context_parallel_zigzag_indices_cover_sequence():
    seq_len = 32
    world_size = 4
    shards = [
        context_parallel_indices(seq_len, rank, world_size)
        for rank in range(world_size)
    ]
    assert all(shard.numel() == seq_len // world_size for shard in shards)
    gathered = context_parallel_global_indices(seq_len, world_size)
    torch.testing.assert_close(gathered, torch.cat(shards))
    torch.testing.assert_close(
        gathered.sort().values,
        torch.arange(seq_len),
    )


def test_megatron_math_attention_forward_backward():
    config = SimpleNamespace(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        intermediate_size=64,
        vocab_size=128,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        attention_backend="megatron_math",
    )
    model = Qwen3ForCausalLM(config)
    tokens = torch.randint(0, config.vocab_size, (2, 8))
    output = model(tokens, labels=tokens, return_logits=False)
    assert output["logits"] is None
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
