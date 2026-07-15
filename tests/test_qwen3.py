from types import SimpleNamespace

import torch

from nano_megatron.qwen3 import Qwen3ForCausalLM
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
