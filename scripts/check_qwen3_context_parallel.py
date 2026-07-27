#!/usr/bin/env python3
"""Compare two-rank Qwen3 context parallel numerics with a full reference."""

import argparse
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist

from nano_megatron.parallel.context_parallel import (
    context_parallel_indices,
    parallelize_qwen3_context,
    sync_context_parallel_grads,
)
from nano_megatron.qwen3 import Qwen3ForCausalLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    if args.dtype == "float16" and not use_cuda:
        raise RuntimeError("the float16 context-parallel check requires CUDA")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if use_cuda:
        torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl" if use_cuda else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if use_cuda:
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if world_size != 2:
        raise ValueError(f"this check expects exactly 2 ranks, got {world_size}")

    config = SimpleNamespace(
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        vocab_size=256,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        attention_backend="megatron_math",
    )
    torch.manual_seed(1234)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    reference = Qwen3ForCausalLM(config).to(device=device, dtype=dtype)
    candidate = Qwen3ForCausalLM(config).to(device=device, dtype=dtype)
    candidate.load_state_dict(reference.state_dict())
    candidate = parallelize_qwen3_context(candidate)
    reference.enable_gradient_checkpointing()
    candidate.enable_gradient_checkpointing()

    generator = torch.Generator(device=device).manual_seed(5678)
    tokens = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
        generator=generator,
        device=device,
    )
    labels = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
        generator=generator,
        device=device,
    )

    reference_output = reference(
        tokens,
        labels=labels,
        return_logits=True,
        labels_shifted=True,
    )
    candidate_output = candidate(
        tokens,
        labels=labels,
        return_logits=True,
        labels_shifted=True,
    )
    local_indices = context_parallel_indices(
        tokens.shape[1],
        rank,
        world_size,
        device=device,
    )
    expected_logits = reference_output["logits"].index_select(1, local_indices)
    logit_max_diff = (
        candidate_output["logits"].float() - expected_logits.float()
    ).abs().max()
    dist.all_reduce(logit_max_diff, op=dist.ReduceOp.MAX)
    loss_diff = (
        candidate_output["loss"].float() - reference_output["loss"].float()
    ).abs()
    dist.all_reduce(loss_diff, op=dist.ReduceOp.MAX)
    logits_atol = 2e-3 if dtype == torch.float16 else 2e-5
    logits_rtol = 5e-3 if dtype == torch.float16 else 2e-5
    torch.testing.assert_close(
        candidate_output["logits"],
        expected_logits,
        rtol=logits_rtol,
        atol=logits_atol,
    )
    torch.testing.assert_close(
        candidate_output["loss"],
        reference_output["loss"],
        rtol=2e-4 if dtype == torch.float16 else 2e-6,
        atol=2e-4 if dtype == torch.float16 else 2e-6,
    )

    reference_output["loss"].backward()
    candidate_output["loss"].backward()
    sync_context_parallel_grads(candidate)
    grad_max_diff = torch.zeros((), device=device)
    min_grad_cosine = torch.ones((), device=device)
    for (reference_name, reference_parameter), (candidate_name, candidate_parameter) in zip(
        reference.named_parameters(),
        candidate.named_parameters(),
    ):
        assert reference_name == candidate_name
        reference_grad = reference_parameter.grad.float()
        candidate_grad = candidate_parameter.grad.float()
        grad_max_diff = torch.maximum(
            grad_max_diff,
            (candidate_grad - reference_grad).abs().max(),
        )
        reference_norm = torch.linalg.vector_norm(reference_grad)
        candidate_norm = torch.linalg.vector_norm(candidate_grad)
        if reference_norm > 0 and candidate_norm > 0:
            cosine = torch.dot(
                candidate_grad.reshape(-1),
                reference_grad.reshape(-1),
            ) / (candidate_norm * reference_norm)
            min_grad_cosine = torch.minimum(min_grad_cosine, cosine)
        torch.testing.assert_close(
            candidate_parameter.grad,
            reference_parameter.grad,
            rtol=5e-2 if dtype == torch.float16 else 5e-5,
            atol=5e-4 if dtype == torch.float16 else 5e-6,
            msg=lambda message: f"{candidate_name}: {message}",
        )
    dist.all_reduce(grad_max_diff, op=dist.ReduceOp.MAX)
    dist.all_reduce(min_grad_cosine, op=dist.ReduceOp.MIN)

    if rank == 0:
        print(
            f"Context parallel {args.dtype} correctness passed: local logits, "
            "global loss, and synchronized parameter gradients match the full "
            f"reference. max_logit_diff={logit_max_diff.item():.8g}, "
            f"loss_diff={loss_diff.item():.8g}, "
            f"max_grad_diff={grad_max_diff.item():.8g}, "
            f"min_grad_cosine={min_grad_cosine.item():.10f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
