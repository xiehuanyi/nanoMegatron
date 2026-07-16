#!/usr/bin/env python3
"""Compare Transformer Engine FusedAdam with fused torch AdamW on CUDA."""

import argparse
import json
from pathlib import Path

import torch
from transformer_engine.pytorch.optimizers import FusedAdam


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    actual64 = actual.double()
    reference64 = reference.double()
    diff = (actual64 - reference64).abs()
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "cosine": torch.nn.functional.cosine_similarity(
            actual64.flatten(), reference64.flatten(), dim=0
        ).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--numel", type=int, default=1_000_003)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(1234)
    device = torch.device("cuda")
    initial = torch.randn(args.numel, device=device, dtype=torch.float32)
    torch_param = torch.nn.Parameter(initial.clone())
    te_param = torch.nn.Parameter(initial.clone())
    torch_optimizer = torch.optim.AdamW(
        [torch_param],
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        fused=True,
    )
    te_optimizer = FusedAdam(
        [te_param],
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        adam_w_mode=True,
    )

    first_step = None
    for step in range(args.steps):
        generator = torch.Generator(device=device).manual_seed(9000 + step)
        grad = torch.randn(args.numel, device=device, generator=generator)
        torch_param.grad = grad
        te_param.grad = grad.clone()
        torch_optimizer.step()
        te_optimizer.step()
        if step == 0:
            torch.cuda.synchronize()
            first_step = _metrics(te_param, torch_param)

    torch.cuda.synchronize()
    torch_state = torch_optimizer.state[torch_param]
    te_state = te_optimizer.state[te_param]
    result = {
        "schema_version": 1,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numel": args.numel,
        "steps": args.steps,
        "hyperparameters": {
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
        "after_first_step": first_step,
        "after_final_step": {
            "parameter": _metrics(te_param, torch_param),
            "exp_avg": _metrics(te_state["exp_avg"], torch_state["exp_avg"]),
            "exp_avg_sq": _metrics(te_state["exp_avg_sq"], torch_state["exp_avg_sq"]),
        },
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
