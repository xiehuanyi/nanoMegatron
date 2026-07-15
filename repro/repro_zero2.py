"""Check flat ZeRO-2 gradient accumulation (2 CPU ranks, gloo).

Reference: DDP semantics — grad = avg over ranks of (sum over micro-steps).
Test: 2 micro-step backwards before optimizer.step(), same data on both ranks
(so the correct averaged grad == 2x single-backward grad).

Run:  torchrun --nproc_per_node=2 repro/repro_zero2.py
"""
import sys
import copy
import os
import torch
import torch.nn as nn
import torch.distributed as dist

sys.path.insert(0, ".")
from nano_megatron.parallel.zero import FP16OptimizerWrapper, ZeROOptimizer


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)
        self.c = nn.Linear(8, 4)

    def forward(self, x):
        return self.c(torch.relu(self.b(torch.relu(self.a(x)))))


def run_backwards(model, x, n):
    for _ in range(n):
        model(x).pow(2).mean().backward()


def main():
    use_cuda = os.environ.get("USE_CUDA") == "1"
    dist.init_process_group("nccl" if use_cuda else "gloo")
    rank = dist.get_rank()
    if use_cuda:
        torch.cuda.set_device(rank)
    device = torch.device("cuda", rank) if use_cuda else torch.device("cpu")
    torch.manual_seed(0)

    model = Tiny().to(device)
    if use_cuda:
        model.half()
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    ref = copy.deepcopy(model)
    ref_opt = FP16OptimizerWrapper(ref.parameters(), lr=1e-3, weight_decay=0.0)

    x = torch.randn(4, 8, device=device, dtype=next(model.parameters()).dtype)
    dist.broadcast(x, src=0)

    # reference: plain accumulation over 2 micro steps (same data on each rank
    # -> DDP-averaged grad == local accumulated grad)
    run_backwards(ref, x, 2)

    opt = ZeROOptimizer(model, lr=1e-3, weight_decay=0.0, stage=2)
    run_backwards(model, x, 2)
    opt._finish_stage2_grad_sync()

    expected_shards = []
    ref_by_id = {
        id(p): rp.grad for p, rp in zip(opt.all_params, ref.parameters())
    }
    for bucket in opt._grad_bucket.buckets:
        expected = torch.zeros(
            bucket.padded_numel, device=device, dtype=bucket.grad_buffer.dtype
        )
        for p in bucket.params:
            offset, n = bucket.entries[id(p)]
            expected[offset:offset + n].copy_(ref_by_id[id(p)].view(-1))
        start = rank * bucket.shard_numel
        expected_shards.append(expected[start:start + bucket.shard_numel])
    expected = torch.cat(expected_shards)
    actual = torch.cat([
        bucket.local_grad for bucket in opt._grad_bucket.buckets
    ])
    diff = (actual - expected).abs().max().item()
    tolerance = 2e-3 if use_cuda else 1e-5
    bad = int(diff >= tolerance)
    print(f"[rank {rank}] flat-shard maxdiff={diff:.2e} "
          f"{'ok' if not bad else 'WRONG'}")

    ref_opt.step()
    opt.step()
    opt.finish_param_sync()
    update_diff = max(
        (p - rp).abs().max().item()
        for p, rp in zip(opt.all_params, ref.parameters())
    )
    update_bad = int(update_diff >= tolerance)
    bad += update_bad
    print(f"[rank {rank}] optimizer-step maxdiff={update_diff:.2e} "
          f"{'ok' if not update_bad else 'WRONG'}")

    total_bad = torch.tensor([bad], device=device)
    dist.all_reduce(total_bad)
    if rank == 0:
        print("ZeRO-2 grad-accum OK" if total_bad.item() == 0
              else f"ZeRO-2 grad-accum BROKEN ({total_bad.item()} params wrong)")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
