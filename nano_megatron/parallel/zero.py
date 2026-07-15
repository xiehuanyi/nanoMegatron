"""ZeRO optimizers used by nanoMegatron.

Stage 2 follows the distributed-optimizer layout used by production training
stacks: parameters and gradients live in contiguous bucket buffers, every rank
owns an equally-sized slice of each bucket, gradients are reduce-scattered
during backward, and updated parameter slices are all-gathered after Adam.
"""

from collections import defaultdict

import torch
import torch.distributed as dist


class _FlatBucket:
    """One padded parameter bucket and its reduce-scatter buffers."""

    def __init__(self, params, rank, world_size, group):
        self.params = list(params)
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.entries = {}

        numel = sum(p.numel() for p in self.params)
        self.shard_numel = (numel + world_size - 1) // world_size
        self.padded_numel = self.shard_numel * world_size
        first = self.params[0]

        self.param_buffer = torch.zeros(
            self.padded_numel, dtype=first.dtype, device=first.device
        )
        self.grad_buffer = torch.zeros(
            self.padded_numel, dtype=first.dtype, device=first.device
        )

        offset = 0
        with torch.no_grad():
            for p in self.params:
                n = p.numel()
                view = self.param_buffer[offset:offset + n].view_as(p)
                view.copy_(p.data)
                p.data = view
                self.entries[id(p)] = (offset, n)
                offset += n

        start = rank * self.shard_numel
        self.local_param = self.param_buffer[start:start + self.shard_numel]
        self.local_grad = self.grad_buffer[start:start + self.shard_numel]
        self.ready = 0
        self.work = None

    def accumulate(self, p):
        # A second micro-batch may reach this bucket while its previous
        # reduce-scatter is still in flight. Finish it before reusing storage.
        if self.work is not None:
            raise RuntimeError("gradient buffer reused before ReduceScatter completed")
        offset, n = self.entries[id(p)]
        self.grad_buffer[offset:offset + n].add_(p.grad.detach().view(-1))
        p.grad = None
        self.ready += 1

    def launch(self):
        if self.work is not None:
            self.finish()
        self.work = dist.reduce_scatter_tensor(
            self.local_grad,
            self.grad_buffer,
            op=dist.ReduceOp.SUM,
            group=self.group,
            async_op=True,
        )
        self.ready = 0

    def finish(self):
        if self.work is None:
            return
        self.work.wait()
        self.local_grad.div_(self.world_size)
        self.work = None

    def reset(self):
        if self.work is not None:
            self.finish()
        self.grad_buffer.zero_()
        self.ready = 0


class _FlatGradReducer:
    """Flat model-dtype gradients with ordered asynchronous ReduceScatter."""

    def __init__(self, all_params, rank, world_size, group, bucket_size_mb):
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.buckets = []
        self.param_to_bucket = {}
        self.next_to_launch = 0
        self.synced = False
        self.accumulated_grads = None

        # Buckets follow backward order. A parameter larger than the target
        # remains intact so its post-accumulate hook is sufficient for readiness.
        limit = bucket_size_mb * 1024 * 1024 // 4
        current = []
        current_numel = 0
        for p in reversed(all_params):
            different_storage = current and (
                p.dtype != current[0].dtype or p.device != current[0].device
            )
            if current and (
                different_storage or current_numel + p.numel() > limit
            ):
                self._add_bucket(current)
                current = []
                current_numel = 0
            current.append(p)
            current_numel += p.numel()
            if current_numel >= limit:
                self._add_bucket(current)
                current = []
                current_numel = 0
        if current:
            self._add_bucket(current)

        for p in all_params:
            p.register_post_accumulate_grad_hook(self._make_hook(p))

    def _add_bucket(self, params):
        index = len(self.buckets)
        bucket = _FlatBucket(
            params, self.rank, self.world_size, self.group
        )
        self.buckets.append(bucket)
        for p in params:
            self.param_to_bucket[id(p)] = index

    def _make_hook(self, p):
        def hook(_):
            self.synced = False
            if self.next_to_launch == len(self.buckets):
                self._start_next_microbatch()
            bucket = self.buckets[self.param_to_bucket[id(p)]]
            bucket.accumulate(p)
            self._try_launch()
        return hook

    def _start_next_microbatch(self):
        # ReduceScatter writes its output into the rank-local slice of the
        # input buffer, as NCCL supports in-place collectives. Allocate an
        # accumulation shard only when more than one micro-batch is observed.
        for bucket in self.buckets:
            bucket.finish()
        if self.accumulated_grads is None:
            self.accumulated_grads = [
                torch.zeros_like(bucket.local_grad) for bucket in self.buckets
            ]
        for accumulated, bucket in zip(self.accumulated_grads, self.buckets):
            accumulated.add_(bucket.local_grad)
            bucket.grad_buffer.zero_()
        self.next_to_launch = 0

    def _try_launch(self):
        while self.next_to_launch < len(self.buckets):
            bucket = self.buckets[self.next_to_launch]
            if bucket.ready != len(bucket.params):
                break
            bucket.launch()
            self.next_to_launch += 1

    def finish_all(self):
        """Launch missing/unused gradients in the same order on every rank."""
        if self.synced:
            return
        while self.next_to_launch < len(self.buckets):
            self.buckets[self.next_to_launch].launch()
            self.next_to_launch += 1
        for bucket in self.buckets:
            bucket.finish()
        if self.accumulated_grads is not None:
            for accumulated, bucket in zip(self.accumulated_grads, self.buckets):
                bucket.local_grad.add_(accumulated)
        self.next_to_launch = len(self.buckets)
        self.synced = True

    # Kept as a small compatibility alias for existing debug scripts.
    flush_all = finish_all

    def reset(self):
        for bucket in self.buckets:
            bucket.reset()
        self.next_to_launch = 0
        self.synced = False
        self.accumulated_grads = None


class ZeROOptimizer:
    """ZeRO Stage 1 and flat-buffer Stage 2 optimizer."""

    def __init__(
        self,
        model,
        lr: float,
        weight_decay: float,
        stage: int,
        bucket_size_mb: int = 160,
        process_group=None,
        overlap_param_gather: bool = True,
    ):
        if stage not in (1, 2):
            raise ValueError(f"ZeRO stage must be 1 or 2, got {stage}")
        self.stage = stage
        self.group = process_group
        self.rank = dist.get_rank(group=process_group)
        self.world_size = dist.get_world_size(group=process_group)

        self.all_params = [p for p in model.parameters() if p.requires_grad]
        if not self.all_params:
            raise ValueError("ZeROOptimizer requires at least one trainable parameter")
        self.param_to_owner = {
            id(p): (i % self.world_size) for i, p in enumerate(self.all_params)
        }
        self._zero1_synced = False
        self._stage2_grads_copied = False
        self._grad_bucket = None
        self._param_gather_work = {}
        self._forward_hook_handles = []

        if stage == 2:
            self._init_stage2(lr, weight_decay, bucket_size_mb)
            if overlap_param_gather:
                self._register_param_gather_hooks(model)
        else:
            self._init_stage1(lr, weight_decay)

        self.param_groups = self.optimizer.param_groups

    def _init_stage1(self, lr, weight_decay):
        self.local_params = [
            p for i, p in enumerate(self.all_params)
            if i % self.world_size == self.rank
        ]
        self.fp32_copies = [
            p.data.float().clone().requires_grad_(True) for p in self.local_params
        ]
        self.optimizer = torch.optim.AdamW(
            self.fp32_copies,
            lr=lr,
            weight_decay=weight_decay,
            fused=self.fp32_copies[0].is_cuda,
        )

    def _register_param_gather_hooks(self, model):
        for module in model.modules():
            bucket_indices = []
            for p in module.parameters(recurse=False):
                index = self._grad_bucket.param_to_bucket.get(id(p))
                if index is not None and index not in bucket_indices:
                    bucket_indices.append(index)
            if not bucket_indices:
                continue

            def wait_for_params(_, __, indices=tuple(bucket_indices)):
                self._wait_param_buckets(indices)

            self._forward_hook_handles.append(
                module.register_forward_pre_hook(wait_for_params)
            )

    def _wait_param_buckets(self, bucket_indices):
        for index in bucket_indices:
            work = self._param_gather_work.pop(index, None)
            if work is not None:
                work.wait()

    def finish_param_sync(self):
        """Wait for parameter gathers not consumed by forward pre-hooks."""
        for index in self._param_gather_order:
            work = self._param_gather_work.pop(index, None)
            if work is not None:
                work.wait()

    def _init_stage2(self, lr, weight_decay, bucket_size_mb):
        self._grad_bucket = _FlatGradReducer(
            self.all_params,
            self.rank,
            self.world_size,
            self.group,
            bucket_size_mb,
        )
        self.fp32_copies = [
            torch.nn.Parameter(bucket.local_param.float(), requires_grad=True)
            for bucket in self._grad_bucket.buckets
        ]
        self.fp32_grads = [torch.empty_like(p) for p in self.fp32_copies]
        self.local_params = self.fp32_copies
        self.local_shard_numel = sum(p.numel() for p in self.fp32_copies)
        self.optimizer = torch.optim.AdamW(
            self.fp32_copies,
            lr=lr,
            weight_decay=weight_decay,
            fused=self.fp32_copies[0].is_cuda,
        )

        # Launch order follows the first forward use rather than backward
        # bucket order. Later gathers can then overlap earlier layer compute.
        seen = set()
        self._param_gather_order = []
        for p in self.all_params:
            index = self._grad_bucket.param_to_bucket[id(p)]
            if index not in seen:
                seen.add(index)
                self._param_gather_order.append(index)

    def _finish_stage2_grad_sync(self):
        if self._stage2_grads_copied:
            return
        self._grad_bucket.finish_all()
        for master, grad, bucket in zip(
            self.fp32_copies, self.fp32_grads, self._grad_bucket.buckets
        ):
            grad.copy_(bucket.local_grad)
            master.grad = grad
        self._stage2_grads_copied = True

    def _ensure_grads(self):
        """确保所有参数都有梯度张量。仅 ZeRO-1 在 step 前需要。"""
        for p in self.all_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)

    def _copy_fp16_grads_to_fp32(self):
        """拷贝 fp16 梯度到 fp32（立即释放 fp16 梯度省显存）。"""
        for fp32_p, fp16_p in zip(self.fp32_copies, self.local_params):
            if fp16_p.grad is not None:
                fp32_p.grad = fp16_p.grad.float()
                fp16_p.grad = None

    def _sync_fp32_to_fp16(self):
        """把 fp32 更新后的参数同步回 fp16 模型参数。"""
        for fp32_p, fp16_p in zip(self.fp32_copies, self.local_params):
            fp16_p.data.copy_(fp32_p.data)

    def _reduce_zero1_grads(self):
        """ZeRO-1：AllReduce 所有梯度（AVG），非 owner 释放。幂等。"""
        if self._zero1_synced:
            return
        self._ensure_grads()
        for p in self.all_params:
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=self.group)
        for i, p in enumerate(self.all_params):
            if i % self.world_size != self.rank:
                p.grad = None
        self._zero1_synced = True

    def clip_grad_norm(self, max_norm: float):
        """分布式 grad clipping（ZeRO-1/2 通用）。

        裁剪必须基于 reduce 后的全局梯度：先确保梯度已 reduce 到 owner，
        每个 rank 算自己 owner 参数的 norm²，AllReduce 后得到全局 norm。
        然后用全局 norm 计算 clip coefficient，应用到各 rank 自己的 owner 参数上。
        """
        if self.stage == 1:
            self._reduce_zero1_grads()
            grads = [p.grad for p in self.local_params if p.grad is not None]
        else:
            self._finish_stage2_grad_sync()
            grads = [p.grad for p in self.fp32_copies]

        device = grads[0].device
        local_norm_sq = torch.zeros(1, device=device, dtype=torch.float32)
        for grad in grads:
            local_norm_sq += grad.float().pow(2).sum()

        dist.all_reduce(local_norm_sq, op=dist.ReduceOp.SUM, group=self.group)
        global_norm = local_norm_sq.sqrt().item()

        if global_norm > max_norm and global_norm > 0:
            clip_coef = max_norm / global_norm
            for grad in grads:
                grad.mul_(clip_coef)
        return global_norm

    def step(self):
        if self.stage == 1:
            self._reduce_zero1_grads()
        else:
            self._finish_stage2_grad_sync()
        self._zero1_synced = False

        if self.stage == 1:
            self._copy_fp16_grads_to_fp32()
        self.optimizer.step()

        if self.stage == 1:
            self._sync_fp32_to_fp16()
            per_owner = defaultdict(list)
            for i, p in enumerate(self.all_params):
                per_owner[i % self.world_size].append(p)
            for owner in range(self.world_size):
                params = per_owner[owner]
                flat = torch.cat([p.data.view(-1) for p in params])
                dist.broadcast(flat, src=owner, group=self.group)
                offset = 0
                for p in params:
                    n = p.numel()
                    p.data.copy_(flat[offset:offset + n].view(p.shape))
                    offset += n
        else:
            self._all_gather_stage2_params()

    def _all_gather_stage2_params(self):
        self.finish_param_sync()
        with torch.no_grad():
            for master, bucket in zip(self.fp32_copies, self._grad_bucket.buckets):
                bucket.local_param.copy_(master)
            for index in self._param_gather_order:
                bucket = self._grad_bucket.buckets[index]
                self._param_gather_work[index] = dist.all_gather_into_tensor(
                    bucket.param_buffer,
                    bucket.local_param,
                    group=self.group,
                    async_op=True,
                )

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        for p in self.all_params:
            p.grad = None
        if self.stage == 2:
            self._grad_bucket.reset()
            self._stage2_grads_copied = False


def setup_zero(model, config, stage: int):
    """用 ZeRO 包装模型。"""
    local_rank = dist.get_rank()
    model = model.to(local_rank)

    optimizer = ZeROOptimizer(
        model,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        stage=stage,
    )
    return model, optimizer


# ============================================================
# FP16 Optimizer Wrapper (其他策略用)
# ============================================================

class FP16OptimizerWrapper:
    """fp16 模型的 fp32 优化器包装。

    问题：fp16 参数直接用 Adam 会导致 grad² 溢出 → NaN。
    方案：维护一份 fp32 参数副本做 Adam 更新，每步同步回 fp16。
    """

    def __init__(self, params, lr: float, weight_decay: float):
        self.fp16_params = list(params)
        self.fp32_params = [p.data.float().clone().requires_grad_(True) for p in self.fp16_params]
        self.optimizer = torch.optim.AdamW(self.fp32_params, lr=lr, weight_decay=weight_decay)
        self.param_groups = self.optimizer.param_groups

    def step(self):
        for fp32_p, fp16_p in zip(self.fp32_params, self.fp16_params):
            if fp16_p.grad is not None:
                fp32_p.grad = fp16_p.grad.float()
                fp16_p.grad = None
        self.optimizer.step()
        for fp32_p, fp16_p in zip(self.fp32_params, self.fp16_params):
            fp16_p.data.copy_(fp32_p.data)

    def zero_grad(self):
        self.optimizer.zero_grad()
        for p in self.fp16_params:
            p.grad = None
