#!/usr/bin/env python3
"""Megatron pretrain entry point with a rank-local fixed benchmark batch."""

from functools import partial

import torch

from megatron.core.enums import ModelType
from megatron.training import get_args, inprocess_restart, pretrain

import pretrain_gpt
from gpt_builders import gpt_builder
from model_provider import model_provider


_CACHED_BATCHES = {}
_ORIGINAL_GET_BATCH = pretrain_gpt.get_batch


def _get_fixed_batch(data_iterator, vp_stage=None):
    key = -1 if vp_stage is None else vp_stage
    if key not in _CACHED_BATCHES:
        original = tuple(_ORIGINAL_GET_BATCH(data_iterator, vp_stage))
        if original[0] is None:
            _CACHED_BATCHES[key] = original
        else:
            args = get_args()
            device = original[0].device
            generator = torch.Generator(device=device).manual_seed(
                args.seed + torch.distributed.get_rank()
            )
            token_stream = torch.randint(
                0,
                args.padded_vocab_size,
                (args.micro_batch_size, args.seq_length + 1),
                device=device,
                generator=generator,
            )
            tokens = token_stream[:, :-1].contiguous()
            labels = token_stream[:, 1:].contiguous()
            loss_mask = torch.ones_like(labels, dtype=torch.float32)
            position_ids = torch.arange(args.seq_length, device=device).expand_as(tokens)
            if torch.distributed.get_rank() == 0:
                print(
                    "FIXED_BATCH "
                    f"sum={token_stream.sum().item()} "
                    f"prefix={token_stream[0, :8].tolist()}",
                    flush=True,
                )
            _CACHED_BATCHES[key] = (tokens, labels, loss_mask, None, position_ids)
    return _CACHED_BATCHES[key]


def main():
    pretrain_gpt.get_batch = _get_fixed_batch
    pretrain_gpt.train_valid_test_datasets_provider.is_distributed = True
    wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
    wrapped_pretrain(
        pretrain_gpt.train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        pretrain_gpt.forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
        extra_args_provider=(
            pretrain_gpt.add_modelopt_args if pretrain_gpt.has_nvidia_modelopt else None
        ),
        store=store,
    )


if __name__ == "__main__":
    main()
