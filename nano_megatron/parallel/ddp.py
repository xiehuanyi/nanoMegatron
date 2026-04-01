"""
DDP（Distributed Data Parallel）—— 最基础的数据并行。

原理：
1. 每个 GPU 持有完整的模型副本
2. 每个 GPU 处理不同的数据 batch
3. 反向传播后，AllReduce 同步梯度（取平均）
4. 所有 GPU 用相同的梯度更新参数，保持模型一致

这是最简单也最常用的并行方式。PyTorch 的 DistributedDataParallel 会自动
在 backward 时 hook 梯度同步，开发者无需手动处理。
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(model, config):
    """用 DDP 包装模型。

    Returns:
        (wrapped_model, None)  —— 第二项为 train_step_fn，DDP 用默认的即可
    """
    local_rank = dist.get_rank()
    model = model.to(local_rank)

    # DDP 包装：自动在 backward() 时插入 AllReduce 同步梯度
    model = DDP(model, device_ids=[local_rank])

    return model, None  # None 表示用 Trainer 的默认 train_step
