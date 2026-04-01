"""工具函数。"""

import yaml
from types import SimpleNamespace


def load_config(path: str) -> SimpleNamespace:
    """加载 YAML 配置文件，返回嵌套的 SimpleNamespace 方便用 . 访问。"""
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _dict_to_namespace(raw)


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = _dict_to_namespace(v)
    return SimpleNamespace(**d)


def is_main_process() -> bool:
    """判断是否为主进程（rank 0 或单卡）。"""
    import torch.distributed as dist
    return not dist.is_initialized() or dist.get_rank() == 0
