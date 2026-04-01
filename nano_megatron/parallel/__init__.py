"""并行策略模块。每个文件实现一种并行策略。"""

from nano_megatron.parallel.ddp import setup_ddp
from nano_megatron.parallel.zero import setup_zero
from nano_megatron.parallel.tensor_parallel import setup_tp
from nano_megatron.parallel.sequence_parallel import setup_sp
from nano_megatron.parallel.pipeline_parallel import setup_pp
from nano_megatron.parallel.expert_parallel import setup_ep

# 策略名 -> (setup 函数, 是否需要自定义 train_step)
STRATEGIES = {
    "ddp":   setup_ddp,
    "zero1": lambda m, c: setup_zero(m, c, stage=1),
    "zero2": lambda m, c: setup_zero(m, c, stage=2),
    "zero3": lambda m, c: setup_zero(m, c, stage=3),
    "tp":    setup_tp,
    "sp":    setup_sp,
    "pp":    setup_pp,
    "ep":    setup_ep,
}
