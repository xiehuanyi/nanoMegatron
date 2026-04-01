"""
性能指标收集：throughput、peak memory、GPU 利用率。
"""

import torch


class MetricsTracker:
    """收集训练过程中的性能指标。"""

    def __init__(self):
        self.losses = []
        self.tokens_per_sec_history = []

    def update(self, loss: float, tokens_per_sec: float):
        self.losses.append(loss)
        self.tokens_per_sec_history.append(tokens_per_sec)

    def summary(self):
        """打印训练性能汇总。"""
        if not self.losses:
            return

        avg_loss = sum(self.losses[-100:]) / len(self.losses[-100:])
        avg_tps = sum(self.tokens_per_sec_history) / len(self.tokens_per_sec_history)
        peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        print("\n" + "=" * 50)
        print("Training Summary")
        print("=" * 50)
        print(f"  Final avg loss (last 100):  {avg_loss:.4f}")
        print(f"  Avg throughput:             {avg_tps:.0f} tokens/sec")
        print(f"  Peak GPU memory:            {peak_mem:.2f} GB")

        try:
            handle = torch.cuda.nvml.nvmlDeviceGetHandleByIndex(0)
            util = torch.cuda.nvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"  GPU utilization:            {util.gpu}%")
        except Exception:
            pass

        print("=" * 50)
