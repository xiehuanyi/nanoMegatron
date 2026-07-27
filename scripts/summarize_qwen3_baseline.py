#!/usr/bin/env python3
"""Turn nano/Megatron logs and nvidia-smi samples into one comparison JSON."""

import argparse
import json
import re
import statistics
from pathlib import Path


def peak_mib(path: Path):
    values = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                values.append(float(line.rsplit(",", 1)[-1]))
            except ValueError:
                pass
    return max(values) if values else None


def megatron_step_times(path: Path):
    pattern = re.compile(r"elapsed time per iteration \(ms\):\s*([0-9.]+)")
    return [float(match.group(1)) for match in pattern.finditer(path.read_text(encoding="utf-8"))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--global-tokens-per-step", type=int, default=1024)
    parser.add_argument("--protocol", default="B0-portable-single-gpu")
    args = parser.parse_args()
    nano = json.loads((args.output_dir / "nano.json").read_text(encoding="utf-8"))
    mcore_times = megatron_step_times(args.output_dir / "megatron.log")
    if len(mcore_times) < 4:
        raise RuntimeError("Megatron log does not contain enough measured iterations")
    # Megatron creates optimizer state during iteration 1; discard the first 3 iterations.
    mcore_times = mcore_times[3:13]
    mcore_mean_ms = statistics.mean(mcore_times)
    nano_smi = peak_mib(args.output_dir / "nano_memory.csv")
    mcore_smi = peak_mib(args.output_dir / "megatron_memory.csv")
    result = {
        "schema_version": 1,
        "protocol": args.protocol,
        "nano": {
            "tokens_per_second": nano["tokens_per_second"],
            "mean_step_ms": nano["mean_step_ms"],
            "peak_smi_mib": nano_smi,
            "peak_allocated_gib": nano["peak_memory_allocated_gib"],
        },
        "megatron": {
            "tokens_per_second": args.global_tokens_per_step / (mcore_mean_ms / 1000),
            "mean_step_ms": mcore_mean_ms,
            "peak_smi_mib": mcore_smi,
        },
        "parity": {
            "throughput_ratio_nano_over_megatron": mcore_mean_ms / nano["mean_step_ms"],
            "memory_ratio_nano_over_megatron": (
                nano_smi / mcore_smi if nano_smi is not None and mcore_smi else None
            ),
        },
        "megatron_measured_step_ms": mcore_times,
    }
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("SUMMARY " + json.dumps(result))


if __name__ == "__main__":
    main()
