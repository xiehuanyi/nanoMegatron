#!/usr/bin/env python3
"""Aggregate independent Qwen3 baseline summaries with medians."""

import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = [json.loads(path.read_text(encoding="utf-8")) for path in args.summaries]
    protocols = {run["protocol"] for run in runs}
    if len(protocols) != 1:
        raise ValueError(f"mixed protocols: {sorted(protocols)}")

    def median(path):
        values = []
        for run in runs:
            value = run
            for key in path:
                value = value[key]
            values.append(value)
        return statistics.median(values)

    result = {
        "schema_version": 1,
        "protocol": protocols.pop(),
        "run_count": len(runs),
        "source_summaries": [str(path) for path in args.summaries],
        "median": {
            "nano_tokens_per_second": median(("nano", "tokens_per_second")),
            "megatron_tokens_per_second": median(("megatron", "tokens_per_second")),
            "throughput_ratio_nano_over_megatron": median(
                ("parity", "throughput_ratio_nano_over_megatron")
            ),
            "nano_peak_smi_mib": median(("nano", "peak_smi_mib")),
            "megatron_peak_smi_mib": median(("megatron", "peak_smi_mib")),
            "memory_ratio_nano_over_megatron": median(
                ("parity", "memory_ratio_nano_over_megatron")
            ),
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["median"], indent=2))


if __name__ == "__main__":
    main()
