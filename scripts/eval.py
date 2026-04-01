"""
评估入口脚本。

用法：
  python scripts/eval.py --config configs/default.yaml --checkpoint checkpoints/checkpoint_1000.pt
"""

import argparse
import torch
from transformers import AutoTokenizer

from nano_megatron.utils import load_config
from nano_megatron.model import PhiMoEForCausalLM, load_hf_weights
from nano_megatron.evaluate import evaluate_gsm8k


def main():
    parser = argparse.ArgumentParser(description="nanoMegatron GSM8k 评估")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="SFT checkpoint 路径")
    parser.add_argument("--max_samples", type=int, default=-1, help="评估样本数，-1 为全部")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    print("[MODEL] Building model...")
    model = PhiMoEForCausalLM(config.model)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"[MODEL] Loaded checkpoint from {args.checkpoint}")
    else:
        load_hf_weights(model, config.model.name)

    model = model.to(device)

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 评估
    print("[EVAL] Evaluating on GSM8k test set...")
    accuracy = evaluate_gsm8k(model, tokenizer, config, device, max_samples=args.max_samples)
    print(f"[EVAL] GSM8k Accuracy: {accuracy:.2%} ({args.max_samples if args.max_samples > 0 else 'all'} samples)")


if __name__ == "__main__":
    main()
