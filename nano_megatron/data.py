"""
GSM8k 数据加载模块。

把 GSM8k 的 question/answer 转成 SFT chat 格式，然后 tokenize。
answer 字段里的 <<...>> 计算注释会被去掉，保留自然语言推理链和 #### 最终答案。
"""

import re
import torch
from torch.utils.data import Dataset, DataLoader


def load_gsm8k(split: str = "train"):
    """从 HuggingFace 加载 GSM8k 数据集。"""
    from datasets import load_dataset
    return load_dataset("openai/gsm8k", "main", split=split)


def format_chat(question: str, answer: str) -> str:
    """把 question/answer 拼成 SFT 格式。

    格式：
      <|user|>\n{question}<|end|>\n<|assistant|>\n{answer}<|end|>

    同时去掉 answer 中的 <<...>> 计算标注（如 <<48/2=24>>）。
    """
    # 去掉 <<...>> 标注
    answer_clean = re.sub(r"<<.*?>>", "", answer)
    return f"<|user|>\n{question}<|end|>\n<|assistant|>\n{answer_clean}<|end|>"


class GSM8kDataset(Dataset):
    """GSM8k SFT 数据集。每个样本是一段 tokenize 后的 input_ids + labels。"""

    def __init__(self, tokenizer, split: str = "train", max_seq_len: int = 512):
        raw = load_gsm8k(split)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # 预处理：格式化 + tokenize
        self.samples = []
        for item in raw:
            text = format_chat(item["question"], item["answer"])
            tokens = tokenizer(text, truncation=True, max_length=max_seq_len,
                               return_tensors="pt", padding="max_length")
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)

            # Labels = input_ids，padding 位置设为 -100（不计算 loss）
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            self.samples.append({"input_ids": input_ids, "labels": labels})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def create_dataloader(tokenizer, config, split: str = "train") -> DataLoader:
    """创建 DataLoader。分布式训练时自动用 DistributedSampler。"""
    dataset = GSM8kDataset(tokenizer, split=split, max_seq_len=config.data.max_seq_len)

    sampler = None
    shuffle = (split == "train")
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            from torch.utils.data import DistributedSampler
            sampler = DistributedSampler(dataset, shuffle=shuffle)
            shuffle = False  # sampler 负责 shuffle
    except Exception:
        pass

    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=getattr(config.data, "num_workers", 0),
        pin_memory=True,
        drop_last=True,
    )
