"""
GSM8k 评估模块。

评估方式：
1. 给模型输入 question，让它自回归生成 answer
2. 从生成的文本中提取 #### 后面的数字
3. 和标准答案做 exact match
"""

import re
import torch
from nano_megatron.data import load_gsm8k


def extract_answer(text: str) -> str:
    """从模型生成的文本中提取最终答案（#### 后面的数字）。"""
    match = re.search(r"####\s*([+-]?\d[\d,]*\.?\d*)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    # 兜底：试着找最后一个数字
    numbers = re.findall(r"[+-]?\d[\d,]*\.?\d*", text)
    return numbers[-1].replace(",", "").strip() if numbers else ""


@torch.no_grad()
def evaluate_gsm8k(model, tokenizer, config, device, max_samples: int = 100) -> float:
    """在 GSM8k test set 上评估 exact match accuracy。

    Args:
        max_samples: 为了快速评估，默认只测前 100 题。设为 -1 测全部。
    """
    model.eval()
    dataset = load_gsm8k("test")

    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0

    for item in dataset:
        question = item["question"]
        gold = extract_answer(item["answer"])

        # 构造 prompt
        prompt = f"<|user|>\n{question}<|end|>\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # 自回归生成
        output_ids = generate(model, inputs["input_ids"], max_new_tokens=256,
                              eos_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        pred = extract_answer(response)
        if pred == gold:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def generate(model, input_ids: torch.Tensor, max_new_tokens: int = 256,
             eos_token_id: int = None, temperature: float = 0.0) -> torch.Tensor:
    """最简单的自回归生成（greedy / sampling）。"""
    for _ in range(max_new_tokens):
        output = model(input_ids)
        logits = output["logits"][:, -1, :]  # 取最后一个位置的 logits

        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits.argmax(dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return input_ids
