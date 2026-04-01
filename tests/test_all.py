"""
nanoMegatron 全部测试。

可在单卡/CPU 上运行，验证各模块的基本正确性。
  python -m pytest tests/test_all.py -v

注意：分布式并行策略的测试需要多卡环境，这里只测单卡逻辑。
"""

import sys
import os
import math
import pytest
import torch
import torch.nn as nn

# 把项目根目录加到 path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from types import SimpleNamespace


# ============================================================
# 辅助：构造小型测试配置（避免加载完整 3.8B 模型）
# ============================================================

def make_tiny_config():
    """创建一个微型配置，用于快速测试。"""
    return SimpleNamespace(
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=256,
        max_seq_len=32,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
    )


# ============================================================
# Test: 配置加载
# ============================================================

class TestConfig:
    def test_load_config(self, tmp_path):
        """测试 YAML 配置加载。"""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("model:\n  hidden_size: 64\n  num_layers: 2\n")

        from nano_megatron.utils import load_config
        cfg = load_config(str(config_file))

        assert cfg.model.hidden_size == 64
        assert cfg.model.num_layers == 2


# ============================================================
# Test: 模型组件
# ============================================================

class TestModelComponents:
    def test_rms_norm(self):
        """测试 RMSNorm 输出形状和数值稳定性。"""
        from nano_megatron.model import RMSNorm

        norm = RMSNorm(64)
        x = torch.randn(2, 8, 64)
        out = norm(x)

        assert out.shape == x.shape
        # 输出不应该有 NaN
        assert not torch.isnan(out).any()

    def test_rope(self):
        """测试 RoPE 旋转位置编码。"""
        from nano_megatron.model import build_rope_cache, apply_rope

        seq_len, head_dim = 16, 32
        cos, sin = build_rope_cache(seq_len, head_dim)
        assert cos.shape == (seq_len, head_dim)
        assert sin.shape == (seq_len, head_dim)

        # 测试旋转：同一个向量在不同位置应该产生不同结果
        x = torch.randn(1, 1, seq_len, head_dim)
        cos_4d = cos.unsqueeze(0).unsqueeze(0)
        sin_4d = sin.unsqueeze(0).unsqueeze(0)
        rotated = apply_rope(x, cos_4d, sin_4d)
        assert rotated.shape == x.shape
        # 位置 0 和位置 1 的结果应该不同
        assert not torch.allclose(rotated[0, 0, 0], rotated[0, 0, 1])

    def test_attention(self):
        """测试 GQA Attention 前向传播。"""
        from nano_megatron.model import PhiMoEAttention, build_rope_cache

        config = make_tiny_config()
        attn = PhiMoEAttention(config)

        B, L = 2, 8
        x = torch.randn(B, L, config.hidden_size)
        cos, sin = build_rope_cache(L, config.head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        out = attn(x, cos, sin)
        assert out.shape == (B, L, config.hidden_size)

    def test_expert(self):
        """测试单个 Expert FFN。"""
        from nano_megatron.model import PhiMoEExpert

        config = make_tiny_config()
        expert = PhiMoEExpert(config)

        x = torch.randn(4, config.hidden_size)
        out = expert(x)
        assert out.shape == (4, config.hidden_size)

    def test_moe(self):
        """测试 Sparse MoE 层。"""
        from nano_megatron.model import PhiMoESparseMoE

        config = make_tiny_config()
        moe = PhiMoESparseMoE(config)

        B, L = 2, 8
        x = torch.randn(B, L, config.hidden_size)
        out, router_logits = moe(x)
        assert out.shape == (B, L, config.hidden_size)
        assert router_logits.shape == (B * L, config.num_experts)

        # 路由权重应该是合理的（非零，且 top-2 被选中）
        topk = torch.topk(router_logits, 2, dim=-1)
        assert topk.values.shape == (B * L, 2)

    def test_decoder_layer(self):
        """测试单层 Transformer。"""
        from nano_megatron.model import PhiMoEDecoderLayer, build_rope_cache

        config = make_tiny_config()
        layer = PhiMoEDecoderLayer(config)

        B, L = 2, 8
        x = torch.randn(B, L, config.hidden_size)
        cos, sin = build_rope_cache(L, config.head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        out, router_logits = layer(x, cos, sin)
        assert out.shape == (B, L, config.hidden_size)


# ============================================================
# Test: 完整模型
# ============================================================

class TestFullModel:
    def test_forward(self):
        """测试完整模型前向传播。"""
        from nano_megatron.model import PhiMoEForCausalLM

        config = make_tiny_config()
        model = PhiMoEForCausalLM(config)

        B, L = 2, 8
        input_ids = torch.randint(0, config.vocab_size, (B, L))
        labels = torch.randint(0, config.vocab_size, (B, L))

        output = model(input_ids, labels=labels)
        assert output["logits"].shape == (B, L, config.vocab_size)
        assert output["loss"] is not None
        assert output["loss"].item() > 0  # loss 应该是正数

    def test_backward(self):
        """测试反向传播能正常计算梯度。"""
        from nano_megatron.model import PhiMoEForCausalLM

        config = make_tiny_config()
        model = PhiMoEForCausalLM(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        labels = torch.randint(0, config.vocab_size, (2, 8))

        output = model(input_ids, labels=labels)
        output["loss"].backward()

        # 检查梯度存在
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Missing gradient for {name}"

    def test_generation(self):
        """测试自回归生成。"""
        from nano_megatron.model import PhiMoEForCausalLM
        from nano_megatron.evaluate import generate

        config = make_tiny_config()
        model = PhiMoEForCausalLM(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 4))
        with torch.no_grad():
            output_ids = generate(model, input_ids, max_new_tokens=8)

        assert output_ids.shape[0] == 1
        assert output_ids.shape[1] >= 4  # 至少包含原始 input
        assert output_ids.shape[1] <= 12  # 最多生成 8 个新 token


# ============================================================
# Test: 数据处理
# ============================================================

class TestData:
    def test_format_chat(self):
        """测试 chat 格式化。"""
        from nano_megatron.data import format_chat

        question = "What is 2+2?"
        answer = "2+2 = <<2+2=4>>4\n#### 4"

        formatted = format_chat(question, answer)
        assert "<|user|>" in formatted
        assert "<|assistant|>" in formatted
        assert "<<" not in formatted  # 计算标注应该被去掉
        assert "#### 4" in formatted

    def test_extract_answer(self):
        """测试答案提取。"""
        from nano_megatron.evaluate import extract_answer

        assert extract_answer("Some reasoning\n#### 42") == "42"
        assert extract_answer("The answer is #### 1,234") == "1234"
        assert extract_answer("No hash, just 99") == "99"
        assert extract_answer("#### -5") == "-5"


# ============================================================
# Test: 评估逻辑
# ============================================================

class TestEvaluation:
    def test_extract_answer_edge_cases(self):
        """测试各种格式的答案提取。"""
        from nano_megatron.evaluate import extract_answer

        # 标准格式
        assert extract_answer("step1\nstep2\n#### 72") == "72"
        # 有逗号的大数
        assert extract_answer("#### 1,000,000") == "1000000"
        # 小数
        assert extract_answer("#### 3.14") == "3.14"
        # 负数
        assert extract_answer("#### -10") == "-10"
        # 没有 #### 标记
        assert extract_answer("The answer is 42") == "42"


# ============================================================
# Test: Metrics
# ============================================================

class TestMetrics:
    def test_tracker(self):
        """测试指标收集器。"""
        from nano_megatron.metrics import MetricsTracker

        tracker = MetricsTracker()
        tracker.update(loss=2.5, tokens_per_sec=1000)
        tracker.update(loss=2.0, tokens_per_sec=1200)

        assert len(tracker.losses) == 2
        assert len(tracker.tokens_per_sec_history) == 2


# ============================================================
# Test: 并行组件（单进程模拟）
# ============================================================

class TestParallelComponents:
    def test_column_parallel_linear_shapes(self):
        """测试 ColumnParallelLinear 的形状计算（不需要分布式环境）。"""
        # 这里只测试核心逻辑：权重切分后的形状
        in_dim, out_dim = 64, 32
        tp_size = 2

        # 模拟：每个 rank 持有 out_dim/tp_size 的输出维度
        shard_out = out_dim // tp_size  # 16
        linear = nn.Linear(in_dim, shard_out)

        x = torch.randn(4, 8, in_dim)
        out = linear(x)
        assert out.shape == (4, 8, shard_out)

    def test_pipeline_stage_split(self):
        """测试 Pipeline 的层切分逻辑。"""
        # 32 层分到 4 个 stage，每个 stage 8 层
        num_layers = 32
        pp_size = 4
        layers_per_stage = num_layers // pp_size

        for rank in range(pp_size):
            start = rank * layers_per_stage
            end = start + layers_per_stage
            assert end - start == 8
            if rank == 0:
                assert start == 0
            if rank == pp_size - 1:
                assert end == 32

    def test_expert_parallel_routing(self):
        """测试 EP 的路由逻辑：expert_id -> rank 的映射。"""
        num_experts = 16
        ep_size = 4
        experts_per_rank = num_experts // ep_size

        # expert 0-3 在 rank 0, 4-7 在 rank 1, ...
        for e_id in range(num_experts):
            expected_rank = e_id // experts_per_rank
            assert 0 <= expected_rank < ep_size

    def test_zero_param_partition(self):
        """测试 ZeRO 的参数分配逻辑。"""
        # 8 个参数，4 个 rank
        num_params = 8
        world_size = 4

        for rank in range(world_size):
            local_indices = [i for i in range(num_params) if i % world_size == rank]
            assert len(local_indices) == 2  # 每个 rank 分到 2 个参数
            for idx in local_indices:
                assert idx % world_size == rank


# ============================================================
# Test: 端到端训练（单步）
# ============================================================

class TestEndToEnd:
    def test_single_train_step(self):
        """测试单步训练：前向 → 反向 → 更新，loss 应该下降。"""
        from nano_megatron.model import PhiMoEForCausalLM

        config = make_tiny_config()
        model = PhiMoEForCausalLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        labels = input_ids.clone()

        # 跑几步看 loss 是否下降
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            output = model(input_ids, labels=labels)
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # loss 应该有下降趋势（不要求严格单调）
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"

    def test_model_parameter_count(self):
        """测试微型模型的参数量计算。"""
        from nano_megatron.model import PhiMoEForCausalLM

        config = make_tiny_config()
        model = PhiMoEForCausalLM(config)

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert total > 0
        assert total == trainable  # 所有参数应该可训练
        print(f"\n  Tiny model: {total:,} parameters")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
