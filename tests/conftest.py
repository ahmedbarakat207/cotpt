import torch
import pytest
from transformers import Qwen3Config, Qwen3ForCausalLM


@pytest.fixture
def tiny_config():
    return Qwen3Config(
        vocab_size=200,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
    )


@pytest.fixture
def tiny_model(tiny_config):
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_config)
    model.eval()
    return model


class MockTokenizer:
    """Minimal stand-in exposing exactly the interface cotpt.inference uses,
    so generation can be tested without downloading a real tokenizer."""

    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.eos_token_id = None

    def __call__(self, text, return_tensors="pt"):
        words = text.split()
        ids = [abs(hash(w)) % self.vocab_size for w in words] or [0]
        return type("Enc", (), {"input_ids": torch.tensor([ids])})()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"<{i}>" for i in ids)


@pytest.fixture
def mock_tokenizer(tiny_config):
    return MockTokenizer(vocab_size=tiny_config.vocab_size)
