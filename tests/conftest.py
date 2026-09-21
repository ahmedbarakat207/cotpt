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
        self._special = {}

    def __len__(self):
        return self.vocab_size

    def add_special_tokens(self, special_dict):
        added = 0
        for tok in special_dict.get("additional_special_tokens", []):
            if tok not in self._special:
                self._special[tok] = self.vocab_size
                self.vocab_size += 1
                added += 1
        return added

    def encode(self, text, add_special_tokens=False):
        if text in self._special:
            return [self._special[text]]
        words = text.split()
        base = self.vocab_size - len(self._special)
        ids = [abs(hash(w)) % max(1, base) for w in words] or [0]
        return ids

    def convert_tokens_to_ids(self, token):
        return self._special.get(token)

    def __call__(self, text, return_tensors="pt"):
        return type("Enc", (), {"input_ids": torch.tensor([self.encode(text)])})()

    def decode(self, ids, skip_special_tokens=True):
        special_ids = set(self._special.values())
        if skip_special_tokens:
            ids = [i for i in ids if i not in special_ids]
        return " ".join(f"<{i}>" for i in ids)


@pytest.fixture
def mock_tokenizer(tiny_config):
    return MockTokenizer(vocab_size=tiny_config.vocab_size)
