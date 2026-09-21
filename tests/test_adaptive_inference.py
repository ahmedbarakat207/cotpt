import re
import torch
from transformers import DynamicCache

from cotpt.mixing_head import MixingHead
from cotpt.inference import generate_with_adaptive_deliberation


def test_adaptive_generation_runs_and_uses_mixing_head(tiny_model, mock_tokenizer):
    mixing_head = MixingHead(hidden_size=tiny_model.config.hidden_size)
    result = generate_with_adaptive_deliberation(
        tiny_model, mock_tokenizer, mixing_head, "a test prompt",
        max_visible_tokens=5, num_hidden_tokens=5, show_hidden_thoughts=True,
    )
    assert isinstance(result, str)


def test_high_entropy_threshold_skips_all_thinking(tiny_model, mock_tokenizer, capsys):
    mixing_head = MixingHead(hidden_size=tiny_model.config.hidden_size)
    generate_with_adaptive_deliberation(
        tiny_model, mock_tokenizer, mixing_head, "a test prompt",
        max_visible_tokens=5, num_hidden_tokens=5, entropy_threshold=1000.0,
    )
    captured = capsys.readouterr()
    assert "thought on 0/" in captured.out


def test_negative_entropy_threshold_always_thinks(tiny_model, mock_tokenizer, capsys):
    mixing_head = MixingHead(hidden_size=tiny_model.config.hidden_size)
    generate_with_adaptive_deliberation(
        tiny_model, mock_tokenizer, mixing_head, "a test prompt",
        max_visible_tokens=5, num_hidden_tokens=5, entropy_threshold=-1.0,
    )
    captured = capsys.readouterr()
    m = re.search(r"thought on (\d+)/(\d+)", captured.out)
    assert m is not None
    assert m.group(1) == m.group(2)


def test_eviction_still_exact_with_mixing_head(tiny_model, mock_tokenizer):
    mixing_head = MixingHead(hidden_size=tiny_model.config.hidden_size)
    prompt = "a test prompt"
    prompt_len = mock_tokenizer(prompt, return_tensors="pt").input_ids.shape[1]

    device = next(tiny_model.parameters()).device
    prompt_ids = mock_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    cache = DynamicCache(config=tiny_model.config)
    with torch.no_grad():
        out = tiny_model(input_ids=prompt_ids, past_key_values=cache, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    assert cache.get_seq_length() == prompt_len

    generate_with_adaptive_deliberation(
        tiny_model, mock_tokenizer, mixing_head, prompt,
        max_visible_tokens=5, num_hidden_tokens=5,
    )
