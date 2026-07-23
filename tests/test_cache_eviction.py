import torch
from transformers import DynamicCache

from cotpt.inference import forward_step
from cotpt.model_utils import sample_token


def test_eviction_is_bit_exact(tiny_model):
    """Grow the cache with hidden tokens, crop back, and confirm the result
    is numerically identical to a cache that never saw the hidden tokens."""
    model = tiny_model
    torch.manual_seed(1)
    prompt_ids = torch.randint(0, model.config.vocab_size, (1, 7))

    num_hidden, num_visible = 5, 6

    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    last_logits = out.logits[:, -1, :]
    current_len = prompt_ids.shape[1]
    assert cache.get_seq_length() == current_len

    committed_real_tokens = []
    for _ in range(num_visible):
        checkpoint_len = current_len
        logits = last_logits
        for _ in range(num_hidden):
            h_id = sample_token(logits, temperature=1.0, do_sample=True)
            cache, logits = forward_step(model, h_id, cache)

        assert cache.get_seq_length() == checkpoint_len + num_hidden

        real_id = sample_token(logits, temperature=0.0, do_sample=False)  # greedy, deterministic

        cache.crop(checkpoint_len)
        assert cache.get_seq_length() == checkpoint_len

        cache, last_logits = forward_step(model, real_id, cache)
        current_len = checkpoint_len + 1
        committed_real_tokens.append(real_id.item())

    assert cache.get_seq_length() == prompt_ids.shape[1] + num_visible

    # Reference: a plain forward pass over ONLY prompt + committed real tokens.
    full_seq = torch.cat([prompt_ids, torch.tensor([committed_real_tokens])], dim=1)
    ref_cache = DynamicCache(config=model.config)
    with torch.no_grad():
        ref_out = model(input_ids=full_seq, past_key_values=ref_cache, use_cache=True)
    ref_cache = ref_out.past_key_values

    max_diff = 0.0
    for layer_idx in range(len(cache.layers)):
        max_diff = max(
            max_diff,
            (cache.layers[layer_idx].keys - ref_cache.layers[layer_idx].keys).abs().max().item(),
            (cache.layers[layer_idx].values - ref_cache.layers[layer_idx].values).abs().max().item(),
        )
    assert max_diff < 1e-5, f"eviction left residue in the KV cache! max diff = {max_diff}"


def test_generate_with_hidden_deliberation_runs(tiny_model, mock_tokenizer):
    """The public generation function runs end to end without error, with and
    without showing hidden thoughts, and num_hidden_tokens=0 degrades gracefully."""
    from cotpt.inference import generate_with_hidden_deliberation

    for show, num_hidden in [(True, 5), (False, 3), (True, 0)]:
        result = generate_with_hidden_deliberation(
            tiny_model, mock_tokenizer, "a test prompt",
            max_visible_tokens=4, num_hidden_tokens=num_hidden, show_hidden_thoughts=show,
        )
        assert isinstance(result, str)
