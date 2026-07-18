"""
Per-Token Hidden Deliberation with Aggressive KV-Cache Eviction
==================================================================

For every visible output token, the model:
  A. checkpoints its (clean) KV cache,
  B. secretly generates NUM_HIDDEN_THOUGHT_TOKENS "scratch" tokens, letting
     each one attend to the ones before it (normal autoregressive KV growth),
  C. uses the logits produced after the last scratch token to pick ONE real
     visible token,
  D. evicts the scratch tokens from the cache -- rolling it back to exactly
     the state it was in at (A), byte-for-byte,
  E. commits only the single real token to the now-clean cache and prints it,
  F. repeats.

See README.md for the validation this went through and how position/mask
handling works without any manual bookkeeping (transformers derives
position_ids from cache.get_seq_length(), which reads tensor shape directly,
so cropping the cache is sufficient on its own).
"""

import torch
from transformers import DynamicCache

from .config import (
    MAX_VISIBLE_TOKENS,
    NUM_HIDDEN_THOUGHT_TOKENS,
    THINKING_TEMPERATURE,
    THINKING_DO_SAMPLE,
    REAL_TOKEN_TEMPERATURE,
    REAL_TOKEN_DO_SAMPLE,
    SHOW_HIDDEN_THOUGHTS,
)
from .model_utils import sample_token, is_eos


def forward_step(model, token_id: torch.Tensor, cache: DynamicCache):
    """
    Feed exactly ONE new token through the model, growing `cache` by one
    position. Returns (updated_cache, logits_for_the_position_right_after_it).
    """
    with torch.no_grad():
        out = model(input_ids=token_id.view(1, 1), past_key_values=cache, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :]


def crop_cache_manually(cache: DynamicCache, target_length: int) -> None:
    """
    Not used below (cache.crop() already does this correctly and is the
    current transformers API) -- included so you can see the raw tensor
    slicing this project keeps asking for. This is what
    cache.crop(target_length) does internally, per layer:
    """
    for layer in cache.layers:
        layer.keys = layer.keys[..., :target_length, :]      # dim=-2 = seq_len
        layer.values = layer.values[..., :target_length, :]   # dim=-2 = seq_len


def generate_with_hidden_deliberation(
    model,
    tokenizer,
    prompt: str,
    max_visible_tokens: int = MAX_VISIBLE_TOKENS,
    num_hidden_tokens: int = NUM_HIDDEN_THOUGHT_TOKENS,
    thinking_temperature: float = THINKING_TEMPERATURE,
    thinking_do_sample: bool = THINKING_DO_SAMPLE,
    real_temperature: float = REAL_TOKEN_TEMPERATURE,
    real_do_sample: bool = REAL_TOKEN_DO_SAMPLE,
    show_hidden_thoughts: bool = SHOW_HIDDEN_THOUGHTS,
) -> str:
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    last_logits = out.logits[:, -1, :]

    print(prompt, end="", flush=True)
    generated_text = ""

    for _ in range(max_visible_tokens):
        # ------------------------- STEP A: Checkpoint -------------------------
        checkpoint_len = cache.get_seq_length()

        # -------------------- STEP B: Hidden thought generation ---------------
        hidden_ids = []
        logits = last_logits
        for _ in range(num_hidden_tokens):
            hidden_token = sample_token(logits, thinking_temperature, thinking_do_sample)
            cache, logits = forward_step(model, hidden_token, cache)
            hidden_ids.append(hidden_token.item())

        # ----------------------- STEP C: Pick the real token -------------------
        real_token = sample_token(logits, real_temperature, real_do_sample)
        real_token_id = real_token.item()

        if is_eos(real_token_id, tokenizer, model):
            break

        if show_hidden_thoughts:
            thought_text = tokenizer.decode(hidden_ids, skip_special_tokens=True)
            print(f"\033[2m\u27ea{thought_text}\u27eb\033[0m", end="", flush=True)

        # ------------------------ STEP D: Evict the thoughts --------------------
        cache.crop(checkpoint_len)
        assert cache.get_seq_length() == checkpoint_len, "eviction did not land exactly on the checkpoint"

        # -------------------- STEP E: Commit the real token & print -------------
        cache, last_logits = forward_step(model, real_token, cache)
        token_text = tokenizer.decode([real_token_id], skip_special_tokens=True)
        print(token_text, end="", flush=True)
        generated_text += token_text

        # ------------------------------ STEP F: loop -----------------------------

    print()
    return generated_text
