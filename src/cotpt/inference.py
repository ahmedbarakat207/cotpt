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


def forward_step_with_hidden(model, token_id: torch.Tensor, cache: DynamicCache):
    """Same as forward_step, but also returns the post-final-norm hidden
    state at that position (needed by the mixing head)."""
    with torch.no_grad():
        out = model(input_ids=token_id.view(1, 1), past_key_values=cache, use_cache=True, output_hidden_states=True)
    return out.past_key_values, out.logits[:, -1, :], out.hidden_states[-1][:, -1, :]


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


def generate_with_adaptive_deliberation(
    model,
    tokenizer,
    mixing_head,
    prompt: str,
    max_visible_tokens: int = MAX_VISIBLE_TOKENS,
    num_hidden_tokens: int = NUM_HIDDEN_THOUGHT_TOKENS,
    thinking_temperature: float = THINKING_TEMPERATURE,
    thinking_do_sample: bool = THINKING_DO_SAMPLE,
    real_temperature: float = REAL_TOKEN_TEMPERATURE,
    real_do_sample: bool = REAL_TOKEN_DO_SAMPLE,
    show_hidden_thoughts: bool = SHOW_HIDDEN_THOUGHTS,
    entropy_threshold: float = None,
) -> str:
    """
    A second inference loop that actually uses the trained MixingHead,
    instead of always fully committing to the post-thought prediction like
    generate_with_hidden_deliberation does. Two things this adds:

      - The real token is sampled from a BLEND of the "no thought" and
        "post thought" predictions (mixed_hidden = (1-w)*before + w*after,
        projected through the model's own lm_head), where `w` is the
        trained mixing head's own judgment of how useful this particular
        thought was -- so a bad thought can be down-weighted instead of
        always fully trusted.
      - If `entropy_threshold` is set, thinking is skipped entirely when the
        base ("no thought") prediction is already confident (entropy below
        the threshold) -- a real compute saving, not just a quality change,
        since most tokens in ordinary text don't need deliberation.

    Cache eviction (Steps A/D) still happens exactly as in
    generate_with_hidden_deliberation whenever thinking does happen -- the
    mixing head changes what's predicted, not whether hidden tokens persist.
    """
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    last_logits = out.logits[:, -1, :]
    last_hidden = out.hidden_states[-1][:, -1, :]

    print(prompt, end="", flush=True)
    generated_text = ""
    num_visible, num_thought = 0, 0

    for _ in range(max_visible_tokens):
        checkpoint_len = cache.get_seq_length()
        num_visible += 1

        base_probs = torch.softmax(last_logits, dim=-1)
        entropy = -(base_probs * torch.log(base_probs + 1e-10)).sum(dim=-1).item()
        should_think = entropy_threshold is None or entropy > entropy_threshold

        if not should_think:
            real_token = sample_token(last_logits, real_temperature, real_do_sample)
            real_token_id = real_token.item()
            if is_eos(real_token_id, tokenizer, model):
                break
            if show_hidden_thoughts:
                print(f"\033[2m\u27ea(skipped, entropy={entropy:.2f})\u27eb\033[0m", end="", flush=True)
            cache, last_logits, last_hidden = forward_step_with_hidden(model, real_token, cache)
            token_text = tokenizer.decode([real_token_id], skip_special_tokens=True)
            print(token_text, end="", flush=True)
            generated_text += token_text
            continue

        num_thought += 1
        hidden_ids = []
        logits = last_logits
        post_thought_hidden = last_hidden
        for _ in range(num_hidden_tokens):
            hidden_token = sample_token(logits, thinking_temperature, thinking_do_sample)
            cache, logits, post_thought_hidden = forward_step_with_hidden(model, hidden_token, cache)
            hidden_ids.append(hidden_token.item())

        w = mixing_head(last_hidden, post_thought_hidden)
        mixed_hidden = (1 - w) * last_hidden + w * post_thought_hidden
        mixed_logits = model.lm_head(mixed_hidden)
        real_token = sample_token(mixed_logits, real_temperature, real_do_sample)
        real_token_id = real_token.item()

        if is_eos(real_token_id, tokenizer, model):
            break

        if show_hidden_thoughts:
            thought_text = tokenizer.decode(hidden_ids, skip_special_tokens=True)
            print(f"\033[2m\u27ea{thought_text}, w={w.item():.2f}\u27eb\033[0m", end="", flush=True)

        cache.crop(checkpoint_len)
        assert cache.get_seq_length() == checkpoint_len, "eviction did not land exactly on the checkpoint"

        cache, last_logits, last_hidden = forward_step_with_hidden(model, real_token, cache)
        token_text = tokenizer.decode([real_token_id], skip_special_tokens=True)
        print(token_text, end="", flush=True)
        generated_text += token_text

    print()
    if entropy_threshold is not None:
        print(f"[adaptive: thought on {num_thought}/{num_visible} visible tokens]")
    return generated_text
