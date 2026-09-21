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
    MIXING_MODE,
)
from .mixing_head import blend_logits
from .model_utils import sample_token, is_eos


def forward_step(model, token_id: torch.Tensor, cache: DynamicCache):
    with torch.no_grad():
        out = model(input_ids=token_id.view(1, 1), past_key_values=cache, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :]


def forward_step_with_hidden(model, token_id: torch.Tensor, cache: DynamicCache):
    with torch.no_grad():
        out = model(input_ids=token_id.view(1, 1), past_key_values=cache, use_cache=True, output_hidden_states=True)
    return out.past_key_values, out.logits[:, -1, :], out.hidden_states[-1][:, -1, :]


def evict_hidden_tokens(cache: DynamicCache, checkpoint_len: int) -> None:
    """Evict hidden-thought tokens added after ``checkpoint_len``.

    Uses the negative ``crop(-n)`` form ("remove n tokens") instead of the
    deprecated positive ``crop(length)`` form, which logs a warning on every
    step in current transformers and will be removed in 5.18.
    """
    to_evict = cache.get_seq_length() - checkpoint_len
    if to_evict > 0:
        cache.crop(-to_evict)


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
    print_prompt: bool = True,
    use_thought_tokens: bool = False,
    start_thought_id: int = None,
    end_thought_id: int = None,
) -> str:
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    last_logits = out.logits[:, -1, :]

    if print_prompt:
        print(prompt, end="", flush=True)
    generated_text = ""
    bracket = use_thought_tokens and num_hidden_tokens > 0 and start_thought_id is not None and end_thought_id is not None

    for _ in range(max_visible_tokens):
        checkpoint_len = cache.get_seq_length()

        hidden_ids = []
        logits = last_logits
        if bracket:
            start_tok = torch.tensor(start_thought_id, device=device)
            cache, logits = forward_step(model, start_tok, cache)
        for _ in range(num_hidden_tokens):
            hidden_token = sample_token(logits, thinking_temperature, thinking_do_sample)
            cache, logits = forward_step(model, hidden_token, cache)
            hidden_ids.append(hidden_token.item())
        if bracket:
            end_tok = torch.tensor(end_thought_id, device=device)
            cache, logits = forward_step(model, end_tok, cache)

        real_token = sample_token(logits, real_temperature, real_do_sample)
        real_token_id = real_token.item()

        if is_eos(real_token_id, tokenizer, model):
            break

        if show_hidden_thoughts:
            thought_text = tokenizer.decode(hidden_ids, skip_special_tokens=True)
            print(f"\033[2m\u27ea{thought_text}\u27eb\033[0m", end="", flush=True)

        evict_hidden_tokens(cache, checkpoint_len)
        assert cache.get_seq_length() == checkpoint_len

        cache, last_logits = forward_step(model, real_token, cache)
        token_text = tokenizer.decode([real_token_id], skip_special_tokens=True)
        print(token_text, end="", flush=True)
        generated_text += token_text

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
    print_prompt: bool = True,
    mixing_mode: str = MIXING_MODE,
    use_thought_tokens: bool = False,
    start_thought_id: int = None,
    end_thought_id: int = None,
) -> str:
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    last_logits = out.logits[:, -1, :]
    last_hidden = out.hidden_states[-1][:, -1, :]

    if print_prompt:
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
        device = last_logits.device
        bracket = use_thought_tokens and num_hidden_tokens > 0 and start_thought_id is not None and end_thought_id is not None
        if bracket:
            start_tok = torch.tensor(start_thought_id, device=device)
            cache, logits, post_thought_hidden = forward_step_with_hidden(model, start_tok, cache)
        for _ in range(num_hidden_tokens):
            hidden_token = sample_token(logits, thinking_temperature, thinking_do_sample)
            cache, logits, post_thought_hidden = forward_step_with_hidden(model, hidden_token, cache)
            hidden_ids.append(hidden_token.item())
        if bracket:
            end_tok = torch.tensor(end_thought_id, device=device)
            cache, logits, post_thought_hidden = forward_step_with_hidden(model, end_tok, cache)

        w = mixing_head(last_hidden, post_thought_hidden)
        if mixing_mode == "logit":
            mixed_logits = blend_logits(w, last_logits, logits)
        else:
            mixed_hidden = (1 - w) * last_hidden + w * post_thought_hidden
            mixed_logits = model.lm_head(mixed_hidden)
        real_token = sample_token(mixed_logits, real_temperature, real_do_sample)
        real_token_id = real_token.item()

        if is_eos(real_token_id, tokenizer, model):
            break

        if show_hidden_thoughts:
            thought_text = tokenizer.decode(hidden_ids, skip_special_tokens=True)
            print(f"\033[2m\u27ea{thought_text}, w={w.item():.2f}\u27eb\033[0m", end="", flush=True)

        evict_hidden_tokens(cache, checkpoint_len)
        assert cache.get_seq_length() == checkpoint_len

        cache, last_logits, last_hidden = forward_step_with_hidden(model, real_token, cache)
        token_text = tokenizer.decode([real_token_id], skip_special_tokens=True)
        print(token_text, end="", flush=True)
        generated_text += token_text

    print()
    if entropy_threshold is not None:
        print(f"[adaptive: thought on {num_thought}/{num_visible} visible tokens]")
    return generated_text
