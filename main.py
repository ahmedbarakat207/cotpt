"""
Per-Token Hidden Deliberation with Aggressive KV-Cache Eviction
==================================================================

For every visible output token, the model:
  A. checkpoints its (clean) KV cache,
  B. secretly generates `NUM_HIDDEN_THOUGHT_TOKENS` "scratch" tokens, letting
     each one attend to the ones before it (normal autoregressive KV growth),
  C. uses the logits produced after the last scratch token to pick ONE real
     visible token,
  D. evicts the scratch tokens from the cache -- rolling it back to exactly
     the state it was in at (A), byte-for-byte,
  E. commits only the single real token to the now-clean cache and prints it,
  F. repeats.

Net effect: the model gets `NUM_HIDDEN_THOUGHT_TOKENS` steps of private
"thinking" per output token, but future tokens can never attend to that
thinking -- it's computed and then wiped, every single step. This is inspired
by Quiet-STaR-style rationale tokens, but where Quiet-STaR mixes hidden-state
representations of the rationale into the next-token prediction, this
prototype evicts the rationale from the KV cache entirely and relies only on
whatever the *last* hidden token's forward pass encoded into its output
logits.

REQUIREMENTS
    pip install torch "transformers>=4.51.0"
    (developed & verified against transformers 5.14 -- see NOTES at the
    bottom of this file if you're on an older version)

A NOTE ON THE TARGET MODEL
    "Qwen/Qwen3.5-0.8B" is a real, very recently released model, but it is a
    *hybrid* architecture (24 layers arranged as 6 x (3x Gated-DeltaNet -> 1x
    Attention)). Only the Attention layers keep a normal per-token KV cache
    that can be sliced along the sequence dimension the way this script does
    (and the way you described). The DeltaNet layers instead keep a
    fixed-size *recurrent state* that has already irreversibly mixed in every
    past token -- there is no "last 5 tokens" to slice out of it, only a
    "restore the whole state from before" operation. Evicting hidden
    computation from a hybrid model like that is a genuinely different (and
    more involved) problem than the one this script solves.

    So this script targets "Qwen/Qwen3-0.6B" instead: a small, dense,
    ordinary GQA + RoPE transformer, currently real and available, where
    "slice the KV cache along dim=-2" is exactly correct for every layer. It
    is a very close match to your "or equivalent ultra-lightweight causal LM"
    fallback. Swap MODEL_ID below if you want a different dense model --
    anything using a standard `DynamicCache` will work unmodified. If you
    specifically want to fight with Qwen3.5's hybrid cache, that's a
    reasonable follow-up but needs its own eviction path for the DeltaNet
    layers; ask and I'll build that version separately.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# ============================== CONFIG ==================================
# Tweak these freely.

MODEL_ID = "Qwen/Qwen3-0.6B"

NUM_HIDDEN_THOUGHT_TOKENS = 5      # how many scratch tokens per visible token
MAX_VISIBLE_TOKENS = 80            # generation length cap

THINKING_TEMPERATURE = 0.8         # sampling temperature during hidden thought
THINKING_DO_SAMPLE = True          # False = greedy scratch tokens (less useful)

REAL_TOKEN_TEMPERATURE = 0.7       # sampling temperature for the visible token
REAL_TOKEN_DO_SAMPLE = False       # False = greedy (deterministic, more coherent)

SHOW_HIDDEN_THOUGHTS = True        # print each hidden thought (dimmed) for
                                    # debugging; set False for the "real" hidden
                                    # behavior this paradigm is meant to have

SEED = 42


# ============================== CORE HELPERS =============================

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto").to(device)
    model.eval()
    return model, tokenizer


def sample_token(logits: torch.Tensor, temperature: float, do_sample: bool) -> torch.Tensor:
    """logits: [1, vocab_size] -> a [1]-shaped LongTensor token id."""
    if not do_sample or temperature <= 0:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def is_eos(token_id: int, tokenizer, model) -> bool:
    """Handles models (like most Qwen chat variants) that define multiple
    stop tokens via generation_config in addition to tokenizer.eos_token_id.
    """
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        ids = tokenizer.eos_token_id
        eos_ids.update(ids if isinstance(ids, list) else [ids])
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if gen_eos is not None:
        eos_ids.update(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    return token_id in eos_ids


def forward_step(model, token_id: torch.Tensor, cache: DynamicCache):
    """
    Feed exactly ONE new token through the model, growing `cache` by one
    position. Returns (updated_cache, logits_for_the_position_right_after_it).

    Deliberately NOT passing position_ids / attention_mask: when omitted,
    the model derives position_ids from `cache.get_seq_length()`, and that
    length is read directly off the cache tensors' own shape (not a separate
    counter). So once we crop the cache in Step D, the very next forward call
    automatically computes the correct position -- no manual bookkeeping,
    and no way for it to drift out of sync. Verified empirically (see the
    bottom of this file for how to reproduce that check).
    """
    with torch.no_grad():
        out = model(input_ids=token_id.view(1, 1), past_key_values=cache, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :]


# ========================= THE DELIBERATION LOOP ==========================

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

    # Prime the cache with the full prompt in one shot. This is the "standard
    # forward pass" that establishes the initial past_key_values (Step A, for
    # the very first visible token only -- on later iterations, Step A is
    # free: the cache is already sitting at the clean state we'd checkpoint).
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(input_ids=prompt_ids, past_key_values=cache, use_cache=True)
    cache = out.past_key_values
    last_logits = out.logits[:, -1, :]

    print(prompt, end="", flush=True)
    generated_text = ""

    for _ in range(max_visible_tokens):
        # ------------------------- STEP A: Checkpoint -------------------------
        # The cache currently holds only committed, real tokens. Remember its
        # length so Step D can roll back to exactly this point.
        checkpoint_len = cache.get_seq_length()

        # -------------------- STEP B: Hidden thought generation ---------------
        # Ordinary autoregressive generation for `num_hidden_tokens` steps:
        # each hidden token attends to the real context AND every hidden
        # token generated before it in this same burst, so the model can
        # build on its own scratch reasoning.
        hidden_ids = []
        logits = last_logits
        for _ in range(num_hidden_tokens):
            hidden_token = sample_token(logits, thinking_temperature, thinking_do_sample)
            cache, logits = forward_step(model, hidden_token, cache)
            hidden_ids.append(hidden_token.item())
        # `logits` now holds the distribution for "what comes right after the
        # last hidden token" -- exactly what Step C asks us to use.

        # ----------------------- STEP C: Pick the real token -------------------
        real_token = sample_token(logits, real_temperature, real_do_sample)
        real_token_id = real_token.item()

        if is_eos(real_token_id, tokenizer, model):
            break

        if show_hidden_thoughts:
            thought_text = tokenizer.decode(hidden_ids, skip_special_tokens=True)
            print(f"\033[2m\u27ea{thought_text}\u27eb\033[0m", end="", flush=True)

        # ------------------------ STEP D: Evict the thoughts --------------------
        # Roll the cache back to exactly its pre-thinking length. Under the
        # hood, `Cache.crop()` slices every layer's key/value tensors along
        # the sequence dimension: `keys = keys[..., :checkpoint_len, :]`
        # (dim=-2 of a [batch, num_heads, seq_len, head_dim] tensor), which
        # is precisely the "safe slice to roll back the cache size" this
        # was asked to demonstrate. See `crop_cache_manually` below for the
        # by-hand equivalent of what this call does.
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


def crop_cache_manually(cache: DynamicCache, target_length: int) -> None:
    """
    Not used above (cache.crop() already does this correctly and is the
    current, version-safe transformers API) -- included because you
    specifically asked to see the raw tensor slicing. This is what
    `cache.crop(target_length)` does internally, per layer:
    """
    for layer in cache.layers:
        layer.keys = layer.keys[..., :target_length, :]     # dim=-2 = seq_len
        layer.values = layer.values[..., :target_length, :]  # dim=-2 = seq_len
    # Nothing else to update: get_seq_length() reads keys.shape[-2] directly,
    # so there's no separate counter that could fall out of sync.


# ================================== MAIN ===================================

if __name__ == "__main__":
    torch.manual_seed(SEED)

    device = pick_device()
    print(f"Loading {MODEL_ID} on {device} ...")
    model, tokenizer = load_model_and_tokenizer(MODEL_ID, device)

    mock_prompt = (
        "Q: I am an odd number. Take away one letter and I become even. "
        "What number am I?\nA: Let's think step by step."
    )

    print("=" * 70)
    print(f"hidden thinking tokens / visible token : {NUM_HIDDEN_THOUGHT_TOKENS}")
    print(f"max visible tokens                     : {MAX_VISIBLE_TOKENS}")
    print(f"thinking temp / do_sample               : {THINKING_TEMPERATURE} / {THINKING_DO_SAMPLE}")
    print(f"real-token temp / do_sample             : {REAL_TOKEN_TEMPERATURE} / {REAL_TOKEN_DO_SAMPLE}")
    print("(dimmed text in \u27ea angle brackets\u27eb = hidden thoughts, shown here for")
    print(" debugging only -- set SHOW_HIDDEN_THOUGHTS = False to hide them)")
    print("=" * 70 + "\n")

    generate_with_hidden_deliberation(model, tokenizer, mock_prompt)

# ---------------------------------------------------------------------------
# NOTES
#
# * Version compatibility: this exact code (down to `cache.crop()` and the
#   `cache.layers[i].keys/.values` attribute names used in
#   `crop_cache_manually`) was run and passed the bit-exactness check
#   described above on both transformers 5.14.0 and 4.57.6. If you're on
#   something older and `cache.crop` doesn't exist or `cache.layers` raises
#   an AttributeError, `pip install -U transformers` first -- the Cache
#   object's internal layout has changed more than once, and I haven't
#   checked versions below 4.57.
#
# * To verify the eviction is exact on your machine: run the loop with
#   REAL_TOKEN_DO_SAMPLE = False and THINKING_DO_SAMPLE = False (fully
#   greedy), record the chosen real tokens, then separately run
#   `model(torch.cat([prompt_ids, torch.tensor([real_tokens])], dim=1))`
#   through a fresh cache and diff the two caches' `.layers[i].keys/values`
#   and final logits. They should match to float precision -- this is
#   exactly how this script was validated before being handed to you.
# ---------------------------------------------------------------------------
