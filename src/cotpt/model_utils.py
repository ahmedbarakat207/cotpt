import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_ID, START_THOUGHT_TOKEN, END_THOUGHT_TOKEN


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_thought_token_ids(tokenizer, start_token: str = START_THOUGHT_TOKEN, end_token: str = END_THOUGHT_TOKEN):
    """Return (start_id, end_id) if both thought tokens exist in the tokenizer, else (None, None)."""
    start_ids = tokenizer.encode(start_token, add_special_tokens=False) if hasattr(tokenizer, "encode") else None
    end_ids = tokenizer.encode(end_token, add_special_tokens=False) if hasattr(tokenizer, "encode") else None
    # Mock tokenizers in tests may not implement encode; fall back to convert_tokens_to_ids.
    if start_ids is None or end_ids is None:
        try:
            s = tokenizer.convert_tokens_to_ids(start_token)
            e = tokenizer.convert_tokens_to_ids(end_token)
            if s is None or e is None:
                return None, None
            return s, e
        except Exception:
            return None, None
    if len(start_ids) == 1 and len(end_ids) == 1:
        return start_ids[0], end_ids[0]
    return None, None


def ensure_thought_tokens(
    tokenizer,
    model,
    start_token: str = START_THOUGHT_TOKEN,
    end_token: str = END_THOUGHT_TOKEN,
):
    """Add start/end thought tokens if missing, resize embeddings, init sensibly.

    Quiet-STaR inits both to the em-dash embedding. We do the same when an
    em-dash token exists, else fall back to the mean of existing embeddings.
    Returns (start_id, end_id).
    """
    existing, _ = get_thought_token_ids(tokenizer, start_token, end_token)
    if existing is not None:
        # Already present; still ensure model vocab matches.
        try:
            if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
                model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass
        return get_thought_token_ids(tokenizer, start_token, end_token)

    # HF tokenizers: add as special tokens so they are not split.
    try:
        num_added = tokenizer.add_special_tokens({"additional_special_tokens": [start_token, end_token]})
    except Exception:
        # Minimal mock tokenizers used in tests.
        if hasattr(tokenizer, "add_special_tokens"):
            num_added = tokenizer.add_special_tokens({"additional_special_tokens": [start_token, end_token]})
        else:
            return None, None
    if num_added > 0:
        try:
            model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass
        # Init new rows to em-dash embedding when available, else mean.
        try:
            with torch.no_grad():
                emb = model.get_input_embeddings().weight
                vocab_size = emb.shape[0]
                new_ids = list(range(vocab_size - num_added, vocab_size))
                init_vec = None
                try:
                    dash_ids = tokenizer.encode("—", add_special_tokens=False)
                    if len(dash_ids) == 1:
                        init_vec = emb[dash_ids[0]].clone()
                except Exception:
                    init_vec = None
                if init_vec is None:
                    init_vec = emb[:-num_added].mean(dim=0)
                for nid in new_ids:
                    emb[nid].copy_(init_vec)
        except Exception:
            pass
    return get_thought_token_ids(tokenizer, start_token, end_token)


def load_model_and_tokenizer(
    model_id: str,
    device: str,
    use_thought_tokens: bool = False,
    start_thought_token: str = START_THOUGHT_TOKEN,
    end_thought_token: str = END_THOUGHT_TOKEN,
):
    if os.path.isdir(model_id) and os.path.exists(os.path.join(model_id, "adapter_config.json")):
        from peft import PeftModel
        with open(os.path.join(model_id, "adapter_config.json")) as f:
            cfg = json.load(f)
        base_id = cfg.get("base_model_name_or_path") or MODEL_ID
        tok_id = model_id if os.path.exists(os.path.join(model_id, "tokenizer_config.json")) else base_id
        tokenizer = AutoTokenizer.from_pretrained(tok_id)
        base_model = AutoModelForCausalLM.from_pretrained(base_id, dtype="auto").to(device)
        if use_thought_tokens:
            ensure_thought_tokens(tokenizer, base_model, start_thought_token, end_thought_token)
        model = PeftModel.from_pretrained(base_model, model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto").to(device)
        if use_thought_tokens:
            ensure_thought_tokens(tokenizer, model, start_thought_token, end_thought_token)
    model.eval()
    return model, tokenizer


def sample_token(logits: torch.Tensor, temperature: float, do_sample: bool) -> torch.Tensor:
    if not do_sample or temperature <= 0:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def is_eos(token_id: int, tokenizer, model) -> bool:
    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        ids = tokenizer.eos_token_id
        eos_ids.update(ids if isinstance(ids, list) else [ids])
    gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if gen_eos is not None:
        eos_ids.update(gen_eos if isinstance(gen_eos, list) else [gen_eos])
    return token_id in eos_ids
