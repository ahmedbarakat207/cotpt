import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODEL_ID


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(model_id: str, device: str):
    if os.path.isdir(model_id) and os.path.exists(os.path.join(model_id, "adapter_config.json")):
        from peft import PeftModel
        with open(os.path.join(model_id, "adapter_config.json")) as f:
            cfg = json.load(f)
        base_id = cfg.get("base_model_name_or_path") or MODEL_ID
        tok_id = model_id if os.path.exists(os.path.join(model_id, "tokenizer_config.json")) else base_id
        tokenizer = AutoTokenizer.from_pretrained(tok_id)
        base_model = AutoModelForCausalLM.from_pretrained(base_id, dtype="auto").to(device)
        model = PeftModel.from_pretrained(base_model, model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto").to(device)
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
