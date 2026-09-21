import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .mixing_head import MixingHead
from .value_head import ValueHead


def is_lora_checkpoint(checkpoint_dir: str) -> bool:
    return os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json"))


def save_checkpoint(
    output_dir,
    model,
    tokenizer,
    mixing_head,
    value_head=None,
    optimizer=None,
    step: int = 0,
    extra_meta: dict = None,
):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save(mixing_head.state_dict(), os.path.join(output_dir, "mixing_head.pt"))
    if value_head is not None:
        torch.save(value_head.state_dict(), os.path.join(output_dir, "value_head.pt"))
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    meta = {"step": step}
    if extra_meta:
        meta.update(extra_meta)
    with open(os.path.join(output_dir, "training_state.json"), "w") as f:
        json.dump(meta, f)


def load_checkpoint_for_resume(
    checkpoint_dir: str,
    base_model_id: str,
    device: str,
    use_value_head: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    if is_lora_checkpoint(checkpoint_dir):
        from peft import PeftModel
        base_model = AutoModelForCausalLM.from_pretrained(base_model_id, dtype="auto").to(device)
        try:
            # If tokenizer was extended with thought tokens, resize base vocab first.
            if len(tokenizer) != base_model.get_input_embeddings().weight.shape[0]:
                base_model.resize_token_embeddings(len(tokenizer))
        except Exception:
            pass
        model = PeftModel.from_pretrained(base_model, checkpoint_dir, is_trainable=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, dtype="auto").to(device)
    try:
        if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))
    except Exception:
        pass
    model.train()
    hidden_size = model.config.hidden_size

    mixing_head = MixingHead(hidden_size).to(device)
    mixing_head_path = os.path.join(checkpoint_dir, "mixing_head.pt")
    if os.path.exists(mixing_head_path):
        mixing_head.load_state_dict(torch.load(mixing_head_path, map_location=device))

    value_head = None
    if use_value_head:
        value_head = ValueHead(hidden_size).to(device)
        value_head_path = os.path.join(checkpoint_dir, "value_head.pt")
        if os.path.exists(value_head_path):
            value_head.load_state_dict(torch.load(value_head_path, map_location=device))

    optimizer_state_dict = None
    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
    if os.path.exists(optimizer_path):
        optimizer_state_dict = torch.load(optimizer_path, map_location=device)

    step = 0
    meta_path = os.path.join(checkpoint_dir, "training_state.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            step = json.load(f).get("step", 0)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "mixing_head": mixing_head,
        "value_head": value_head,
        "optimizer_state_dict": optimizer_state_dict,
        "step": step,
    }
