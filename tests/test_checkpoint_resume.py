import json
import os

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model

from cotpt.mixing_head import MixingHead
from cotpt.value_head import ValueHead
from cotpt.checkpoint_utils import save_checkpoint, load_checkpoint_for_resume
import cotpt.checkpoint_utils as ckpt_mod


class DummyTokenizerForSaveLoad:
    """Real tokenizers already have working save/load; this stands in for
    one so the test doesn't need network access to a real tokenizer."""

    def save_pretrained(self, path):
        with open(os.path.join(path, "dummy_tokenizer.json"), "w") as f:
            json.dump({"ok": True}, f)

    @classmethod
    def from_pretrained(cls, path):
        assert os.path.exists(os.path.join(path, "dummy_tokenizer.json"))
        return cls()


def _make_config():
    return Qwen3Config(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128)


def test_plain_checkpoint_roundtrip(tmp_path):
    ckpt_dir = str(tmp_path / "plain_ckpt")
    torch.manual_seed(0)
    config = _make_config()
    model = Qwen3ForCausalLM(config)
    model.train()
    mixing_head = MixingHead(config.hidden_size)
    value_head = ValueHead(config.hidden_size)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": 1e-4},
        {"params": mixing_head.parameters(), "lr": 1e-3},
        {"params": value_head.parameters(), "lr": 1e-3},
    ])
    for _ in range(2):
        loss = model(input_ids=torch.randint(0, 100, (1, 10)), labels=torch.randint(0, 100, (1, 10))).loss
        loss = loss + mixing_head(torch.randn(1, 32), torch.randn(1, 32)).sum() + value_head(torch.randn(1, 32)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    save_checkpoint(ckpt_dir, model, DummyTokenizerForSaveLoad(), mixing_head, value_head, optimizer, step=17)

    orig_auto_tok = ckpt_mod.AutoTokenizer
    ckpt_mod.AutoTokenizer = DummyTokenizerForSaveLoad
    try:
        result = load_checkpoint_for_resume(ckpt_dir, base_model_id="unused-for-plain",
                                             device="cpu", use_value_head=True)
    finally:
        ckpt_mod.AutoTokenizer = orig_auto_tok

    orig_sd, loaded_sd = model.state_dict(), result["model"].state_dict()
    assert max((orig_sd[k] - loaded_sd[k]).abs().max().item() for k in orig_sd) == 0.0
    assert max((mixing_head.state_dict()[k] - result["mixing_head"].state_dict()[k]).abs().max().item()
               for k in mixing_head.state_dict()) == 0.0
    assert max((value_head.state_dict()[k] - result["value_head"].state_dict()[k]).abs().max().item()
               for k in value_head.state_dict()) == 0.0
    assert result["step"] == 17

    new_optimizer = torch.optim.AdamW([
        {"params": result["model"].parameters(), "lr": 1e-4},
        {"params": result["mixing_head"].parameters(), "lr": 1e-3},
        {"params": result["value_head"].parameters(), "lr": 1e-3},
    ])
    new_optimizer.load_state_dict(result["optimizer_state_dict"])  # should not raise


def test_lora_checkpoint_roundtrip(tmp_path):
    ckpt_dir = str(tmp_path / "lora_ckpt")
    torch.manual_seed(1)
    config = _make_config()
    base_model = Qwen3ForCausalLM(config)
    lora_cfg = LoraConfig(task_type="CAUSAL_LM",
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                           r=8, lora_alpha=16, lora_dropout=0.0)
    peft_model = get_peft_model(base_model, lora_cfg)
    peft_model.train()
    mixing_head = MixingHead(config.hidden_size)

    out = peft_model(input_ids=torch.randint(0, 100, (1, 10)), labels=torch.randint(0, 100, (1, 10)))
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=1e-2)
    out.loss.backward()
    opt.step()

    save_checkpoint(ckpt_dir, peft_model, DummyTokenizerForSaveLoad(), mixing_head, None, opt, step=3)

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, dtype=None):
            torch.manual_seed(1)  # reconstruct identical base weights, standing in for a real download
            return Qwen3ForCausalLM(_make_config())

    orig_auto_model, orig_auto_tok = ckpt_mod.AutoModelForCausalLM, ckpt_mod.AutoTokenizer
    ckpt_mod.AutoModelForCausalLM = _FakeAutoModel
    ckpt_mod.AutoTokenizer = DummyTokenizerForSaveLoad
    try:
        result = load_checkpoint_for_resume(ckpt_dir, base_model_id="fake-base",
                                             device="cpu", use_value_head=False)
    finally:
        ckpt_mod.AutoModelForCausalLM = orig_auto_model
        ckpt_mod.AutoTokenizer = orig_auto_tok

    assert hasattr(result["model"], "peft_config")
    with torch.no_grad():
        test_ids = torch.randint(0, 100, (1, 8))
        logits_orig = peft_model(input_ids=test_ids).logits
        logits_loaded = result["model"](input_ids=test_ids).logits
    assert (logits_orig - logits_loaded).abs().max().item() < 1e-5
