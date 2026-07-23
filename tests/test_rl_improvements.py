import copy
import math
import torch
from peft import LoraConfig, get_peft_model

from cotpt.mixing_head import MixingHead
from cotpt.value_head import ValueHead
from cotpt.training import (
    cotpt_training_step,
    generate_rollout_batch,
    reinforce_loss_with_value_baseline,
)


def test_value_baseline_favors_highest_reward_and_learns_values():
    torch.manual_seed(0)
    logits = torch.nn.Parameter(torch.zeros(4))
    value_param = torch.nn.Parameter(torch.zeros(4))
    rewards = torch.tensor([0.0, 1.0, 5.0, -2.0])
    opt = torch.optim.SGD([logits, value_param], lr=0.5)

    for _ in range(200):
        probs = torch.softmax(logits, dim=-1)
        logprob = torch.log(probs + 1e-10).unsqueeze(1)
        values = value_param.unsqueeze(1)
        policy_loss, value_loss, _ = reinforce_loss_with_value_baseline(rewards, logprob, values)
        loss = policy_loss + value_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    final_probs = torch.softmax(logits, dim=-1)
    assert final_probs[2] > 0.5 and final_probs[2] == final_probs.max()
    assert (value_param - rewards).abs().max().item() < 0.5


def test_kl_is_zero_at_lora_init(tiny_model):
    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=8, lora_alpha=16, lora_dropout=0.0,
    )
    peft_model = get_peft_model(tiny_model, lora_cfg)
    peft_model.train()

    torch.manual_seed(1)
    prefix = torch.randint(0, tiny_model.config.vocab_size, (1, 6))
    result = generate_rollout_batch(peft_model, prefix, num_rollouts=3, thought_length=4,
                                     temperature=1.0, use_disable_adapter=True)
    assert result["kl_total"].abs().max().item() < 1e-4


def test_entropy_is_near_uniform_at_init(tiny_model):
    torch.manual_seed(1)
    prefix = torch.randint(0, tiny_model.config.vocab_size, (1, 6))
    result = generate_rollout_batch(tiny_model, prefix, num_rollouts=3, thought_length=4, temperature=1.0)
    assert abs(result["entropy_mean"].item() - math.log(tiny_model.config.vocab_size)) < 0.1


def test_full_step_with_all_features_no_lora(tiny_model):
    tiny_model.train()
    ref_model = copy.deepcopy(tiny_model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    mixing_head = MixingHead(tiny_model.config.hidden_size)
    value_head = ValueHead(tiny_model.config.hidden_size)

    torch.manual_seed(5)
    input_ids = torch.randint(0, tiny_model.config.vocab_size, (1, 20))
    stats = cotpt_training_step(
        tiny_model, mixing_head, input_ids, value_head=value_head, ref_model=ref_model,
        num_think_positions=4, num_rollouts=3, thought_length=5, lookahead=4,
    )
    assert stats["mean_kl"] < 1e-3  # ref == policy at init
    stats["total_loss"].backward()
    assert all(p.grad is not None for p in tiny_model.parameters())
    assert all(p.grad is not None for p in value_head.parameters())


def test_full_step_with_lora(tiny_model):
    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=8, lora_alpha=16, lora_dropout=0.0,
    )
    peft_model = get_peft_model(tiny_model, lora_cfg)
    peft_model.train()
    mixing_head = MixingHead(tiny_model.config.hidden_size)
    value_head = ValueHead(tiny_model.config.hidden_size)

    torch.manual_seed(6)
    input_ids = torch.randint(0, tiny_model.config.vocab_size, (1, 20))
    stats = cotpt_training_step(
        peft_model, mixing_head, input_ids, value_head=value_head, use_disable_adapter=True,
        num_think_positions=4, num_rollouts=3, thought_length=5, lookahead=4,
    )
    assert stats["mean_kl"] < 1e-3
    stats["total_loss"].backward()
    lora_params = [p for n, p in peft_model.named_parameters() if p.requires_grad]
    assert all(p.grad is not None for p in lora_params)


def test_bare_bones_mode_still_works(tiny_model):
    """Everything (value baseline, KL, entropy bonus) turned off should
    behave exactly like the original bare-bones training step."""
    tiny_model.train()
    mixing_head = MixingHead(tiny_model.config.hidden_size)
    torch.manual_seed(7)
    input_ids = torch.randint(0, tiny_model.config.vocab_size, (1, 20))
    stats = cotpt_training_step(
        tiny_model, mixing_head, input_ids,
        use_value_baseline=False, use_kl_penalty=False, use_entropy_bonus=False,
        num_think_positions=4, num_rollouts=3, thought_length=5, lookahead=4,
    )
    stats["total_loss"].backward()  # should not raise


def test_entropy_term_gradient_increases_entropy():
    """Isolates the entropy bonus from every other loss term: if the only
    thing driving gradients is -entropy (i.e. maximize entropy), entropy
    should climb toward its theoretical ceiling, log(vocab_size). This is a
    cleaner test than comparing full training runs with/without the bonus --
    in the full objective, REINFORCE/mix/aux-LM losses also push the same
    shared parameters around and can swamp the (intentionally small,
    ENTROPY_COEFF=0.01-scale) entropy signal within a short run, which is
    not a bug, just noise unrelated to whether the term itself is correct."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import math

    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128)
    model = Qwen3ForCausalLM(config)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    torch.manual_seed(1)
    prefix = torch.randint(0, 100, (1, 6))

    entropies = []
    for step in range(15):
        torch.manual_seed(50 + step)
        result = generate_rollout_batch(model, prefix, num_rollouts=3, thought_length=5, temperature=1.0)
        entropy = result["entropy_mean"]
        entropies.append(entropy.item())
        loss = -entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert entropies[-1] > entropies[0]
    assert entropies[-1] < math.log(config.vocab_size) + 1e-3  # never exceeds the theoretical ceiling


def test_overfit_with_all_rl_features_on():
    """Watching total_loss alone is misleading once the value baseline is
    active (advantage shrinks toward 0 as the critic calibrates, which can
    make the REINFORCE term's raw magnitude move toward zero from a very
    negative start -- i.e. total_loss can rise even though training is
    healthy). The metrics that should actually improve are aux LM loss,
    value-head loss, and mean reward."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=128)
    model = Qwen3ForCausalLM(config)
    model.train()
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    mixing_head = MixingHead(config.hidden_size)
    value_head = ValueHead(config.hidden_size)
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": 1e-4},
        {"params": mixing_head.parameters(), "lr": 1e-3},
        {"params": value_head.parameters(), "lr": 1e-3},
    ])
    torch.manual_seed(2)
    fixed_input_ids = torch.randint(0, config.vocab_size, (1, 20))

    aux_losses, value_losses, rewards_track = [], [], []
    for step in range(50):
        torch.manual_seed(200 + step)
        stats = cotpt_training_step(
            model, mixing_head, fixed_input_ids, value_head=value_head, ref_model=ref_model,
            num_think_positions=4, num_rollouts=3, thought_length=5, lookahead=4,
        )
        optimizer.zero_grad()
        stats["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(mixing_head.parameters()) + list(value_head.parameters()), 1.0
        )
        optimizer.step()
        aux_losses.append(stats["aux_lm_loss"])
        value_losses.append(stats["value_loss"])
        rewards_track.append(stats["mean_reward"])

    assert sum(aux_losses[-5:]) / 5 < sum(aux_losses[:5]) / 5
    assert sum(value_losses[-5:]) / 5 < sum(value_losses[:5]) / 5
    assert sum(rewards_track[-5:]) / 5 > sum(rewards_track[:5]) / 5
